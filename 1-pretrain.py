import os
import platform
import argparse
import time
import math
import warnings
from queue import Queue

import pandas as pd
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer
from tqdm import tqdm

from model.model import Transformer
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        tqdm.write(content)


def get_lr(it, all):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate * args.lr_min_ratio

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    if args.lr_decay_ratio is None:
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    else:
        decay_ratio = args.lr_decay_ratio ** ((it - warmup_iters) / args.lr_decay_interval)
    assert 0 <= decay_ratio <= 1
    if args.lr_decay_ratio is None:
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)
    else:
        return max(min_lr, args.learning_rate * decay_ratio)


def train_epoch(epoch, wandb):
    start_time = time.time()
    t_bar = tqdm(total=len(train_loader), desc=f'Epoch:[{epoch}/{args.epochs}]')
    loss_recorded, loss_recorded_value, loss_count_max = Queue(), 0.0, args.accumulation_steps * args.batch_size
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            
            loss_recorded.put(loss.item() * args.accumulation_steps)
            loss_recorded_value += loss.item() * args.accumulation_steps
            if loss_recorded.qsize() > loss_count_max:
                loss_recorded_value -= loss_recorded.get()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
        
        log_loss = loss_recorded_value / max(loss_recorded.qsize(), 1)
        if log_loss >= 0.1:
            log_loss = f'{log_loss:.3f}'
        else:
            log_loss = f'{log_loss:.3e}'
        t_bar.update()
        t_bar.set_postfix_str(f'loss:{log_loss}')

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{} lr:{:.3e} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    log_loss,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

    model = Transformer(lm_config).to(args.device)
    if args.resume_path is not None and os.path.exists(args.resume_path):
        try:
            model.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        except:
            Logger(f'Load model weights from {args.resume_path} failed. Start from scratch.')
    # moe_path = '_moe' if lm_config.use_moe else ''

    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_data.csv", help="Path to training data")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio. Overrides warmup_iters. Recommended 0.01.")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    parser.add_argument('--resume_path', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.99, help='learning rate decay ratio')
    parser.add_argument('--lr_decay_interval', type=int, default=None, help='learning rate decay interval')
    parser.add_argument('--lr_min_ratio', type=float, default=0.1, help='minimum learning_rate ratio')

    args = parser.parse_args()
    
    if args.lr_decay_interval is None:
        args.lr_decay_interval = args.batch_size * args.accumulation_steps
    

    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model()
    df = pd.read_csv(args.data_path)
    df = df.sample(frac=1.0)
    train_ds = PretrainDataset(df, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    if args.warmup_ratio is not None:
        args.warmup_iters = int(args.epochs * len(train_loader) * args.warmup_ratio)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
