import csv
import itertools
import re
import json
import jsonlines
import psutil
import ujson
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset

bos_token = "<s>"
eos_token = "</s>"


def pretrain_process(chunk_size=50000):
    chunk_idx = 0

    with jsonlines.open(f'{dataset_dir}/mobvoi_seq_monkey_general_open_corpus.jsonl') as reader:
        with open(f'{dataset_dir}/pretrain_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['text'])

            while True:
                chunk = list(itertools.islice(reader, chunk_size))
                if not chunk:
                    break

                for idx, obj in enumerate(chunk):
                    try:
                        content = obj.get('text', '')
                        if len(content) > 512:
                            continue
                        writer.writerow([content])
                    except UnicodeDecodeError as e:
                        print(f"Skipping invalid line {chunk_idx * chunk_size + idx + 1}: {e}")
                        continue
                chunk_idx += 1
                print('chunk:', ((chunk_idx - 1) * chunk_size, chunk_idx * chunk_size), 'process end')


def sft_process(contain_history=False):
    file_name = 'sft_data.csv'
    if not contain_history:
        file_name = 'sft_data_single.csv'

    def chinese_ratio(text):
        # 匹配所有中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 中文字符数量占比
        return len(chinese_chars) / len(text) if text else 0

    def process_and_write_data(data):
        q_lst, a_lst, history_lst = [], [], []
        for per in data:
            history, q, a = per['history'], per['q'], per['a']

            if (contain_history and not history) or not q or not a:
                continue
            if len(q) < 10 or len(a) < 5:
                continue
            if len(q) > 256 or len(a) > 256:
                continue
            # 判断q和a中中文字符占比是否超过70%
            if not (chinese_ratio(q) > 0.9 and chinese_ratio(a) > 0.9):
                continue

            q_lst.append(q)
            a_lst.append(a)
            if contain_history:
                history_lst.append(history)
            else:
                history_lst.append([])

        # 创建DataFrame并追加到CSV文件
        df = pd.DataFrame({'history': history_lst, 'q': q_lst, 'a': a_lst})
        df.to_csv(f'{dataset_dir}/{file_name}', mode='a', header=False, index=False, lineterminator='\r\n')

    chunk_size = 1000  # 每次处理的记录数
    data = []

    with open(f'{dataset_dir}/{file_name}', 'w', encoding='utf-8') as f:
        f.write('history,q,a\n')

    sft_datasets = [f'{dataset_dir}/sft_data_zh.jsonl']
    if not contain_history:
        sft_datasets = [f'{dataset_dir}/sft_data_zh.jsonl']

    for path in sft_datasets:
        with jsonlines.open(path) as reader:
            for idx, obj in enumerate(reader):
                try:
                    data.append({
                        'history': obj.get('history', ''),
                        'q': obj.get('input', '') + obj.get('q', ''),
                        'a': obj.get('output', '') + obj.get('a', '')
                    })

                    if len(data) >= chunk_size:
                        process_and_write_data(data)
                        data = []
                except jsonlines.InvalidLineError as e:
                    print(f"Skipping invalid JSON line {idx + 1}: {e}")
                    continue

            if data:
                process_and_write_data(data)
                data = []


def rl_process():
    ################
    # Dataset
    ################

    dataset_path = [f'{dataset_dir}/dpo/dpo_zh_demo.json',
                    f'{dataset_dir}/dpo/train_data.json',
                    f'{dataset_dir}/dpo/huozi_rlhf_data.json', ]

    train_dataset = load_dataset('json', data_files=dataset_path)

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["reject"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = train_dataset.map(
        process,
        load_from_cache_file=False,
    )

    output_dataset_path = f'{dataset_dir}/dpo/train_data.json'
    ds['train'].to_json(output_dataset_path, force_ascii=False, orient='records', lines=True)


if __name__ == "__main__":
    dataset_dir = "./datasets"
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer', use_fast=False)
    print('tokenizer词表大小：', len(tokenizer))

    ################
    # 1: pretrain
    # 2: sft
    # 3: RL
    ################
    process_type = 1

    if process_type == 1:
        pretrain_process()
    if process_type == 2:
        sft_process(contain_history=False)
    if process_type == 3:
        rl_process()
