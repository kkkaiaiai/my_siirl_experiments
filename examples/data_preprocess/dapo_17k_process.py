# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""
Preprocess the DAPO-17k dataset:
1. 去重（按 content 去重）
2. 替换提示语为: Let's think step by step, and write the final answer in the format \boxed{...}.
3. 保存为 parquet 格式
"""

import argparse
import os
import re
from collections import Counter

import datasets

from siirl.utils.extras.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="./data/DAPO-17k-processed",
        help="本地保存路径",
    )
    parser.add_argument("--hdfs_dir", default=None, help="HDFS保存路径")
    parser.add_argument(
        "--data_source",
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        help="原始数据路径",
    )

    args = parser.parse_args()

    # 1. 加载数据
    dataset = datasets.load_dataset(args.data_source, "default")
    train_dataset = dataset["train"]

    # 2. 统计重复情况
    contents = [p[0]["content"] if p else "" for p in train_dataset["prompt"]]
    counter = Counter(contents)
    num_total = len(contents)
    num_unique = len(counter)
    print(f"总样本数: {num_total}")
    print(f"唯一题目数: {num_unique}")
    print(f"重复题目数: {num_total - num_unique}")
    print(f"重复比例: {(num_total - num_unique) / num_total:.2%}")

    # 3. 去重
    seen = set()
    unique_indices = []
    for i, c in enumerate(contents):
        if c not in seen:
            seen.add(c)
            unique_indices.append(i)
    dedup_dataset = train_dataset.select(unique_indices)
    print(f"去重后样本数: {len(dedup_dataset)}")

    # 4. 替换提示语
    # 旧的开头 prompt
    old_prompt_start = (
        "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    )

    # 新的开头 prompt
    new_prompt_start = (
        "Solve the following math problem step by step.\n"
        "Show all reasoning clearly.\n\n"
        "At the very end of your response, write the final answer in the format:\n\n"
        "Answer: \\boxed{FINAL_ANSWER}\n\n"
        "Where:\n"
        "- `Answer:` is written exactly like this (capital \"A\", colon, space).\n"
        "- The final answer is enclosed in `\\boxed{...}`.\n"
        "- Nothing else should appear after the closing `}`."
    )

    # 旧的结尾提示
    old_str = 'Remember to put your answer on its own line after "Answer:".'

    # 新的结尾提示（加 box 格式提醒）
    new_str = (
        'Remember to put your answer on its own line after "Answer:",and enclose it in \\boxed{...}.'
    )

    def replace_prompt(example, idx):
        new_prompt = []
        for p in example["prompt"]:
            if "content" in p:
                content = p["content"]
                # 开头替换
                content = content.replace(old_prompt_start, new_prompt_start)
                # 结尾替换
                content = content.replace(old_str, new_str)
                p = {**p, "content": content}
            new_prompt.append(p)

        return {
            "data_source": args.data_source,
            "prompt": new_prompt,
            "ability": example["ability"],
            "reward_model": example["reward_model"],
            "extra_info": {**example.get("extra_info", {}), "index": idx},
        }
    processed_dataset = dedup_dataset.map(
        replace_prompt, with_indices=True, num_proc=8  # 多进程加速
    )

    # 5. 保存为 parquet
    os.makedirs(args.local_dir, exist_ok=True)
    processed_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))

    # 6. 如果指定了 hdfs_dir，则上传
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(args.local_dir, args.hdfs_dir)

    print("处理完成！")
    print(f"本地保存路径: {args.local_dir}")
