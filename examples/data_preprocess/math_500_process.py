# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from siirl.utils.extras.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def strip_left_right(latex_str: str) -> str:
    # 去掉所有的 \left 和 \right（保留后面的括号符号）
    latex_str = re.sub(r'\\left\s*', '', latex_str)
    latex_str = re.sub(r'\\right\s*', '', latex_str)
    
    return latex_str
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math_500")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "HuggingFaceH4/MATH-500"

    dataset = datasets.load_dataset(data_source, "default")
    

    test_dataset = dataset["test"]
    old_prompt_start = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    new_str = (
        'Remember to put your answer on its own line after "Answer:",and enclose it in \\boxed{...}.'
    )


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            question = old_prompt_start+ question_raw + " " + new_str

            answer_raw = example.pop("answer")
            solution = strip_left_right(answer_raw)
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": example["unique_id"],
                    "answer": answer_raw,
                    "question": question_raw,
                    "subject": example["subject"],
                    "level" : example["level"]
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
