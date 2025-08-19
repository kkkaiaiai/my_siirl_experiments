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


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/aime_2024")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "HuggingFaceH4/aime_2024"
    
    dataset = datasets.load_dataset(data_source, "default")

    test_dataset = dataset["train"]
    old_prompt_start = "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    new_str = (
        'Remember to put your answer on its own line after "Answer:",and enclose it in \\boxed{...}.'
    )


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # import pdb;pdb.set_trace()
            question_raw = example.pop("problem")

            # question = old_prompt_start+ question_raw + " " + new_str
            question = question_raw

            answer_raw = example.pop("answer")
            solution = answer_raw
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": example["id"],
                    "answer": answer_raw,
                    "question": question_raw,
                    "url": example["url"],
                    "year": example["year"],
                    "true_solution": example["solution"],
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
