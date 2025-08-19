import os
import time
import json
import random
import argparse
import ray
import re
import regex
import logging
from datasets import Dataset
import pandas as pd
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from tqdm import tqdm
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import torch


os.environ["TOKENIZERS_PARALLELISM"] = "false"

ds_collections = {
    "aime_2025": {
        "path": "/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/siiRL/data/aime_2025/test.parquet",
    },
    "MATH-500": {
        "path": "/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/siiRL/data/MATH-500/test.parquet",
    },
    "aime_2024": {
        "path": "/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/siiRL/data/aime_2024/test.parquet",
    },
}

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{{}}."

@ray.remote(num_gpus=1)
class LLMActor:
    def __init__(self, model_path, tensor_parallel_size, stop_token_ids):
        self.llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=tensor_parallel_size)
        self.stop_token_ids = stop_token_ids
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def generate(self, inputs, sampling_params):
        return self.llm.generate(inputs, sampling_params=sampling_params)


def _extract_boxed_answer(solution_str):
    match = regex.findall(
        r"(\\boxed\{(?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*\})", 
        solution_str, 
        regex.DOTALL
    )
    return match[-1][0].strip() if match else ""

def compute_score(solution_str: str, ground_truth: str) -> dict:
    pred = _extract_boxed_answer(solution_str)
    reward = 0.0
    if pred == ground_truth:
        reward = 1.0
    try:
        gt_parsed = parse(ground_truth, extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()])
        pred_parsed = parse(pred, extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()])
        if verify(gt_parsed, pred_parsed):
            reward = 1.0
    except Exception as e:
        logging.debug(f"Error in math_verify: {e}")
    return reward, pred

def verify_single_output(data_item, outputs):
    predictions = [out.text for out in outputs]
    ground_truth = data_item["reward_model"]["ground_truth"]

    preds_with_score = []
    acc_list = []
    for pred_text in predictions:
        score, pred_str = compute_score(pred_text, ground_truth)
        preds_with_score.append({
            "text": pred_text,
            "score": score,
            "pred": pred_str,
        })
        acc_list.append(score)

    data_item["responses"] = preds_with_score
    return data_item, acc_list

@ray.remote
def verify_task(data_item, outputs):
    return verify_single_output(data_item, outputs)

def main(args):
    ray.init()

    # 读取数据集
    for ds_name in args.datasets:
        df = pd.read_parquet(ds_collections[ds_name]["path"])
        data = Dataset.from_pandas(df)

        processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
        inputs = []
        for data_item in data:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data_item["prompt"][0]["content"]},
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs.append({"prompt": prompt})

        gpu_count = torch.cuda.device_count()
        llm_actors = [
            LLMActor.options(num_gpus=1).remote(args.checkpoint, args.tensor_parallel_size, None)
            for _ in range(gpu_count)
        ]

        sampling_params = SamplingParams(
            temperature=args.temperature,
            seed=args.seed,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_token,
            n=args.num_samples,
        )

        # 提交推理任务
        inference_futures = []
        for i, inp in enumerate(inputs):
            actor = llm_actors[i % gpu_count]
            inference_futures.append(actor.generate.remote([inp], sampling_params))

        # pass@k 计算准备
        ks = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] if k <= args.num_samples]
        pass_at_k = {k: 0 for k in ks}
        average_at_k = {k: 0 for k in ks}
        total_items = 0
        all_outputs = {}

        # 推理完成后直接提交验证
        verify_futures = []
        for i, inf_future in enumerate(inference_futures):
            outputs = ray.get(inf_future)  # 每个任务输出
            verify_futures.append(verify_task.remote(dict(data[i]), outputs[0].outputs))
        
        remaining_refs = list(verify_futures)
        pbar = tqdm(total=len(remaining_refs), desc="Verifying")
        while remaining_refs:
            ready, remaining_refs = ray.wait(remaining_refs, num_returns=1)
            ref = ready[0]
            try:
                data_item, acc_list = ray.get(ref)  # 拿到结果
                total_items += 1
                all_outputs[data_item["id"]] = data_item

                for k in ks:
                    pass_at_k[k] += any(acc_list[:k])
                    average_at_k[k] += sum(acc_list[:k]) / k

            except Exception as e:
                print(f"Error verifying sample (ref={ref}): {e}")

            pbar.update(1)

        pbar.close()

        # 归一化
        for k in ks:
            pass_at_k[k] /= total_items
            average_at_k[k] /= total_items

        print(f"Dataset {ds_name} pass@k:", pass_at_k)
        print(f"Dataset {ds_name} average@k:", average_at_k)

        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"{args.model_name}_{ds_name}_{time_prefix}_{args.num_samples}.json"
        output_path = os.path.join(args.out_dir, results_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=4, ensure_ascii=False)
        print("Results saved to {}".format(output_path))

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="qwen8b-base")
    parser.add_argument("--datasets", type=str, default="aime_2025")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=float, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_token", type=int, default=4090)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)

    main(args)