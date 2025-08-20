import argparse
import json
import os
import random
import time
from tqdm import tqdm

from datasets import load_dataset, Dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import pandas as pd

import logging

import regex
import re
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def evaluate_chat_model_pipeline(args):
    random.seed(args.seed)

    for ds_name in args.datasets:
        # 1. 加载数据
        df = pd.read_parquet(ds_collections[ds_name]["path"])
        data = Dataset.from_pandas(df)

        # 2. 构造 prompts（主进程做，避免子进程 tokenizer 警告）
        inputs = []
        for data_item in tqdm(data, desc=f"Building prompts for {ds_name}"):
            # messages = [
            #     {"role": "system", "content": SYSTEM_PROMPT},
            #     {"role": "user", "content": data_item["prompt"][1]["content"]},
            # ]
            messages = data_item["prompt"]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs.append({"prompt": prompt})

        # 3. 统计指标初始化
        ks = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] if k <= args.num_samples]
        pass_at_k = {k: 0 for k in ks}
        average_at_k = {k: 0.0 for k in ks}
        total_items = 0
        all_outputs = {}

        # 4. 多进程验证
        executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        futures = []

        # 5. 批量推理 + 提交验证任务
        for start in tqdm(range(0, len(inputs), args.batch_size), desc="Inference batches"):
            end = min(start + args.batch_size, len(inputs))
            batch_inputs = inputs[start:end]

            sampling_params = SamplingParams(
                temperature=args.temperature,
                seed=args.seed,
                top_k=args.top_k,
                top_p=args.top_p,
                max_tokens=args.max_token,
                n=args.num_samples,
                stop_token_ids=stop_token_ids
            )

            model_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

            for i, model_output in enumerate(model_outputs, start=start):
                futures.append(
                    executor.submit(verify_single_output, dict(data[i]), model_output.outputs)
                )

        # 6. 收集结果并直接累计指标
        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
            try:
                data_item, acc_list = future.result()
                total_items += 1
                all_outputs[data_item["id"]] = data_item

                for k in ks:
                    pass_at_k[k] += any(acc_list[:k])
                    average_at_k[k] += sum(acc_list[:k]) / k

            except Exception as e:
                print(f"Error verifying sample: {e}")

        executor.shutdown(wait=True)

        # 7. 归一化
        for k in ks:
            pass_at_k[k] /= total_items
            average_at_k[k] /= total_items

        # 8. 打印结果
        print("\n=== Metrics ===")
        for k in ks:
            print(f"pass@{k}: {pass_at_k[k]:.4f}  average@{k}: {average_at_k[k]:.4f}")

        # 9. 保存结果
        result_data = {
            "metrics": {
                "pass_at_k": pass_at_k,
                "average_at_k": average_at_k
            },
            "outputs": all_outputs
        }
        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"{args.model_name}_{ds_name}_{time_prefix}_{args.num_samples}.json"
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(result_data, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        print(f"Results saved to {output_path}")

def _extract_boxed_answer(solution_str):
    """Extract boxed answer using advanced regex from math_verify_box.py."""
    match = regex.findall(
        r"(\\boxed\{(?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*\})", 
        solution_str, 
        re.DOTALL
    )
    return match[-1][0].strip() if match else ""

def compute_score(solution_str: str, ground_truth: str) -> dict:
    """
    Compute score using math_verify_box.py approach with advanced mathematical verification.
    """
    # Extract boxed answer using advanced regex
    pred = _extract_boxed_answer(solution_str)
    reward = 0.0
    
    if pred == ground_truth:
        reward = 1.0
    try:
        # Use math_verify for advanced mathematical comparison
        gt_parsed = parse(
            ground_truth,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
        pred_parsed = parse(
            pred,
            extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
        )
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

def evaluate_chat_model(args):
    random.seed(args.seed)

    for ds_name in args.datasets:
        # data = load_dataset(ds_collections[ds_name]["root"], cache_dir=os.path.join(os.getcwd(), "data/MMK12/"))[
        #     ds_collections[ds_name]["split"]
        # ]
        df = pd.read_parquet(ds_collections[ds_name]["path"])
        data = Dataset.from_pandas(df)
        # data = load_dataset("parquet", data_files=ds_collections[ds_name]["path"], split="train")

        inputs = []
        for data_item in tqdm(data):
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": data_item["prompt"][0]["content"],
                },
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs.append(
                {
                    "prompt": prompt,
                }
            )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            seed=args.seed,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_token,
            n=args.num_samples,
            stop_token_ids=stop_token_ids
        )
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        
        total, passed = 0, 0
        outputs = []
        for data_item, model_output in zip(data, model_outputs): 
            data_item, acc_list = verify_single_output(data_item, model_output.outputs)
            outputs.append(data_item)

        accuracy = passed / total if total > 0 else 0.0
        print(f"pass@{args.num_samples}: {accuracy}")

        temp = {item["id"]: item for item in outputs if item is not None}
        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"{args.model_name}_{ds_name}_{time_prefix}_{args.num_samples}.json"
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        print("Results saved to {}".format(output_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--model_name", type=str, default="qwen8b-base")
    parser.add_argument("--datasets", type=str, default="aime_2025")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=float, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_token", type=int, default=8192)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(",")

    print("datasets:", args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    evaluate_chat_model_pipeline(args)
