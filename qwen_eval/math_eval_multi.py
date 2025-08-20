#!/usr/bin/env python3
"""
Math Evaluation with Ray Data Parallelism
Based on math_eval.py but enhanced with Ray for distributed inference across multiple GPUs
"""

import random
random.seed(42)
import os
import argparse
import time
import json
import logging
import csv
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import ray
import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from libs.utils import myrefl_sys3, set_seed, load_jsonl, save_jsonl, construct_prompt
from libs.parser import *
from libs.trajectory import *
from libs.data_loader import load_data
from libs.python_executor import PythonExecutor
from libs.model_utils import load_hf_lm_and_tokenizer, generate_completions


def choice_answer_clean(code: str) -> str:
    """Extract choice answer from code."""
    # Simple implementation - extract A, B, C, D, E from code
    for choice in ["A", "B", "C", "D", "E"]:
        if choice in code:
            return choice
    return "A"  # Default fallback


def extract_answer(text: str, data_name: str) -> str:
    """Extract answer from text."""
    # This is a simplified version - in practice you'd use the full parser
    import re
    
    # Look for boxed answers
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for final answer patterns
    answer_patterns = [
        r'(?i)(?:the )?(?:final )?answer is:?\s*([^\n]+)',
        r'(?i)(?:therefore|thus|hence),?\s*(?:the )?(?:answer|result) is:?\s*([^\n]+)',
        r'(?i)answer:\s*([^\n]+)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    # Fallback: look for last number or expression
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    
    return text.strip()[:50]  # Return first 50 chars as fallback


def run_execute(executor: PythonExecutor, code: str, prompt_type: str, data_name: str) -> Tuple[str, str]:
    """Execute code and extract prediction."""
    if prompt_type in ["cot"]:
        # For CoT, extract answer directly from text
        pred = extract_answer(code, data_name)
        return pred, ""
    else:
        # For code-based approaches, execute the code
        try:
            result, report = executor.apply(code)
            if result:
                pred = extract_answer(result, data_name)
            else:
                pred = extract_answer(code, data_name)
            return pred, report
        except Exception as e:
            return extract_answer(code, data_name), str(e)


# compute_progressive_metrics function removed - now handled by evaluate() function


def is_answer_correct(pred: str, gold: str) -> bool:
    """
    检查预测答案是否正确
    使用更复杂的匹配逻辑
    """
    if pred is None or gold is None:
        return False
    
    pred_str = str(pred).strip()
    gold_str = str(gold).strip()
    
    # 精确匹配
    if pred_str == gold_str:
        return True
    
    # 去除空格后匹配
    if pred_str.replace(' ', '') == gold_str.replace(' ', ''):
        return True
    
    # 数值匹配（如果都是数字）
    try:
        pred_num = float(pred_str)
        gold_num = float(gold_str)
        return abs(pred_num - gold_num) < 1e-6
    except (ValueError, TypeError):
        pass
    
    # 提取数字进行匹配
    import re
    pred_numbers = re.findall(r'-?\d+\.?\d*', pred_str)
    gold_numbers = re.findall(r'-?\d+\.?\d*', gold_str)
    
    if pred_numbers and gold_numbers:
        try:
            return abs(float(pred_numbers[-1]) - float(gold_numbers[-1])) < 1e-6
        except (ValueError, IndexError):
            pass
    
    return False


def save_results_to_csv(model_name: str, dataset_results: Dict[str, Dict], csv_file: str = "math_eval_results.csv"):
    """
    保存结果到 CSV 文件（追加模式）
    
    Args:
        model_name: 模型名称
        dataset_results: 每个数据集的结果字典
        csv_file: CSV 文件路径
    """
    # 准备数据行
    row_data = {"model": model_name}
    
    # 添加每个数据集的指标
    for dataset_name, metrics in dataset_results.items():
        if dataset_name == "avg":
            continue  # 跳过平均值，单独处理
        
        # 添加基础准确率
        if "acc" in metrics:
            acc_value = metrics["acc"]
            if isinstance(acc_value, (int, float)):
                row_data[f"{dataset_name}_acc"] = round(float(acc_value), 6)
        
        # 添加 pass_at_k 指标
        if "pass_at_k" in metrics:
            for k, value in metrics["pass_at_k"].items():
                if isinstance(value, (int, float)):
                    row_data[f"{dataset_name}_pass@{k}"] = round(float(value), 6)
        
        # 添加 average_at_k 指标
        if "average_at_k" in metrics:
            for k, value in metrics["average_at_k"].items():
                if isinstance(value, (int, float)):
                    row_data[f"{dataset_name}_average@{k}"] = round(float(value), 6)
    
    # 添加平均指标
    if "avg" in dataset_results:
        avg_metrics = dataset_results["avg"]
        
        # 添加平均准确率
        if "acc" in avg_metrics:
            acc_value = avg_metrics["acc"]
            if isinstance(acc_value, (int, float)):
                row_data["avg_acc"] = round(float(acc_value), 6)
        
        # 添加平均 pass_at_k 指标
        if "pass_at_k" in avg_metrics:
            for k, value in avg_metrics["pass_at_k"].items():
                if isinstance(value, (int, float)):
                    row_data[f"avg_pass@{k}"] = round(float(value), 6)
        
        # 添加平均 average_at_k 指标
        if "average_at_k" in avg_metrics:
            for k, value in avg_metrics["average_at_k"].items():
                if isinstance(value, (int, float)):
                    row_data[f"avg_average@{k}"] = round(float(value), 6)
    
    # 检查文件是否存在
    file_exists = os.path.exists(csv_file)
    
    # 如果文件存在，读取现有的表头以保持一致性
    existing_fieldnames = []
    if file_exists:
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_fieldnames = next(reader, [])
        except Exception as e:
            logging.warning(f"Could not read existing CSV headers: {e}")
    
    # 确定最终的字段名（保持顺序一致）
    if existing_fieldnames:
        # 使用现有的字段名，添加新的字段
        all_fieldnames = existing_fieldnames[:]
        for key in row_data.keys():
            if key not in all_fieldnames:
                all_fieldnames.append(key)
    else:
        # 新文件，使用固定顺序
        all_fieldnames = ["model"]
        
        # 按数据集名称排序添加字段
        dataset_names = sorted([name for name in dataset_results.keys() if name != "avg"])
        
        # 获取所有可用的 k 值
        all_k_values = set()
        for dataset_name in dataset_names:
            metrics = dataset_results[dataset_name]
            if "pass_at_k" in metrics:
                all_k_values.update(metrics["pass_at_k"].keys())
        all_k_values = sorted(all_k_values)
        
        for dataset_name in dataset_names:
            all_fieldnames.append(f"{dataset_name}_acc")
            for k in all_k_values:
                all_fieldnames.extend([f"{dataset_name}_pass@{k}", f"{dataset_name}_average@{k}"])
        
        # 添加平均指标
        all_fieldnames.append("avg_acc")
        for k in all_k_values:
            all_fieldnames.extend([f"avg_pass@{k}", f"avg_average@{k}"])
    
    # 写入 CSV
    try:
        with open(csv_file, 'w' if not file_exists else 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction='ignore')
            
            # 如果是新文件或重写，写入表头
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row_data)
        
        logging.info(f"Results saved to {csv_file}")
    except Exception as e:
        logging.error(f"Failed to save results to CSV: {e}")
        # 保存到备用文件
        backup_file = csv_file.replace('.csv', f'_backup_{int(time.time())}.csv')
        try:
            with open(backup_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
                writer.writeheader()
                writer.writerow(row_data)
            logging.info(f"Results saved to backup file: {backup_file}")
        except Exception as backup_e:
            logging.error(f"Failed to save backup file: {backup_e}")


def calculate_average_progressive_metrics(dataset_results: Dict[str, Dict]) -> Dict[str, float]:
    """计算所有数据集的平均指标"""
    all_metrics = {}
    dataset_count = 0
    
    for dataset_name, metrics in dataset_results.items():
        if dataset_name == "avg":
            continue
        
        dataset_count += 1
        
        # 处理基础准确率
        if "acc" in metrics:
            if "acc" not in all_metrics:
                all_metrics["acc"] = 0.0
            all_metrics["acc"] += metrics["acc"]
        
        # 处理 pass@k 和 pass@1 (保持向后兼容)
        if "pass@k" in metrics:
            if "pass@k" not in all_metrics:
                all_metrics["pass@k"] = 0.0
            all_metrics["pass@k"] += metrics["pass@k"]
        
        if "pass@1" in metrics:
            if "pass@1" not in all_metrics:
                all_metrics["pass@1"] = 0.0
            all_metrics["pass@1"] += metrics["pass@1"]
        
        # 处理新的 pass_at_k 指标
        if "pass_at_k" in metrics:
            if "pass_at_k" not in all_metrics:
                all_metrics["pass_at_k"] = {}
            for k, value in metrics["pass_at_k"].items():
                if k not in all_metrics["pass_at_k"]:
                    all_metrics["pass_at_k"][k] = 0.0
                all_metrics["pass_at_k"][k] += value
        
        # 处理新的 average_at_k 指标
        if "average_at_k" in metrics:
            if "average_at_k" not in all_metrics:
                all_metrics["average_at_k"] = {}
            for k, value in metrics["average_at_k"].items():
                if k not in all_metrics["average_at_k"]:
                    all_metrics["average_at_k"][k] = 0.0
                all_metrics["average_at_k"][k] += value
    
    # 计算平均值
    if dataset_count > 0:
        # 基础指标平均值
        for metric_name in ["acc", "pass@k", "pass@1"]:
            if metric_name in all_metrics:
                all_metrics[metric_name] /= dataset_count
        
        # pass_at_k 平均值
        if "pass_at_k" in all_metrics:
            for k in all_metrics["pass_at_k"]:
                all_metrics["pass_at_k"][k] /= dataset_count
        
        # average_at_k 平均值
        if "average_at_k" in all_metrics:
            for k in all_metrics["average_at_k"]:
                all_metrics["average_at_k"][k] /= dataset_count
    
    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Math Evaluation with Ray Data Parallelism")
    
    # Original arguments from math_eval.py
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=0.85, type=float)
    parser.add_argument("--max_tokens_per_call", default=80000, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--max_func_call", type=int, default=1)
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template to prompt.")
    
    # Ray-specific arguments
    parser.add_argument("--num_workers", type=int, default=None, 
                       help="Number of Ray workers (default: number of GPUs)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size per worker")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                       help="Pipeline parallel size (deprecated in multi-worker setup)")
    parser.add_argument("--gpus_per_worker", type=int, default=1,
                       help="Number of GPUs per Ray worker")
    parser.add_argument("--ray_init_timeout", type=int, default=600,
                       help="Ray initialization timeout in seconds")
    parser.add_argument("--batch_size_per_worker", type=int, default=None,
                       help="Batch size per worker (auto-calculated if not provided)")
    
    # Few-shot and adaptation
    parser.add_argument("--adapt_few_shot", action="store_true",
                       help="Few shot for multiple-choice questions, zero shot for others.")
    
    # Progressive evaluation is now handled automatically by evaluate() function
    parser.add_argument("--csv_output", type=str, default="math_eval_results.csv",
                       help="CSV file path for saving results")
    
    # Debug arguments
    parser.add_argument("--debug_print_samples", action="store_true",
                       help="Print generated code and predictions for debugging")
    
    args = parser.parse_args()
    args.top_p = (1 if args.temperature == 0 else args.top_p)  # top_p must be 1 when using greedy sampling (vllm)
    
    # Set default number of workers to number of available GPUs
    if args.num_workers is None:
        args.num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
    return args


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'math_eval_multi_{time.strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )


@ray.remote(num_gpus=1)
class MathEvalWorker:
    """Ray remote worker for distributed math evaluation."""
    
    def __init__(self, model_name_or_path: str, tensor_parallel_size: int = 1, 
                 worker_id: int = 0, args_dict: Dict = None):
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_id = worker_id
        self.args = argparse.Namespace(**args_dict) if args_dict else None
        
        logging.info(f"Worker {worker_id}: Initializing with model {model_name_or_path}")
        
        # Initialize model
        try:
            self.llm = LLM(
                model=model_name_or_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                gpu_memory_utilization=0.85,  # Leave some memory for other operations
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            logging.info(f"Worker {worker_id}: Model initialized successfully")
        except Exception as e:
            logging.error(f"Worker {worker_id}: Failed to initialize model: {e}")
            raise
        
        # Initialize Python executor
        if self.args and "pal" in self.args.prompt_type:
            self.executor = PythonExecutor(get_answer_expr="solution()")
        else:
            self.executor = PythonExecutor(get_answer_from_stdout=True)
    
    def process_samples(self, samples: List[Dict], data_name: str, worker_args: Dict = None) -> List[Dict]:
        """Process a batch of samples with specified sampling count."""
        if not samples:
            return []
        
        # Use worker-specific args if provided, otherwise use default
        if worker_args:
            n_sampling = worker_args.get('n_sampling', self.args.n_sampling)
        else:
            n_sampling = self.args.n_sampling
        
        logging.info(f"Worker {self.worker_id}: Processing {len(samples)} samples with {n_sampling} sampling each")
        
        # Prepare prompts
        input_prompts = []
        for sample in samples:
            for _ in range(n_sampling):
                input_prompts.append(sample["prompt"])
        
        # Generate responses
        results = self._generate_responses(input_prompts, data_name, n_sampling)
        
        # Process results back to samples
        processed_samples = []
        for i, sample in enumerate(samples):
            start_idx = i * n_sampling
            end_idx = (i + 1) * n_sampling
            
            sample_results = results[start_idx:end_idx]
            codes = [r["code"] for r in sample_results]
            preds = [r["pred"] for r in sample_results]
            reports = [r["report"] for r in sample_results]
            
            # Handle multi-choice questions
            for j in range(len(preds)):
                if sample["gold"] in ["A", "B", "C", "D", "E"] and preds[j] not in ["A", "B", "C", "D", "E"]:
                    preds[j] = choice_answer_clean(codes[j])
                elif self._is_multi_choice(sample["gold"]) and not self._is_multi_choice(preds[j]):
                    if preds[j] is not None:
                        preds[j] = "".join([c for c in preds[j] if c in ["A", "B", "C", "D", "E"]])
            
            # Create processed sample
            processed_sample = sample.copy()
            processed_sample.pop("prompt", None)
            processed_sample.update({
                "code": codes,
                "pred": preds,
                "report": reports,
                "worker_id": self.worker_id,
                "worker_sampling": n_sampling
            })
            processed_samples.append(processed_sample)
        
        logging.info(f"Worker {self.worker_id}: Completed processing {len(processed_samples)} samples")
        return processed_samples
    
    def _generate_responses(self, input_prompts: List[str], data_name: str, n_sampling: int = None) -> List[Dict]:
        """Generate responses for input prompts."""
        # Determine stop words based on prompt type
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        code_prompt_type = ['numina', "qwen25-math-autocode", "pal", "tool-integrated", 
                           "qwen25-math-pot", "jiuzhang_tora", "pot_trigger", "below_pot_trigger",
                           "ds_pot_trigger", "mix_trigger", "cot_prompt", "below_cot_prompt"]
        
        if self.args.prompt_type in ["cot"]:
            stop_words.append("\n\nQuestion:")
        if self.args.prompt_type in code_prompt_type:
            stop_words.extend(["\n\n---", "```output"])
        elif self.args.prompt_type in ["wizard_zs", "platypus_fs"]:
            stop_words.extend(["Instruction", "Response"])
        elif "jiuzhang" in self.args.prompt_type:
            stop_words.append("\n\n## Question")
        elif "numina" in self.args.prompt_type:
            stop_words.append("\nProblem")
        elif "pure" in self.args.prompt_type:
            stop_words.append("\n\n\n")
        elif self.args.prompt_type == 'dpsk':
            stop_words = ["<|im_end|>", "<|endoftext|>", "<|end_of_solution|>", self.tokenizer.eos_token]
        
        # Determine max function calls
        max_func_call = self.args.max_func_call
        if self.args.prompt_type in ["below_pot_trigger", "below_cot_prompt", "below_cot_trigger"]:
            max_func_call = 1
        if self.args.prompt_type in ["tool-integrated", "qwen25-math-pot", 'numina']:
            max_func_call = 4
        
        # Initialize processing state
        remain_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]
        end_prompts = []
        
        # Multi-turn generation loop
        for epoch in range(max_func_call):
            if not remain_prompts:
                break
            
            current_prompts = remain_prompts
            prompts = [item[1] for item in current_prompts]
            
            # Generate with vLLM
            outputs = self.llm.generate(
                prompts,
                SamplingParams(
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643] if "qwen2" in self.model_name_or_path.lower() else None
                    ),
                ),
            )
            
            # Sort outputs by request_id
            outputs = sorted(outputs, key=lambda x: int(x.request_id))
            outputs = [output.outputs[0] for output in outputs]
            
            # Process outputs
            remain_prompts = []
            remain_codes = []
            
            for (i, query), output in zip(current_prompts, outputs):
                output_text = output.text.rstrip()
                query += output_text
                
                # Handle different prompt types
                if self.args.prompt_type in code_prompt_type:
                    if "```python" in output_text or 'print(' in output_text:
                        remain_prompts.append((i, query))
                        code = self._extract_code(output_text)
                        remain_codes.append(code)
                    else:
                        end_prompts.append((i, query))
                elif self.args.prompt_type == "cot":
                    end_prompts.append((i, query))
                elif "boxed" not in output_text and output_text.endswith("```"):
                    program = self._extract_code(output_text)
                    remain_prompts.append((i, query))
                    remain_codes.append(program)
                else:
                    end_prompts.append((i, query))
            
            # Execute code if any
            if remain_codes:
                remain_results = self.executor.batch_apply(remain_codes)
                for k in range(len(remain_prompts)):
                    i, query = remain_prompts[k]
                    res, report = remain_results[k]
                    exec_result = res if res else report
                    
                    if max_func_call == 1:
                        exec_result = "\\boxed{" + exec_result + "}"
                        query += exec_result
                        end_prompts.append((i, query))
                        continue
                    
                    if "```python" in query:
                        exec_result = f"\n```output\n{exec_result}\n```\n"
                        query += exec_result
                    
                    if epoch == max_func_call - 1:
                        query += "\nReach max function call limit."
                    remain_prompts[k] = (i, query)
                
                if max_func_call == 1:
                    remain_prompts = []
        
        # Add any remaining prompts to end_prompts
        end_prompts.extend(remain_prompts)
        end_prompts = sorted(end_prompts, key=lambda x: x[0])
        
        # Extract final codes and run execution
        codes = []
        for i in range(len(input_prompts)):
            _, end_prompt = end_prompts[i]
            code = end_prompt.split(input_prompts[i])[-1].strip()
            codes.append(code)
        
        # Final execution and result extraction
        final_results = [
            run_execute(self.executor, code, self.args.prompt_type, data_name) 
            for code in codes
        ]
        
        return [
            {
                "code": code,
                "pred": result[0],
                "report": result[1]
            }
            for code, result in zip(codes, final_results)
        ]
    
    def _extract_code(self, output_text: str) -> str:
        """Extract code from output text."""
        nvidia_code_start_token, nvidia_code_end_token = '<llm-code>', '</llm-code>'
        code_start_token, code_end_token = "```python", "```"
        
        begin_cot = '<start>'
        string = output_text.split(begin_cot)[-1]
        string = string.replace('\\n', '\n')
        
        if nvidia_code_start_token in string:
            string = string.split(nvidia_code_start_token)[-1]
            string = string.split(nvidia_code_end_token)[0]
        if code_start_token in string:
            string = string.split(code_start_token)[-1]
            string = string.split(code_end_token)[0]
        
        string = string.strip('\n').strip()
        lines = string.strip().split("\n")
        if lines and "print" not in lines[-1]:
            lines[-1] = f"print({lines[-1]})"
            string = "\n".join(lines)
        
        return string
    

    
    def _is_multi_choice(self, answer: str) -> bool:
        """Check if answer is multi-choice format."""
        if answer is None:
            return False
        for c in answer:
            if c not in ["A", "B", "C", "D", "E"]:
                return False
        return True


class MathEvalMulti:
    """Main class for distributed math evaluation."""
    
    def __init__(self, args):
        self.args = args
        setup_logging()
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    }
                }
            )
            logging.info(f"Ray initialized with {args.num_workers} workers")
    
    def prepare_data(self, data_name: str) -> Tuple[List[Dict], List[Dict], str]:
        """Prepare data for evaluation (same as original)."""
        examples = load_data(data_name, self.args.split, self.args.data_dir)

        # Sample `num_test_sample` from dataset
        if self.args.num_test_sample > 0:
            examples = examples[:self.args.num_test_sample]

        # Shuffle
        if self.args.shuffle:
            random.seed(datetime.now().timestamp())
            random.shuffle(examples)

        # Select start and end
        examples = examples[self.args.start : len(examples) if self.args.end == -1 else self.args.end]

        # Get out_file name
        dt_string = datetime.now().strftime("%m-%d_%H-%M")
        model_name = "/".join(self.args.model_name_or_path.split("/")[-2:])
        out_file_prefix = f"{self.args.split}_{self.args.prompt_type}_{self.args.num_test_sample}_seed{self.args.seed}_t{self.args.temperature}"
        output_dir = self.args.output_dir.split('/')[-1]
        if not os.path.exists(output_dir):
            output_dir = f"/home/ma-user/work/zhouzhijian/DRL/qwen_eval/{output_dir}"
        out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{self.args.start}_e{self.args.end}_{self.args.num_shots}shots_multi.jsonl"
        
        os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

        # Load processed samples
        processed_samples = []
        if not self.args.overwrite:
            processed_files = [
                f for f in os.listdir(f"{output_dir}/{data_name}/")
                if f.endswith(".jsonl") and f.startswith(out_file_prefix)
            ]
            for f in processed_files:
                processed_samples.extend(list(load_jsonl(f"{output_dir}/{data_name}/{f}")))

        # Deduplicate
        processed_samples = {sample["idx"]: sample for sample in processed_samples}
        processed_idxs = list(processed_samples.keys())
        processed_samples = list(processed_samples.values())
        examples = [example for example in examples if example["idx"] not in processed_idxs]
        
        return examples, processed_samples, out_file
    
    def prepare_samples(self, examples: List[Dict], data_name: str) -> List[Dict]:
        """Prepare samples with prompts."""
        samples = []
        
        # System templates
        system_templates = {
            "qwen3": "Please reason step by step, and put your final answer within \\boxed{{}}.",
            "llama": "Please reason step by step, and put your final answer within \\boxed{{}}.",
            "llama_longcot": myrefl_sys3,
        }
        
        # We need a tokenizer for chat template - create a dummy one for now
        # In practice, this should be initialized properly
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, trust_remote_code=True)
            self.tokenizer = tokenizer
        except Exception as e:
            logging.warning(f"Could not load tokenizer: {e}. Using construct_prompt instead.")
        
        for example in tqdm(examples, desc="Preparing samples"):
            idx = example["idx"]

            # Parse question and answer
            example["question"] = parse_question(example, data_name)
            if example["question"] == "":
                continue
            gt_cot, gt_ans = parse_ground_truth(example, data_name)
            example["gt_ans"] = gt_ans
            q = example["question"] if 'question' in example else example['problem']
            
            # Construct prompt
            if (self.args.apply_chat_template or 'llama' in self.args.prompt_type.lower()) and hasattr(self, 'tokenizer'):
                full_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_templates.get(self.args.prompt_type, system_templates["qwen3"])},
                     {"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif self.args.prompt_type == 'qwen3' and hasattr(self, 'tokenizer'):
                full_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_templates[self.args.prompt_type]},
                     {"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                full_prompt = construct_prompt(example, data_name, self.args)

            sample = {
                "idx": idx,
                "question": q,
                "reference_solution": gt_cot,
                "gold": gt_ans,
                "prompt": full_prompt,
            }

            # Add additional fields
            for key in ["level", "type", "unit", "solution_type", "choices", "solution",
                       "ques_type", "ans_type", "answer_type", "dataset", "subfield",
                       "filed", "theorem", "answer", "domain", "difficulty", "problem",
                       "source", 'final_answer']:
                if key in example:
                    sample[key] = example[key]
            
            samples.append(sample)
        
        return samples
    
    def distribute_work(self, samples: List[Dict], data_name: str) -> List[Dict]:
        """
        Distribute sampling work across Ray workers.
        Each worker processes ALL samples but generates different portions of samples.
        """
        if not samples:
            return []
        
        total_sampling = self.args.n_sampling
        num_workers = self.args.num_workers
        
        # Calculate samples per worker
        samples_per_worker = max(1, total_sampling // num_workers)
        remaining_samples = total_sampling % num_workers
        
        logging.info(f"Total sampling: {total_sampling}, Workers: {num_workers}")
        logging.info(f"Samples per worker: {samples_per_worker}, Remaining: {remaining_samples}")
        
        # Create workers
        workers = []
        args_dict = vars(self.args)
        for i in range(self.args.num_workers):
            worker = MathEvalWorker.remote(
                model_name_or_path=self.args.model_name_or_path,
                tensor_parallel_size=self.args.tensor_parallel_size,
                worker_id=i,
                args_dict=args_dict
            )
            workers.append(worker)
        
        # Distribute sampling work
        futures = []
        
        for i, worker in enumerate(workers):
            # Each worker processes ALL samples, but generates different number of samples per question
            worker_n_sampling = samples_per_worker + (1 if i < remaining_samples else 0)
            
            if worker_n_sampling > 0:
                # Create a copy of args for this worker with adjusted n_sampling
                worker_args = args_dict.copy()
                worker_args['n_sampling'] = worker_n_sampling
                
                future = worker.process_samples.remote(samples, data_name, worker_args)
                futures.append((future, i, worker_n_sampling))
                
                logging.info(f"Worker {i}: processing {len(samples)} problems, {worker_n_sampling} samples each")
        
        # Collect results from all workers
        worker_results = []
        for future, worker_id, n_sampling in tqdm(futures, desc="Collecting worker results"):
            batch_results = ray.get(future)
            worker_results.append((worker_id, n_sampling, batch_results))
        
        # Merge results: combine predictions from all workers for each sample
        merged_results = []
        for sample_idx, original_sample in enumerate(samples):
            merged_sample = original_sample.copy()
            merged_sample.pop("prompt", None)
            
            # Collect all predictions from all workers for this sample
            all_codes = []
            all_preds = []
            all_reports = []
            
            for worker_id, n_sampling, worker_samples in worker_results:
                if sample_idx < len(worker_samples):
                    worker_sample = worker_samples[sample_idx]
                    
                    # Extract predictions from this worker
                    codes = worker_sample.get("code", [])
                    preds = worker_sample.get("pred", [])
                    reports = worker_sample.get("report", [])
                    
                    # Ensure they are lists
                    if not isinstance(codes, list):
                        codes = [codes]
                    if not isinstance(preds, list):
                        preds = [preds]
                    if not isinstance(reports, list):
                        reports = [reports]
                    
                    all_codes.extend(codes)
                    all_preds.extend(preds)
                    all_reports.extend(reports)
            
            # Update merged sample with all predictions
            merged_sample.update({
                "code": all_codes,
                "pred": all_preds,
                "report": all_reports,
                "total_workers": len(worker_results),
                "total_sampling": len(all_preds)
            })
            merged_results.append(merged_sample)
        
        logging.info(f"Merged results: {len(merged_results)} samples, total sampling: {sum(len(s.get('pred', [])) for s in merged_results)}")
        return merged_results
    
    def evaluate_dataset(self, data_name: str) -> Dict:
        """Evaluate a single dataset."""
        logging.info(f"Evaluating dataset: {data_name}")
        
        # Prepare data
        examples, processed_samples, out_file = self.prepare_data(data_name)
        logging.info(f"Dataset: {data_name}, remaining samples: {len(examples)}")
        
        if not examples:
            logging.info(f"No new samples to process for {data_name}")
            if processed_samples:
                # Evaluate existing processed samples
                all_samples, result_json = evaluate(
                    samples=processed_samples,
                    data_name=data_name,
                    prompt_type=self.args.prompt_type,
                    execute=True,
                )
                
                # Progressive evaluation is now handled by evaluate() function
                
                return result_json
            else:
                return {"acc": 0.0}
        
        # Prepare samples
        samples = self.prepare_samples(examples, data_name)
        if not samples:
            logging.warning(f"No valid samples prepared for {data_name}")
            return {"acc": 0.0}
        
        # Distribute work and process
        start_time = time.time()
        processed_results = self.distribute_work(samples, data_name)
        time_use = time.time() - start_time
        
        # Combine with existing processed samples
        all_samples = processed_results + processed_samples
        
        # Evaluate results
        all_samples, result_json = evaluate(
            samples=all_samples,
            data_name=data_name,
            prompt_type=self.args.prompt_type,
            execute=True,
        )
        
        # Progressive evaluation is now handled by evaluate() function
        # Log the new pass@k and average@k metrics
        if "pass_at_k" in result_json:
            logging.info(f"Pass@k metrics for {data_name}:")
            for k, value in result_json["pass_at_k"].items():
                logging.info(f"  pass@{k}: {value:.4f}")
        if "average_at_k" in result_json:
            logging.info(f"Average@k metrics for {data_name}:")
            for k, value in result_json["average_at_k"].items():
                logging.info(f"  average@{k}: {value:.4f}")
        
        # Save outputs
        if self.args.save_outputs and processed_results:
            save_jsonl(all_samples, out_file)
        
        # Add timing information
        result_json["time_use_in_second"] = time_use
        result_json["time_use_in_minute"] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
        result_json["num_workers"] = self.args.num_workers
        
        # Save metrics
        with open(out_file.replace(".jsonl", f"_{self.args.prompt_type}_metrics_multi.json"), "w") as f:
            json.dump(result_json, f, indent=4)
        
        logging.info(f"Dataset {data_name} evaluation completed. Accuracy: {result_json.get('acc', 0):.1f}")
        return result_json
    
    def run(self):
        """Run the complete evaluation pipeline."""
        logging.info("Starting distributed math evaluation")
        
        # Process each dataset
        data_list = self.args.data_names.split(",")
        results = []
        dataset_results = {}
        
        for data_name in data_list:
            result = self.evaluate_dataset(data_name)
            results.append(result)
            dataset_results[data_name] = result
        
        # Calculate average metrics
        avg_result = calculate_average_progressive_metrics(dataset_results)
        
        dataset_results["avg"] = avg_result
        results.append(avg_result)
        data_list.append("avg")
        
        # Print summary
        pad = max([len(data_name) for data_name in data_list])
        print("\n" + "="*50)
        print("DISTRIBUTED EVALUATION SUMMARY")
        print("="*50)
        print(f"Workers used: {self.args.num_workers}")
        print(f"Tensor parallel size per worker: {self.args.tensor_parallel_size}")
        
        # Print basic accuracy
        print("\nBasic Accuracy:")
        print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
        print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
        
        # Print pass@k and average@k metrics
        print("\nPass@k Results:")
        # Get all available k values from first result that has pass_at_k
        k_values = []
        for result in results:
            if "pass_at_k" in result:
                k_values = sorted(result["pass_at_k"].keys())
                break
        
        if k_values:
            for k in k_values:
                print(f"\npass@{k}:")
                pass_values = []
                for result in results:
                    if "pass_at_k" in result and k in result["pass_at_k"]:
                        pass_values.append(f"{result['pass_at_k'][k]:.3f}")
                    else:
                        pass_values.append("N/A")
                print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
                print("\t".join([val.ljust(pad, " ") for val in pass_values]))
                
                print(f"\naverage@{k}:")
                avg_values = []
                for result in results:
                    if "average_at_k" in result and k in result["average_at_k"]:
                        avg_values.append(f"{result['average_at_k'][k]:.3f}")
                    else:
                        avg_values.append("N/A")
                print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
                print("\t".join([val.ljust(pad, " ") for val in avg_values]))
        
        # Save to CSV
        model_name = self.args.model_name_or_path.split('/')[-1]  # Get model name from path
        save_results_to_csv(model_name, dataset_results, self.args.csv_output)
        print(f"\nResults saved to {self.args.csv_output}")
        
        logging.info("Distributed evaluation completed successfully")


def main():
    args = parse_args()
    set_seed(args.seed)
    
    evaluator = MathEvalMulti(args)
    evaluator.run()


if __name__ == "__main__":
    main()
