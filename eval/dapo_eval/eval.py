#!/usr/bin/env python3
"""
DAPO Dataset Evaluation Pipeline
Evaluates math problem solving using vLLM and Ray for distributed inference
"""

import argparse
import json
import logging
import re
import regex
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio
import time

import pandas as pd
import ray
from vllm import LLM, SamplingParams
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Import math_verify for advanced mathematical verification
try:
    from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    logging.warning("math_verify not available, falling back to basic verification")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Math answer verification utilities using math_verify_box.py approach
def _extract_boxed_answer(solution_str):
    """Extract boxed answer using advanced regex from math_verify_box.py."""
    match = regex.findall(
        r"(\\boxed\{(?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*\})", 
        solution_str, 
        re.DOTALL
    )
    return match[-1][0].strip() if match else ""


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string (legacy compatibility)."""
    return _extract_boxed_answer(string) or None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km", "units",
    "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm",
    "gm", "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}",
    "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, 
    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> Tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria."""
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[List[int]] = None
) -> Tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria."""
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, 
    pause_tokens_index: Optional[List[int]] = None
) -> Tuple[bool, str]:
    """Verify if the solution is correct."""
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score_math_verify(solution_str: str, ground_truth: str) -> Dict:
    """
    Compute score using math_verify_box.py approach with advanced mathematical verification.
    """
    # Extract boxed answer using advanced regex
    pred = _extract_boxed_answer(solution_str)
    reward = 0.0
    
    if pred == ground_truth:
        reward = 1.0
    elif MATH_VERIFY_AVAILABLE:
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
    
    return {
        "score": reward,
        "acc": reward == 1.0,
        "pred": pred,
    }


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[List[int]] = None,
    use_math_verify: bool = True,
) -> Dict:
    """Compute the reward score for a solution."""
    
    if use_math_verify and MATH_VERIFY_AVAILABLE:
        # Use math_verify_box.py approach
        return compute_score_math_verify(solution_str, ground_truth)
    else:
        # Fallback to original verification approach
        # Limit solution length for efficiency
        solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

        # Verify the solution using original approach
        correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

        reward = 1.0 if correct else 0.0
        acc = correct

        return {
            "score": reward,
            "acc": acc,
            "pred": pred,
        }


@ray.remote(num_gpus=1)
class VLLMWorker:
    """Ray remote worker for vLLM inference."""
    
    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        logging.info(f"Initialized vLLM worker with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    def generate_samples(
        self, 
        prompts: List[List[Dict]], 
        num_samples: int = 10,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_tokens: int = 2048,
    ) -> List[List[Dict]]:
        """Generate multiple samples for each prompt with token counts."""
        sampling_params = SamplingParams(
            temperature = temperature,
            top_p = top_p,
            max_tokens=max_tokens,
            n=num_samples,  # Generate num_samples completions per prompt
        )
        text = [self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Set to False to strictly disable thinking
        ) for prompt in prompts]
        outputs = self.llm.generate(text, sampling_params)
        results = []
        
        for output in outputs:
            samples = []
            for completion in output.outputs:
                # Calculate token count for this completion
                token_count = len(self.tokenizer.encode(completion.text))
                samples.append({
                    'text': completion.text,
                    'token_count': token_count
                })
            results.append(samples)
        
        return results


class DatasetEvaluator:
    """Main evaluation pipeline for DAPO dataset."""
    
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        self.results = []
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'eval_{time.strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        
        # Log verification method being used
        if getattr(self.args, 'use_math_verify', True) and MATH_VERIFY_AVAILABLE:
            logging.info("Using math_verify library for advanced mathematical verification")
        else:
            logging.info("Using legacy verification method")
        
    def load_dataset(self) -> pd.DataFrame:
        """Load and return the dataset."""
        logging.info(f"Loading dataset from {self.args.file}")
        df = pd.read_parquet(self.args.file)
        logging.info(f"Loaded {len(df)} samples")
        
        # Take subset if specified
        if self.args.max_samples and self.args.max_samples < len(df):
            df = df.head(self.args.max_samples)
            logging.info(f"Using subset of {len(df)} samples")
            
        return df
    
    def extract_prompts(self, df: pd.DataFrame) -> List[List[Dict]]:
        """Extract prompts from the dataset format."""
        prompts = []
        for idx, row in df.iterrows():
            # Keep the full message format for apply_chat_template
            prompts.append(row['prompt'])
        return prompts
    
    def batch_evaluate(self, df: pd.DataFrame, num_workers: int = 1) -> List[Dict]:
        """Evaluate dataset using Ray workers."""
        logging.info("Initializing Ray cluster...")
        if not ray.is_initialized():
            ray.init()
        
        # Create workers
        workers = [
            VLLMWorker.remote(
                model_name=self.args.model,
                tensor_parallel_size=self.args.tensor_parallel_size
            ) 
            for _ in range(num_workers)
        ]
        
        # Extract prompts and prepare batches
        prompts = self.extract_prompts(df)
        batch_size = max(1, len(prompts) // num_workers)
        
        logging.info(f"Processing {len(prompts)} prompts with {num_workers} workers")
        
        # Distribute work across workers
        futures = []
        for i, worker in enumerate(workers):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < num_workers - 1 else len(prompts)
            batch_prompts = prompts[start_idx:end_idx]
            
            if batch_prompts:
                future = worker.generate_samples.remote(
                    batch_prompts,
                    num_samples=self.args.num_samples,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens
                )
                futures.append((future, start_idx, end_idx))
        
        # Collect results
        all_generations = [None] * len(prompts)
        
        for future, start_idx, end_idx in tqdm(futures, desc="Collecting results"):
            batch_results = ray.get(future)
            for local_idx, generations in enumerate(batch_results):
                global_idx = start_idx + local_idx
                all_generations[global_idx] = generations
        
        # Evaluate each sample
        results = []
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), desc="Evaluating samples")):
            ground_truth = row['reward_model']['ground_truth']
            generations = all_generations[idx]
            
            sample_results = []
            for gen_idx, generation_data in enumerate(generations):
                # Extract text and token count from generation data
                generation_text = generation_data['text']
                token_count = generation_data['token_count']
                
                score_data = compute_score(
                    generation_text, 
                    ground_truth, 
                    strict_box_verify=self.args.strict_box_verify,
                    use_math_verify=getattr(self.args, 'use_math_verify', True)
                )
                
                sample_result = {
                    'sample_idx': idx,
                    'generation_idx': gen_idx,
                    'prompt': prompts[idx][:100] + "..." if len(prompts[idx]) > 100 else prompts[idx],
                    'generation': generation_text,
                    'token_count': token_count,
                    'ground_truth': ground_truth,
                    'predicted_answer': score_data['pred'],
                    'score': score_data['score'],
                    'correct': score_data['acc'],
                    'data_source': row['data_source'],
                    'ability': row['ability'],
                }
                sample_results.append(sample_result)
            
            results.append({
                'sample_idx': idx,
                'ground_truth': ground_truth,
                'generations': sample_results,
                'best_score': max(r['score'] for r in sample_results),
                'avg_score': float(np.mean([r['score'] for r in sample_results])),
                'accuracy_at_n': any(r['correct'] for r in sample_results),
                'avg_accuracy': float(np.mean([r['correct'] for r in sample_results])),
                'avg_tokens_per_generation': float(np.mean([r['token_count'] for r in sample_results])),
                'min_tokens': min(r['token_count'] for r in sample_results),
                'max_tokens': max(r['token_count'] for r in sample_results),
                'data_source': row['data_source'],
                'ability': row['ability'],
            })
        
        return results
    
    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute overall evaluation metrics."""
        total_samples = len(results)
        correct_at_1 = sum(1 for r in results if r['generations'][0]['correct'])
        correct_at_n = sum(1 for r in results if r['accuracy_at_n'])
        
        avg_best_score = float(np.mean([r['best_score'] for r in results]))
        avg_score_across_samples = float(np.mean([r['avg_score'] for r in results]))
        avg_accuracy_across_samples = float(np.mean([r['avg_accuracy'] for r in results]))
        
        # Compute token statistics
        all_token_counts = []
        for result in results:
            for generation in result['generations']:
                all_token_counts.append(generation['token_count'])
        
        avg_tokens_per_generation = float(np.mean(all_token_counts))
        std_tokens_per_generation = float(np.std(all_token_counts))
        min_tokens_overall = int(np.min(all_token_counts))
        max_tokens_overall = int(np.max(all_token_counts))
        avg_tokens_per_sample = float(np.mean([r['avg_tokens_per_generation'] for r in results]))
        
        # Compute metrics by data source and ability
        by_source = {}
        by_ability = {}
        
        for result in results:
            source = result['data_source']
            ability = result['ability']
            
            if source not in by_source:
                by_source[source] = {'correct': 0, 'total': 0}
            if ability not in by_ability:
                by_ability[ability] = {'correct': 0, 'total': 0}
                
            by_source[source]['total'] += 1
            by_ability[ability]['total'] += 1
            
            if result['accuracy_at_n']:
                by_source[source]['correct'] += 1
                by_ability[ability]['correct'] += 1
        
        metrics = {
            'total_samples': total_samples,
            'accuracy_at_1': correct_at_1 / total_samples,
            'accuracy_at_n': correct_at_n / total_samples,
            'average_best_score': avg_best_score,
            'average_score_across_samples': avg_score_across_samples,
            'average_accuracy_across_samples': avg_accuracy_across_samples,
            'token_statistics': {
                'avg_tokens_per_generation': avg_tokens_per_generation,
                'std_tokens_per_generation': std_tokens_per_generation,
                'min_tokens_overall': min_tokens_overall,
                'max_tokens_overall': max_tokens_overall,
                'avg_tokens_per_sample': avg_tokens_per_sample,
                'total_generations': len(all_token_counts),
            },
            'by_data_source': {k: v['correct']/v['total'] for k, v in by_source.items()},
            'by_ability': {k: v['correct']/v['total'] for k, v in by_ability.items()},
        }
        
        return metrics
    
    def save_results(self, results: List[Dict], metrics: Dict):
        """Save evaluation results and metrics."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # Save metrics
        metrics_file = output_dir / f"eval_metrics_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        logging.info(f"Results saved to {results_file}")
        logging.info(f"Metrics saved to {metrics_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Accuracy@1: {metrics['accuracy_at_1']:.4f}")
        print(f"Accuracy@N (pass@{self.args.num_samples}): {metrics['accuracy_at_n']:.4f}")
        print(f"Average best score: {metrics['average_best_score']:.4f}")
        print(f"Average score across samples: {metrics['average_score_across_samples']:.4f}")
        print(f"Average accuracy across samples: {metrics['average_accuracy_across_samples']:.4f}")
        
        # Token statistics
        token_stats = metrics['token_statistics']
        print(f"\nToken Statistics:")
        print(f"  Average tokens per generation: {token_stats['avg_tokens_per_generation']:.2f}")
        print(f"  Standard deviation: {token_stats['std_tokens_per_generation']:.2f}")
        print(f"  Average tokens per sample: {token_stats['avg_tokens_per_sample']:.2f}")
        print(f"  Min tokens: {token_stats['min_tokens_overall']}")
        print(f"  Max tokens: {token_stats['max_tokens_overall']}")
        print(f"  Total generations: {token_stats['total_generations']}")
        
        print(f"\nBy Data Source:")
        for source, acc in metrics['by_data_source'].items():
            print(f"  {source}: {acc:.4f}")
            
        print(f"\nBy Ability:")
        for ability, acc in metrics['by_ability'].items():
            print(f"  {ability}: {acc:.4f}")
    
    def run(self):
        """Run the complete evaluation pipeline."""
        logging.info("Starting DAPO dataset evaluation")
        
        # Load dataset
        df = self.load_dataset()
        
        # Run evaluation
        results = self.batch_evaluate(df, num_workers=self.args.num_workers)
        
        # Compute metrics
        metrics = self.compute_metrics(results)
        
        # Save results
        self.save_results(results, metrics)
        
        logging.info("Evaluation completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DAPO dataset using vLLM")
    
    # Dataset arguments
    parser.add_argument("--file", default="./data/DAPO-17k-processed/train.parquet", 
                       help="Path to dataset file")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    
    # Model arguments
    parser.add_argument("--model", required=True,
                       help="Model name or path for vLLM")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size for vLLM")
    
    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="top_p")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum tokens to generate")
    
    # Evaluation arguments
    parser.add_argument("--strict_box_verify", action="store_true",
                       help="Use strict box verification for answers")
    parser.add_argument("--use_math_verify", action="store_true", default=True,
                       help="Use math_verify library for advanced mathematical verification (default: True)")
    parser.add_argument("--disable_math_verify", action="store_true",
                       help="Disable math_verify and use legacy verification")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of Ray workers to use")
    parser.add_argument("--output_dir", default="./eval_results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Handle math_verify argument logic
    if args.disable_math_verify:
        args.use_math_verify = False
    
    evaluator = DatasetEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
