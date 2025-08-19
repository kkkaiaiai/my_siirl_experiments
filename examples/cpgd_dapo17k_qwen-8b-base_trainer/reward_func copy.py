import re
import regex
import logging
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from concurrent.futures import ProcessPoolExecutor
from functools import wraps

logging.getLogger("math_verify").setLevel(logging.CRITICAL)

_GLOBAL_EXECUTOR = ProcessPoolExecutor(max_workers=32)


def _extract_boxed_answer(solution_str):
    match = regex.findall(
        r"(\\boxed\{(?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*\})", 
        solution_str, 
        re.DOTALL
    )
    return match[-1][0].strip() if match else ""


def _check_format_validity(solution_str):
    pattern = (
        r"(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r"^<think>.+?</think>\s*<answer>.+?</answer>$"
    )
    return 0.5 if re.search(pattern, solution_str, re.DOTALL) else 0.0


def _compute_rewards(solution_str, ground_truth):
    # Format reward
    # format_reward = _check_format_validity(solution_str)

    # Accuracy reward
    pred = _extract_boxed_answer(solution_str)
    reward = 0.0
    if pred == ground_truth:
        reward = 1.0
    else:
        try:
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
            print("Error in verify:", e)

    # return {
    #     "score": format_reward + reward,
    #     "format_score": format_reward,
    #     "accuracy_score": reward,
    # }
    return reward


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    return _GLOBAL_EXECUTOR.submit(_compute_rewards, solution_str, ground_truth).result()
