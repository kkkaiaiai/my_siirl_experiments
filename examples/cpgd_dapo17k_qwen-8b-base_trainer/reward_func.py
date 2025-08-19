import re
import regex
import logging
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
from concurrent.futures import ProcessPoolExecutor
from functools import wraps

logging.getLogger("math_verify").setLevel(logging.CRITICAL)


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


def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
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
    
    repetition_1gram = compute_ngram_repetition_rate(solution_str, n=1)
    repetition_3gram = compute_ngram_repetition_rate(solution_str, n=3)
    repetition_5gram = compute_ngram_repetition_rate(solution_str, n=5)

    # return {
    #     "score": format_reward + reward,
    #     "format_score": format_reward,
    #     "accuracy_score": reward,
    # }
    # return reward
    return {
        "score": reward, 
        "repetition_1gram": repetition_1gram,
        "repetition_3gram": repetition_3gram,
        "repetition_5gram": repetition_5gram,
    }


# def compute_score(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
#     return _GLOBAL_EXECUTOR.submit(_compute_rewards, solution_str, ground_truth).result()

#####
##### 重复率计算
#####


def compute_ngram_repetition_rate(text: str, n: int = 3) -> float:
    """计算n-gram重复率"""
    if not text:
        return 0.0
    
    # 清理文本
    text = re.sub(r'\s+', ' ', text.strip().lower())
    words = text.split()
    
    if len(words) < n:
        return 0.0
    
    # 提取n-gram
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    # 计算重复率
    unique_ngrams = set(ngrams)
    repetition_rate = 1.0 - (len(unique_ngrams) / len(ngrams))
    
    return repetition_rate
