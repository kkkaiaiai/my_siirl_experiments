import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import pdb
from libs.grader import *

from libs.parser import *
from libs.utils import load_jsonl
from libs.python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 
  
    with ProcessPool(max_workers=48) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0

    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])
    
    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    table = np.array(score_mat)
    col_means= table.mean(axis=0)
    
    # 初始化多个 k 值的指标
    max_k = table.shape[1]  # 最大可用的 k 值
    ks = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256] if k <= max_k]
    pass_at_k = {}
    average_at_k = {}
    
    # 计算各个 k 值的指标
    for k in ks:
        # pass@k: 前k个预测中至少有一个正确的样本比例
        pass_at_k[k] = (table[:, :k] > 0.5).any(axis=1).astype(float).mean()
        # average@k: 前k个预测的平均准确率
        average_at_k[k] = (table[:, :k] > 0.5).astype(float).mean()
    
    # 保持原有变量以兼容现有代码
    row_means = pass_at_k[ks[0]] if ks else 0.0  # 使用最小k值的pass@k作为row_means
    
    # nrow, ncol = table.shape 
    scores = (table.reshape(-1)>0.5).astype(float)
    pass1 = scores.mean()
    mean_score = list(np.round(col_means * 10000, decimals=4))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0],
        "pass@k": np.round(row_means*10000, decimals=4),  # 保持原有格式兼容性
        "pass@1": np.round(pass1, decimals=4),
        # 新增多个k值的详细指标
        "pass_at_k": {k: np.round(v, decimals=4) for k, v in pass_at_k.items()},
        "average_at_k": {k: np.round(v, decimals=4) for k, v in average_at_k.items()}
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
