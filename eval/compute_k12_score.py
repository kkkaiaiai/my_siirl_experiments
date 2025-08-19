import json
from collections import defaultdict

data_path = "/inspire/hdd/global_user/shaowenqi-shaowenqi/liuzongkai/MM-EUREKA-Qwen/eval/results/K12_cpgd_001_5epi/MMK12_250512221126_extract.json"
# data_path = "/inspire/hdd/global_user/shaowenqi-shaowenqi/liuzongkai/MM-EUREKA-Qwen/eval/results/cpgd001_32B-5episode/MMK12_250611082220_extract.json"
# 读取 JSON 数据
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 用于统计每个 subject 的总数和正确数
subject_stats = defaultdict(lambda: {'total': 0, 'true': 0})

# 遍历每个条目
for item in data.values():
    subject = item.get('subject')
    score = item.get('score')
    if subject:
        subject_stats[subject]['total'] += 1
        if score is True:
            subject_stats[subject]['true'] += 1

# 计算并输出每个 subject 的 true 比例
for subject, stats in subject_stats.items():
    total = stats['total']
    true_count = stats['true']
    ratio = true_count / total if total else 0
    print(f"{subject}: {true_count}/{total} = {ratio:.2%}")
