
qwen_score = [68.2, 47.9, 25.4, 20.2, 62.1, 53.6]

# 提取数据（格式：模型名称在前，6个指标数值在后）
models_data = [
    ["GPT-4o", 63.8, 50.2, 30.4, 35.0, 68.8, 49.9],
    ["o1", 73.9, 57.0, 60.3, 68.0, 98.7, 73.9],
    ["InternVL2.5-8B", 64.4, 39.5, 19.7, 12.3, 53.5, 45.6],
    ["QwenVL2.5-7B", 68.2, 47.9, 25.4, 20.2, 62.1, 53.6],
    ["InternVL2.5-MPO-8B", 68.9, 35.5, 21.5, 7.8, 53.5, 34.5],
    ["Adora (7B)", 73.5, 50.1, 23.0, 20.1, 64.2, 58.1],
    ["R1-Onevision (7B)", 64.1, 47.1, 23.5, 17.3, 61.8, 39.8],
    ["OpenVLThinker (7B)", 70.2, 47.9, 25.3, 20.1, 64.3, 60.6],
    ["MM-Eureka-7B", 73.0, 50.3, 26.9, 20.1, 66.1, 64.5],
    ["RLOO", 68.6, 48.3, 23.0, 19.5, 65.8, 61.3],
    ["REINFORCE++", 63.9, 45.5, 18.2, 17.8, 66.7, 64.3],
    ["GRPO", 70.3, 51.4, 25.9, 18.5, 67.4, 65.1],
    ["GRPO w/ drift", 71.9, 49.4, 26.3, 20.9, 67.7, 63.7],
    ["CPG", 72.7, 52.3, 27.6, 20.8, 70.7, 66.2],
    ["0.1+filter-like weighting", 73.4, 51.4, 25.9, 21.5, 70.2, 67.3],
    ["0.01+filter-like weighting", 72.9, 51, 20.7, 20.3, 70.5, 66.5],
    ["0.001+filter-like weighting", 72.8, 50.0, 25.2, 21.8, 68.5, 68.1],
    ["0.005+filter-like weighting", 71.3, 52.0, 23.6, 19.7, 67.2, 66.3],
    ["CPGD_0.1", 73.6, 51.0, 22.7, 22.7, 70.8, 65.9],
    ["CPGD_0.05", 72.9, 50.6, 22.9, 22, 67.8, 66.8], 
    ["CPGD_0.01", 74, 50.6, 28.3, 21.4, 68.3, 65.3],
    ["CPGD_0.01 600", 74.5, 51.0, 28.4, 21.9, 68.3, 65.4],
    ["CPGD_0.005", 73.2, 52.2, 27.9, 20.8, 68, 67.5], 
    ["CPGD_0.001", 74.4, 51.7, 26.0, 22.2, 68.9, 65.4],
    ["CPGD_0.0001", 74.4, 51.1, 25.9, 20, 68.7, 66.5],
    ["CPG IS", 73.1, 51.1, 26.7, 21.1, 66.2, 65.4], 
    ["CPGD 0.1 IS", 70.5, 49.3, 23.8, 20.6, 64.9, 67.8],
    ["CPGD 0.001 IS", 74, 51.5, 28.1, 21.4, 68.3, 65.5], 
    ["CPGD 0.0005 IS", 70.7, 49.2, 23.8, 21, 67.4, 64.7],
    ["0.1 dual clip IS", 71.4, 49.6, 25.9, 20.4, 65.7, 65.1],
    ["CPG verl packing", 72.1, 47.6, 19.4, 19.3, 66.5, 67.1],
    ["CPG verl", 71.1, 46.4, 20.6, 19.5, 63.9, 66.7],
]


ablation_data = [
    ["PG", 67.8, 42.0, 22.5, 8.0, 58.6, 65.9],
    ["PGD", 64.2, 41.1, 20.8, 7.5, 58.3, 67.3],
    ["CPG", 72.7, 52.3, 27.6, 20.8, 70.7, 66.2],
    ["CPGD", 73.6, 51.0, 22.7, 22.7, 70.8, 65.9],
    ["unprocessed rewards", 69.1, 40.2, 21.8, 3.5, 59.7, 67.2],
    ["equal weighting", 73.1, 51.1, 27.2, 20.8, 67.9, 65.8],
    ["std", 73.6, 51.0, 22.7, 22.7, 70.8, 65.9],
    ["filter-like weighting", 73.4, 51.4, 25.9, 21.5, 70.2, 67.3],
    ["w/ reference constraint", 71.8, 50.0, 21.0, 21.2, 69.8, 65.8],
    ["w/o reference constraint", 73.6, 51.0, 22.7, 22.7, 70.8, 65.9]
]

qwen_score = [71.7, 49.9, 40.1, 30.0, 69.1, 66.8]

large_model_data = [
    # ["GPT-4o", 63.8, 50.2, 30.4, 35.0, 68.8, 49.9],
    # ["o1", 73.9, 57.0, 60.3, 68.0, 98.7, 73.9],
    ["InternVL2.5-38B", 71.9, 49.4, 31.8, 32.0, 67.5, 58.0],
    ["QwenVL2.5-32B", 71.7, 49.9, 40.1, 30.0, 69.1, 66.8],
    ["InternVL2.5-MPO-38B", 73.8, 46.5, 32.3, 25.6, 66.2, 48.3],
    ["QVQ-72B-Preview", 71.4, 48.2, 35.9, 33.2, 65.4, 61.5],
    ["MM-Eureka-32B", 74.8, 56.5, 34.4, 35.9, 73.4, 72.7],
    ["CPGD 2epi", 72.0, 56.9, 28.7, 40.3, 71.8, 73.9],
    ["CPGD-32B 0.001", 75.5, 58.2, 31.9, 40.8, 74.4, 76.5], 
]

leading_model_data = [
    ["GPT-4o", 63.8, 50.2, 30.4, 35.0, 68.8, 49.9],
    ["o1", 73.9, 57.0, 60.3, 68.0, 98.7, 73.9],
    ["Claude3.7-Sonnet", 66.8, 52.0, 41.3, 48.9, 72.6, 55.3],
    ["Gemini2-flash", 70.4, 59.3, 41.3, 51.0, 71.4, 65.2],
]

# leading_model_data_woK12 = [
#     ["GPT-4o", 63.8, 50.2, 30.4, 35.0, 68.8],
#     ["o1", 73.9, 57.0, 60.3, 68.0, 98.7],
# ]

# models_data_woK12 = [
#     ["InternVL2.5-8B", 64.4, 39.5, 19.7, 12.3, 53.5],
#     ["QwenVL2.5-7B", 68.2, 47.9, 25.4, 20.2, 62.1],
#     ["InternVL2.5-MPO-8B", 68.9, 35.5, 21.5, 7.8, 53.5],
#     ["Adora (7B)", 73.5, 50.1, 23.0, 20.1, 64.2],
#     ["R1-Onevision (7B)", 64.1, 47.1, 23.5, 17.3, 61.8],
#     ["OpenVLThinker (7B)", 70.2, 47.9, 25.3, 20.1, 64.3],
#     ["MM-Eureka-7B", 73.0, 50.3, 26.9, 20.1, 66.1],
#     ["RLOO", 68.6, 48.3, 23.0, 19.5, 65.8],
#     ["REINFORCE++", 63.9, 45.5, 18.2, 17.8, 66.7],
#     ["GRPO", 70.3, 51.4, 25.9, 18.5, 67.4],
#     # ["CPGD (ours)", 73.6, 51.0, 22.7, 22.7, 70.8, 65.9]
#     ["filter-like weighting", 73.4, 51.4, 25.9, 21.5, 70.2],
# ]

def calculate_overall(models):
    # 提取所有数值并转置为按列排列
    scores = [[row[1], row[2], row[3], row[4], row[5], row[6]] for row in models]
    columns = list(zip(*scores))
    
    # # 计算每列的Min-Max
    # min_vals = [min(col) for col in columns]
    # max_vals = [max(col) for col in columns]

    # min_vals = [0 for col in columns]
    # max_vals = [1 for col in columns]
    
    # 计算每个模型的综合得分
    results = []
    for name, *metrics in models:
        normalized = []
        for i, val in enumerate(metrics):
            norm_val = val / qwen_score[i]
            normalized.append(norm_val)
        
        overall = round(sum(normalized)/6, 2)  # 保留两位小数
        results.append((name, overall))
    
    return results

# 执行计算
# overall_scores = calculate_overall(models_data)
# overall_scores = calculate_overall(ablation_data)
overall_scores = calculate_overall(large_model_data)


# 打印结果（按原始表格顺序）
for name, score in overall_scores:
    print(f"{name.ljust(25)}: {score:.2f}")


overall_scores = calculate_overall(leading_model_data)

# 打印结果（按原始表格顺序）
for name, score in overall_scores:
    print(f"{name.ljust(25)}: {score:.2f}")