import os
import shutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def merge_checkpoints(checkpoint_dirs, output_dir, model_name="merged_model"):
    """
    合并多个LLM检查点为一个新的检查点
    
    Args:
        checkpoint_dirs: 包含多个检查点目录的列表
        output_dir: 输出目录
        model_name: 合并后的模型名称
    """
    # 创建输出目录
    output_path = os.path.join(output_dir, model_name)
    os.makedirs(output_path, exist_ok=True)
    
    # 加载第一个检查点作为参考，获取参数名和配置
    ref_checkpoint = checkpoint_dirs[0]
    config = AutoConfig.from_pretrained(ref_checkpoint)
    
    # 保存配置文件
    config.save_pretrained(output_path)
    
    # 复制分词器相关文件
    tokenizer_files = [
        "tokenizer_config.json", "tokenizer.json", "vocab.json",
        "special_tokens_map.json", "model.safetensors.index.json",
        "merges.txt", "added_tokens.json", "config.json", "generation_config.json"
    ]
    for file in tokenizer_files:
        src_path = os.path.join(ref_checkpoint, file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(output_path, file))
    
    # 先加载第一个模型
    base_model = AutoModelForCausalLM.from_pretrained(checkpoint_dirs[0])

    # 累加参数
    with torch.no_grad():
        for other_path in checkpoint_dirs[1:]:
            other_model = AutoModelForCausalLM.from_pretrained(other_path)
            for p_base, p_other in tqdm(
                zip(base_model.parameters(), other_model.parameters()),
                total=sum(1 for _ in base_model.parameters()),  # 参数总数
                desc=f"  -> 处理 {other_path}",
                leave=False
            ):
                p_base.add_(p_other)  # inplace 累加
            del other_model
            print(f"finish adding ckpt {other_path}")

    # 做平均
    with torch.no_grad():
        print(f"begin div")
        for p in tqdm(base_model.parameters(), total=sum(1 for _ in base_model.parameters()), desc="平均参数"):
            p.div_(len(checkpoint_dirs))
        print(f"finish div")

    # 存到新地址（原有 ckpt 不会被改动）
    base_model.save_pretrained(output_path, safe_serialization=True)

    print(f"所有检查点已成功合并，保存至: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 要合并的检查点目录列表
    checkpoint_directories = [
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_40/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_50/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_60/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_70/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_80/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_90/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_100/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_110/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_120/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_130/actor_agent_0/huggingface",
        # "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_140/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_150/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_160/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_170/actor_agent_0/huggingface",
    ]
    
    # 输出目录
    output_directory = "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/merage_models/"
    
    # 执行合并
    merge_checkpoints(checkpoint_directories, output_directory, "start150_end170_step10_merged")
