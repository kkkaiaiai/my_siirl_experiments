import os
import shutil
from transformers import AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
import torch

def merge_checkpoints(checkpoint_dirs, output_dir, model_name="merged_model", alpha=0.05):
    """
    使用指数平均 (EMA) 合并多个LLM检查点为一个新的检查点
    
    Args:
        checkpoint_dirs: 包含多个检查点目录的列表
        output_dir: 输出目录
        model_name: 合并后的模型名称
        alpha: EMA 的更新系数 (默认 0.05)
    """
    # 创建输出目录
    output_path = os.path.join(output_dir, model_name)
    os.makedirs(output_path, exist_ok=True)
    
    # 加载第一个检查点作为参考，获取参数名和配置
    ref_checkpoint = checkpoint_dirs[0]
    config = AutoConfig.from_pretrained(ref_checkpoint)
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
            
    # 初始化一个模型（全零权重）
    base_model = AutoModelForCausalLM.from_pretrained(ref_checkpoint)
    with torch.no_grad():
        for p in base_model.parameters():
            p.zero_()

    # 逐个 ckpt 做 EMA
    with torch.no_grad():
        for ckpt_path in checkpoint_dirs:
            other_model = AutoModelForCausalLM.from_pretrained(ckpt_path)
            for p_base, p_other in tqdm(
                zip(base_model.parameters(), other_model.parameters()),
                total=sum(1 for _ in base_model.parameters()),
                desc=f"EMA 合并 {ckpt_path}",
                leave=False
            ):
                p_base.mul_(1 - alpha).add_(p_other, alpha=alpha)
            del other_model
            print(f"✅ 完成 EMA 更新: {ckpt_path}")


    base_model.save_pretrained(output_path, safe_serialization=True)
    print(f"🎉 所有检查点已使用 EMA 成功合并，保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    checkpoint_directories = [
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_40/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_50/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_60/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_70/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_80/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_90/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_100/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_110/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_120/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_130/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_140/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_150/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_160/actor_agent_0/huggingface",
        "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_170/actor_agent_0/huggingface",
    ]
    
    output_directory = "ckpts/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/merage_models/"
    
    # alpha = 2 / (N + 1) = 2 / 12 = 0.16667
    merge_checkpoints(checkpoint_directories, output_directory, "start40_end170_step10_ema-right", alpha=0.15)
