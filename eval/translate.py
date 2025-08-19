from siirl.model_merger.base_model_merger import ModelMergerConfig
from siirl.model_merger.fsdp_model_merger import FSDPModelMerger

local_dir = "/home/ma-user/work/zhouzhijian/DRL/experiment/checkpoints/verl_ppo_distributional/qwen3-4B-base-17k-ppo-0/global_step_60/actor"
target_dir = "/home/ma-user/work/zhouzhijian/DRL/experiment/checkpoints/verl_ppo_distributional/qwen3-4B-base-17k-ppo-0/global_step_60/actor/huggingface"
target_path = "./checkpoint/drl-60"
# 配置转换参数
config = ModelMergerConfig(
    operation="merge",
    backend="fsdp",
    local_dir=local_dir,  # 您的FSDP checkpoint目录
    target_dir=target_path,       # 输出目录
    # 关键：确保hf_model_config_path指向正确的路径
    hf_model_config_path=target_dir ,  # 原始HF模型路径
    trust_remote_code=True  # 如果需要的话
)

# 执行转换
merger = FSDPModelMerger(config)
merger.merge_and_save()
merger.cleanup()