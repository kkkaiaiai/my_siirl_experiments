
# CHECKPOINT="K12_clipgrad_Gradppo_k3"
# RESULT_DIR="K12_cpgd_5epi"

# CHECKPOINT=fix_rloo_WOKL_5/ckpt/global_step250_hf
# RESULT_DIR="K12_rloo_wokl_5epi_250step"

# CHECKPOINT=reinforce_WOKL_5
# RESULT_DIR="K12_reinforce_wokl_5epi"


# CHECKPOINT=K12_Grad
# RESULT_DIR="K12_pg_5epi"

# CHECKPOINT=K12_clipGrad
# RESULT_DIR="K12_cpg_5epi"

# CHECKPOINT=K12_grad_Gradppo_k3
# RESULT_DIR="K12_pgd_5epi"

# CHECKPOINT=K12_clipgrad_Gradppo_woSTD
# RESULT_DIR="K12_cpgd_woStd_5epi"

# CHECKPOINT=K12_clipgrad_Gradppo_woNorm
# RESULT_DIR="K12_cpgd_woGroupNorm_5epi"

# CHECKPOINT=K12_clipgrad_Gradppo_k3_weight_clip
# RESULT_DIR="K12_cpgd_filterClipWeight_5epi"

# CHECKPOINT=K12_clipgrad_Gradppo_kl_k3
# RESULT_DIR="K12_cpgd_reference_5epi"

# CHECKPOINT=K12_cpgd_001
# RESULT_DIR="K12_cpgd_001_5epi"

# CHECKPOINT=K12_cpgd_0001
# RESULT_DIR="K12_cpgd_0001_5epi"

# CHECKPOINT=K12_cpgd_005
# RESULT_DIR="K12_cpgd_005_5epi"

# CHECKPOINT=K12_cpgd_0005
# RESULT_DIR="K12_cpgd_0005_5epi"

# CHECKPOINT=K12_cpgd_0001_weightClip
# RESULT_DIR="K12_cpgd_0001_weight_clip_5epi"

# CHECKPOINT=K12_cpgd_001_weightClip
# RESULT_DIR="K12_cpgd_001_weight_clip_5epi"

# CHECKPOINT=log/grpo_drift_001_5
# RESULT_DIR="grpo_drift_001_5epi"


# ================ #
# ====== 7B ====== #
# ================ #

# python mmk12/evaluate_mmk12.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python mathvista/evaluate_mathvista.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python wemath/evaluate_wemath_vllm_qwen.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python olympiadbench/evaluate_olympiadbench.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python mathverse/evaluate_mathverse.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python mathvision/evaluate_mathvision.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}


# CHECKPOINT=../..//CPGD/log/K12_cpgd_32b_5epi-new
# RESULT_DIR="cpgd001_32B-5episode"

# python mmk12/evaluate_mmk12.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python mathvista/evaluate_mathvista.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python wemath/evaluate_wemath_vllm_qwen.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python olympiadbench/evaluate_olympiadbench.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python mathverse/evaluate_mathverse.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
# python mathvision/evaluate_mathvision.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}

# python evaluate_gaokao.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}

CHECKPOINT=../ckpts/qwen3-8b_cpgd_deepscaler_hybrid/nodes4_train_bsz512_mini_bsz512_rollout_n16_clip_eps_high0.2_lr_warmupconstant_total_epoch10/global_step_600/actor_agent_0/huggingface
RESULT_DIR="cpgd001_7b_deepscaler/nodes4_train_bsz512_mini_bsz512_rollout_n16_clip_eps_high0.2_lr_warmupconstant_total_epoch10_step600"

python olympiadbench/evaluate_olympiadbench.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR}