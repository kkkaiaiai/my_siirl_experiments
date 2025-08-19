HOME=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/siiRL/ckpts

# CHECKPOINT=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/models/Qwen3-8B-Base
# RESULT_DIR="dapo17k"
# model_name=qwen3_8b_base

# python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 4 --batch_size 256 --model_name $model_name --datasets aime_2024

# CHECKPOINT=$HOME/Qwen3-8B-Base_cpgd_DAPO-Math-17k_hybrid/nodes4_train_bsz1024_mini_bsz1024_rollout_n16_clip_eps_high0.2_lr_warmupconstant_total_epoch10/global_step_30/actor_agent_0/huggingface
# RESULT_DIR="dapo17k"
# model_name=cpgd_30step

# python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 32 --batch_size 256 --model_name $model_name --datasets aime_2024
# # python dapo_eval/eval_math_ray.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 4 --batch_size 256 --model_name $model_name --datasets aime_2024

# CHECKPOINT=$HOME/Qwen3-8B-Base_cpgd_DAPO-Math-17k_hybrid/nodes4_train_bsz1024_mini_bsz1024_rollout_n16_clip_eps_high0.2_lr_warmupconstant_total_epoch10/global_step_160/actor_agent_0/huggingface
# RESULT_DIR="dapo17k"
# model_name=cpgd_160step
# python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 256 --batch_size 256 --model_name $model_name

# CHECKPOINT=$HOME/Qwen3-8B-Base_cpgd_DAPO-Math-17k_hybrid/nodes4_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.25_lr_warmup_styleconstant_total_epoch10/global_step_40/actor_agent_0/huggingface
# RESULT_DIR="dapo17k"
# model_name=cpgd_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.25_40step
# python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 256 --batch_size 256 --model_name $model_name

# CHECKPOINT=$HOME/Qwen3-8B-Base_cpgd_DAPO-Math-17k_hybrid/nodes4_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.25_lr_warmup_styleconstant_total_epoch10/global_step_60/actor_agent_0/huggingface
# RESULT_DIR="dapo17k"
# model_name=cpgd_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.25_60step
# python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 256 --batch_size 256 --model_name $model_name

# CHECKPOINT=$HOME/Qwen3-8B-Base_cpgd_DAPO-Math-17k_hybrid/nodes4_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10_positiveNLL0.1/global_step_30/actor_agent_0/huggingface
# RESULT_DIR="dapo17k"
# model_name=positiveNLL0.1_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_30step
# python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 256 --batch_size 256 --model_name $model_name

# CHECKPOINT=$HOME/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_30/actor_agent_0/huggingface
# RESULT_DIR="dapo17k"
# model_name=passk_klCov01coef_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_30step
# python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 256 --batch_size 256 --model_name $model_name

CHECKPOINT=$HOME/Qwen3-8B-Base_cpgd_passk_DAPO-Math-17k_hybrid/nodes4_passk_klCov01coef_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_lr_warmup_styleconstant_total_epoch10/global_step_160/actor_agent_0/huggingface
RESULT_DIR="dapo17k"
model_name=passk_klCov01coef_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.2_160step
python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 256 --batch_size 256 --model_name $model_name