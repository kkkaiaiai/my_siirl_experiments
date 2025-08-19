HOME=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/siiRL/ckpts

CHECKPOINT=$HOME/Qwen3-8B-Base_cpgd_DAPO-Math-17k_hybrid/nodes4_train_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.25_lr_warmup_styleconstant_total_epoch10/global_step_40/actor_agent_0/huggingface
RESULT_DIR="dapo17k"
model_name=cpgd_bsz512_mini_bsz128_rollout_n16_clip_eps_high0.25_40step
python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.7 --num_samples 256 --batch_size 256 --model_name $model_name