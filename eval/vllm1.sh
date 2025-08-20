HOME=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/siiRL/ckpts
qwen3base=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/models/Qwen3-8B-Base

CHECKPOINT=$qwen3base
RESULT_DIR="dapo17k"
model_name=qwen3_base
python dapo_eval/eval_aime25.py --checkpoint ${CHECKPOINT} --out-dir results/${RESULT_DIR} --temperature 0.6 --top_p 0.95 --num_samples 256 --batch_size 256 --model_name $model_name
