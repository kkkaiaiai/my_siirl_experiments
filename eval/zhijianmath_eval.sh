#!/bin/bash

export HOME="/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/siiRL/"

# MODLE=$HOME/../models/Qwen3-8B-Base
MODLE=/inspire/hdd/project/qproject-multireasoning/shaowenqi-shaowenqi/liuzongkai/siiRL/ckpts/qwen3-8b_cpgd_deepscaler_hybrid/nodes4_train_bsz512_mini_bsz512_rollout_n16_clip_eps_high0.2_lr_warmupconstant_total_epoch10/global_step_60/actor_agent_0/huggingface
DATA_NAME=aime_2025
FAILE_PATH="$HOME/data/$DATA_NAME/test.parquet"
cd $HOME

ray start --head

python ./eval/dapo_eval/eval.py \
    --model $MODLE \
    --file $FAILE_PATH \
    --num_samples 32 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 4096 \
    --num_workers 8 \
    --output_dir "./eval/eval_results/$DATA_NAME/30step" \
    --strict_box_verify
    
ray stop