cd "/home/ma-user/work/DRL/qwen_eval"

WORKSPACE="/home/ma-user/work"
export PATH="$WORKSPACE/anaconda3/bin:$PATH"
source "/home/ma-user/anaconda3/etc/profile.d/conda.sh"
conda activate "$WORKSPACE/anaconda3/envs/drl"

temperature=${temperature:-"0.6"}
nsamples=${nsamples:-"256"}
topp=${topp:-"0.95"}
num_workers=${num_workers:-"8"}
tensor_parallel_size=${tensor_parallel_size:-"1"}
start=${start:-'0'}
end=${end:-'-1'}
PROMPT_TYPE="qwen3"
# MODEL_NAME_OR_PATH="/home/ma-user/work/DRL/Model/Qwen/Qwen3-4B-Base"
MODEL_NAME_OR_PATH="/home/ma-user/work/DRL/checkpoint/drl-60"
DATA_NAMES="aime25"
NUM_TEST_SAMPLE=-1
MAX_TURN=${4:-"1"}
max_tokens_per_call=${max_tokens_per_call:-'8096'}
OUTPUT_DIR="./outputs"
SPLIT="test"
csv_output=${csv_output:-"./math_eval_results_new.csv"}

# Environment variables for better performance
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -u math_eval_multi.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_names ${DATA_NAMES} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 42 \
    --temperature ${temperature} \
    --n_sampling ${nsamples} \
    --top_p ${topp} \
    --start ${start} \
    --end ${end} \
    --save_outputs \
    --num_shots 0 \
    --overwrite \
    --num_workers ${num_workers} \
    --tensor_parallel_size ${tensor_parallel_size} \
    --max_tokens_per_call ${max_tokens_per_call} \
    --max_func_call ${MAX_TURN} \
    --csv_output ${csv_output}
