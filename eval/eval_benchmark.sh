set -x

CHECKPOINT=${1}
DATASET=${2}
CHECKPOINT="$(pwd)/${CHECKPOINT}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Save original arguments
ARGS=("$@")

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto)
      GPUS=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
echo "GPUS: ${GPUS}"


if [ ${DATASET} == "mathvista-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      mathvista/evaluate_mathvista.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathverse-testmini" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      mathverse/evaluate_mathverse.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mathvision-test" ]; then
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --master_addr=127.0.0.1 \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      mathvision/evaluate_mathvision.py --checkpoint ${CHECKPOINT} "${ARGS[@]:2}"
fi

if [ ${DATASET} == "OlympiadBench" ]; then
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc_per_node=${GPUS} \
        --master_port=${MASTER_PORT} \
        olympiadbench/evaluate_olympiadbench.py --checkpoint ${CHECKPOINT}  "${ARGS[@]:2}"
fi

if [ ${DATASET} == "mmk12" ]; then
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --nproc_per_node=${GPUS} \
        --master_port=${MASTER_PORT} \
        mmk12/evaluate_mmk12.py --checkpoint ${CHECKPOINT}  "${ARGS[@]:2}"
fi

if [ ${DATASET} == "wemath" ]; then
    python \
      wemath/evaluate_wemath_vllm_qwen.py --checkpoint ${CHECKPOINT} --datasets wemath_testmini "${ARGS[@]:2}"
fi

if [ ${DATASET} == "wemath-32b" ]; then
    python \
      wemath/evaluate_wemath_vllm_qwen_32b.py --checkpoint ${CHECKPOINT} --datasets wemath_testmini "${ARGS[@]:2}"
fi