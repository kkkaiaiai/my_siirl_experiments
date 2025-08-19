CHECKPOINT=../../MM-EUREKA-verl/checkpoints/cpgd-verl-debug-multi-node/CPG-vllm-4node-entropy00001-2025-06-01_08-12-47/global_step_610/actor/huggingface
RESULT_DIR=cpg_verl-en00001

python mmk12/evaluate_mmk12.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python mathvista/evaluate_mathvista.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python wemath/evaluate_wemath_vllm_qwen.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python olympiadbench/evaluate_olympiadbench.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python mathverse/evaluate_mathverse.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python mathvision/evaluate_mathvision.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}