CHECKPOINT=../../CPGD/log/K12_cpgd_001_IS
RESULT_DIR=cpg_IS

python mmk12/evaluate_mmk12.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python mathvista/evaluate_mathvista.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python wemath/evaluate_wemath_vllm_qwen.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python olympiadbench/evaluate_olympiadbench.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python mathverse/evaluate_mathverse.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}
python mathvision/evaluate_mathvision.py --checkpoint ../log_new/${CHECKPOINT} --out-dir results/${RESULT_DIR}