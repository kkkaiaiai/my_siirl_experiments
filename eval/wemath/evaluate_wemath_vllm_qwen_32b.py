import argparse
import json
import os
import random
import time
import re

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

ds_collections = {
    # 'zh_exam_k12_single_test': {
    #     'root': '/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/shaowenqi-shaowenqi/ReasoningData/MMCoT',
    #     'annotation': '/inspire/hdd/global_user/shaowenqi-shaowenqi/mengfanqing/InternVL/internvl_chat/eval/zh_exam_k12/merged_cot_single_test.json',
    # },
    'wemath_testmini': {
        'root': '/inspire/hdd/global_user/shaowenqi-shaowenqi/mengfanqing/We-Math/data',
        'annotation': '/inspire/hdd/global_user/shaowenqi-shaowenqi/mengfanqing/We-Math/testmini.json',
    }
}

SYSTEM_PROMPT = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. Th answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer assistant's output should start with <answer> and end with </answer>."

def evaluate_chat_model():
    random.seed(args.seed)

    for ds_name in args.datasets:
        data_tmp = json.load(open(ds_collections[ds_name]['annotation'], encoding='utf-8'))
        inputs = []
        data = []
        cnt = 0
        for datatmp in data_tmp:
            datatmp["ID"] = str(cnt)
            cnt += 1
            data.append(datatmp)
            
        print(len(data),'len(data)')
        for data_item in data:
            image_path = 'file://' + os.path.join(ds_collections[ds_name]['root'], data_item["image_path"])
            
            data_item['query'] = data_item["question"] + '\nOptions: ' + data_item["option"]
            
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path
                        },
                        {
                            "type": "text",
                            "text": data_item['query']
                        },
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_data, _ = process_vision_info(messages)

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image_data
                },
            })
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids, skip_special_tokens=False)
        # sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, stop_token_ids=stop_token_ids, skip_special_tokens=False)
        # sampling_params = SamplingParams(temperature=0.9, top_p=0.9, top_k=50, max_tokens=2048, stop_token_ids=stop_token_ids, skip_special_tokens=False)
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
        
        outputs = []
        for data_item, model_output in zip(data, model_outputs):
            data_item['response'] = model_output.outputs[0].text
            outputs.append(data_item)

    
        temp = {}
        for data_item in outputs:
            id = data_item['ID']
            temp[id] = data_item

        print(f'Evaluating {ds_name} ...')
        time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
        results_file = f'{ds_name}_{time_prefix}.json'
        output_path = os.path.join(args.out_dir, results_file)
        json.dump(temp, open(output_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
        print('Results saved to {}'.format(output_path))

        # cmd = f'python eval/mathvista/extract_answer.py --output_file {results_file}'
        # print(cmd)
        # os.system(cmd)
        #
        # cmd = f'python eval/mathvista/calculate_score.py --output_file {results_file} --score_file {results_file[:-5]}_score.json'
        # print(cmd)
        # os.system(cmd)

        # cmd = f'python eval/zh_exam_k12/extract_calculate.py --output_file {results_file}'
        # print(cmd)
        # os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='wemath_testmini')
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=4,
        # max_model_len=32768,
        limit_mm_per_prompt={"image": 1},
        # mm_processor_kwargs={"max_dynamic_patch": 6},
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None
    
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
