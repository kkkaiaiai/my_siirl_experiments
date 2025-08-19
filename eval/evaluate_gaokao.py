import argparse
import json
import os
import random
import time

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from PIL import Image


SYSTEM_PROMPT_32B = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. Th answer should be enclosed within <answer> </answer> tags, i.e., Since $1+1=2$, so the answer is $2$. <answer> The answer is $\\\\boxed{2}$ </answer>, which means the final answer assistant's output should start with <answer> and end with </answer>."
SYSTEM_PROMPT = "Solve the question. The user asks a question, and you solves it. You first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$. The final answer must be wrapped using the \\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output should start with <think> and end with </answer>."

data_item = {
    "id": "6", 
    "question": "In a sailing race, athletes can use an anemometer to measure the magnitude and direction of the wind speed. The measured result is called the apparent wind speed in navigation. The vector corresponding to the apparent wind speed is the sum of the vector corresponding to the true wind speed and the vector corresponding to the ship's traveling wind speed. Among them, the magnitude of the vector corresponding to the ship's traveling wind speed is equal to that of the vector corresponding to the ship's speed, and the directions are opposite. Figure 1 shows the corresponding relationship between some wind force levels, names, and wind speed magnitudes. It is known that the vectors corresponding to the apparent wind speed and the ship's speed measured by a certain sailing athlete at a certain moment are shown in Figure 2 (the magnitude of the wind speed is the same as that of the vector, and the unit is m/s). Then what is the true wind? \nA. Gentle breeze \nB.Breeze \nC. Moderate breeze \nD. Fresh breeze",
    "image": Image.open("./data/gaokao/6thtiankong.png").convert("RGB"),
}
## "answer": "A",


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, default="gaokao")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # args.datasets = args.datasets.split(",")

    # print("datasets:", args.datasets)
    
    random.seed(args.seed)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=4,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    stop_token_ids = None

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_item["image"]},
                {"type": "text", "text": data_item["question"]},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_data, _ = process_vision_info(messages)

    input_data = {
        "prompt": prompt,
        "multi_modal_data": {"image": image_data},
    }
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids)
    model_outputs = llm.generate(input_data, sampling_params=sampling_params)

    print("Model response:")
    print(model_outputs[0].outputs[0].text)
    
    print(f"Evaluating ...")
    time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
    results_file = f"gaokao_{time_prefix}.json"
    output_path = os.path.join(args.out_dir, results_file)
    data_item["response"] = model_outputs[0].outputs[0].text
    del data_item["image"]
    json.dump(data_item, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    print("Results saved to {}".format(output_path))