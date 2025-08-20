import os
import json
import random
import json
import os
import pdb
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any

from examples import get_examples


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def lower_keys(example):
    new_example = {}
    for key, value in example.items():
        if key != key.lower():
            new_key = key.lower()
            new_example[new_key] = value
        else:
            new_example[key] = value
    return new_example


EXAMPLES = get_examples()


def load_prompt(data_name, prompt_type, num_shots):
    if not num_shots:
        return []

    if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
        data_name = "gsm8k"
    if data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
        data_name = "math"
    if data_name in ["sat_math"]:
        data_name = "mmlu_stem"
    if data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        data_name = "gaokao"

    if prompt_type in ["tool-integrated"]:
        prompt_type = "tora"
    if prompt_type == 'pal':
        data_name+='_pal'
    return EXAMPLES[data_name][:num_shots]

refl_sys = """Your role as an assistant involves thoroughly exploring questions through a systematic long
thinking process before providing the final precise and accurate solutions. This requires
engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection,
backtracing, and iteration to develop well-considered thinking process.
Please structure your response into two main sections: Thought and Solution.
In the Thought section, detail your reasoning process using the specified format:
‘‘‘
<|begin_of_thought|>
{{thought with steps separated with "\n\n"}}
<|end_of_thought|>
’’’
Each step should include detailed considerations such as analisying questions, summarizing
relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining
any errors, and revisiting previous steps.
In the Solution section, based on various attempts, explorations, and reflections from the Thought
section, systematically present the final solution that you deem correct. The solution should
remain a logical, accurate, concise expression style and detail necessary step needed to reach the
conclusion, formatted as follows:
‘‘‘
<|begin_of_solution|>
{{final formatted, precise, and clear solution}}
<|end_of_solution|>
’’’
Now, try to solve the following question through the above guidelines:
"""

myrefl_sys = """You are a clever student tasked with developing a well-reasoned approach to solving the given problem. Throughout your response, you must use **first-person narrative** to express your thinking as if you are talking to yourself.

As you work through the problem, engage in a thorough cycle of analysis by:
- Carefully analyzing the problem step by step.
- Regularly verifying and reassessing your reasoning, calculations, and conclusions.
- Revisiting and adjusting any earlier assumptions or errors in logic as you move forward.

Show the details of each step of your thought process clearly. Act as though you are thinking aloud by using the **first-person narrative**.
Once you are confident in your final solution, enclose it within \\boxed{{}}.
"""
myrefl_sys2 = """
You are a clever student tasked with developing a well-reasoned approach to solving the given problem. Throughout your response, you must use **first-person narrative** to express your thinking, such as “I need to...”, “I think...”, and “I believe...”. 

As you work through the problem, engage in a thorough cycle of analysis by:
- Carefully analyzing the problem step by step.
- Regularly verifying and reassessing your reasoning, calculations, and conclusions.
- Revisiting and adjusting any earlier assumptions or errors in logic as you move forward.

Remember to explain each step of your thought process clearly, and act as **self-thinking** by using the **first-person narrative**.
Once you are confident in your final solution, enclose it within \\boxed{{}}.
"""
myrefl_sys3 = """
You are a thoughtful and diligent student tasked with solving a problem. As you work through the problem, document your thought process in a reflective, first-person narrative. Think of yourself as talking to yourself through each step. Consider each step carefully, question your reasoning, and adjust as needed to arrive at a sound solution. Here's how you should proceed:

1. **Step-by-Step Analysis**: Start by thoroughly understanding the problem. Identify what is provided and what is being asked. Consider high-level strategies or approaches first, and then break them down into smaller, manageable steps. Ensure you address each component one at a time and do not skip over any details.
   
2. **Self-Questioning**: As you work through each step, ask yourself reflective questions like, "Is this correct?", "Does it make sense?", or "What might I be overlooking?" Be critical of your own reasoning, and adjust your approach as needed. Use <confidence></confidence> notation to express your confidence and evaluate the progress about solving the problem.
   
3. **Reassessment**: If you notice a mistake or feel uncertain about your approach, reassess your work. Go back and revise your assumptions, logic, or calculations to correct any missteps, ensuring you're on the right track.

4. **Alternative Approaches**: If you find yourself stuck or unsure about the current method, consider alternative approaches. Look at the problem from different angles, and if one method feels insufficient, explore others.

5. **Clear Detailing**: For each step, explain your reasoning clearly and in simple language. Make sure anyone who follows your work can easily understand the logic behind your decisions and the steps you've taken.

6. **Final Solution**: Once you're confident in your solution, enclose it in \\boxed{} to highlight your final answer.

**Your goal is to approach the problem in a reflective, iterative manner, ensuring that no steps are skipped and no assumptions go unchecked.**
"""
PROMPT_TEMPLATES = {
    "direct": ("Question: {input}\nAnswer: ", "{output}", "\n\n"),
    "cot": ("Question: {input}\nAnswer: ", "{output}", "\n\n\n"),
    "pal": ("Question: {input}\n\n", "{output}", "\n---\n"),
    "tool-integrated": ("Question: {input}\n\nSolution:\n", "{output}", "\n---\n"),
    "self-instruct": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "tora": ("<|user|>\n{input}\n<|assistant|>\n", "{output}", "\n"),
    "pot_trigger_direct": ("User:{input}\nLet's write a program.\nAssistant:\n\n", "{output}", "\n\n"),
    "pot_trigger": ("User:{input}\nLet's write a program.\nAssistant:\n\n", "{output}", "\n\n"),
    "ds_autocode_prompt": ("User:{input}\nAssistant:", "{output}", "\n\n"),
    "ds_directly": ("User:{input}\nPlease either write a program or reason step by step.\nAssistant:\n\n", "{output}", "\n\n"),
    "ds_directly_4shot": ("User:{input}\nPlease either write a program or reason step by step.\nAssistant:\n\n", "{output}", "\n\n"),
    "cot_trigger": ("User:{input}\nLet's reason step by step.\nAssistant:\n\n", "{output}", "\n\n"),
    
    "below_pot_trigger": ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input}\nLet's write a program.\n\n\n### Response:", "{output}", "\n\n"),
    "below_cot_prompt":  ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input}\n\n### Response:", "{output}", "\n\n"),
    "below_cot_trigger": ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input}\nLet's reason step by step.\n\n\n### Response:","{output}", "\n\n"),
    "mix_trigger": ("User:{input}\nYou should solve this problem using CoT (think step by step) or PoT (write a program).\nAssistant:\n", "{output}", "\n\n"),
    
    "ds_pot_trigger": (
        "User: {input}\n"
        "\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "wizard_zs": (
        "### Instruction:\n{input}\n\n### Response: Let's think step by step.",
        "{output}",
        "\n\n\n",
    ),
    "platypus_fs": (
        "### Instruction:\n{input}\n\n### Response:\n",
        "{output}",
        "\n\n\n",
    ),
    "deepseek-math": (
        "User: {input}\nPlease reason step by step, "
        "and put your final answer within \\boxed{{}}.\n\nAssistant:",
        "{output}",
        "\n\n\n",
    ),
    "kpmath": (
        "User: Please reason step by step and put your final answer at the end "
        'with "The answer is: ".\n\n{input}\n\nAssistant:',
        "{output}",
    ),
    "jiuzhang": (
        "## Question\n{input}\n\n## Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_tora": (
        "## Question\n{input}\n\n## Code Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "jiuzhang_nl": (
        "## Question\n{input}\n\n## Natural Language Solution\n",
        "{output}",
        "\n\n\n",
    ),
    "mmiqc": (
        'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{input}\n\n',
        "{output}",
        "\n\n\n",
    ),
    "qwen25-math-cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-autocode-none": (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "dpsk": (
        "<｜begin▁of▁sentence｜>"
        "<｜User｜>{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        "<｜Assistant｜>",
        "{output}",
        "\n\n",
    ),
    "qwen25-none": (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-reflection": (
        "<|im_start|>system\nPlease respond in a self-thinking manner.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    
    "our_o1_longcot": (
        "<|im_start|>system\nYou are an advanced AI language model specializing in solving math and programming problems step by step. Carefully analyze each part of the problem, verify the accuracy of your reasoning with relevant facts and data, and provide clear, logical solutions. Reflect on and review your approach throughout the problem-solving process to ensure precision and thoroughness. Always think through the problem step by step and provide your answers accordingly.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    
    "32b-reflection": (
        f"<|im_start|>system\n{refl_sys}<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "7b-reflection": (
        f"<|im_start|>system\n{myrefl_sys}<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "7b-reflection-0": (
        f"<|im_start|>system\n{myrefl_sys2}<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "7b-reflection-1": (
        f"<|im_start|>system\n{myrefl_sys3.replace(r'{}',r'{{}}')}<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen_default": (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-pot": (
        "<|im_start|>system\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "qwen25-math-autocode": (
        "<|im_start|>system\nPlease either write a program or reason step by step.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "abel": (
        "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
        "{output}",
        "\n\n",
    ),
    "shepherd": ("{input}\n", "{output}", "\n\n\n"),
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mathstral": (
        "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
        "{output}",
        "\n\n",
    ),
    "internlm-math-fs": ("Question:{input}\nAnswer:", "{output}", "\n"),
    "internlm-math-chat": (
        "<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    ),
    "mistral": (
        "[INST] {input}[/INST]",
        "{output}",
        "\n\n",
    ),
    "numina": ("Problem:\n{input}\nSolution:", " {output}", "\n\n"),
}


def construct_prompt(example, data_name, args):
    if args.adapt_few_shot and data_name in [
        "gaokao2024_I",
        "gaokao2024_II",
        "gaokao_math_qa",
        "gaokao2024_mix",
        "cn_middle_school",
    ]:
        demos = load_prompt(data_name, args.prompt_type, 5)
    else:
        demos = load_prompt(data_name, args.prompt_type, args.num_shots)
    prompt_type = args.prompt_type
    if prompt_type == "platypus_fs":
        prompt_type = "cot"
    if prompt_type == "tool-integrated":
        prompt_type = "tora"
    if args.prompt_type == 'ds_cot_trigger':
        args.prompt_type = 'deepseek-math'
    prompt_temp = PROMPT_TEMPLATES[args.prompt_type]

    splitter = prompt_temp[2]
    input_template, output_template, splitter = (
        prompt_temp[0],
        prompt_temp[1],
        prompt_temp[2],
    )
   
    demo_prompt = splitter.join(
        [
            input_template.format(input=q) + output_template.format(output=a)
            for q, a in demos
        ]
    )
    
    context = input_template.format(input=example["question"])

    if len(demo_prompt) == 0 :
        full_prompt = context
    else:
        full_prompt = demo_prompt + splitter + context

    if args.prompt_type == "platypus_fs":
        full_prompt_temp = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        )
        full_prompt = full_prompt_temp.format(instruction=full_prompt)

    if prompt_type == "tora":
        full_prompt = (
            """Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:

- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a `\\boxed{}` without any units.
- Utilize the `pi` symbol and `Rational`` from Sympy for $\pi$ and fractions, and simplify all fractions and square roots without converting them to decimal values.

Here are some examples you may refer to:

---

"""
            + full_prompt
        )

    return full_prompt.strip(" ")  # important!


key_map = {
    "gt": "Ground Truth",
    "pred": "Prediction",
    "gt_cot": "Reference CoT",
    "score": "Score",
}


def show_sample(sample, print_all_preds=False):
    print("==" * 20)
    for key in ["idx", "type", "level", "dataset"]:
        if key in sample:
            # capitalize
            print("{}: {}".format(key[0].upper() + key[1:], sample[key]))
    print("Question:", repr(sample["question"]))
    if "code" in sample:
        if print_all_preds:
            for code in sample["code"]:
                print("-" * 20)
                print("code:", code)
            print("Execution:", sample["report"])
        else:
            print("Solution:\n", sample["code"][0])
            print("Execution:", sample["report"][0])
    if "pred" in sample:
        print("Prediction:", repr(sample["pred"][0]))
    for key in ["gt", "score", "unit", "gt_cot"]:
        if key in sample:
            _key = key_map.get(key, key)
            print("{}: {}".format(_key, repr(sample[key])))
    print()
