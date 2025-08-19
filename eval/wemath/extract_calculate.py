import argparse
import string
import logging
import copy as cp
from collections import defaultdict
import pandas as pd
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import regex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# OpenAI
import openai
from tqdm import tqdm
from utilities import *

openai.api_key = os.getenv('OPENAI_API_KEY')
# openai.base_url = 'https://api.claudeshop.top/v1/'
# openai.api_key = 'sk-f2a6NrUuZ5ul1F5DzDohrFmdg2SN4oOq9jieJAFSZDuusi9O'
# openai.base_url = 'https://open.xiaojingai.com/v1/'
# openai.base_url = 'http://35.220.164.252:3888/v1/'
# openai.api_key = 'sk-ziEcEd3wh3wfRkVkRfCzhPEXyGx7BsDgNYeSbRvEzDFMh3f5'
# openai.api_key = 'sk-bPpiqqm8EYFmUS0F77oTdvWcgkNujsMCw1bJCVIL4aSLUjRj'
# openai.base_url = 'http://35.220.164.252:3888/v1/'

# openai.base_url = 'http://35.220.164.252:3888/v1/'
# openai.api_key = 'sk-Cm4nWQYOac4BnF7VKVdq3QHnTKvTkELwlLSlYDP75fqRga93'

openai.base_url = 'http://35.220.164.252:3888/v1/'
openai.api_key = 'sk-ePnCpGDm4YMqhNCL3kxcLJCLXkdIhiKzBg5d5Zzag79dRgQX'
# print(openai.api_key)



def build_zh_exam_k12_gpt4_prompt(question_data):
#     prompt = """You are given a question, the solution, the correct answer. Please determine if the solution matches the correct answer.
# Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
# The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
# Return only "Yes" if the solution is correct or "No" if it is incorrect.
# Only return "Yes" or "No" with no additional text or formatting.

# Question: 
# {question}
# --------------------------------
# Correct Answer:
# {answer}
# --------------------------------
# Solution: 
# {solution}
# --------------------------------
# """
    prompt = """You are given a question, the correct answer and a model's answer. Please determine if the model's answer matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the model's answer and the correct answer.
Only the correctness of the model's answer matters.
Return only "Yes" if the model's answer is correct or "No" if it is incorrect.
Only return "Yes" or "No" with no additional text or formatting.

Question:
{question}
--------------------------------
Correct Answer:
{correct_answer}
--------------------------------
Model's Answer:
{solution}
--------------------------------"""
#     task_description = """
# Please review the question, its corresponding answer, and the answer details to determine if the provided answer is correct. 
# Ignoring any differences in formatting, including LaTeX, symbols, or spacing. Focus only on whether the provided answer matches the correct answer.
# You only need to return Yes or No.
# """
    question = question_data['query']
    answer = question_data['answer']
    response = str(question_data['response'])
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        completion_match = regex.findall(
            r"\\boxed\{((?:[^{}]+|(?P<BRACES>\{(?:[^{}]+|(?P>BRACES))*\}))*)\}", response, re.DOTALL
        )
        response = completion_match[-1][0].strip() if completion_match else response

    print(response)
        
#     prompt = task_description
#     prompt += f'Question: {question}' + '\n'
#     prompt += f'Answer: {answer}' + '\n'
#     prompt += f'Answer detail: {answer_detail}' + '\n'
#     prompt += f'Provided answer: {response}' + '\n'
#     prompt += 'Correctness:'
    prompt = prompt.format(question=question, correct_answer=answer, solution=response)
    return prompt



def score_answer(response, problem):
    prompt = build_zh_exam_k12_gpt4_prompt(problem)
    logging.info(f"id: {problem['ID']}")
    completion = get_chat_response_vlmevalkit(prompt)
    if completion.lower() == 'yes':
        return True, problem['ID']
    elif completion.lower() == 'no':
        return False, problem['ID']

def ZhExamK12_acc(results):
    correct = 0
    total = len(results)
    for result in results:
        if result['score']:
            correct += 1
    return {'correct': correct, 'total': total, 'accuracy': correct / total}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    # model
    parser.add_argument('--llm_engine', type=str, default='gpt-4o', help='llm engine',
                        choices=['gpt-3.5-turbo', 'gpt-3.5', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-turbo', 'gpt-4o'])
    parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    # output
    parser.add_argument('--output_label', type=str, default='extract', help='label for the output file')
    args = parser.parse_args()

    # args
    label = args.response_label
    result_file = os.path.join(args.output_dir, args.output_file)

    if args.output_label != '':
        output_file = result_file.replace('.json', f'_{args.output_label}.json')
    else:
        output_file = result_file

    # read results
    print(f'Reading {result_file}...')
    results = read_json(result_file)

    # full pids
    test_ids = list(results.keys())
    if args.number > 0:
        test_ids = test_ids[:min(args.number, len(test_ids))]
    print('Number of testing problems:', len(test_ids))

    # # tqdm, enumerate results
    # for i, pid in enumerate(tqdm(test_pids)):
    #     problem = results[pid]

    #     assert label in problem
    #     response = problem[label]

    #     extraction = extract_answer(response, problem)
    #     results[pid]['extraction'] = extraction
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(score_answer, results[sample_id][label], results[sample_id]) for sample_id in test_ids]
        
        for future in as_completed(futures):
            score, id = future.result()
            results[id]['score'] = score

    print(f'Saving results to {output_file}...')
    save_json(results, output_file)
    print(f'Results saved.')



    results = [v for _, v in results.items()]
    scores = ZhExamK12_acc(results)
    print(scores)
    print(f"Saving scores to {result_file.replace('.json', f'_score.json')}...")
    save_json(scores, result_file.replace('.json', f'_score.json'))
    print(f'Scores saved.')
