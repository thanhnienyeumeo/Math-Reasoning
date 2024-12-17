import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, Trainer, pipeline
from peft import LoraConfig
from datasets import Dataset
import datasets
from trl import SFTTrainer, PPOTrainer

from tqdm import tqdm
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='qwen/Qwen2.5-1.5B-Instruct')
argparser.add_argument('--dataset', type=str, default='gsm8k')
argparser.add_argument('--model_path', type=str, default='qwen1.5_checkpoint')
args = argparser.parse_args()


model_name = args.model_name
model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(model_path,
                                            low_cpu_mem_usage=True,
    # torch_dtype=torch.float16,
    # quantization_config=quant_config,
    # torch_dtype=torch.float16,
    device_map={"":0}
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

#load dataset:
dataset = None
if args.dataset == 'gsm8k':
    # dataset = datasets.load_dataset('gsm8k', "main")
    dataset = datasets.load_dataset("gsm8k", "main")

train_dataset, test_dataset = dataset['train'], dataset['test']





import os
import time
from groq import Groq

client = Groq(
    api_key='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' #get tokenn from groq
)

def generate_answer_groq(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=2048,
    )

    return chat_completion.choices[0].message.content

prefix_prompt = '''Your task is to evaluate a generated answer from a model by comparing it with a correct reference answer. Determine if the generated answer matches the correct answer. If the generated answer is correct, respond with 1. If it is incorrect, respond with 0. Do not provide any other responses besides 1 or 0.'''

def generate_prompt(question, answer, i):
        input = f'''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{question}\nPut your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n'''
        output = model.generate(tokenizer(input, 
        return_tensors='pt').input_ids.to('cuda'), 
        max_new_tokens = 2048,
        eos_token_id = tokenizer("<|im_end|>")['input_ids'],
        temperature = 0.2
        )
        answer_ = tokenizer.decode(output[0])
        # print(answer_model)
        answer_model = ''
        try:
            answer_model = answer_.split("assistant")[2]
        except:
            print(f'ans from question {i} is not in the correct format')
            answer_model = answer_
            print(answer_)
        return f'''{prefix_prompt}\n\ntrue answer: {answer}\n\ngenerated answer: {answer_model}''', answer_

import time
from tqdm import tqdm

cnt = 0
all_ans = []
question = 'question'
answer = 'answer'
for i in tqdm(range(len(test_dataset))):
        
        prompt, ans = generate_prompt(test_dataset[question][i], test_dataset[answer][i],i)
        # print(prompt)
        score = generate_answer_groq(prompt)
        if score == 1 or score == '1':
            cnt += 1
        elif score == 0 or score == '0':
            pass
        else:
            # print('Error: ', score)
            raise ValueError('Error: ', score)
        time.sleep(0.2)
        # print("\n")
        ans = ans + '\n' + str(score) + '\n'
        all_ans.append(ans)
        # print('ans: ', ans)
        if i % 100 == 0:
            print("Accuracy until sample {}: ".format(i), cnt/(i+1))
            with open(f'all_ans_{model_path}.txt', 'w') as f:
                f.write('\n'.join(all_ans))
            print("Saved all answers")

print("Accuracy: ", cnt/len(test_dataset))
        