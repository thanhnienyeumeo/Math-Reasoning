import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, Trainer, pipeline
from peft import LoraConfig
from datasets import Dataset
import datasets
from trl import SFTTrainer, PPOTrainer
from format import prompt_phi, prompt_qwen, prompt_llama
from tqdm import tqdm
import argparse
from copy import deepcopy
import json

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='qwen/Qwen2.5-1.5B-Instruct')
argparser.add_argument('--dataset', type=str, default='gsm8k')
argparser.add_argument('--model_path', type=str, default='checkpoint/qwen1.5_checkpoint')
argparser.add_argument('--type', type=str, default='qwen')
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


prompt = prompt_qwen if args.type == 'qwen' else prompt_phi if args.type == 'phi' else prompt_llama


cnt = 0
all_ans = []
all_true_ans = []
wrong_ans = []
#save all_ans and all_true_ans to file
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

cnt_not_format = 0
for question, answer in test_dataset:
    for i in tqdm(range(len(test_dataset))):
    # for i in tqdm(range(22,40)):

        input = prompt.format(instruction = test_dataset[question][i])
        # print(input)
        output = model.generate(tokenizer(input, return_tensors='pt').input_ids.to('cuda'), early_stopping=True, max_new_tokens=1024, eos_token_id = tokenizer("<|im_end|>")['input_ids'], temperature = 0.2)
        
        ori_ans = tokenizer.decode(output[0])
        ori_ans = ori_ans[ori_ans.index('<|im_start|>assistant\n'):]
        # print(ori_ans)
        ans = deepcopy(ori_ans)
        all_ans.append(ori_ans)
        true_ans = test_dataset[answer][i]
        true_ans = true_ans[true_ans.rindex(r'####') + 4:].strip()
        all_true_ans.append(true_ans)
        if "The answer is" in ans or "the answer is" in ans:
            ans = ans[ans.index("The answer is") + len("The answer is:"):]
            if '<|endoftext|>' in ans or '<|im_end|>' in ans:
                ans = ans[:ans.index('<')]
            ans = ans.strip()
        else:
            try:
                ans = ans[ans.rindex(r'\boxed{') + 7:ans.rindex('}')]
                if ans == '':
                    print('not having answer')
                    wrong_ans.append(ans)
                    print(ori_ans)
                    continue
            except:
                print('not format. Evaluating using groq....')
                print(ans)
                prompt2, _ = generate_prompt(test_dataset[question][i], test_dataset[answer][i],i)
                score = generate_answer_groq(prompt2)
                if score == 1 or score == '1':
                    cnt += 1
                elif score == 0 or score == '0':
                    wrong_ans.append(str(i) + ': ' + ans + ' | ' + test_dataset[answer][i])
                    
                else:
                    print('Error: ', score)
                    
                    # raise ValueError('Error: ', score)
                cnt_not_format +=1 
                continue
        
        
        
        if ans == true_ans:
            cnt += 1
        else:
            wrong_ans.append(str(i) + ': ' + ans + ' | ' + true_ans)
        
        if i % 100 == 0 or i == len(test_dataset) -1: 
          print('Current accuracy is: ', cnt/(i+1))
          with open('all_ans.json', 'w') as f:
              json.dump(all_ans, f)
          with open('all_true_ans.json', 'w') as f:
              json.dump(all_true_ans, f)
          with open('wrong_ans.json', 'w') as f:
                json.dump(wrong_ans, f)
        
    break


print("Accuracy: ", cnt/len(test_dataset))
print("Number of    not format: ", cnt_not_format)
        