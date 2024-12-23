import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, Trainer, pipeline
from peft import LoraConfig
from datasets import Dataset
import datasets
from trl import SFTTrainer, PPOTrainer

from tqdm import tqdm
#load model name
# model_name = "qwen/Qwen2.5-0.5B"
import argparse
from format import prompt_phi, prompt_qwen, prompt_llama

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
argparser.add_argument('--dataset', type=str, default='gsm8k')
argparser.add_argument('--model_path', type=str, default='checkpoint/qwen0.5/results/checkpoint-37365')
argparser.add_argument('--question', type=str, default=None)
args = argparser.parse_args()

question_train = 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?'
question2 = 'Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?'
question = 'JHenry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?'
if args.question is not None:
    question = args.question
prompt = prompt_qwen

prompt_COT = '''<|system|>\nYou are a helpful assistant.
<|user|>\n{question}\n
Reason step by step, and your final answer within \\boxed{{}}.'''


input = prompt.format(instruction = question2) #use for qwen
input_phi = prompt_phi.format(instruction = question_train) #use for phi
input_COT = prompt_COT.format(question = question_train) #use for COT
model_name = args.model_name
model_path = args.model_path

based_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                   low_cpu_mem_usage=True,
    # quantization_config=quant_config,
    # torch_dtype=torch.float16,
    # torch_dtype = torch.float32,
    device_map={"":0}
)
# based_model.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)



output = based_model.generate(tokenizer(input, return_tensors='pt').input_ids.to('cuda'), early_stopping=True, eos_token_id = tokenizer("<|im_end|>")['input_ids'], max_new_tokens = 512)


print(tokenizer.decode(output[0]))