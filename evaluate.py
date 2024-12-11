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


argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', type=str, default='qwen/Qwen2.5-1.5B')
argparser.add_argument('--dataset', type=str, default='gsm8k')
argparser.add_argument('--model_path', type=str, default='checkpoint-1.5B')
args = argparser.parse_args()


question = 'James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?'
prompt = '''<|im_start|>user
{question}<|im_end|>

<|im_start|>assistant'''

input = prompt.format(question = question)

model_name = args.model_name

based_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                   low_cpu_mem_usage=True,
    # quantization_config=quant_config,
    # torch_dtype=torch.float16,
    device_map={"":0}
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
based_model.from_pretrained(args.model_path)


output = based_model.generate(tokenizer(input, return_tensors='pt').input_ids.to('cuda'), early_stopping=True)


print(tokenizer.decode(output[0]))