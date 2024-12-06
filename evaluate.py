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
argparser.add_argument('--model_path', type=str, default='checkpoint-37365-20241206T082926Z-001.zip')
args = argparser.parse_args()

model_name = args.model_name

based_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                   low_cpu_mem_usage=True,
    # quantization_config=quant_config,
    # torch_dtype=torch.float16,
    device_map={"":0}
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
based_model.load_state_dict(torch.load(args.model_path))

prompt = '''<|im_start|>user
Jen buys and sells candy bars. She buys candy bars for 80 cents each and sells them for a dollar each. If she buys 50 candy bars and sells 48 of them, how much profit does she make in cents?<|im_end|>

<|im_start|>assistant'''
output = based_model.generate(tokenizer(prompt, return_tensors='pt').input_ids.to('cuda'), early_stopping=True)


print(tokenizer.decode(output[0]))