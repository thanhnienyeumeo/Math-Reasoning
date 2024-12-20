# -*- coding: utf-8 -*-
"""Train.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17SY4I5fp4wMEntkmDOGuDN01ZLTi2B7P
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, Trainer, pipeline, BitsAndBytesConfig
from peft import LoraConfig
from datasets import Dataset
import datasets
from trl import SFTTrainer, PPOTrainer
from format import prompt_phi, prompt_qwen, prompt_llama
from tqdm import tqdm
#load model name
# model_name = "qwen/Qwen2.5-0.5B"
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', '-m', type=str, default='microsoft/Phi-3.5-mini-instruct')
argparser.add_argument('--dataset', '-d', type=str, default='metamath')
argparser.add_argument('--model_path', type=str, default=None)
argparser.add_argument('--type', type=str, default='phi')
# argparser.add_argument('--bf16', type=bool, default=False)
argparser.add_argument('--quant', '-q', type=bool, default=False)
#rank of lora
argparser.add_argument('--rank', '-r', type=int, default=64)
argparser.add_argument('--num_samples', '-n', type=int, default=None)
argparser.add_argument('--save_path', type=str, default=None)
argparser.add_argument('--save_strategy', '-s', type=str, default='epoch')
argparser.add_argument('--save_steps', '-ss', type=int, default=60000)
args = argparser.parse_args()

model_name = args.model_name
model_path = args.model_path
type = args.type
# model_name = "bkai-foundation-models/vietnamese-llama2-7b-120GB"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)
if model_path is not None:
    based_model = AutoModelForCausalLM.from_pretrained(model_path,
    quantization_config=quant_config if args.quant else None,
    torch_dtype=torch.float16,
    device_map={'': torch.cuda.current_device()}
    )
    # Nếu checkpoint có chứa optimizer state
    # optimizer = AdamW(based_model.parameters(), lr=2e-4)
    # checkpoint = torch.load(f"{checkpoint_path}/optimizer.pt")
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # optimizer = optimizer.to("cuda")

    # scheduler = checkpoint.get("scheduler_state_dict", None)
    # if scheduler:
    #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
# Nếu bạn dùng learning rate scheduler
else:
    based_model = AutoModelForCausalLM.from_pretrained(model_name,
    quantization_config=quant_config if args.quant else None,
    #   torch_dtype=torch.float32,
    torch_dtype=torch.float16,
    #   device_map={"":0}
    device_map={'': torch.cuda.current_device()}
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_params = LoraConfig(
    r= args.rank,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
"q_proj",
"k_proj",
"v_proj",
"o_proj",
"gate_proj",
"up_proj",
"down_proj",
"lm_head",
],
)
import numpy as np
# import matplotlib.pyplot as plt
question='problem'
answer='solution'
suffix = "<|im_end|>"
if type == 'llama':
    suffix = "<|eot_id|>"
elif type == 'phi':
        suffix = "<|endoftext|>"
if args.dataset == 'gsm8k':
    dataset = datasets.load_dataset('gsm8k', "main")
    train_dataset, test_dataset = dataset['train'], dataset['test']
    question = 'problem'
    answer = 'answer'
elif args.dataset == 'metamath':
    dataset = datasets.load_dataset("Colder203/meta_math_smaller_than_512")
    train_dataset = dataset['train']
    if args.num_samples is not None:
        train_dataset = train_dataset.select(np.random.choice(len(train_dataset), args.num_samples))
    print(train_dataset)
    dataset = train_dataset.train_test_split(test_size=0.1)
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    question = 'query'
    answer = 'response'
else:
    dataset = datasets.load_dataset("lighteval/MATH", "all")
    train_dataset, test_dataset = dataset['train'], dataset['test']
    question = 'problem'
    answer = 'solution'
print('load dataset ok')
def preprocess_function(examples):
    
    if type == 'qwen':
        prompt = prompt_qwen
    elif type == 'phi':
        prompt = prompt_phi
            
    elif type == 'llama':
        prompt = prompt_llama


    inputs = [prompt.format(instruction = quest) for quest in examples[question]]
    # print(inputs[0])
    
    targets = [f"{completion}" + suffix for completion in examples[answer]]
    # print(targets[0])
    # model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs = tokenizer(inputs,
                            #  max_length=512, padding = True
                             )
    # labels = tokenizer(targets, max_length=512, truncation=True, padding = True)
    labels = tokenizer(targets,
                      #  max_length=512, padding = True
                       )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

tokenized_eval_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# %cd /content/drive/MyDrive/qwen

training_params = TrainingArguments(
    output_dir=f"./{type}/results" if args.save_path is None else args.save_path,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=200,
    learning_rate=2e-4,
    logging_dir=f"./{type}/logs",
    save_strategy="epoch" if args.save_strategy == 'epoch' else "steps",
    save_steps=60000 if args.save_steps is None else args.save_steps,
    # fp32=True,
    bf16=torch.cuda.is_bf16_supported(),
    fp16 = not torch.cuda.is_bf16_supported(),
    evaluation_strategy="epoch",

    
)

trainer = SFTTrainer(
    model=based_model,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_eval_dataset,
    peft_config=peft_params , #full finetuning
    # max_seq_length=2048,
    args=training_params,
    packing=False,
    tokenizer = tokenizer,
    # optimizers=(optimizer, scheduler) if model_path is not None else None,
    # overlap_comm = False 
)

#turn off wandb
trainer.is_wandb_on = False

# trainer.train() #This is 0.5B
if args.model_path is not None:
    trainer.train(resume_from_checkpoint=args.model_path)
else:
    trainer.train() #This is 1.5B

trainer.save_model(model_name)
