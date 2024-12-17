import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, Trainer, pipeline,  BitsAndBytesConfig
from peft import LoraConfig
from datasets import Dataset
import datasets
from trl import SFTTrainer, PPOTrainer
#load model name
model_name = "microsoft/Phi-3.5-mini-instruct"
from tqdm import tqdm
model_path = "/content/drive/MyDrive/qwen/phi/results/checkpoint-37365"
# model_name = "bkai-foundation-models/vietnamese-llama2-7b-120GB"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_use_double_quant=False,
)



based_model = AutoModelForCausalLM.from_pretrained(model_name,
  # low_cpu_mem_usage = True,
  quantization_config=quant_config,
  torch_dtype=torch.float16,
  device_map={"":0}
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_params = LoraConfig(
    r=256,
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


metaMath = datasets.load_dataset("meta-math/MetaMathQA")
metaMath_list = metaMath['train'].to_list()
filter_list = []
for i in tqdm(range(len(metaMath_list))):
    query = metaMath_list[i]['query']
    query_tokens = tokenizer(query, return_tensors='pt')
    if len(query_tokens['input_ids']) > 1024:
        continue
    filter_list.append(metaMath_list[i])
train_dataset = Dataset.from_list(filter_list)
# train_dataset = metaMath['train']
def preprocess_function(examples, question = "query", answer = "response"):
    prompt = (
            "<|system|>\nYou are a helpful assistant.\n"
            "<|user|>\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}\n"
            "<|assistant|>\n"
        )
    inputs = [prompt.format(instruction = question) for question in examples[question]]
    print(inputs[0])
    targets = [f"{completion}\n" for completion in examples[answer]]
    print(targets[0])
    # model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs = tokenizer(inputs,
                            #  max_length=512, padding = False, truncation = True
                             )
    #remove all sample which having more than 1024 tokens
    # model_inputs = {k: [v for i, v in enumerate(model_inputs[k]) if len(v) < 1024] for k in model_inputs}
    # labels = tokenizer(targets, max_length=512, truncation=True, padding = True)
    labels = tokenizer(targets,
                    #    max_length=512, padding = False, truncation = True
                       )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
# tokenizer.padding_side = "right"
training_params = TrainingArguments(

    output_dir="./results",

    num_train_epochs=3,

    per_device_train_batch_size=8,

    gradient_accumulation_steps=1,

    logging_steps=100,

    learning_rate=2e-4,

    logging_dir="./logs",

    save_strategy="steps",
    save_steps=1/12,

    # fp16=True,

    bf16 = True,
    

)



trainer = SFTTrainer(

    model=based_model,

    train_dataset=tokenized_dataset,

    # eval_dataset=tokenized_eval_dataset,

    peft_config=peft_params,

    # max_seq_length=2048,

    args=training_params,

    # packing=False,

)


trainer.train()