import sys

import os
os.environ['TRANSFORMERS_CACHE'] = './cache/'
import transformers
from torch import nn
import torch
from be_great.multihead_models import MOEModelForCausalLM
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback, BitsAndBytesConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from matplotlib import pyplot as plt
from tqdm import tqdm 
import argparse
import datetime
import json
from be_great import GReaT
from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_trainer import GReaTTrainer
import re
from shutil import copy
from sklearn import preprocessing, pipeline, ensemble, compose
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

path = sys.argv[-2]
outpath = sys.argv[-1]
print('will read from', path, 'and save to', outpath)
n_samples = 5000

# %%
modelname = 'meta-llama/Llama-3.2-1B'
tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {"bos_token": "<BOS>", 'eos_token': '<EOS>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# %%
dgpt2 = transformers.AutoModelForCausalLM.from_pretrained(modelname, device_map='auto',
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
dgpt2.resize_token_embeddings(len(tokenizer))

# %%
model = MOEModelForCausalLM(dgpt2, num_experts=4, moe=True, multihead=False)

def apply_efficient_finetuning(model):
    lora_config = LoraConfig(
        r=1,  
        lora_alpha=256,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,  # this is specific for gpt2 model, to be adapted
    )
    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
    print('applying lora, model type now', type(model))

# %%
model = apply_efficient_finetuning(model)

# %%
ckpt_path = os.path.join(path, 'model.pt')
sd = torch.load(ckpt_path, map_location=torch.device('cuda:0'))

# %%
for name, param in model.named_parameters():
    param.data.copy_(sd[name])

# %%
model.eval() 
columns = ['Year','Station','Month','Rainfall']
column_names_tokens = tokenizer(columns).input_ids
model.set_generation_mode(token_heads=list(range(4)), column_names_tokens=column_names_tokens)

# %%
sbs=1
inputs = torch.full((sbs, 1), tokenizer.bos_token_id).to(model.device)
samples = []
for i in tqdm(range(0, n_samples, sbs)):
    toks = model.generate(inputs, do_sample=True, num_beams=1, max_length=1000,#dataconfig['max_col_length']*len(dataconfig['cols']), 
                        pad_token_id=tokenizer.eos_token_id)[...,1:] # remove BOS token
    outs = tokenizer.batch_decode(toks)
    samples.extend(outs)
    if len(samples)%100 == 0:
        with open(os.path.join(outpath, 'samples.txt'), 'a+') as f:
            f.write('\n'.join(samples))
        samples = []


outs = samples
with open(outpath, 'w') as f:
    f.write('\n'.join(outs))
with open(os.path.join(path, 'samples.txt'), 'w') as f:
    f.write('\n'.join(outs))
