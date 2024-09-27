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

path = sys.argv[-2]
outpath = sys.argv[-1]
print('will read from', path, 'and save to', outpath)

# %%
modelname = 'meta-llama/Meta-Llama-3-8B'
tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='left')
special_tokens_dict = {"bos_token": "<BOS>", 'eos_token': '<EOS>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# %%
tokenizer.eos_token_id

# %%
dgpt2 = transformers.AutoModelForCausalLM.from_pretrained(modelname, device_map='auto',
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))

# %%
great= GReaT.load_from_dir(path,
                           model = dgpt2)

# %%
great.model.resize_token_embeddings(len(tokenizer))

# %%
great.model = MOEModelForCausalLM(great.model, num_experts=6, moe=False, multihead=True)

# %%
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

def apply_efficient_finetuning(model):
    lora_config = LoraConfig(
        r=1,  
        lora_alpha=256,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head.layers.0', 'lm_head.layers.1','lm_head.layers.2', 'lm_head.layers.3', 'lm_head.layers.4', 'lm_head.layers.5'],
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
great.model = apply_efficient_finetuning(great.model)

# %%
ckpt_path = os.path.join(path, 'model.pt')
sd = torch.load(ckpt_path)
# dgpt2copy.load_state_dict(torch.load(ckpt_path, weights_only=True))
sd.keys()

# %%
for name, param in great.model.named_parameters():
    param.data.copy_(sd[name])

# %%
great.model.eval() 
great.tokenizer = tokenizer
# columns = ['bedrooms','occupancy','value_median_house']
# column_names_tokens = tokenizer(columns).input_ids
# great.model.set_generation_mode(token_heads=list(range(6)), column_names_tokens=column_names_tokens)

# %%
outs = great.sample(10000, k=1, max_length = 1000)

# %%
pre = len('<|begin_of_text|>')
outs = [s[pre:] for s in outs]
with open(outpath, 'w') as f:
    f.write('\n'.join(outs))
with open(os.path.join(path, 'samples.txt'), 'w') as f:
    f.write('\n'.join(outs))
