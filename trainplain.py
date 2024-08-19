import transformers
from torch import nn
import torch
from be_great.multihead_models import MOEModelForCausalLM
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from matplotlib import pyplot as plt
from tqdm import tqdm 
import os
import argparse
import datetime
import json
from be_great import GReaT
from sklearn.datasets import fetch_california_housing
import re
from shutil import copy
    
parser = argparse.ArgumentParser(
                    prog='Train-Plain',
                    description='Basic program to train LLMs and MOE LLMs on tabular data',
                    epilog='Contact sonia at cromp@wisc.edu with questions!')
parser.add_argument('-p', '--path',
                    default=None, help='where to store/access model checkpoints, samples, etc')
parser.add_argument('-d', '--dataset',
                    default='adult', help='adult or diabetes')
parser.add_argument('-m', '--moe', action='store_true',
                    default=False, help='whether to use a MOE model')
parser.add_argument('-t', '--train', action='store_true',
                    default=False, help='whether to train: train on trainset')
parser.add_argument('-v', '--valtrain', action='store_true',
                    default=False, help='whether to train: train on valset (for fast debugging purposes only)')
parser.add_argument('-g', '--great', action='store_true',
                    default=False, help='whether to use GReaT-style training/sampling')
parser.add_argument('-n', '--n-samples', type=int,
                    default=10, help='number of samples to synthesize (or 0 to skip this)')
# dataset, dgpt2 vs llama, ...
args = parser.parse_args()
print(args)

if args.train and args.valtrain:
    raise Exception('choose only -t or -v')

now = datetime.datetime.now()
if args.path == None:
    tiny = '-tiny' if args.valtrain else ''
    great = 'great' if args.great else 'plain'
    variant = 'moe' if args.moe else 'oh'
    outpath = f'./ckpts/{args.dataset}{tiny}/{great}/{variant}/{now.month}-{now.day}({now.hour}:{now.minute})'
print('outpath', outpath)
os.makedirs(outpath, exist_ok=True)

# Load the dataset (needed even just for sampling, to get column names)
if args.dataset == 'adult':
    file_path = '/home/sonia/be_great/data/adult/latest'  # Update this with the correct path
elif args.dataset == 'diabetes':
    file_path = '/home/sonia/be_great/data/diabetes/latest'
if args.valtrain:
    data = pd.read_csv(os.path.join(file_path, 'val.csv'))
else:
    data = pd.read_csv(os.path.join(file_path, 'train.csv'))
with open(os.path.join(file_path, 'config.json'), 'r') as f:
    dataconfig = json.load(f)

if args.train:
    copy(os.path.join(file_path, 'config.json'), os.path.join(outpath, 'dataconfig.json'))


def parse(raws, args, file_path, outpath):
    real = pd.read_csv(os.path.join(file_path, 'all.csv'))
    cols  = set(real.columns)
    
    def parse_line(l):
        entries = l[:-1].split('.<EOS>') # remove newline at end
        # print(entries)
        words = [c.split(' ') for c in entries] #'name', 'is', 'value'
        # print(words)
        d = {c[0]:c[2] for c in words if len(c)==3 and c[0] in cols}
        if set(d.keys()) == cols:
            return d 
        else:
            return None

    line_dicts = [parse_line(l) for l in raws]
    line_dicts = [l for l in line_dicts if l is not None]
    print(len(raws)-len(line_dicts), 'problem lines')
    df = pd.DataFrame.from_records(line_dicts)

    with open(os.path.join(file_path, 'config.json'), 'r') as f:
        dataconfig = json.load(f)
    ords = dataconfig['ords']

    ordvals = {col:set(real[col].unique()) for col in ords}
    for col in ordvals:
        ordvals[col] = [str(val).strip() for val in ordvals[col]]

    for col in ordvals:
        df = df[df[col].isin(ordvals[col])]
        print(col, len(df))
        
    df.to_csv(os.path.join(path, 'samplesclean.csv'), index=False)

if not args.great:
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {"bos_token": "<BOS>", 'eos_token': '<EOS>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    dgpt2 = transformers.AutoModelForCausalLM.from_pretrained('distilgpt2')
    dgpt2.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.moe:
        num_experts = len(data.columns)
        print('create', num_experts, 'head moe model')
        dgpt2copy = MOEModelForCausalLM(dgpt2, num_experts=num_experts, multihead=True)
        model = dgpt2copy # don't forget to change tokenizer name and optimizer too
        model.set_train_mode()
    else:
        model = dgpt2
    
    if args.train:
        lr = 5e-6
        epochs = 1
        config = {
            'file_path': file_path,
            'creation_time': str(now),
            'lr': lr,
            'epochs': epochs,
            'args': vars(args)
        }
        
        model.train()
        # Move the model to the device (GPU if available)
        model.to(device)

        # Data stuff
        # Preprocess the data: Convert each row to a string
        def row_to_col_sentences(row):
            return [str(col).strip() + " is " + str(val).strip() + '.<EOS>' for col, val in zip(row.index, row.values)]

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, cols=None, max_col_length=10, do_moe_format=True):
                self.texts = texts
                self.tokenizer = tokenizer
                self.cols = cols # "None" for all cols, else a list of desired cols' names
                self.max_col_length = max_col_length
                self.do_moe_format = do_moe_format

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                if self.cols is None:
                    text = row_to_col_sentences(data.iloc[idx])
                else:
                    text = row_to_col_sentences(data[self.cols].iloc[idx]) # ['age is 39.', 'workclass is State-gov.', ...]
                if self.do_moe_format:
                    tokenized_text = self.tokenizer(text, truncation=True, max_length=self.max_col_length, padding='max_length', return_tensors="pt")
                    prompt = torch.full((1,), #batch_size x token
                                        self.tokenizer.bos_token_id)
                    return {'input_ids': prompt, 'labels': tokenized_text.input_ids.squeeze()}
                else:
                    text = tokenizer.bos_token + ''.join(text)
                    tokenized_text = self.tokenizer(text, truncation=True, padding='longest', return_tensors='pt')
                    return {'input_ids': tokenized_text.input_ids.squeeze(), 'attention_mask': tokenized_text.attention_mask.squeeze(),
                            'labels': tokenized_text.input_ids.squeeze()}
                    

        text_data = data.apply(row_to_col_sentences, axis=1).tolist()
        dataset = TextDataset(text_data, tokenizer, max_col_length=dataconfig['max_col_length'], do_moe_format=args.moe)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


        # Set up the optimizer and learning rate scheduler
        optimizer = AdamW(model.parameters(), lr=lr)
        ins = tokenizer(tokenizer.bos_token, return_tensors='pt')

        lossesmoe = []
        for epoch in range(epochs):  
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                batch = {k:v.to(device) for (k,v) in batch.items()}

                outputs = model(**batch)
                # outputs = model.debug_forward(ins['input_ids'].to(device), ins['attention_mask'].to(device), labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                lossesmoe.append(loss.item())
                if len(lossesmoe) % 1000 == 0:
                    torch.save(model.state_dict(), os.path.join(outpath, f'{len(lossesmoe)}.pt'))
                    try:
                        plt.close()
                    except:
                        pass
                    plt.plot(lossesmoe)
                    plt.savefig(os.path.join(outpath, 'loss.png'))
    if not args.train: # load in checkpoint so we can sample
        ckpt_ints = [int(f.split('.')[0]) for f in os.listdir(outpath) if f.endswith('.pt')] #steps where epochs saved
        max_ckpt = max(ckpt_ints)
        ckpt_path = os.path.join(outpath, f'{max_ckpt}.pt')
        print('loading from', ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
        model.to(device)

    if args.n_samples > 0:
        model.eval()
        column_names_tokens = tokenizer(list(data.columns)).input_ids
        if args.moe:
            model.set_generation_mode(column_names_tokens=column_names_tokens)

        samples = []
        for i in tqdm(range(args.n_samples)):
            toks = model.generate(do_sample=True, num_beams=1, max_length=data_config['max_col_length']*len(data_config['cols']), 
                                pad_token_id=tokenizer.eos_token_id)[...,1:] # remove BOS token
            samples.append(tokenizer.batch_decode(toks)[0])
            if len(samples)%100 == 0:
                with open(os.path.join(outpath, 'samples.txt'), 'a+') as f:
                    f.write('\n'.join(samples))
                samples = []
            
        with open(os.path.join(outpath, 'samples.txt'), 'a+') as f:
            f.write('\n'.join(samples))
            
        print('samples saved to', os.path.join(outpath, 'samples.txt'))
        
else: #use great
    if args.train:
        model = GReaT(llm='distilgpt2', batch_size=1,  
              epochs=1, save_steps=3225,
              experiment_dir=outpath, multihead=args.moe)
        model.fit(data)
        model.save(outpath)
    elif not args.train:
        model = GReaT.load_from_dir(outpath)
        
    if args.n_samples > 0:
        synthetic_data = model.sample(n_samples=args.n_samples, parse=False, k=1, 
                                      max_length=data_config['max_col_length']*len(data_config['cols']))
        synthetic_data = [l[0]+'\n' for l in synthetic_data] #remove [] around batch of 1 sample

        # if not args.moe:
        #     with open(os.path.join(outpath, 'samplesclean.csv'), 'w') as f: # pre-parsed
        #         f.writelines(synthetic_data)
        # else:
        with open(os.path.join(outpath, 'samples.txt'), 'w') as f: # not pre-parsed
            f.writelines(synthetic_data)
            
        raws = [re.sub('is\?', 'is ?', raw) for raw in synthetic_data] # fix that "is ?" is decoded to "is?" by tokenizer
            
        # parsing
        problem = 0
        def parse_line(l):
            cols = l.split('.<EOS>')
            words = [c.split(' ') for c in cols] #'name', 'is', 'value'
            words = [w for w in words if len(w)==3]
            if len(words) > 15: #some models put extra stuff at the end
                words = words[:15]
            if len(words) == 15:
                return {c[0]:c[2] for c in words}
            else:
                problem += 1
                return {}

        line_dicts = [parse_line(l) for l in raws]
        df = pd.DataFrame.from_records(line_dicts)
        print(problem, 'problem lines')
        print(df.columns, df.shape)

        real = pd.read_csv(os.path.join(file_path, 'all.csv'))
        with open(os.path.join(file_path, 'config.json'), 'r') as f:
            dataconfig = json.load(f)
        ords = dataconfig['ords']

        ordvals = {col:set(real[col].unique()) for col in ords}
        for col in ordvals:
            ordvals[col] = [str(val).strip() for val in ordvals[col]]

        for col in ordvals:
            df = df[df[col].isin(ordvals[col])]
            print(col, len(df))
            
        df.to_csv(os.path.join(outpath, 'samplesclean.csv'), index=False)
        