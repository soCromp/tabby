import transformers
from torch import nn
import torch
from be_great.multihead_models import MOEModelForCausalLM
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from matplotlib import pyplot as plt
from tqdm import tqdm 
import os
import argparse
import datetime
import json
from be_great import GReaT
from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_trainer import GReaTTrainer
import re
from shutil import copy
from sklearn import preprocessing, pipeline, ensemble, compose
    
parser = argparse.ArgumentParser(
                    prog='Train-Plain',
                    description='Basic program to train LLMs and MOE LLMs on tabular data',
                    epilog='Contact sonia at cromp@wisc.edu with questions!')
parser.add_argument('-p', '--path',
                    default=None, help='where to store/access model checkpoints, samples, etc')
parser.add_argument('-d', '--dataset',
                    default=None, help='adult or diabetes')
parser.add_argument('-m', '--moe', action='store_true',
                    default=False, help='whether to use a MOE model')
parser.add_argument('-mh', '--mh', action='store_true',
                    default=False, help='whether to use a Multi-Head model')
parser.add_argument('-t', '--train', action='store_true',
                    default=False, help='whether to train: train on trainset')
parser.add_argument('-v', '--valtrain', action='store_true',
                    default=False, help='whether to train: train on valset (for fast debugging purposes only)')
parser.add_argument('-g', '--great', action='store_true',
                    default=False, help='whether to use GReaT-style training/sampling')
parser.add_argument('-r', '--pre', action='store_true',
                    default=False, help='whether to use the pRetrained (distilled) gpt2 tabular model from TapTap')
parser.add_argument('-c', '--ec', action='store_true',
                    default=False, help='whether to Encode the Categorical columns à la Tabula')
parser.add_argument('-lr', '--lr', type=float,
                    default=1e-6, help='training learning rate')
parser.add_argument('-n', '--n-samples', type=int,
                    default=10, help='number of samples to synthesize (or 0 to skip this)')
parser.add_argument('--parse', action='store_true',
                    help='just read in samples.txt and try to parse it- this option is for debugging purposes')
parser.add_argument('-validation', '--validation', action='store_true',
                    help='just run the validation- for debugging purposes')
args = parser.parse_args()
print(args)

now = datetime.datetime.now()
if args.path == None:
    import socket 
    drivedict = {'brandy_old_fashioned':'/mnt/data/sonia'}
    drive = drivedict.get(socket.gethostname(), '.')
    tiny = '-tiny' if args.valtrain else ''
    great = 'great' if args.great else 'plain'
    variant = 'moe' if args.moe else 'oh'
    outpath = f'{drive}/ckpts/{args.dataset}{tiny}/{now.month}-{now.day}/{great}/{variant}/{now.month}-{now.day}.{now.hour}'
else:
    outpath = args.path
print('outpath', outpath)

if args.train and args.valtrain:
    raise Exception('choose only -t or -v')
if args.train or args.valtrain:
    os.makedirs(outpath, exist_ok=True)
if (args.train or args.valtrain) and args.dataset == None:
    raise Exception('specify dataset')

# Load the dataset (needed even just for sampling, to get column names)
if not (args.train or args.valtrain) and args.dataset == None:
    # load version of dataset that the pre-existing model was trained on
    with open(os.path.join(args.path, 'dataconfig.json'), 'r') as f:
        dataconfig = json.load(f)
    version = '.'.join(dataconfig['creation_time'].split(' '))
    file_path = f'./data/{dataconfig["dataset_name"]}/{version}'
    print(file_path)
else:
    try:
        file_path = f'./data/{args.dataset}/latest'
    except:
        raise Exception('unsupported dataset', args.dataset)
    
if args.valtrain:
    data = pd.read_csv(os.path.join(file_path, 'val.csv'))
    valdata = pd.read_csv(os.path.join(file_path, 'val.csv'))
else:
    data = pd.read_csv(os.path.join(file_path, 'train.csv'))
    valdata = pd.read_csv(os.path.join(file_path, 'val.csv'))
with open(os.path.join(file_path, 'config.json'), 'r') as f:
    dataconfig = json.load(f)

if args.train or args.valtrain:
    copy(os.path.join(file_path, 'config.json'), os.path.join(outpath, 'dataconfig.json'))
    
if args.pre:
    modelname = 'ztphs980/taptap-distill'
else:
    modelname = 'distilgpt2'


def parse(raws, args, file_path, outpath):
    real = pd.read_csv(os.path.join(file_path, 'all.csv'))
    cols  = set(real.columns)
    
    def parse_line(l):
        entries = l[:-1].split('<EOS>') # remove newline at end
        # print(entries)
        words = [c.split(' ') for c in entries] #'name', 'is', 'value'
        # print(words)
        d = dict()
        for c in words:
            if c[0] in cols and len(c) == 3 and c[0] not in d: # keep only first occurence
                d[c[0]] = c[2]
        # d = {c[0]:c[2] for c in words if len(c)==3 and c[0] in cols}

        if set(d.keys()) == cols:
            return d 
        else:
            return None

    line_dicts = [parse_line(l) for l in raws]
    line_dicts = [l for l in line_dicts if l is not None]
    print(len(raws)-len(line_dicts), 'problem lines')
    if len(line_dicts) == 0:
        print('did not successfully parse samples. returning')
        return
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
        if len(df) == 0:
            print('did not successfully parse samples. returning')
            return
        
    df.to_csv(os.path.join(outpath, 'samplesclean.csv'), index=False)

if args.parse:
    with open(os.path.join(args.path, 'samples.txt'), 'r') as f:
        raws = f.readlines()
    parse(raws, args, file_path, args.path)

elif not args.great:
    tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens_dict = {"bos_token": "<BOS>", 'eos_token': '<EOS>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    dgpt2 = transformers.AutoModelForCausalLM.from_pretrained(modelname, device_map='auto')
    dgpt2.resize_token_embeddings(len(tokenizer))
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.moe or args.mh:
        num_experts = len(data.columns)
        print('create', num_experts, 'head moe model')
        dgpt2copy = MOEModelForCausalLM(dgpt2, num_experts=num_experts, moe=args.moe, multihead=args.mh)
        model = dgpt2copy # don't forget to change tokenizer name and optimizer too
        model.set_train_mode()
    else:
        model = dgpt2
        
    print(model)
    
    if args.train or args.valtrain:
        epochs = 50
        config = {
            'file_path': file_path,
            'creation_time': str(now),
            'lr': args.lr,
            'epochs': epochs,
            'args': vars(args)
        }
        with open(os.path.join(outpath, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        model.train()

        # Data stuff
        # Preprocess the data: Convert each row to a string
        def row_to_col_sentences(row):
            return [str(col).strip() + " is " + str(val).strip() + '<EOS>' for col, val in zip(row.index, row.values)]

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
        do_moe_format = args.moe or args.mh
        dataset = TextDataset(text_data, tokenizer, max_col_length=dataconfig['max_col_length'], do_moe_format=do_moe_format)
        
        text_valdata = valdata.apply(row_to_col_sentences, axis=1).tolist()
        valdataset = TextDataset(text_valdata, tokenizer, max_col_length=dataconfig['max_col_length'], do_moe_format=do_moe_format)
        
        
        targs = TrainingArguments(output_dir=outpath, overwrite_output_dir=True, do_train=True, save_steps=5000,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, 
                                  learning_rate=args.lr, num_train_epochs=epochs,
                                  load_best_model_at_end = True, evaluation_strategy='steps', eval_steps=5000,
                                  save_total_limit = 5, metric_for_best_model='eval_loss',)
        trainer = Trainer(model, targs, train_dataset=dataset, eval_dataset=valdataset,
                                  callbacks = [EarlyStoppingCallback(early_stopping_threshold=0, early_stopping_patience=2)])
        trainer.train()
        torch.save(model.state_dict(), os.path.join(outpath, f'model.pt'))
        
        valresult = trainer.evaluate(valdataset)
        print('valresult', valresult)
        config['validation_eval'] = valresult
        with open(os.path.join(outpath, 'config.json'), 'w') as f:
            json.dump(config, f)
            
        pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(outpath, 'losses.csv'))
    
    if not args.train and not args.valtrain: # load in checkpoint so we can validate or sample
        # ckpt_ints = [int(f.split('.')[0]) for f in os.listdir(outpath) if f.endswith('.pt')] #steps where epochs saved
        # max_ckpt = max(ckpt_ints)
        ckpt_path = os.path.join(outpath, 'model.pt')
        print('loading from', ckpt_path)
        model.load_state_dict(torch.load(ckpt_path))
        # model.to(device)
        
    if args.validation:
        raise NotImplementedError()

    if args.n_samples > 0:
        model.eval()
        column_names_tokens = tokenizer(list(data.columns)).input_ids
        if args.moe or args.mh:
            token_heads = list(range( len(data.columns) ))
            model.set_generation_mode(token_heads=token_heads, column_names_tokens=column_names_tokens)
            sbs = 1
        else: 
            sbs = min(100, args.n_samples)

        inputs = torch.full((sbs, 1), tokenizer.bos_token_id).to(model.device)
        samples = []
        for i in tqdm(range(0, args.n_samples, sbs)):
            
            toks = model.generate(inputs, do_sample=True, num_beams=1, max_length=1000,#dataconfig['max_col_length']*len(dataconfig['cols']), 
                                pad_token_id=tokenizer.eos_token_id)[...,1:] # remove BOS token
            outs = tokenizer.batch_decode(toks)
            samples.extend(outs)
            if len(samples)%100 == 0:
                with open(os.path.join(outpath, 'samples.txt'), 'a+') as f:
                    f.write('\n'.join(samples))
                samples = []
            
        with open(os.path.join(outpath, 'samples.txt'), 'a+') as f:
            f.write('\n'.join(samples))
            
        print('samples saved to', os.path.join(outpath, 'samples.txt'))
        samples = [s+'\n' for s in samples]
        parse(samples, args, file_path, outpath)
        
else: #use great
    if args.train or args.valtrain:
        config = {
            'file_path': file_path,
            'creation_time': str(now),
            'lr': args.lr,
            'args': vars(args)
        }
        with open(os.path.join(outpath, 'trainplain_config.json'), 'w') as f:
            json.dump(config, f)
        
        model = GReaT(llm=modelname, batch_size=1, per_device_eval_batch_size=1,
              epochs=50, save_steps=5000,
              experiment_dir=outpath, multihead=args.mh, moe=args.moe, fp16=True, learning_rate=args.lr,
                load_best_model_at_end = True, evaluation_strategy='steps', eval_steps=5000,
                save_total_limit = 5, metric_for_best_model='eval_loss',)
            #   efficient_finetuning='lora')
        trainer = model.fit(data, eval_dataset=valdata, conditional_col=dataconfig['labs'][0])
        model.save(outpath)
        
        
        great_valds = GReaTDataset.from_pandas(valdata)
        great_valds.set_stuff(model.tokenizer, args.moe) 
        valresult = trainer.evaluate(great_valds)
        print('valresult', valresult)
        with open(os.path.join(outpath, 'validationeval.json'), 'w') as f:
            json.dump(valresult, f)
            
        pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(outpath, 'losses.csv'))
    elif not args.train and not args.valtrain:
        model = GReaT.load_from_dir(outpath)
        
    if args.validation:
        training_args = TrainingArguments(
            model.experiment_dir,
            num_train_epochs=model.epochs,
            per_device_train_batch_size=model.batch_size,
            **model.train_hyperparameters,
        )
        great_valds = GReaTDataset.from_pandas(valdata)
        great_valds.set_stuff(model.tokenizer, args.moe) 
        trainer = GReaTTrainer(
            model.model,
            training_args,
            train_dataset=great_valds,
            tokenizer=model.tokenizer,
            data_collator=GReaTDataCollator(model.tokenizer),
        )
        valresult = trainer.evaluate(great_valds)
        print('valresult', valresult)
        with open(os.path.join(outpath, 'validationeval.json'), 'w') as f:
            json.dump(valresult, f)
        pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(outpath, 'losses_val.csv'))
        
    if args.n_samples > 0:
        sbs = 100 #sample batch size
        max_length = dataconfig['max_col_length']*len(dataconfig['cols'])
        if args.moe or args.mh:
            sbs = 1
            max_length = 1000 #since moe stops on its own
        synthetic_data = model.sample(n_samples=args.n_samples, k=sbs, 
                                      max_length=max_length)
        synthetic_data = [l+'\n' for l in synthetic_data] #add newlines

        # if not args.moe:
        #     with open(os.path.join(outpath, 'samplesclean.csv'), 'w') as f: # pre-parsed
        #         f.writelines(synthetic_data)
        # else:
        with open(os.path.join(outpath, 'samples.txt'), 'w') as f: # not pre-parsed
            f.writelines(synthetic_data)
            
        raws = [re.sub('is\?', 'is ?', raw) for raw in synthetic_data] # fix that "is ?" is decoded to "is?" by tokenizer
            
        # parsing
        parse(raws, args, file_path, outpath)
        
print(outpath)
