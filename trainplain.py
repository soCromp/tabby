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
from transformers import DataCollatorForTokenClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.nn.utils.rnn import pad_sequence
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from accelerate import PartialState
os.environ["WANDB_DISABLED"] = 'true'

    
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
parser.add_argument('-l8', '--llama8', action='store_true',
                    default=False, help='use llama3 8B')
parser.add_argument('-l1', '--llama1', action='store_true',
                    default=False, help='use llama3.2 1B')
parser.add_argument('-gpt2', '--gpt2', action='store_true',
                    default=False, help='use non-distilled GPT2')
parser.add_argument('-eff', '--efficient', action='store_true',
                    default=False, help='use LORA and reduced precision')
parser.add_argument('-lr', '--lr', type=float,
                    default=1e-6, help='training learning rate')
parser.add_argument('-n', '--n-samples', type=int,
                    default=10, help='number of samples to synthesize (or 0 to skip this)')
parser.add_argument('--parse', action='store_true',
                    help='just read in samples.txt and try to parse it- this option is for debugging purposes')
parser.add_argument('-validation', '--validation', action='store_true',
                    help='just run the validation- for debugging purposes')
parser.add_argument('-resume', '--resume', action='store_true', default=False,
                    help='resume training run')
parser.add_argument('-e', '--epochs', type=int,
                    default=50, help='number of epochs to train')
parser.add_argument('-local', '--local', action='store_true', default=False,
                    help='whether to use local copies of model weights. For CHTC runs.')
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
    data = pd.read_csv(os.path.join(file_path, 'val.csv')).iloc[:,-3:]
    valdata = pd.read_csv(os.path.join(file_path, 'val.csv')).iloc[:,-3:]
    alldata = None
    # used *uniquely* for making sure all possible values are encoded
    # with tabula:
    if args.ec:
        alldata = pd.read_csv(os.path.join(file_path, 'all.csv')) 
else:
    data = pd.read_csv(os.path.join(file_path, 'train.csv'))
    valdata = pd.read_csv(os.path.join(file_path, 'val.csv'))
    alldata = None
    # used *uniquely* for making sure all possible values are encoded
    # with tabula:
    if args.ec:
        alldata = pd.read_csv(os.path.join(file_path, 'all.csv')) 
with open(os.path.join(file_path, 'config.json'), 'r') as f:
    dataconfig = json.load(f)

if args.train or args.valtrain:
    copy(os.path.join(file_path, 'config.json'), os.path.join(outpath, 'dataconfig.json'))
    

if os.path.exists('./accesstoken.txt'):
    with open('./accesstoken.txt', 'r') as f:
        accesstoken = f.read()
    accesstoken = accesstoken.split(' ')[-1][:-1]
else:
    accesstoken = None
    
if args.pre:
    modelname = 'ztphs980/taptap-distill'
elif args.llama8:
    if args.local:
        modelname = 'Meta-Llama-3-8B'
    else:
        modelname = 'meta-llama/Meta-Llama-3-8B'
elif args.llama1:
    if args.local:
        modelname = 'Llama-3.2-1B'
    else:
        modelname = 'meta-llama/Llama-3.2-1B'
elif args.gpt2:
    modelname = 'gpt2'
else:
    modelname = 'distilgpt2'

## In case the dataset has blanks in the csv, there will be nan key errors if we don't replace them
def fill_na(df):
    for col in df:
        ## Don't fill numerical, or it'll mess with the distribution
        if df[col].dtype not in [int, float]:
            # df[col].fillna("?", inplace=True)
            df.fillna({col: '?'}, inplace=True)
fill_na(data)
fill_na(valdata)
if alldata is not None:
    fill_na(alldata)

def make_label_encoders(data, categorical_columns):
    label_encoder_list = []
    for column_index, column in enumerate(data.columns):
        if column in categorical_columns:
            label_encoder = preprocessing.LabelEncoder()
            data[column] = data[column].astype(str)
            label_encoder.fit(data[column])
            current_label_encoder = dict()
            current_label_encoder['column'] = column
            current_label_encoder['label_encoder'] = label_encoder
            label_encoder_list.append(current_label_encoder)
    return label_encoder_list
            
def encode_categorical_columns(data, label_encoder_list): 
    # pass the dataframe of data to encode and label_encoder_list
    for i in range(len(label_encoder_list)):
        label_encoder = label_encoder_list[i]['label_encoder']
        column_name = label_encoder_list[i]['column']
        
        transformed_column = label_encoder.transform(data[column_name])
        data[column_name] = transformed_column
    return data

def decode_categorical_columns(data, label_encoder_list):
    # pass the data to decode and the label_encoder_list 
    for i in range(len(label_encoder_list)):
        le = label_encoder_list[i]["label_encoder"]
        allowed_values = list(range(len(le.classes_)))
        
        # delete rows that should generate numeric value but generate other data type
        data[label_encoder_list[i]['column']] = pd.to_numeric(data[label_encoder_list[i]['column']], errors='coerce')
        data = data.dropna(subset=[label_encoder_list[i]['column']])

        # delete rows that generate category that is out of boundary
        data[label_encoder_list[i]['column']] = data[label_encoder_list[i]['column']].astype(float)
        data = data[data[label_encoder_list[i]['column']].isin(allowed_values)]

    for i in range(len(label_encoder_list)):
        le = label_encoder_list[i]["label_encoder"]
        data[label_encoder_list[i]["column"]] = data[label_encoder_list[i]["column"]].astype(int)
        data[label_encoder_list[i]["column"]] = le.inverse_transform(data[label_encoder_list[i]["column"]])
        
    return data

label_encoder_list = None
if args.ec: # use tabula ordinalization of categorical columns
    if dataconfig['task'] == 'regression':
        label_encoder_list = make_label_encoders(alldata, dataconfig['ords'])
        data[dataconfig['ords']] = data[dataconfig['ords']].map(lambda x: str(x))
        valdata[dataconfig['ords']] = valdata[dataconfig['ords']].map(lambda x: str(x))
    elif dataconfig['task'] == 'classification':
        label_encoder_list = make_label_encoders(alldata, dataconfig['ords']+dataconfig['labs'])
        data[dataconfig['ords']+dataconfig['labs']] = \
            data[dataconfig['ords']+dataconfig['labs']].map(lambda x: str(x))
        valdata[dataconfig['ords']+dataconfig['labs']] = \
            valdata[dataconfig['ords']+dataconfig['labs']].map(lambda x: str(x))
    alldata = None
    print(label_encoder_list[-1]['label_encoder'].classes_)
    data = encode_categorical_columns(data, label_encoder_list)
    valdata = encode_categorical_columns(valdata, label_encoder_list)
    

def parse(raws, args, file_path, outpath):
    real = pd.read_csv(os.path.join(file_path, 'all.csv'))
    cols  = set(real.columns)
    
    def parse_line(l):
        entries = l[:-1].split(';')[1:] # remove newline at end
        # print(entries)
        words = [c.split(' ') for c in entries] #'name', 'is', 'value'
        # print(words)
        d = dict()
        for c in words:
            if c[0] in cols and len(c) == 3 and c[0] not in d: # keep only first occurence
                d[c[0]] = c[2]
        # d = {c[0]:c[2] for c in words if len(c)==3 and c[0] in cols}
        # print(d)
        # print(set(d.keys()), cols)

        if set(d.keys()) == cols:
            print('success')
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

    if args.ec:
        df = decode_categorical_columns(df, label_encoder_list)
    else:
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
    if accesstoken is not None:
        tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='left', token=accesstoken)
    else:
        tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    # special_tokens_dict = {"bos_token": "<BOS>", 'eos_token': '<EOC>'}
    # num_added_toks = tokenizer.add_special_tokens({"bos_token": "<BOS>"})
    bos_token_id = tokenizer(';', add_special_tokens=False).input_ids[0]#len(tokenizer)-1
    
    if args.efficient:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        quantization_config = None

    if accesstoken is not None:
        dgpt2 = transformers.AutoModelForCausalLM.from_pretrained(modelname, token=accesstoken, device_map={"": PartialState().process_index},
                                                                  quantization_config=quantization_config)
    else:
        dgpt2 = transformers.AutoModelForCausalLM.from_pretrained(modelname, device_map={"": PartialState().process_index},
                                                                  quantization_config=quantization_config)
    print(dgpt2, type(dgpt2))
    # dgpt2.resize_token_embeddings(len(tokenizer))
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.moe or args.mh:
        num_experts = len(data.columns)
        print('create', num_experts, 'head moe model')
        dgpt2copy = MOEModelForCausalLM(dgpt2, num_experts=num_experts, moe=args.moe, multihead=args.mh, 
                                        pad=tokenizer.pad_token_id, eoc=tokenizer(';', add_special_tokens=False).input_ids[0])
        model = dgpt2copy # don't forget to change tokenizer name and optimizer too
        model.set_train_mode()
    else:
        model = dgpt2
        
    if args.efficient:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters(): # so the LM head is still fully finetuned
            param.requires_grad = True
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
                linear_layers.append(name)
        
        lora_config = LoraConfig(
            r=1,  
            lora_alpha=256,
            target_modules=linear_layers,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,  # this is specific for gpt2 model, to be adapted
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print('applying lora, model type now', type(model))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        
    print(model)
    
    if args.train or args.valtrain:
        config = {
            'file_path': file_path,
            'creation_time': str(now),
            'lr': args.lr,
            'epochs': args.epochs,
            'args': vars(args)
        }
        with open(os.path.join(outpath, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        model.train()

        # Data stuff
        # Preprocess the data: Convert each row to a string
        def row_to_col_sentences(row):
            return [str(col).strip() + " is " + str(val).strip() + ';' for col, val in zip(row.index, row.values)]

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
                    text = row_to_col_sentences(data[self.cols].iloc[idx]) # ['age is 39', 'workclass is State-gov', ...]
                if self.do_moe_format:
                    # print(text)
                    tokenized_text = self.tokenizer(text, truncation=True, max_length=self.max_col_length, padding='max_length', return_tensors="pt",
                                                    add_special_tokens=False)
                    prompt = torch.full((1,), #batch_size x token
                                        bos_token_id)
                    return {'input_ids': prompt, 'labels': tokenized_text.input_ids.squeeze()}
                else:
                    text = tokenizer.decode([bos_token_id])[0] + ''.join(text)
                    # print(text)
                    tokenized_text = self.tokenizer(text, truncation=True, padding='longest', return_tensors='pt')
                    return {'input_ids': tokenized_text.input_ids.squeeze(), 'attention_mask': tokenized_text.attention_mask.squeeze(),
                            'labels': tokenized_text.input_ids.squeeze()}
                    
                    
        class CustomDataCollator(DataCollatorForTokenClassification):
            def __call__(self, features):
                input_ids = [item['input_ids'] for item in features]
                labels = [item['labels'] for item in features]
                
                # Pad sequences
                input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                labels_padded = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

                return {
                    'input_ids': input_ids_padded,
                    'labels': labels_padded
                }
        

                    

        text_data = data.apply(row_to_col_sentences, axis=1).tolist()
        do_moe_format = args.moe or args.mh
        dataset = TextDataset(text_data, tokenizer, max_col_length=dataconfig['max_col_length'], do_moe_format=do_moe_format,)
        
        text_valdata = valdata.apply(row_to_col_sentences, axis=1).tolist()
        valdataset = TextDataset(text_valdata, tokenizer, max_col_length=dataconfig['max_col_length'], do_moe_format=do_moe_format)
        
        
        targs = TrainingArguments(output_dir=outpath, overwrite_output_dir=True, do_train=True, save_steps=5000,
                                  per_device_train_batch_size=1, per_device_eval_batch_size=1, 
                                  learning_rate=args.lr, num_train_epochs=args.epochs,
                                  load_best_model_at_end = True, evaluation_strategy='steps', eval_steps=5000,
                                  save_total_limit = 3, metric_for_best_model='eval_loss', bf16=args.efficient, ddp_find_unused_parameters=False, gradient_checkpointing=False, gradient_checkpointing_kwargs={"use_reentrant": False})
        trainer = Trainer(model, targs, train_dataset=dataset, eval_dataset=valdataset, #data_collator=CustomDataCollator(tokenizer=tokenizer),
                                  callbacks = [EarlyStoppingCallback(early_stopping_threshold=0, early_stopping_patience=2)])
        trainer.train(resume_from_checkpoint=args.resume)

        torch.save(model.state_dict(), os.path.join(outpath, f'model.pt'))
        
        valresult = trainer.evaluate(valdataset)
        print('valresult', valresult)
        config['validation_eval'] = valresult
        with open(os.path.join(outpath, 'config.json'), 'w') as f:
            json.dump(config, f)
            
        pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(outpath, 'losses.csv'))
    
    if not args.train and not args.valtrain: # load in checkpoint so we can validate or sample
        ckpt_path = os.path.join(outpath, 'model.pt')
        print('loading from', ckpt_path)
        sd = torch.load(ckpt_path)
        for name, param in model.named_parameters():
            param.data.copy_(sd[name])
        
    if args.validation:
        raise NotImplementedError()

    if args.n_samples > 0:
        from transformers.utils import logging
        logging.set_verbosity_error()
        
        if args.efficient:
            model = model.merge_and_unload()
        model.eval()
        column_names_tokens = tokenizer(list(data.columns), add_special_tokens=False).input_ids
        print(list(data.columns), column_names_tokens)
        if args.moe or args.mh:
            token_heads = list(range( len(data.columns) ))
            model.set_generation_mode(token_heads=token_heads, column_names_tokens=column_names_tokens)
            sbs = 1
        else: 
            sbs = 1#min(1, args.n_samples)

        inputs = torch.full((sbs, 1), bos_token_id).to(model.device)
        if args.llama1:
            inputs = tokenizer(';', return_tensors='pt')['input_ids'].cuda()#.unsqueeze(0)

        samples = []
        startind = 0 # remove BOS token
        # if args.moe or args.mh:
        #     startind=2
        for i in tqdm(range(0, args.n_samples, sbs)):
            
            toks = model.generate(inputs, do_sample=True, num_beams=1, max_length=1000,#dataconfig['max_col_length']*len(dataconfig['cols']), 
                                # pad_token_id=tokenizer.eos_token_id
                                )[...,startind:] # remove BOS token
            outs = tokenizer.batch_decode(toks)
            samples.extend(outs)
            if len(samples)%100 == 0:
                with open(os.path.join(outpath, 'samples.txt'), 'a+') as f:
                    f.write('\n'.join(samples))
                samples = []
            
        with open(os.path.join(outpath, 'samples.txt'), 'a+') as f:
            f.write('\n'.join(samples))
            
        print('samples saved to', os.path.join(outpath, 'samples.txt'))
        with open(os.path.join(args.path, 'samples.txt'), 'r') as f:
            raws = f.readlines()
        parse(raws, args, file_path, outpath)
        
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
            
        ef = False
        if args.efficient:
            ef = 'lora'
        
        model = GReaT(llm=modelname, batch_size=1, per_device_eval_batch_size=1,
            epochs=args.epochs, save_steps=5000,
            experiment_dir=outpath, multihead=args.mh, moe=args.moe, learning_rate=args.lr,
            load_best_model_at_end = True, evaluation_strategy='steps', eval_steps=5000,
            save_total_limit = 1, metric_for_best_model='eval_loss',
            efficient_finetuning=ef, bf16=args.llama1 or args.llama8)
        trainer = model.fit(data, eval_dataset=valdata, conditional_col=dataconfig['labs'][0], resume_from_checkpoint=args.resume)
        model.save(outpath)
        
        
        great_valds = GReaTDataset.from_pandas(valdata)
        great_valds.set_stuff(model.tokenizer, args.moe or args.mh) 
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
        sbs = min(100, args.n_samples) #sample batch size
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
