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

outpath = './ckpts/moemh/dgpt2/adult-allcol/aug13'
os.makedirs(outpath, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2", padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {"bos_token": "<BOS>", 'eos_token': '<EOS>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

dgpt2 = transformers.AutoModelForCausalLM.from_pretrained('distilgpt2')
dgpt2.resize_token_embeddings(len(tokenizer))
num_experts = 15
dgpt2copy = MOEModelForCausalLM(dgpt2, num_experts=num_experts, multihead=True)
model = dgpt2copy # don't forget to change tokenizer name and optimizer too

# model.train()
# model.set_train_mode()
# # Move the model to the device (GPU if available)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# Data stuff
# Load the dataset
file_path = '/hdd3/sonia/data/adult.csv'  # Update this with the correct path
data = pd.read_csv(file_path)

# # Preprocess the data: Convert each row to a string
# def row_to_string(row):
#     return ", ".join([f"{col} is {val}" for col, val in row.items()]) + "."
# def row_to_sentences(row):
#     return '. '.join([str(col).strip() + " is " + str(val).strip() for col, val in zip(row.index, row.values)])
# def row_to_col_sentences(row):
#     return [str(col).strip() + " is " + str(val).strip() + '.<EOS>' for col, val in zip(row.index, row.values)]

# class TextDataset(Dataset):
#     def __init__(self, texts, tokenizer, cols=None, max_col_length=10, do_moe_format=True):
#         self.texts = texts
#         self.tokenizer = tokenizer
#         self.cols = cols # "None" for all cols, else a list of desired cols' names
#         self.max_col_length = max_col_length
#         self.do_moe_format = do_moe_format

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         if self.cols is None:
#             text = row_to_col_sentences(data.iloc[idx])
#         else:
#             text = row_to_col_sentences(data[self.cols].iloc[idx]) # ['age is 39.', 'workclass is State-gov.', ...]
#         if self.do_moe_format:
#             tokenized_text = self.tokenizer(text, truncation=True, max_length=self.max_col_length, padding='max_length', return_tensors="pt")
#             prompt = torch.full((1,), #batch_size x token
#                                 self.tokenizer.bos_token_id)
#             return prompt, tokenized_text.input_ids.squeeze()
#         else:
#             text = tokenizer.bos_token + ' '.join(text)
#             tokenized_text = self.tokenizer(text, truncation=True, padding='longest', return_tensors='pt')
#             return tokenized_text.input_ids.squeeze(), tokenized_text.attention_mask.squeeze()
            

# text_data = data.apply(row_to_col_sentences, axis=1).tolist()
# dataset = TextDataset(text_data, tokenizer, max_col_length=20)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# # Set up the optimizer and learning rate scheduler
# optimizer = AdamW(model.parameters(), lr=5e-6)
# # num_training_steps = len(dataloader) * 1  # Number of epochs
# # lr_scheduler = LinearLR(optimizer, total_iters=num_training_steps)

# ins = tokenizer(tokenizer.bos_token, return_tensors='pt')

# lossesmoe = []
# for epoch in range(1):  # Train for 1 epochs
#     for batch in tqdm(dataloader):
#         optimizer.zero_grad()
#         prompt, labels = batch
#         prompt = prompt.to(device)
#         labels = labels.to(device)

#         outputs = model.multicol_forward(input_ids=prompt, labels=labels)
#         # outputs = model.debug_forward(ins['input_ids'].to(device), ins['attention_mask'].to(device), labels=labels)
#         loss = outputs.loss

#         loss.backward()
#         optimizer.step()

#         lossesmoe.append(loss.item())
#         if len(lossesmoe) % 1000 == 0:
#             torch.save(model.state_dict(), os.path.join(outpath, f'{len(lossesmoe)}.pt'))
#             try:
#                 plt.close()
#             except:
#                 pass
#             plt.plot(lossesmoe)
#             plt.savefig(os.path.join(outpath, 'loss.png'))

model.load_state_dict(torch.load(outpath+'/48000.pt'))
model.eval()
print(list(data.columns))
column_names_tokens = tokenizer(list(data.columns)).input_ids
model.set_generation_mode(column_names_tokens=column_names_tokens)

samples = []
for i in tqdm(range(10000)):
    toks = model.generate(do_sample=True, num_beams=1, max_length=140, 
                          pad_token_id=tokenizer.eos_token_id)[...,1:] # remove BOS token
    samples.append(tokenizer.batch_decode(toks)[0])
    
with open(os.path.join(outpath, 'samples.txt'), 'w') as f:
    f.write('\n'.join(samples))
    