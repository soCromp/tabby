from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import sys

modelname = sys.argv[-3] # 'gpt2'
outpath = sys.argv[-2] # './gpt2icl.csv'
ex_size = int(sys.argv[-1]) # 12

model = AutoModelForCausalLM.from_pretrained(modelname, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='left')
tokenizer.pad_token_id = tokenizer.eos_token_id

train = pd.read_csv('./data/house-new-tiny/latest/train.csv')
def row_to_col_sentences(row):
	return "".join( [str(col).strip() + " is " + str(val).strip() + ', ' for col, val in zip(row.index, row.values)])

batch_size = 32
n = 10000

def parse(row):
	try:
		result = []
		colsraw = row.split(',')[:-1]
		for col in colsraw:
			result.append(col.split(' ')[-1])
		if len(result) == len(train.columns):
			return result
		else:
			return None
	except:
		return None

rowsparsed = []
for i in tqdm(range(0, n, batch_size)):
	cur_batch_size = min(batch_size, n-i)
	prompts = []
	for _ in range(batch_size):
		text_data = train.sample(n=ex_size).apply(row_to_col_sentences, axis=1).tolist()
		prompt = 'Provide the next one row of this tabular dataset:\n' + '\n'.join(text_data) + '\n'
		prompts.append(prompt)

	toks = tokenizer(prompts, return_tensors='pt', padding=True)
	outtoks = model.generate(input_ids=toks.input_ids.cuda(), attention_mask=toks.attention_mask.cuda(), 
							do_sample=True, max_new_tokens=120)
	outwin = tokenizer.batch_decode(outtoks)
	outs = [text[len(prompt):] for text, prompt in zip(outwin, prompts)]
	outstrim = []
	for text in outs:
		lines = text.split('\n')
		if len(lines) >=2:
			outstrim.append(lines[1])

	parsed = [parse(row) for row in outstrim]
	parsed = [p for p in parsed if p is not None]
	rowsparsed.extend(parsed)
 
synth = pd.DataFrame(rowsparsed, columns=train.columns)
print(synth.head())
synth.to_csv(outpath, index=False)
