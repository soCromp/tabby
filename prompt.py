from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm

modelname = 'meta-llama/Meta-Llama-3-8B'
model = AutoModelForCausalLM.from_pretrained(modelname, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(modelname)
tokenizer.pad_token_id = tokenizer.eos_token_id

train = pd.read_csv('./data/diabetes-new/latest/train.csv')
def row_to_col_sentences(row):
	return "".join( [str(col).strip() + " is " + str(val).strip() + ', ' for col, val in zip(row.index, row.values)])

batch_size = 8
ex_size = 20
n = 10000
outpath = './llamaicl.csv'

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

	toks = tokenizer(prompts, return_tensors='pt')
	outtoks = model.generate(input_ids=toks.input_ids.cuda(), attention_mask=toks.attention_mask.cuda(), 
							do_sample=True, max_new_tokens=120)
	outwin = tokenizer.batch_decode(outtoks)
	outs = [text[len(prompt):] for text, prompt in zip(outwin, prompts)]
	outstrim = [text.split('\n')[1] for text in outs]

	parsed = [parse(row) for row in outstrim]
	parsed = [p for p in parsed if p is not None]
	rowsparsed.extend(parsed)
 
synth = pd.DataFrame(rowsparsed, columns=train.columns)
print(synth.head())
synth.to_csv(outpath, index=False)
