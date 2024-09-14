from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')

train = pd.read_csv('./data/diabetes-new/latest/train.csv')

i = 1
examples = train.iloc[:i, :]
print(examples)

def row_to_col_sentences(row):
	return "<BOS>" + "".join( [str(col).strip() + " is " + str(val).strip() + '<EOS>' for col, val in zip(row.index, row.values)])

text_data = examples.apply(row_to_col_sentences, axis=1).tolist()

prompt = 'Provide the following rows of this tabular dataset:\n' + '\n'.join(text_data)

out=model.generate(**tokenizer(prompt, return_tensors='pt'), max_new_tokens=200)
print(tokenizer.batch_decode(out))
