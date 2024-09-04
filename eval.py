import pandas as pd 
import numpy as np
import sys 
import json 
from sklearn import preprocessing, pipeline, ensemble, compose

configpath = sys.argv[-1]
with open(configpath) as f:
    config = json.load(f)
    
with open(f'./data/{config["dataset"]}/latest/config.json') as f: 
    dataconfig = json.load(f)
    
d = {}
d['real'] = pd.read_csv(f'./data/{config["dataset"]}/latest/test.csv')
config.pop('dataset')

for name, path in config.items():
    d[name] = pd.read_csv(path)


nums = dataconfig['nums']
ords = dataconfig['ords']
labs = dataconfig['labs']

categoriesdict = dict() # collect all unique values for each of the ordinal columns
def to_float_or_nan(value):
    try:
        return float(value)
    except ValueError:
        return np.nan
    
labvals = set([l.strip() for l in d['real'][labs[0]].unique()])
for (k, df) in d.items():
    # remove extra spaces around strings, eg ' dog' -> 'dog'
    df = df.map(lambda x: x.strip() if type(x) == str else x)
    for col in ords:
        categoriesdict[col] = categoriesdict.get(col, []) + df[col].unique().tolist()
    
    df.loc[:,nums] = df.loc[:,nums].map(to_float_or_nan)
    df = df[df[labs[0]].isin(labvals)]
    print(k, '\t\t', len(df))
    d[k] = df

categories = []
for col in ords:
    categories.append(list(set(categoriesdict[col])))
ordenc = preprocessing.OrdinalEncoder(categories=categories)
numenc = preprocessing.StandardScaler()
lb = preprocessing.LabelBinarizer()

def create_pipeline(trainset):
    rfc = ensemble.RandomForestClassifier(n_estimators=10, max_depth=4, random_state=dataconfig['random_state'])
    preprocessing_pipeline = compose.ColumnTransformer([
        ("ordinal_preprocessor", ordenc, ords),
        ("numerical_preprocessor", numenc, nums),
    ])
    complete_pipeline = pipeline.Pipeline([
        ("preprocessor", preprocessing_pipeline),
        ("estimator", rfc)
    ])
    
    preprocessed_labels = lb.fit_transform(trainset['income']).ravel()
    complete_pipeline.fit(trainset[ords+nums], preprocessed_labels)
    return complete_pipeline

real = d['real']
labels = lb.fit_transform(real['income'])

for k, df in d.items():
    rfc = create_pipeline(df)
    score = rfc.score(real[ords+nums], labels)
    print(k, '\t\t', score)

