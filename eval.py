import pandas as pd 
import numpy as np
import sys 
import json 
from sklearn import preprocessing, pipeline, ensemble, compose
from sklearn.metrics import *

configpath = sys.argv[-1]
with open(configpath) as f:
    config = json.load(f)
    
with open(f'./data/{config["dataset"]}/latest/config.json') as f: 
    dataconfig = json.load(f)
    
d = {}
d['real'] = pd.read_csv(f'./data/{config["dataset"]}/latest/test.csv')
config.pop('dataset')

for name, paths in config.items():
    d[name] = pd.read_csv(paths[0])


nums = dataconfig['nums']
ords = dataconfig['ords']
labs = dataconfig['labs']

categoriesdict = dict() # collect all unique values for each of the ordinal columns
def to_float_or_nan(value):
    try:
        return float(value)
    except ValueError:
        return np.nan
    
if dataconfig['task'] == 'classification':
    labvals = set([l.strip() for l in d['real'][labs[0]].unique()])
    
for (k, df) in d.items():
    # remove extra spaces around strings, eg ' dog' -> 'dog'
    df = df.map(lambda x: x.strip() if type(x) == str else x)
    for col in ords:
        categoriesdict[col] = categoriesdict.get(col, []) + df[col].unique().tolist()
    
    df.loc[:,nums] = df.loc[:,nums].map(to_float_or_nan)
    
    if dataconfig['task'] == 'classification':
        df = df[df[labs[0]].isin(labvals)]
    else:
        df.loc[:,labs[0]] = df.loc[:,labs[0]].map(to_float_or_nan)
        df = df[~df.isna()[labs[0]]]
        
    df = df.dropna()
        
    # print(k, df[labs[0]].mean(), df[labs[0]].std())
    df = df[d['real'].columns]
    # df = df.sample(2000)
    
    for colname in df.columns:
        df[colname] = df[colname].astype(d['real'][colname].dtype)
        
    # print(k, '\t\t', len(df))
    d[k] = df

categories = []
for col in ords:
    categories.append(list(set(categoriesdict[col])))
ordenc = preprocessing.OrdinalEncoder(categories=categories)
numenc = preprocessing.StandardScaler()

if dataconfig['task'] == 'classification':
    lb = preprocessing.LabelBinarizer()

    def create_classification_pipeline(trainset):
        rfc = ensemble.RandomForestClassifier(n_estimators=10, max_depth=4, random_state=dataconfig['random_state'])
        preprocessing_pipeline = compose.ColumnTransformer([
            ("ordinal_preprocessor", ordenc, ords),
            ("numerical_preprocessor", numenc, nums),
        ])
        complete_pipeline = pipeline.Pipeline([
            ("preprocessor", preprocessing_pipeline),
            ("estimator", rfc)
        ])
        
        preprocessed_labels = lb.fit_transform(trainset[labs[0]]).ravel()
        complete_pipeline.fit(trainset[ords+nums], preprocessed_labels)
        return complete_pipeline

    real = d['real']
    labels = lb.fit_transform(real[labs[0]])

    results = []
    columns = ['run', 'n', 'acc']
    for k, df in d.items():
        rfc = create_classification_pipeline(df)
        score = rfc.score(real[ords+nums], labels)
        results.append((k, len(df), score))
        # print(k, '\t\t', score)

else:
    def create_regression_pipeline(trainset):
        rfc = ensemble.RandomForestRegressor(random_state=dataconfig['random_state'])
        preprocessing_pipeline = compose.ColumnTransformer([
            ("ordinal_preprocessor", ordenc, ords),
            ("numerical_preprocessor", numenc, nums),
        ])
        complete_pipeline = pipeline.Pipeline([
            ("preprocessor", preprocessing_pipeline),
            ("estimator", rfc)
        ])
        
        preprocessed_labels = trainset[labs[0]] # (trainset[labs[0]]-trainset[labs[0]].mean()) / trainset[labs[0]].std()
        complete_pipeline.fit(trainset[ords+nums], preprocessed_labels)
        return complete_pipeline
    
    real = d['real']
    labels = real[labs[0]]
    # labels = (real[labs[0]]-real[labs[0]].mean()) / real[labs[0]].std()
    
    results = []
    for k, df in d.items():
        rfc = create_regression_pipeline(df)
        print(df.describe())
        # labels = (df[labs[0]]-df[labs[0]].mean()) / df[labs[0]].std()
        rsq = rfc.score(real[ords+nums], labels)
        y_pred = rfc.predict(real[ords+nums])
        mse = mean_squared_error(y_pred, labels)
        results.append((k, len(df), rsq, mse))
        # print(k, '\t\t', len(df), '\t\t', rsq, '\t\t', mse)
    columns = ['run', 'n', 'rsq', 'mse']

print(pd.DataFrame(results, columns = columns))
