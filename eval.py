import pandas as pd 
import numpy as np
import sys 
import json 
from sklearn import preprocessing, pipeline, ensemble, compose
from sklearn.metrics import *
import os 

ckptpath = sys.argv[-1]
synth = pd.read_csv(os.path.join(ckptpath, 'samplesclean.csv'))
dataname = sys.argv[-2]
with open(f'./data/{dataname}/latest/config.json') as f:
    dataconfig = json.load(f)

trainpath = f'./data/{dataconfig["dataset_name"]}/latest/train.csv'
testpath = f'./data/{dataconfig["dataset_name"]}/latest/test.csv'
train = pd.read_csv(trainpath)
test = pd.read_csv(testpath)

# configpath = sys.argv[-1]
# with open(configpath) as f:
#     config = json.load(f)
    
# with open(f'./data/{config["dataset"]}/latest/config.json') as f: 
#     dataconfig = json.load(f)
    
# d = {}
# d['real'] = pd.read_csv(f'./data/{config["dataset"]}/latest/test.csv')
# config.pop('dataset')

# for name, paths in config.items():
#     d[name] = pd.read_csv(paths[0])


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
    labvals = set([l.strip() for l in train[labs[0]].unique()]) | \
                set([l.strip() for l in test[labs[0]].unique()])
    # print(labvals)
    
def preprocess_df(df, categoriesdict):
    # remove extra spaces around strings, eg ' dog' -> 'dog'
    df = df.map(lambda x: x.strip() if type(x) == str else x)
    df.loc[:,nums] = df.loc[:,nums].map(to_float_or_nan)
    df.loc[:,ords] = df.loc[:,ords].fillna('?')
    for col in ords:
        categoriesdict[col] = categoriesdict.get(col, []) + df[col].unique().tolist()
    
    if dataconfig['task'] == 'classification':
        df = df[df[labs[0]].isin(labvals)]
    else:
        df.loc[:,labs[0]] = df.loc[:,labs[0]].map(to_float_or_nan)
        df = df[~df.isna()[labs[0]]]
        
    # df = df.dropna()
        
    # print(k, df[labs[0]].mean(), df[labs[0]].std())
    df = df[train.columns] # put all in same order
    # df = df.sample(2000)
    
    # for colname in df.columns:
    #     df[colname] = df[colname].astype(train[colname].dtype)
        
    # print(k, '\t\t', len(df))
    return df, categoriesdict

train, categoriesdict = preprocess_df(train, categoriesdict)
test, categoriesdict = preprocess_df(test, categoriesdict)
synth, categoriesdict = preprocess_df(synth, categoriesdict)
# print(train)
# print(test)
# print(synth)

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
    
    labels = lb.fit_transform(test[labs[0]])
    results = []
    columns = ['run', 'n', 'acc']
    
    rfc_real = create_classification_pipeline(train)
    score = rfc_real.score(test[ords+nums], labels)
    results.append(('real', len(train), score))
    
    rfc_synth = create_classification_pipeline(synth)
    score = rfc_synth.score(test[ords+nums], labels)
    results.append(('synth', len(synth), score))

    # real = d['real']
    # labels = lb.fit_transform(real[labs[0]])

    # results = []
    # columns = ['run', 'n', 'acc']
    # for k, df in d.items():
    #     rfc = create_classification_pipeline(df)
    #     score = rfc.score(real[ords+nums], labels)
    #     results.append((k, len(df), score))
    #     # print(k, '\t\t', score)

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
    
    labels = test[labs[0]]
    results = []
    columns = ['run', 'n', 'rsq', 'mse']
    
    rfc_real = create_regression_pipeline(train)
    rsq = rfc_real.score(test[ords+nums], labels)
    y_pred = rfc_real.predict(test[ords+nums])
    mse = mean_squared_error(y_pred, labels)
    results.append(('real', len(train), rsq, mse))
    
    rfc_synth = create_regression_pipeline(synth)
    rsq = rfc_synth.score(test[ords+nums], labels)
    y_pred = rfc_synth.predict(test[ords+nums])
    mse = mean_squared_error(y_pred, labels)
    results.append(('synth', len(synth), rsq, mse))

print(pd.DataFrame(results, columns = columns))
