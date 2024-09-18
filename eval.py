import pandas as pd 
import numpy as np
import sys 
import json 
from sklearn import preprocessing, pipeline, ensemble, compose, tree, linear_model
from sklearn.metrics import *
import os 
from copy import deepcopy

dataname = sys.argv[1]
with open(f'./data/{dataname}/latest/config.json') as f:
    dataconfig = json.load(f)
    
trainpath = f'./data/{dataconfig["dataset_name"]}/latest/train.csv'
testpath = f'./data/{dataconfig["dataset_name"]}/latest/test.csv'
train = pd.read_csv(trainpath)
test = pd.read_csv(testpath)
    
sets = [train,]
names = ['real',]
for i in range(2, len(sys.argv)):
    ckptpath = sys.argv[i]
    names.append(ckptpath[10:])
    if ckptpath.endswith('.csv'):
        synth = pd.read_csv(ckptpath)
    else:
        synth = pd.read_csv(os.path.join(ckptpath, 'samplesclean.csv'))
    sets.append(synth)
    

numcols = dataconfig['nums']
ordcols = dataconfig['ords']
labcols = dataconfig['labs']

categoriesdict = dict() # collect all unique values for each of the ordinal columns
def to_float_or_nan(value):
    try:
        return float(value)
    except ValueError:
        return np.nan
    
if dataconfig['task'] == 'classification':
    labvals = set([l.strip() for l in train[labcols[0]].unique()]) | \
                set([l.strip() for l in test[labcols[0]].unique()])
    # print(labvals)
    
def preprocess_df(df, categoriesdict):
    # remove extra spaces around strings, eg ' dog' -> 'dog'
    df = df.map(lambda x: x.strip() if type(x) == str else x)
    df.loc[:,numcols] = df.loc[:,numcols].map(to_float_or_nan)
    df.loc[:,ordcols] = df.loc[:,ordcols].fillna('?')
    df.loc[:,ordcols] = df.loc[:,ordcols].map(lambda x: x.strip())
    for col in ordcols:
        categoriesdict[col] = categoriesdict.get(col, []) + df[col].unique().tolist()
    
    if dataconfig['task'] == 'classification':
        df = df[df[labcols[0]].isin(labvals)]
    else:
        df.loc[:,labcols[0]] = df.loc[:,labcols[0]].map(to_float_or_nan)
        df = df[~df.isna()[labcols[0]]]
        
    # df = df.dropna()
        
    # print(k, df[labs[0]].mean(), df[labs[0]].std())
    df = df[train.columns] # put all in same order
    # df = df.sample(2000)
    
    # for colname in df.columns:
    #     df[colname] = df[colname].astype(train[colname].dtype)
        
    # print(k, '\t\t', len(df))
    return df, categoriesdict


test, categoriesdict = preprocess_df(test, categoriesdict)
for i in range(len(sets)):
    sets[i], categoriesdict = preprocess_df(sets[i], categoriesdict)
train = sets[0]

categories = []
for col in ordcols:
    categories.append(list(set(categoriesdict[col])))

ordenc = preprocessing.OrdinalEncoder(categories=categories)
numenc = preprocessing.StandardScaler()

def distance_to_closest_record(synth, real):
    # takes a synthetic dataset and the real *train* dataset
    nums = deepcopy(dataconfig['nums'])
    ords = deepcopy(dataconfig['ords'])
    if dataconfig['task'] == 'regression':
        nums.extend(dataconfig['labs'])
    else: # classification
        ords.extend(dataconfig['labs'])
    
    mindists = synth.apply(lambda x: ((x[nums]-real[nums]).abs().sum(axis=1) + \
        (x[ords]!=real[ords]).sum(axis=1)).min(), axis=1)
    return mindists

def discriminate(synth, real):
    size = 2*min(len(synth), len(real))
    synth = deepcopy(synth).sample(size//2, random_state=dataconfig['random_state'])
    synth['set'] = 'synth'
    real = deepcopy(real).sample(size//2, random_state=dataconfig['random_state'])
    real['set'] = 'real'
    data = pd.concat([synth, real]).sample(frac=1, random_state=dataconfig['random_state'])
    train = data[:size*3//4]
    test = data[size*3//4:]
    trainlabels = train.pop('set')
    testlabels = test.pop('set')
    
    discrim_ordcols = deepcopy(dataconfig['ords'])
    discrim_numcols = deepcopy(dataconfig['nums'])
    
    discrim_categories = deepcopy(categories)
    if dataconfig['task'] == 'classification':
        discrim_categories.append(list(set(real[labcols[0]].unique())))
        discrim_ordcols.extend(labcols)
    else: #regression
        discrim_numcols.extend(labcols)

    discrim_ordenc = preprocessing.OrdinalEncoder(categories=discrim_categories)
    discrim_numenc = preprocessing.StandardScaler()
    lb = preprocessing.LabelBinarizer()
    
    model = ensemble.RandomForestClassifier(n_estimators=10, max_depth=4, random_state=dataconfig['random_state'])
    preprocessing_pipeline = compose.ColumnTransformer([
        ("ordinal_preprocessor", discrim_ordenc, discrim_ordcols),
        ("numerical_preprocessor", discrim_numenc, discrim_numcols),
    ])
    complete_pipeline = pipeline.Pipeline([
        ("preprocessor", preprocessing_pipeline),
        ("estimator", model)
    ])
    
    # print(train[ordcols+numcols].head())
    preprocessed_trainlabels = lb.fit_transform(trainlabels).ravel()
    preprocessed_testlabels = lb.fit_transform(testlabels).ravel()
    complete_pipeline.fit(train[discrim_ordcols+discrim_numcols], preprocessed_trainlabels)
    acc = complete_pipeline.score(test[discrim_ordcols+discrim_numcols], preprocessed_testlabels)
    return acc

def k_anon(data):
    n_clusters = 100

if dataconfig['task'] == 'classification':
    lb = preprocessing.LabelBinarizer()

    def create_classification_pipeline(trainset, type='rfc'):
        if type == 'rfc':
            model = ensemble.RandomForestClassifier(n_estimators=10, max_depth=4, random_state=dataconfig['random_state'])
        elif type == 'dt':
            model = tree.DecisionTreeClassifier(min_samples_split=4, random_state=dataconfig['random_state'])
        elif type == 'lr':
            model = linear_model.LogisticRegression(random_state=dataconfig['random_state'])
            
        preprocessing_pipeline = compose.ColumnTransformer([
            ("ordinal_preprocessor", ordenc, ordcols),
            ("numerical_preprocessor", numenc, numcols),
        ])
        complete_pipeline = pipeline.Pipeline([
            ("preprocessor", preprocessing_pipeline),
            ("estimator", model)
        ])
        
        preprocessed_labels = lb.fit_transform(trainset[labcols[0]]).ravel()
        complete_pipeline.fit(trainset[ordcols+numcols], preprocessed_labels)
        return complete_pipeline
    
    labels = lb.fit_transform(test[labcols[0]])
    results = []
    columns = ['run', 'n', 'rfc-acc', 'dt-acc', 'lr-acc', 
            #    'dcr-mean', 'dcr-std', 
               'disc']

    
    for i in range(len(sets)):
        data = sets[i]
        print(names[i])
        # random forest
        rf = create_classification_pipeline(data, type='rfc')
        score_rf = rf.score(test[ordcols+numcols], labels)
        
        # decision tree
        dt = create_classification_pipeline(data, 'dt')
        score_dt = dt.score(test[ordcols+numcols], labels)
        
        # logistic regression
        lr = create_classification_pipeline(data.dropna(axis=0), 'lr')
        score_lr = lr.score(test[ordcols+numcols].dropna(axis=0), labels)
        
        # DCR
        # dcr = distance_to_closest_record(data, train)
        
        # discrimination
        disacc = discriminate(data, train)
        
        results.append((names[i], len(data), score_rf, score_dt, score_lr,
            # dcr.mean(), dcr.std(), 
            disacc))
    

else:
    def create_regression_pipeline(trainset, type='rfr'):
        if type == 'rfr':
            model = ensemble.RandomForestRegressor(random_state=dataconfig['random_state'])
        elif type == 'dt':
            model = tree.DecisionTreeRegressor(min_samples_split=80, random_state=dataconfig['random_state'])
        elif type == 'lr':
            model = linear_model.LinearRegression()
            
        preprocessing_pipeline = compose.ColumnTransformer([
            ("ordinal_preprocessor", ordenc, ordcols),
            ("numerical_preprocessor", numenc, numcols),
        ])
        complete_pipeline = pipeline.Pipeline([
            ("preprocessor", preprocessing_pipeline),
            ("estimator", model)
        ])
        
        preprocessed_labels = trainset[labcols[0]] # (trainset[labs[0]]-trainset[labs[0]].mean()) / trainset[labs[0]].std()
        complete_pipeline.fit(trainset[ordcols+numcols], preprocessed_labels)
        return complete_pipeline
    
    labels = test[labcols[0]]
    results = []
    columns = ['run', 'n', 'rfr-rsq', 'rfr-mse', 'dt-rsq', 'dt-mse',
               'lr-rsq', 'lr-mse', 
            #    'dcr-mean', 'dcr-std'
               'disc']
    
    for i in range(len(sets)):
        data = sets[i]
        print(names[i])
        # random forest
        rf = create_regression_pipeline(data, 'rfr')
        rsq_rf = rf.score(test[ordcols+numcols], labels)
        y_pred = rf.predict(test[ordcols+numcols])
        mse_rf = mean_squared_error(y_pred, labels)
        
        # decision tree
        dt = create_regression_pipeline(data, 'dt')
        rsq_dt = dt.score(test[ordcols+numcols], labels)
        y_pred = dt.predict(test[ordcols+numcols])
        mse_dt = mean_squared_error(y_pred, labels)
        
        # linear regression
        lr = create_regression_pipeline(data.dropna(axis=0), 'lr')
        rsq_lr = lr.score(test[ordcols+numcols].dropna(axis=0), labels)
        y_pred = lr.predict(test[ordcols+numcols].dropna(axis=0))
        mse_lr = mean_squared_error(y_pred, labels)
        
        # DCR
        # dcr = distance_to_closest_record(data, train)
        
        # discrimination
        disacc = discriminate(data, train)
        
        results.append((names[i], len(data), 
            rsq_rf, mse_rf, rsq_dt, mse_dt, rsq_lr, mse_lr, 
            # dcr.mean(), dcr.std())
                       disacc))

df = pd.DataFrame(results, columns = columns)
print(df)
df.to_excel(sys.argv[-1]+'eval.xlsx')

