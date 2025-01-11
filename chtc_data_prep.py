### HOUSE-NEW
 
import pandas as pd
import datetime
import os
import json
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame

#rename cols so none start with same token
cols = ['income_median', 'age_median', 'rooms', 'bedrooms', 'population', 
        'occupancy', 'latitude', 'longitude', 'value_median_house']
df.columns = cols
ints = ['age_median', 'rooms', 'bedrooms', 'population', 'households', 'value_median_house']
df = df.fillna('?')

config = {
    'dataset_name': 'house-new',
    'raw_path': 'fetch_california_housing(as_frame=True).frame',
    'random_state': 42,
    'train_frac': 0.75,
    'val_frac': 0.075,
    'creation_time': str(datetime.datetime.now()),
    'max_col_length': 20,
    'task': 'regression',
}

config['cols'] = list(df.columns)
config["ords"] = []
config["nums"] = ['income_median', 'age_median', 'rooms', 'bedrooms', 'population', 
        'occupancy', 'latitude', 'longitude',]
config["labs"] = ["value_median_house"]
assert set(config['ords']+config['nums']+config['labs'])==set(config['cols']) 
assert len(config['ords'])+len(config['nums'])+len(config['labs']) == len(config['cols'])

df = df.sample(frac=1, random_state=config['random_state'], ignore_index=True)

# split into train/val/test sets
n = len(df)
train_size = int(config['train_frac'] * n)
val_size = int(config['val_frac'] * n)
train = df.iloc[:train_size, :]
val = df.iloc[train_size:train_size+val_size, :]
test = df.iloc[train_size+val_size:, :]
print('train', train.shape, 'val', val.shape, 'test', test.shape)

# write everything out
datedirname = '.'.join(config['creation_time'].split())
outpath_date   = os.path.join('./data/', config['dataset_name'], datedirname)
outpath_latest = os.path.join('./data/', config['dataset_name'], 'latest')

for path in [outpath_date, outpath_latest]:
    os.makedirs(path, exist_ok=True)
    train.to_csv(os.path.join(path, 'train.csv'), index=False)
    val.to_csv(os.path.join(path, 'val.csv'), index=False)
    test.to_csv(os.path.join(path, 'test.csv'), index=False)
    df.to_csv(os.path.join(path, 'all.csv'), index=False)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f)
        
### TRAVEL

import pandas as pd
import datetime
import os
import json

config = {
    'dataset_name': 'travel',
    'raw_path': 'https://www.kaggle.com/datasets/tejashvi14/tour-travels-customer-churn-prediction?resource=download',
    'task': 'classification',
    'random_state': 42,
    'train_frac': 0.75,
    'val_frac': 0.075,
    'creation_time': str(datetime.datetime.now()),
    'max_col_length': 20,
    'cols': ['Age','Frequent-Flyer','Class','Services','Social-Media','Hotel','Target'],
    'ords': ['Frequent-Flyer','Class','Social-Media','Hotel'],
    'nums': ['Age','Services',],
    'labs': ['Target']
}

# read in, rename columns
df = pd.read_csv('travel.csv')
assert set(config['ords']+config['nums']+config['labs'])==set(config['cols']) 
assert len(config['ords'])+len(config['nums'])+len(config['labs']) == len(config['cols'])

df[config['ords']] = df[config['ords']].map(lambda x: '-'.join(x.split(' ')))

# shuffle data
df = df.sample(frac=1, random_state=config['random_state'], ignore_index=True)

# split into train/val/test sets
n = len(df)
train_size = int(config['train_frac'] * n)
val_size = int(config['val_frac'] * n)
train = df.iloc[:train_size, :]
val = df.iloc[train_size:train_size+val_size, :]
test = df.iloc[train_size+val_size:, :]
print('train', train.shape, 'val', val.shape, 'test', test.shape)

# write everything out
datedirname = '.'.join(config['creation_time'].split())
outpath_date   = os.path.join('./data/', config['dataset_name'], datedirname)
outpath_latest = os.path.join('./data/', config['dataset_name'], 'latest')

for path in [outpath_date, outpath_latest]:
    os.makedirs(path, exist_ok=True)
    train.to_csv(os.path.join(path, 'train.csv'), index=False)
    val.to_csv(os.path.join(path, 'val.csv'), index=False)
    test.to_csv(os.path.join(path, 'test.csv'), index=False)
    df.to_csv(os.path.join(path, 'all.csv'), index=False)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f)

### DIABETES-NEW

import openml
import pandas as pd
import datetime
import os
import json

dataset = openml.datasets.get_dataset('diabetes')
df, _, _, _ = dataset.get_data(dataset_format="dataframe")

cols = ['pregnancies', 'glucose-plasma', 'blood-pressure', 'skin-thickness', 'insulin', 'BMI', 'pedigree', 'age', 'diagnosis']
ords = []
labs = ['diagnosis']
nums = ['pregnancies', 'glucose-plasma', 'blood-pressure', 'skin-thickness', 'insulin', 'BMI', 'pedigree', 'age']

df.columns = cols 
df['diagnosis'] = df['diagnosis'].map(lambda x: 'positive' if x=='tested_positive' else 'negative')

config = {
    'dataset_name': 'diabetes-new',
    'task': 'classification',
    'raw_path': "openml.datasets.get_dataset('diabetes')",
    'random_state': 42,
    'train_frac': 0.75,
    'val_frac': 0.075,
    'creation_time': str(datetime.datetime.now()),
    'max_col_length': 20,
    'cols': cols,
    'ords': ords,
    'nums': nums,
    'labs': labs,
}
assert set(config['ords']+config['nums']+config['labs'])==set(config['cols']) 
assert len(config['ords'])+len(config['nums'])+len(config['labs']) == len(config['cols'])

# shuffle data
df = df.sample(frac=1, random_state=config['random_state'], ignore_index=True)

# split into train/val/test sets
n = len(df)
train_size = int(config['train_frac'] * n)
val_size = int(config['val_frac'] * n)
train = df.iloc[:train_size, :]
val = df.iloc[train_size:train_size+val_size, :]
test = df.iloc[train_size+val_size:, :]
print('train', train.shape, 'val', val.shape, 'test', test.shape)

# write everything out
datedirname = '.'.join(config['creation_time'].split())
outpath_date   = os.path.join('./data/', config['dataset_name'], datedirname)
outpath_latest = os.path.join('./data/', config['dataset_name'], 'latest')

for path in [outpath_date, outpath_latest]:
    os.makedirs(path, exist_ok=True)
    train.to_csv(os.path.join(path, 'train.csv'), index=False)
    val.to_csv(os.path.join(path, 'val.csv'), index=False)
    test.to_csv(os.path.join(path, 'test.csv'), index=False)
    df.to_csv(os.path.join(path, 'all.csv'), index=False)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f)
