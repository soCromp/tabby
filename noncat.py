import pandas as pd 
import numpy as np 
import random
import string
import os
import json
import datetime

# Generate 5000 random integers between 0 and 999999
ids = np.random.randint(0, 1000000, size=5000)
# Convert to zero-padded strings
idstrs = [f"{num:06d}" for num in ids]

# Generate years
years = np.random.randint(2000, 2026, size=5000)

# Generate prefix
pres = np.random.randint(0, 2, size=5000)
# Convert to zero-padded strings
prestrs = ['sku' if pre==0 else 'upi' for pre in pres]

code = [f'{pre}_{id}_{year}' for pre, id, year in zip(prestrs, idstrs, years)]

ips = [f'{a}.{b}.{c}-{l}' for a, b, c, l in zip(
    np.random.randint(1, 1000, size=5000),
    np.random.randint(0, 1000, size=5000),
    np.random.randint(0, 1000, size=5000),
    [random.choice(string.ascii_letters) for _ in range(5000)]
)]

df = pd.DataFrame([code, ips]).T
print(df, len(df[0].unique()), len(df[1].unique()))

config = {
    'dataset_name': 'noncat',
    'raw_path': 'noncat.py',
    'task': 'classification',
    'random_state': 42,
    'train_frac': 0,
    'val_frac': 0,
    'creation_time': str(datetime.datetime.now()),
    'max_col_length': 20,
    'cols': [0,1],
    'ords': [0],
    'nums': [],
    'labs': [1]
}

# write everything out
datedirname = '.'.join(config['creation_time'].split())
outpath_date   = os.path.join('./data/', config['dataset_name'], datedirname)
outpath_latest = os.path.join('./data/', config['dataset_name'], 'latest')

for path in [outpath_date, outpath_latest]:
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, 'train.csv'), index=False)
    df.to_csv(os.path.join(path, 'val.csv'), index=False)
    df.to_csv(os.path.join(path, 'test.csv'), index=False)
    df.to_csv(os.path.join(path, 'all.csv'), index=False)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f)
