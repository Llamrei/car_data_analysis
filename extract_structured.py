from pathlib import Path
import pickle as pkl

import numpy as np
import bs4
from numpy.random import default_rng
rng = default_rng()
import sklearn
import csv
import re

import sys

if len(sys.argv) > 1:
    SAVE_PATH = sys.argv[1]
else:
    SAVE_PATH = 'structured_data.csv'

data_dir = Path('/mnt/data/car_auction_data')

data_files =  list(data_dir.glob('backup*'))
rng.permutation(data_files)
data_files_sample = data_files[:30]

pre_process = {
    'mileage': lambda x: float(re.sub('[^(0-9|.)]','',x)),
    'manufactured-year' : lambda x: int(x.split(' ')[0]),
    'engine-size' : lambda x: float(re.sub('[^(0-9|.)]','',x)),
    # About a third dont have ownership data
    'owners' : lambda x: int(x.split(' ')[0]),
    'price' : lambda x:x,
    'make' : lambda x:x
    # 'guid' : lambda x:x
}

if not Path(SAVE_PATH).exists():
    with open('structure_data.csv', 'w') as g:
        to_write = {key: None for key in pre_process}
        writer = csv.DictWriter(g, fieldnames=to_write.keys())
        writer.writeheader()

for idx, file in enumerate(data_files_sample):
    print(f'\r File {idx}/{len(data_files_sample)}      ', end='')
    with open(file, 'rb') as f:
        data = pkl.load(f)
        structured_data = ({**x['vehicle']['keyFacts'], 'price':x['price_float'], 'make': x['vehicle']['make'], 'guid':x['guid']}  for x in data)
    with open(SAVE_PATH, 'a') as g:
        to_write = {key: None for key in pre_process}
        writer = csv.DictWriter(g, fieldnames=to_write.keys())
        for ad in structured_data:
            to_write = {key: None for key in pre_process}
            for key in pre_process:
                if key in ad:
                    to_write[key] = pre_process[key](ad[key])
            writer.writerow(to_write)

print("\nDone")