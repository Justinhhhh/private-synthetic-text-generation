import os
import glob
import json
import numpy as np
from collections import defaultdict
files = glob.glob(os.path.join('results/drugs/eps3/phi', '*_results.jsonl'))

acc_results = defaultdict(list)
mf1_results = defaultdict(list)
for file in files:
    seed, split, _ = os.path.basename(file).split('_')
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            acc_results[split].append(data['test_acc'])
            mf1_results[split].append(data['mf1'])

for key in ['1000', '5000', '10000']:
    print('Samples: ', key)
    print("mean acc: {:.4f}, sd acc: {:.4f}".format(np.mean(acc_results[key]), np.std(acc_results[key])))
    print("mean mf1: {:.4f}, sd mf1: {:.4f}".format(np.mean(mf1_results[key]), np.std(mf1_results[key])))
