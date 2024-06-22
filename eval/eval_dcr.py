import numpy as np
import torch 
import pandas as pd
import json

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

pd.options.mode.chained_assignment = None

import argparse

# Function to calculate distances in batches
def calculate_min_distances(syn_batch, data, batch_size_data):
    min_distances = torch.full((syn_batch.size(0),), float('inf'), device=syn_batch.device)
    for start_idx in range(0, data.size(0), batch_size_data):
        end_idx = min(start_idx + batch_size_data, data.size(0))
        data_batch = data[start_idx:end_idx]
        distances = (syn_batch[:, None] - data_batch).abs().sum(dim=2)
        min_batch_distances, _ = distances.min(dim=1)
        min_distances = torch.min(min_distances, min_batch_distances)
    return min_distances


def eval_dcr(syn_data, real_data, test_data, info, dcr_batch_size=1000):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    num_ranges = []

    real_data.columns = list(np.arange(len(real_data.columns)))
    syn_data.columns = list(np.arange(len(real_data.columns)))
    test_data.columns = list(np.arange(len(real_data.columns)))
    for i in num_col_idx:
        num_ranges.append(real_data[i].max() - real_data[i].min()) 
    
    num_ranges = np.array(num_ranges)


    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]
    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]
    num_test_data = test_data[num_col_idx]
    cat_test_data = test_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')
    num_syn_data_np = num_syn_data.to_numpy()
    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
    num_test_data_np = num_test_data.to_numpy()
    cat_test_data_np = cat_test_data.to_numpy().astype('str')

    if cat_real_data.shape[1] > 0:
        encoder = OneHotEncoder()
        encoder.fit(np.concatenate((cat_real_data_np, cat_syn_data_np, cat_test_data_np), axis=0))


        cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
        cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()
        cat_test_data_oh = encoder.transform(cat_test_data_np).toarray()
    else:
        cat_real_data_oh = np.empty((cat_real_data.shape[0], 0))
        cat_syn_data_oh = np.empty((cat_syn_data.shape[0], 0))
        cat_test_data_oh = np.empty((cat_test_data.shape[0], 0))

    num_real_data_np = num_real_data_np / num_ranges
    num_syn_data_np = num_syn_data_np / num_ranges
    num_test_data_np = num_test_data_np / num_ranges

    real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
    syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
    test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    real_data_th = torch.tensor(real_data_np).to(device)
    syn_data_th = torch.tensor(syn_data_np).to(device)  
    test_data_th = torch.tensor(test_data_np).to(device)

    dcrs_real = []
    dcrs_test = []
    batch_size = dcr_batch_size

    for i in tqdm(range((syn_data_th.shape[0] // batch_size) + 1)):
        if i != (syn_data_th.shape[0] // batch_size):
            batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
        else:
            batch_syn_data_th = syn_data_th[i*batch_size:]
            
        # Calculate distances for real and test data in smaller batches
        dcr_real = calculate_min_distances(batch_syn_data_th, real_data_th, batch_size)
        dcr_test = calculate_min_distances(batch_syn_data_th, test_data_th, batch_size)

        dcrs_real.append(dcr_real)
        dcrs_test.append(dcr_test)
        
    dcrs_real = torch.cat(dcrs_real)
    dcrs_test = torch.cat(dcrs_test)
    
    
    score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]
    
    print('DCR Score, a value closer to 0.5 is better')
    print(f'DCR Score = {score}')
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str, default='adult')
    parser.add_argument('--model', type=str, default='model')
    parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

    args = parser.parse_args()

    dataname = args.dataname
    model = args.model

    if not args.path:
        syn_path = f'synthetic/{dataname}/{model}.csv'
    else:
        syn_path = args.path

    real_path = f'synthetic/{dataname}/real.csv'
    test_path = f'synthetic/{dataname}/test.csv'

    data_dir = f'data/{dataname}' 

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    syn_data = pd.read_csv(syn_path)
    real_data = pd.read_csv(real_path)
    test_data = pd.read_csv(test_path)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    num_ranges = []

    real_data.columns = list(np.arange(len(real_data.columns)))
    syn_data.columns = list(np.arange(len(real_data.columns)))
    test_data.columns = list(np.arange(len(real_data.columns)))
    for i in num_col_idx:
        num_ranges.append(real_data[i].max() - real_data[i].min()) 
    
    num_ranges = np.array(num_ranges)


    num_real_data = real_data[num_col_idx]
    cat_real_data = real_data[cat_col_idx]
    num_syn_data = syn_data[num_col_idx]
    cat_syn_data = syn_data[cat_col_idx]
    num_test_data = test_data[num_col_idx]
    cat_test_data = test_data[cat_col_idx]

    num_real_data_np = num_real_data.to_numpy()
    cat_real_data_np = cat_real_data.to_numpy().astype('str')
    num_syn_data_np = num_syn_data.to_numpy()
    cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
    num_test_data_np = num_test_data.to_numpy()
    cat_test_data_np = cat_test_data.to_numpy().astype('str')

    encoder = OneHotEncoder()
    encoder.fit(cat_real_data_np)


    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()
    cat_test_data_oh = encoder.transform(cat_test_data_np).toarray()

    num_real_data_np = num_real_data_np / num_ranges
    num_syn_data_np = num_syn_data_np / num_ranges
    num_test_data_np = num_test_data_np / num_ranges

    real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
    syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
    test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    real_data_th = torch.tensor(real_data_np).to(device)
    syn_data_th = torch.tensor(syn_data_np).to(device)  
    test_data_th = torch.tensor(test_data_np).to(device)

    dcrs_real = []
    dcrs_test = []
    batch_size = 100

    batch_syn_data_np = syn_data_np[i*batch_size: (i+1) * batch_size]

    for i in range((syn_data_th.shape[0] // batch_size) + 1):
        if i != (syn_data_th.shape[0] // batch_size):
            batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
        else:
            batch_syn_data_th = syn_data_th[i*batch_size:]
            
        dcr_real = (batch_syn_data_th[:, None] - real_data_th).abs().sum(dim = 2).min(dim = 1).values
        dcr_test = (batch_syn_data_th[:, None] - test_data_th).abs().sum(dim = 2).min(dim = 1).values
        dcrs_real.append(dcr_real)
        dcrs_test.append(dcr_test)
        
    dcrs_real = torch.cat(dcrs_real)
    dcrs_test = torch.cat(dcrs_test)
    
    
    score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]
    
    print('DCR Score, a value closer to 0.5 is better')
    print(f'{dataname}-{model}, DCR Score = {score}')
