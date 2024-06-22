import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import lib
import os
import json
import faiss
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.train import Trainer
from scripts.utils_train import get_model, make_dataset_from_df
from tab_ddpm import GaussianMultinomialDiffusion

from tab_ddpm.modules import timestep_embedding
import torch.nn.functional as F
from tab_ddpm import logger
from tab_ddpm.resample import create_named_schedule_sampler

def get_group_data_dict(np_data, group_id_attrs=[0,]):
    group_data_dict = {}
    data_len = len(np_data)
    for i in range(data_len):
        row_id = tuple(np_data[i, group_id_attrs])
        if not row_id in group_data_dict:
            group_data_dict[row_id] = []
        group_data_dict[row_id].append(np_data[i])
    
    return group_data_dict

def get_group_data(np_data, group_id_attrs=[0,]):
    group_data_list = []
    data_len = len(np_data)
    i = 0
    while i < data_len:
        group = []
        row_id = np_data[i, group_id_attrs]

        while (np_data[i, group_id_attrs] == row_id).all():
            group.append(np_data[i])
            i += 1
            if i >= data_len:
                break
        group = np.array(group)
        group_data_list.append(group)
    group_data_list = np.array(group_data_list, dtype=object)

    return group_data_list

def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train = 0, num_test = 0):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    seed = 1234

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1
        
    return train_df, test_df, seed    


def get_info_from_domain(data_df, domain_dict):
    info = {}
    info['num_col_idx'] = []
    info['cat_col_idx'] = []
    columns = data_df.columns.tolist()
    for i in range(len(columns)):
        if domain_dict[columns[i]]['type'] == 'discrete':
            info['cat_col_idx'].append(i)
        else:
            info['num_col_idx'].append(i)

    info['target_col_idx'] = []
    info['task_type'] = 'None'
    info['column_names'] = columns

    return info


def pipeline_process_data(
        name, 
        data_df, 
        info, 
        ratio=0.9,
        save=False
    ):
    num_data = data_df.shape[0]

    column_names = info['column_names'] if info['column_names'] else data_df.columns.tolist()
 
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    # Train/ Test Split, 90% Training, 10% Testing (Validation set will be selected from Training set)
    num_train = int(num_data * ratio)
    num_test = num_data - num_train

    if ratio < 1:
        train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    else:
        train_df = data_df.copy()

    train_df.columns = range(len(train_df.columns))

    if ratio < 1:
        test_df.columns = range(len(test_df.columns))

    if ratio < 1:
        print(name, train_df.shape, test_df.shape, data_df.shape)
    else:
        print(name, train_df.shape, data_df.shape)

    col_info = {}
    
    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(train_df[col_idx].max())
        col_info['min'] = float(train_df[col_idx].min())
     
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(train_df[col_idx]))    

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            col_info['max'] = float(train_df[col_idx].max())
            col_info['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            col_info['categorizes'] = list(set(train_df[col_idx]))      

    info['column_info'] = col_info

    train_df.rename(columns = idx_name_mapping, inplace=True)
    if ratio < 1:
        test_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'

    if ratio < 1:
        for col in num_columns:
            test_df.loc[test_df[col] == '?', col] = np.nan
        for col in cat_columns:
            test_df.loc[test_df[col] == '?', col] = 'nan'

    
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()

    if ratio < 1:
        X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
        X_cat_test = test_df[cat_columns].to_numpy()
        y_test = test_df[target_columns].to_numpy()

 
    if save:
        save_dir = f'data/{name}'
        np.save(f'{save_dir}/X_num_train.npy', X_num_train)
        np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
        np.save(f'{save_dir}/y_train.npy', y_train)

        if ratio < 1:
            np.save(f'{save_dir}/X_num_test.npy', X_num_test)
            np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
            np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)

    if ratio < 1:
        test_df[num_columns] = test_df[num_columns].astype(np.float32)


    if save:
        train_df.to_csv(f'{save_dir}/train.csv', index = False)

        if ratio < 1:
            test_df.to_csv(f'{save_dir}/test.csv', index = False)

        if not os.path.exists(f'synthetic/{name}'):
            os.makedirs(f'synthetic/{name}')
        
        train_df.to_csv(f'synthetic/{name}/real.csv', index = False)

        if ratio < 1:
            test_df.to_csv(f'synthetic/{name}/test.csv', index = False)

    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]

    if ratio < 1:
        info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'


    if task_type == 'regression':
        
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'

    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    if save:
        with open(f'{save_dir}/info.json', 'w') as file:
            json.dump(info, file, indent=4)

    print(f'Processing {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'] if ratio < 1 else info['train_num'])
    print('Train', info['train_num'])
    if ratio < 1:
        print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)

    data = {
        'df': {
            'train': train_df
        },
        'numpy': {
            'X_num_train': X_num_train,
            'X_cat_train': X_cat_train,
            'y_train': y_train
        }
    }

    if ratio < 1:
        data['df']['test'] = test_df
        data['numpy']['X_num_test'] = X_num_test
        data['numpy']['X_cat_test'] = X_cat_test
        data['numpy']['y_test'] = y_test

    return data, info



def load_multi_table(data_dir):
    dataset_meta = json.load(open(os.path.join(data_dir, 'dataset_meta.json'), 'r'))

    relation_order = dataset_meta['relation_order']
    relation_order_reversed = relation_order[::-1]

    tables = {}

    for table, meta in dataset_meta['tables'].items():
        tables[table] = {
            'df': pd.read_csv(os.path.join(data_dir, f'{table}.csv')),
            'domain': json.load(open(os.path.join(data_dir, f'{table}_domain.json'))),
            'children': meta['children'],
            'parents': meta['parents'],
        }
        tables[table]['original_cols'] = list(tables[table]['df'].columns)
        tables[table]['original_df'] = tables[table]['df'].copy()
        id_cols = [col for col in tables[table]['df'].columns if '_id' in col]
        df_no_id = tables[table]['df'].drop(columns=id_cols)
        info = get_info_from_domain(
            df_no_id,
            tables[table]['domain']
        )
        data, info = pipeline_process_data(
            name=table,
            data_df=df_no_id,
            info=info,
            ratio=1,
            save=False
        )
        tables[table]['info'] = info

    return tables, relation_order, dataset_meta

def quantile_normalize_sklearn(matrix):
    transformer = QuantileTransformer(output_distribution='normal', random_state=42)  # Change output_distribution as needed

    normalized_data = np.empty((matrix.shape[0], 0))

    # Apply QuantileTransformer to each column and concatenate the results
    for col in range(matrix.shape[1]):
        column = matrix[:, col].reshape(-1, 1)
        transformed_column = transformer.fit_transform(column)
        normalized_data = np.concatenate((normalized_data, transformed_column), axis=1)

    return normalized_data

def min_max_normalize_sklearn(matrix):
    scaler = MinMaxScaler(feature_range=(-1, 1))

    normalized_data = np.empty((matrix.shape[0], 0))

    # Apply MinMaxScaler to each column and concatenate the results
    for col in range(matrix.shape[1]):
        column = matrix[:, col].reshape(-1, 1)
        transformed_column = scaler.fit_transform(column)
        normalized_data = np.concatenate((normalized_data, transformed_column), axis=1)

    return normalized_data

def sample_from_diffusion(
        df, 
        df_info, 
        diffusion, 
        dataset, 
        label_encoders, 
        sample_size, 
        model_params, 
        T_dict,
        sample_batch_size=8192
    ):
    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    _, empirical_class_dist = torch.unique(torch.from_numpy(dataset.y['train']), return_counts=True)
    x_gen, y_gen = diffusion.sample_all(sample_size, sample_batch_size, empirical_class_dist.float(), ddim=False)
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
    num_numerical_features_sample = num_numerical_features + int(dataset.is_regression and not model_params["is_y_cond"])

    X_num_real = df[df_info['num_cols']].to_numpy().astype(float)
    X_cat_real = df[df_info['cat_cols']].to_numpy().astype(str)
    y_real = np.round(df[df_info['y_col']].to_numpy().astype(float)).astype(int).reshape(-1, 1)

    X_num_ = X_gen

    if num_numerical_features != 0:
        X_num_ = dataset.num_transform.inverse_transform(X_gen[:, :num_numerical_features_sample])
        actual_num_numerical_features = num_numerical_features - len(label_encoders)
        X_num = X_num_[:, :actual_num_numerical_features]
        if len(label_encoders) > 0:
            X_cat = X_num_[:, actual_num_numerical_features:]
            X_cat = np.round(X_cat).astype(int)
            decoded_x_cat = []
            for col in range(X_cat.shape[1]):
                x_cat_col = X_cat[:, col]
                x_cat_col = np.clip(x_cat_col, 0, len(label_encoders[col].classes_) - 1)
                decoded_x_cat.append(label_encoders[col].inverse_transform(x_cat_col))
            X_cat = np.column_stack(decoded_x_cat)
        else:
            X_cat = np.empty((X_num.shape[0], 0))

        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['is_y_cond'] == 'concat':
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = lib.round_columns(X_num_real, X_num, disc_cols)

    y_gen = y_gen.reshape(-1, 1)

    if X_cat_real is not None:
        total_real = np.concatenate((X_num_real, X_cat_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, X_cat, np.round(y_gen).astype(int)), axis=1)
    else:
        total_real = np.concatenate((X_num_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, np.round(y_gen).astype(int)), axis=1)

    df_total = pd.DataFrame(total_real)
    df_gen = pd.DataFrame(gen_real)
    columns = [str(x) for x in list(df_total.columns)]

    df_total.columns = columns
    df_gen.columns = columns

    for col in df_total.columns:
        if int(col) < X_num_real.shape[1]:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)
        elif X_cat_real is not None and int(col) < X_num_real.shape[1] + X_cat_real.shape[1]:
            df_total[col] = df_total[col].astype(str)
            df_gen[col] = df_gen[col].astype(str)
        else:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)

    return df_total, df_gen

def train_model(
        df, 
        df_info, 
        model_params, 
        T_dict, 
        steps,
        batch_size,
        model_type,
        gaussian_loss_type,
        num_timesteps,
        scheduler,
        lr,
        weight_decay,
        device='cuda'
    ):
    T = lib.Transformations(**T_dict)
    dataset, label_encoders, column_orders = make_dataset_from_df(
        df, 
        T,
        is_y_cond=model_params['is_y_cond'],
        ratios=[0.99, 0.005, 0.005], 
        df_info=df_info,
        std=0
    )
    print(dataset.n_features)
    train_loader = lib.prepare_fast_dataloader(
        dataset, 
        split='train', 
        batch_size=batch_size,
        y_type='long'
    )

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)

    print(model_params)
    model = get_model(
        model_type,
        model_params
    )
    model.to(device)

    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()
    
    if model_params['is_y_cond'] == 'concat':
        column_orders = column_orders[1:] + [column_orders[0]]
    else:
        column_orders = column_orders + [df_info['y_col']]

    
    return {
        'diffusion': diffusion,
        'label_encoders': label_encoders,
        'dataset': dataset,
        'column_orders': column_orders
    }

class Classifier(nn.Module):
    def __init__(self, d_in, d_out, dim_t, hidden_sizes, dropout_prob=0.5, num_heads=2, num_layers=1):
        super(Classifier, self).__init__()

        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)

        self.transformer_layer = nn.Transformer(
            d_model=dim_t,
            nhead=num_heads,
            num_encoder_layers=num_layers
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        
        # Create a list to hold the layers
        layers = []
        
        # Add input layer
        layers.append(nn.Linear(dim_t, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))  # Batch Normalization
        layers.append(nn.Dropout(p=dropout_prob))
        
        # Add hidden layers with batch normalization and different activation
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.LeakyReLU())  # Different activation
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))  # Batch Normalization
            layers.append(nn.Dropout(p=dropout_prob))
        
        # Add output layer
        layers.append(nn.Linear(hidden_sizes[-1], d_out))
        
        # Create a Sequential model from the list of layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        x = self.proj(x) + emb
        # x = self.transformer_layer(x, x)
        x = self.model(x)
        return x

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = torch.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def numerical_forward_backward_log(
        classifier, 
        optimizer, 
        data_loader, 
        dataset, 
        schedule_sampler, 
        diffusion, 
        prefix="train",
        remove_first_col=False,
        device='cuda'
):
    
    batch, labels = next(data_loader)
    labels = labels.long().to(device)

    if remove_first_col:
        # Remove the first column of the batch, which is the label.
        batch = batch[:, 1:]
    
    num_batch = batch[:, :dataset.n_num_features].to(device)

    t, _ = schedule_sampler.sample(num_batch.shape[0], device)
    batch = diffusion.gaussian_q_sample(num_batch, t).to(device)

    for i, (sub_batch, sub_labels, sub_t) in enumerate(
        split_microbatches(-1, batch, labels, t)
    ):
        logits = classifier(sub_batch, timesteps=sub_t)
        loss = F.cross_entropy(logits, sub_labels, reduction="none")

        losses = {}
        losses[f"{prefix}_loss"] = loss.detach()
        losses[f"{prefix}_acc@1"] = compute_top_k(
            logits, sub_labels, k=1, reduction="none"
        )
        if logits.shape[1] >= 5:
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
        log_loss_dict(diffusion, sub_t, losses)
        del losses
        loss = loss.mean()
        if loss.requires_grad:
            if i == 0:
                optimizer.zero_grad()
            loss.backward(loss * len(sub_batch) / len(batch))

def train_classifier(
        df, 
        df_info, 
        model_params, 
        T_dict, 
        classifier_steps,
        batch_size,
        gaussian_loss_type,
        num_timesteps,
        scheduler,
        device='cuda',
        cluster_col='cluster',
        d_layers=None,
        dim_t=128,
        lr=0.0001
    ):
    T = lib.Transformations(**T_dict)
    dataset, label_encoders, column_orders = make_dataset_from_df(
        df, 
        T,
        is_y_cond=model_params['is_y_cond'],
        ratios=[0.99, 0.005, 0.005], 
        df_info=df_info,
        std=0
    )
    print(dataset.n_features)
    train_loader = lib.prepare_fast_dataloader(
        dataset, 
        split='train', 
        batch_size=batch_size,
        y_type='long'
    )
    val_loader = lib.prepare_fast_dataloader(
        dataset, 
        split='val', 
        batch_size=batch_size,
        y_type='long'
    )
    test_loader = lib.prepare_fast_dataloader(
        dataset, 
        split='test', 
        batch_size=batch_size,
        y_type='long'
    )

    eval_interval = 5
    log_interval = 10

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = (dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0)
    if model_params['is_y_cond'] == 'concat':
        num_numerical_features -= 1

    classifier = Classifier(
        d_in = num_numerical_features,
        d_out=int(max(df[cluster_col].values) + 1),
        dim_t=dim_t,
        hidden_sizes=d_layers
    ).to(device)

    classifier_optimizer = optim.AdamW(classifier.parameters(), lr=lr)

    empty_diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=None,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    empty_diffusion.to(device)

    schedule_sampler = create_named_schedule_sampler(
        'uniform', empty_diffusion
    )

    classifier.train()
    resume_step = 0
    for step in range(classifier_steps):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * batch_size,
        )
        numerical_forward_backward_log(
            classifier, 
            classifier_optimizer, 
            train_loader, 
            dataset, 
            schedule_sampler, 
            empty_diffusion, 
            prefix="train"
        )

        classifier_optimizer.step()
        if not step % eval_interval:
            with torch.no_grad():
                classifier.eval()
                numerical_forward_backward_log(
                    classifier, 
                    classifier_optimizer, 
                    val_loader, 
                    dataset, 
                    schedule_sampler, 
                    empty_diffusion, 
                    prefix="val"
                )
                classifier.train()

        if not step % log_interval:
            logger.dumpkvs()

    # # test classifier
    classifier.eval()

    correct = 0
    for step in range(3000):
        test_x, test_y = next(test_loader)
        test_y = test_y.long().to(device)
        if model_params['is_y_cond'] == 'concat':
            test_x = test_x[:, 1:].to(device)
        else:
            test_x = test_x.to(device)
        with torch.no_grad():
            pred = classifier(test_x, timesteps=torch.zeros(test_x.shape[0]).to(device))
            correct += (pred.argmax(dim=1) == test_y).sum().item()

    acc = correct / (3000 * batch_size)
    print(acc)

    return classifier


def conditional_sampling_by_group_size(
        df, 
        df_info, 
        dataset,
        label_encoders, 
        classifier, 
        diffusion, 
        group_labels, 
        sample_batch_size,
        group_lengths_prob_dicts,
        is_y_cond,
        classifier_scale
    ):
    def cond_fn(x, t, y=None, remove_first_col=False):
        
        assert y is not None
        with torch.enable_grad():
            if remove_first_col:
                x_in = x[:, 1:].detach().requires_grad_(True).float()
            else:
                x_in = x.detach().requires_grad_(True).float()
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    sampled_group_sizes = []
    ys = []
    for group_label in group_labels:
        if not group_label in group_lengths_prob_dicts:
            sampled_group_sizes.append(0)
            continue
        sampled_group_size = sample_from_dict(group_lengths_prob_dicts[group_label])
        sampled_group_sizes.append(sampled_group_size)
        ys.extend([group_label] * sampled_group_size)


    all_rows = []
    all_clusters = []
    curr_index = 0
    while curr_index < len(ys):
        end_index = min(curr_index + sample_batch_size, len(ys))
        curr_ys = torch.tensor(
            np.array(ys[curr_index:end_index]).reshape(-1, 1),
            requires_grad=False
        )
        curr_model_kwargs = {}
        curr_model_kwargs["y"] = curr_ys
        curr_sample, _ = diffusion.conditional_sample(
            ys=curr_ys,
            model_kwargs=curr_model_kwargs,
            cond_fn=cond_fn
        )
        all_rows.extend([sample.cpu().numpy() for sample in [curr_sample]])
        all_clusters.extend([curr_ys.cpu().numpy() for curr_ys in [curr_ys]])
        curr_index += sample_batch_size

    arr = np.concatenate(all_rows, axis=0)
    cluster_arr = np.concatenate(all_clusters, axis=0)
    
    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0

    X_gen, y_gen = arr, cluster_arr
    num_numerical_features_sample = num_numerical_features + int(dataset.is_regression and not is_y_cond)


    X_num_real = df[df_info['num_cols']].to_numpy().astype(float)
    X_cat_real = df[df_info['cat_cols']].to_numpy().astype(str)
    y_real = np.round(df[df_info['y_col']].to_numpy().astype(float)).astype(int).reshape(-1, 1)

    X_num_ = X_gen

    if num_numerical_features != 0:
        X_num_ = dataset.num_transform.inverse_transform(X_gen[:, :num_numerical_features_sample])
        actual_num_numerical_features = num_numerical_features - len(label_encoders)
        X_num = X_num_[:, :actual_num_numerical_features]
        if len(label_encoders) > 0:
            X_cat = X_num_[:, actual_num_numerical_features:]
            X_cat = np.round(X_cat).astype(int)
            decoded_x_cat = []
            for col in range(X_cat.shape[1]):
                decoded_x_cat.append(label_encoders[col].inverse_transform(X_cat[:, col]))
            X_cat = np.column_stack(decoded_x_cat)

        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if is_y_cond == 'concat':
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = lib.round_columns(X_num_real, X_num, disc_cols)

    y_gen = y_gen.reshape(-1, 1)

    if X_cat_real is not None and X_cat_real.shape[1] > 0:
        total_real = np.concatenate((X_num_real, X_cat_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, X_cat, np.round(y_gen).astype(int)), axis=1)

    else:
        total_real = np.concatenate((X_num_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, np.round(y_gen).astype(int)), axis=1)

    df_total = pd.DataFrame(total_real)
    df_gen = pd.DataFrame(gen_real)
    columns = [str(x) for x in list(df_total.columns)]

    df_total.columns = columns
    df_gen.columns = columns

    for col in df_total.columns:
        if int(col) < X_num_real.shape[1]:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)
        elif X_cat_real is not None and int(col) < X_num_real.shape[1] + X_cat_real.shape[1]:
            df_total[col] = df_total[col].astype(str)
            df_gen[col] = df_gen[col].astype(str)
        else:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)

    return df_total, df_gen, sampled_group_sizes


def conditional_sampling(
        df, 
        df_info, 
        dataset,
        label_encoders, 
        classifier, 
        diffusion, 
        labels, 
        sample_batch_size, 
        num_samples,
        is_y_cond,
        classifier_scale=1.0,
        device='cuda',
    ):
    
    def cond_fn(x, t, y=None, remove_first_col=False):
        assert y is not None
        with torch.enable_grad():
            if remove_first_col:
                x_in = x[:, 1:].detach().requires_grad_(True).float()
            else:
                x_in = x.detach().requires_grad_(True).float()
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    all_rows = []
    all_clusters = []
    _, empirical_class_dist = torch.unique(
        torch.from_numpy(labels), 
        return_counts=True
    )

    while len(all_rows) * sample_batch_size < num_samples:
        classes = torch.randint(
            low=0, high=len(empirical_class_dist), size=(sample_batch_size,), device=device
        )
        model_kwargs = {}
        model_kwargs["y"] = classes
        sample, _ = diffusion.sample(
            num_samples=sample_batch_size,
            y_dist=empirical_class_dist.float(),
            model_kwargs=model_kwargs,
            cond_fn=cond_fn
        )
        all_rows.extend([sample.cpu().numpy() for sample in [sample]])
        all_clusters.extend([classes.cpu().numpy() for classes in [classes]])
        print(f"created {len(all_rows) * sample_batch_size} samples")

    arr = np.concatenate(all_rows, axis=0)
    arr = arr[:num_samples]
    cluster_arr = np.concatenate(all_clusters, axis=0)
    cluster_arr = cluster_arr[:num_samples]

    # test how the condition goes
    classifier.eval()
    correct = 0
    for i in range(len(arr)):
        curr_sample = arr[i]
        curr_label = cluster_arr[i].reshape(-1)
        curr_sample = torch.from_numpy(curr_sample).float().to(device)
        curr_label = torch.from_numpy(curr_label).long().to(device)
        with torch.no_grad():
            pred = classifier(curr_sample, timesteps=torch.zeros(curr_sample.shape[0]).to(device))
            pred = pred.argmax()
            correct += (pred.item() == curr_label[0].item())
    acc = correct / len(arr)
    print('classifier quality:', acc)
    print()

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0

    X_gen, y_gen = arr, cluster_arr
    num_numerical_features_sample = num_numerical_features + int(dataset.is_regression and not is_y_cond)


    X_num_real = df[df_info['num_cols']].to_numpy().astype(float)
    X_cat_real = df[df_info['cat_cols']].to_numpy().astype(str)
    y_real = np.round(df[df_info['y_col']].to_numpy()).astype(int).reshape(-1, 1)

    X_num_ = X_gen

    if num_numerical_features != 0:
        X_num_ = dataset.num_transform.inverse_transform(X_gen[:, :num_numerical_features_sample])
        actual_num_numerical_features = num_numerical_features - len(label_encoders)
        X_num = X_num_[:, :actual_num_numerical_features]
        if len(label_encoders) > 0:
            X_cat = X_num_[:, actual_num_numerical_features:]
            X_cat = np.round(X_cat).astype(int)
            decoded_x_cat = []
            for col in range(X_cat.shape[1]):
                decoded_x_cat.append(label_encoders[col].inverse_transform(X_cat[:, col]))
            X_cat = np.column_stack(decoded_x_cat)
        else:
            X_cat = np.empty((X_num.shape[0], 0))

        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if is_y_cond == 'concat':
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = lib.round_columns(X_num_real, X_num, disc_cols)

    y_gen = y_gen.reshape(-1, 1)

    if X_cat_real is not None and X_cat_real.shape[1] > 0:
        total_real = np.concatenate((X_num_real, X_cat_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, X_cat, np.round(y_gen).astype(int)), axis=1)
    else:
        total_real = np.concatenate((X_num_real, y_real), axis=1)
        gen_real = np.concatenate((X_num, np.round(y_gen).astype(int)), axis=1)

    
    df_total = pd.DataFrame(total_real)
    df_gen = pd.DataFrame(gen_real)
    columns = [str(x) for x in list(df_total.columns)]

    df_total.columns = columns
    df_gen.columns = columns

    for col in df_total.columns:
        if int(col) < X_num_real.shape[1]:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)
        elif X_cat_real is not None and int(col) < X_num_real.shape[1] + X_cat_real.shape[1]:
            df_total[col] = df_total[col].astype(str)
            df_gen[col] = df_gen[col].astype(str)
        else:
            df_total[col] = df_total[col].astype(float)
            df_gen[col] = df_gen[col].astype(float)

    return df_total, df_gen


def sample_from_dict(probabilities):
    # Generate a random number between 0 and 1
    random_number = random.random()
    
    # Initialize cumulative sum and the selected key
    cumulative_sum = 0
    selected_key = None
    
    # Iterate through the dictionary
    for key, probability in probabilities.items():
        cumulative_sum += probability
        if cumulative_sum >= random_number:
            selected_key = key
            break
    
    return selected_key

# a function that converts a dict of frequencies to a dict of probabilities
def freq_to_prob(freq_dict):
    prob_dict = {}
    for key in freq_dict:
        prob_dict[key] = freq_dict[key] / sum(list(freq_dict.values()))
    return prob_dict

def convert_to_unique_indices(indices):
    occurrence = set()
    max_index = len(indices)  # Assuming the range is the length of the list
    replacement_candidates = set(range(max_index)) - set(indices)
    
    for i, num in enumerate(tqdm(indices)):
        if num in occurrence:
            # Find the smallest number not in the list
            replacement = min(replacement_candidates)
            indices[i] = replacement
            replacement_candidates.remove(replacement)
        else:
            occurrence.add(num)
    
    return indices

def match_tables(A, B, n_clusters=25, unique_matching=True, batch_size=100):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    # Dimension of vectors
    d = B.shape[1]

    if unique_matching:
        quantiser = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantiser, d, n_clusters, faiss.METRIC_L2)
    else:
        res = faiss.StandardGpuResources()
        quantiser = faiss.IndexFlatL2(d)
        index_cpu = faiss.IndexIVFFlat(quantiser, d, n_clusters, faiss.METRIC_L2)
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    
    index.train(B)
    index.add(B)

    # Initialize lists to store the results
    all_indices = []
    all_distances = []

    if unique_matching:
        batch_size = 1
        n_batches = (A.shape[0] + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, A.shape[0])
            D, I = index.search(A[start:end], k=1)
            index.remove_ids(I.flatten())
            all_distances.append(D)
            all_indices.append(I)

        # Concatenate the results from all batches
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        distances = all_distances.flatten().tolist()
        indices = all_indices.flatten().tolist()
    else:
        n_batches = (A.shape[0] + batch_size - 1) // batch_size

        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, A.shape[0])
            D, I = index.search(A[start:end], k=1)
            all_distances.append(D)
            all_indices.append(I)

        # Concatenate the results from all batches
        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        distances = all_distances.flatten().tolist()
        indices = all_indices.flatten().tolist()
        indices = convert_to_unique_indices(indices)
        assert len(indices) == len(set(indices))

    return indices, distances

def match_rows(A, B):
    original_indices_A = np.arange(A.shape[0])
    original_indices_B = np.arange(B.shape[0])

    matched_indices_A = []
    matched_indices_B = []

    while A.shape[0] > 0:
        # Find nearest neighbors for the current A in B
        nearest_neighbors_indices, _ = match_tables(A, B, n_clusters=25, unique_matching=False, batch_size=100)

        # Calculate match counts for each row in B
        match_counts = np.bincount(nearest_neighbors_indices, minlength=B.shape[0])

        # Filter out rows in A and B that are uniquely matched
        unique_matches = match_counts[nearest_neighbors_indices] == 1

        # Update the matched indices lists
        matched_indices_A.extend(original_indices_A[unique_matches])
        matched_indices_B.extend(nearest_neighbors_indices[unique_matches])

        # Identify rows in A and B that need to be reconsidered
        reconsider_A = ~unique_matches
        reconsider_B_indices = np.unique(nearest_neighbors_indices[~unique_matches])

        # Update A, B, and their original indices for the next iteration
        A = A[reconsider_A]
        original_indices_A = original_indices_A[reconsider_A]

        B = B[reconsider_B_indices]
        original_indices_B = original_indices_B[reconsider_B_indices]

    return matched_indices_A, matched_indices_B


def get_df_without_id(df):
    id_cols = [col for col in df.columns if '_id' in col]
    return df.drop(columns=id_cols)


def handle_multi_parent(
        child, 
        parents, 
        synthetic_tables, 
        n_clusters, 
        unique_matching=True,
        batch_size=100,
        no_matching=False
    ):
    synthetic_child_dfs = [(synthetic_tables[(parent, child)]['df'].copy(), parent) for parent in parents]
    anchor_index = np.argmin([len(df) for df, _ in synthetic_child_dfs])
    anchor = synthetic_child_dfs[anchor_index]
    synthetic_child_dfs.pop(anchor_index)
    for df, parent in synthetic_child_dfs:
        df_without_ids = get_df_without_id(df)
        anchor_df_without_ids = get_df_without_id(anchor[0])
        df_val = df_without_ids.values.astype(float)
        anchor_val = anchor_df_without_ids.values.astype(float)
        if len(df_val.shape) == 1:
            df_val = df_val.reshape(-1, 1)
            anchor_val = anchor_val.reshape(-1, 1)
        
        indices, _ = match_tables(
            anchor_val,
            df_val,
            n_clusters=n_clusters,
            unique_matching=unique_matching,
            batch_size=batch_size
        )
        if no_matching:
            # randomly shuffle the array
            indices = np.random.permutation(indices)

        df = df.iloc[indices]
        anchor[0][f'{parent}_id'] = df[f'{parent}_id'].values
    return anchor[0]
