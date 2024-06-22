from pipeline_utils import *

import numpy as np
import os
import pandas as pd
import json
import argparse
import pickle
import time
from tab_ddpm.utils import *
from pipeline_modules import *
from sdv.metadata import MultiTableMetadata
from gen_multi_report import gen_multi_report

def clava_clustering(tables, relation_order, save_dir, configs):
    relation_order_reversed = relation_order[::-1]
    all_group_lengths_prob_dicts = {}

    # Clustering
    if os.path.exists(os.path.join(save_dir, 'cluster_ckpt.pkl')):
        print('Clustering checkpoint found, loading...')
        cluster_ckpt = pickle.load(open(os.path.join(save_dir, 'cluster_ckpt.pkl'), 'rb'))
        tables = cluster_ckpt['tables']
        all_group_lengths_prob_dicts = cluster_ckpt['all_group_lengths_prob_dicts']
    else:
        for parent, child in relation_order_reversed:
            if parent is not None:
                print(f'Clustering {parent} -> {child}')
                if isinstance(configs['clustering']['num_clusters'], dict):
                    num_clusters = configs['clustering']['num_clusters'][child]
                else:
                    num_clusters = configs['clustering']['num_clusters']
                parent_df_with_cluster, child_df_with_cluster, group_lengths_prob_dicts = pair_clustering_keep_id(
                    tables[child]['df'], 
                    tables[child]['domain'], 
                    tables[parent]['df'],
                    tables[parent]['domain'],
                    f'{child}_id',
                    f'{parent}_id',
                    num_clusters,
                    configs['clustering']['parent_scale'],
                    1, # not used for now
                    parent,
                    child,
                    clustering_method=configs['clustering']['clustering_method'],
                )
                tables[parent]['df'] = parent_df_with_cluster
                tables[child]['df'] = child_df_with_cluster
                all_group_lengths_prob_dicts[(parent, child)] = group_lengths_prob_dicts

        cluster_ckpt = {
            'tables': tables,
            'all_group_lengths_prob_dicts': all_group_lengths_prob_dicts
        }
        pickle.dump(cluster_ckpt, open(os.path.join(save_dir, 'cluster_ckpt.pkl'), 'wb'))

    return tables, all_group_lengths_prob_dicts

def clava_training(tables, relation_order, save_dir, configs):
    models = {}
    for parent, child in relation_order:
        if os.path.exists(os.path.join(save_dir, f'models/{parent}_{child}_ckpt.pkl')):
            print(f'{parent} -> {child} checkpoint found, loading...')
            models[(parent, child)] = pickle.load(open(os.path.join(save_dir, f'models/{parent}_{child}_ckpt.pkl'), 'rb'))
            continue
        print(f'Training {parent} -> {child}')
        df_with_cluster = tables[child]['df']
        id_cols = [col for col in df_with_cluster.columns if '_id' in col]
        df_without_id = df_with_cluster.drop(columns=id_cols)
        result = child_training(
            df_without_id,
            tables[child]['domain'],
            parent,
            child,
            configs
        )
        models[(parent, child)] = result
        pickle.dump(result, open(os.path.join(save_dir, f'models/{parent}_{child}_ckpt.pkl'), 'wb'))

    for parent, child in relation_order:
        if parent is None:
            tables[child]['df']['placeholder'] = list(range(len(tables[child]['df'])))
    
    return tables, models

def clava_synthesizing(tables, relation_order, save_dir, all_group_lengths_prob_dicts, models, configs, sample_scale=1):
    all_exist = True
    for key, val in tables.items():
        table_dir = os.path.join(
            configs['general']['workspace_dir'], 
            key, 
            f'{configs["general"]["sample_prefix"]}_final'
        )
        if not os.path.exists(os.path.join(table_dir, f'{key}_synthetic.csv')):
            all_exist = False
            break
    if all_exist:
        print('Synthetic tables found, loading...')
        synthetic_tables = {}
        for key, val in tables.items():
            table_dir = os.path.join(
                configs['general']['workspace_dir'], 
                key, 
                f'{configs["general"]["sample_prefix"]}_final'
            )
            synthetic_tables[key] = pd.read_csv(os.path.join(table_dir, f'{key}_synthetic.csv'))
        return synthetic_tables, 0, 0
    
    synthesizing_start_time = time.time()
    synthetic_tables = {}
    if os.path.exists(os.path.join(save_dir, 'before_matching/synthetic_tables.pkl')):
        print('Synthetic tables found, loading...')
        synthetic_tables = pickle.load(open(os.path.join(save_dir, 'before_matching/synthetic_tables.pkl'), 'rb'))

    # Synthesize
    for parent, child in relation_order:
        if (parent, child) in synthetic_tables:
            print(f'{parent} -> {child} synthetic table found, skip...')
            print('---------------------------------------------------')
            continue
        print(f'Generating {parent} -> {child}')
        result = models[(parent, child)]
        df_with_cluster = tables[child]['df']
        df_without_id = get_df_without_id(df_with_cluster)
        if parent is None:
            _, child_generated = sample_from_diffusion(
                df=df_without_id, 
                df_info=result['df_info'], 
                diffusion=result['diffusion'],
                dataset=result['dataset'],
                label_encoders=result['label_encoders'],
                sample_size=int(sample_scale * len(df_without_id)),
                model_params=result['model_params'],
                T_dict=result['T_dict'],
                sample_batch_size=configs['sampling']['batch_size']
            )
            child_keys = list(range(len(child_generated)))
            generated_final_arr = np.concatenate(
                [
                    np.array(child_keys).reshape(-1, 1),
                    child_generated.to_numpy()
                ],
                axis=1
            )
            generated_final_df = pd.DataFrame(
                generated_final_arr,
                columns=[f'{child}_id'] + result['df_info']['num_cols'] + result['df_info']['cat_cols'] + [result['df_info']['y_col']]
            )
            generated_final_df = generated_final_df[tables[child]['df'].columns]
            synthetic_tables[(parent, child)] = {
                'df': generated_final_df,
                'keys': child_keys
            }
        else:
            for key, val in synthetic_tables.items():
                if key[1] == parent:
                    parent_synthetic_df = val['df']
                    parent_keys = val['keys']
                    parent_result = models[key]
                    break

            child_result = models[(parent, child)]
            parent_label_index = parent_result['column_orders'].index(
                child_result['df_info']['y_col']
            )

            parent_synthetic_df_without_id = get_df_without_id(parent_synthetic_df)

            _, child_generated, child_sampled_group_sizes = conditional_sampling_by_group_size(
                df=df_without_id, 
                df_info=child_result['df_info'],
                dataset=child_result['dataset'],
                label_encoders=child_result['label_encoders'],
                classifier=child_result['classifier'],
                diffusion=child_result['diffusion'],
                group_labels=parent_synthetic_df_without_id.values[:, parent_label_index].astype(float).astype(int).tolist(),
                group_lengths_prob_dicts=all_group_lengths_prob_dicts[(parent, child)],
                sample_batch_size=configs['sampling']['batch_size'],
                is_y_cond='none',
                classifier_scale=configs['sampling']['classifier_scale'],
            )

            child_foreign_keys = np.repeat(parent_keys, child_sampled_group_sizes, axis=0).reshape((-1, 1))
            child_foreign_keys_arr = np.array(child_foreign_keys).reshape(-1, 1)
            child_primary_keys_arr = np.arange(
                len(child_generated)
            ).reshape(-1, 1)

            child_generated_final_arr = np.concatenate(
                [
                    child_primary_keys_arr,
                    child_generated.to_numpy(),
                    child_foreign_keys_arr
                ],
                axis=1
            )

            child_final_columns = [f'{child}_id'] + result['df_info']['num_cols'] + \
                result['df_info']['cat_cols'] + [result['df_info']['y_col']] + [f'{parent}_id']

            child_final_df = pd.DataFrame(
                child_generated_final_arr,
                columns=child_final_columns
            )
            original_columns = []
            for col in tables[child]['df'].columns:
                if col in child_final_df.columns:
                    original_columns.append(col)
            child_final_df = child_final_df[original_columns]
            synthetic_tables[(parent, child)] = {
                'df': child_final_df,
                'keys': child_primary_keys_arr.flatten().tolist()
            }
        pickle.dump(synthetic_tables, open(os.path.join(save_dir, 'before_matching/synthetic_tables.pkl'), 'wb'))
    
    synthesizing_end_time = time.time()
    synthesizing_time_spent = synthesizing_end_time - synthesizing_start_time

    matching_start_time = time.time()

    # Matching
    final_tables = {}
    for parent, child in relation_order:
        if child not in final_tables:
            if len(tables[child]['parents']) > 1:
                final_tables[child] = handle_multi_parent(
                    child, 
                    tables[child]['parents'], 
                    synthetic_tables, 
                    configs['matching']['num_matching_clusters'],
                    unique_matching=configs['matching']['unique_matching'],
                    batch_size=configs['matching']['matching_batch_size'],
                    no_matching=configs['matching']['no_matching']
                )
            else:
                final_tables[child] = synthetic_tables[(parent, child)]['df']

    matching_end_time = time.time()
    matching_time_spent = matching_end_time - matching_start_time

    cleaned_tables = {}
    for key, val in final_tables.items():
        cleaned_tables[key] = val[tables[key]['original_cols']]

    for key, val in cleaned_tables.items():
        table_dir = os.path.join(
            configs['general']['workspace_dir'], 
            key, 
            f'{configs["general"]["sample_prefix"]}_final'
        )
        os.makedirs(table_dir, exist_ok=True)
        val.to_csv(os.path.join(table_dir, f'{key}_synthetic.csv'), index=False)

    return cleaned_tables, synthesizing_time_spent, matching_time_spent

def clava_eval(tables, save_dir, configs, relation_order, synthetic_tables=None):
    metadata = MultiTableMetadata()
    for table_name, val in tables.items():
        df = val['original_df']
        metadata.detect_table_from_dataframe(
            table_name,
            df
        )
        id_cols = [col for col in df.columns if '_id' in col]
        for id_col in id_cols:
            metadata.update_column(
                table_name=table_name,
                column_name=id_col,
                sdtype='id'
            )
        domain = tables[table_name]['domain']
        for col, dom in domain.items():
            if col in df.columns:
                if dom['type'] == 'discrete':
                    metadata.update_column(
                        table_name=table_name,
                        column_name=col,
                        sdtype='categorical',
                    )
                elif dom['type'] == 'continuous':
                    metadata.update_column(
                        table_name=table_name,
                        column_name=col,
                        sdtype='numerical',
                    )
                else:
                    raise ValueError(f'Unknown domain type: {dom["type"]}')
        metadata.set_primary_key(
            table_name=table_name,
            column_name=f'{table_name}_id'
        )

    for parent, child in relation_order:
        if parent is not None:
            metadata.add_relationship(
                parent_table_name=parent,
                child_table_name=child,
                parent_primary_key=f'{parent}_id',
                child_foreign_key=f'{parent}_id'
            )

    if synthetic_tables is None:
        synthetic_tables = {}
        for table, meta in dataset_meta['tables'].items():
            table_dir = os.path.join(
                configs['general']['workspace_dir'], 
                table, 
                f'{configs["general"]["sample_prefix"]}_final'
            )
            synthetic_tables[table] = pd.read_csv(os.path.join(table_dir, f'{table}_synthetic.csv'))
    
    report = gen_multi_report(
        configs['general']['data_dir'],
        configs['general']['workspace_dir'],
        'clava'
    )
    
    pickle.dump(metadata, open(os.path.join(save_dir, 'metadata.pkl'), 'wb'))
    return report

def load_configs(config_path):
    configs = json.load(open(config_path, 'r'))

    save_dir = os.path.join(configs['general']['workspace_dir'], configs['general']['exp_name'])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'before_matching'), exist_ok=True)

    with open(os.path.join(save_dir, 'args'), 'w') as file:
        json.dump(configs, file, indent=4)

    return configs, save_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.json')

    args = parser.parse_args()
    clustering_start_time = time.time()
    configs, save_dir = load_configs(args.config_path)

    tables, relation_order, dataset_meta = load_multi_table(configs['general']['data_dir'])

    # Clustering
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, save_dir, configs)

    clustering_end_time = time.time()

    clustering_time_spent = clustering_end_time - clustering_start_time

    training_start_time = time.time()

    # Training

    tables, models = clava_training(tables, relation_order, save_dir, configs)

    training_end_time = time.time()
    training_time_spent = training_end_time - training_start_time

    # Synthesizing

    cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
        tables, 
        relation_order, 
        save_dir, 
        all_group_lengths_prob_dicts, 
        models,
        configs,
        sample_scale=1 if not 'debug' in configs else configs['debug']['sample_scale']
    )

    # Eval
    report = clava_eval(tables, save_dir, configs, relation_order, cleaned_tables)
    
    print('Time spent: ')
    print('Clustering: ', clustering_time_spent)
    print('Training: ', training_time_spent)
    print('Synthesizing: ', synthesizing_time_spent)
    print('Matching: ', matching_time_spent)
    print('Total: ', clustering_time_spent + training_time_spent + synthesizing_time_spent + matching_time_spent)
