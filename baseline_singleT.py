import argparse
import math
import os
import random
import json
import pandas as pd
import pickle
import time
from pipeline_utils import get_info_from_domain, pipeline_process_data
from baseline_utils import *

from synthcity.plugins import Plugins
from pipeline_modules import child_training
from pipeline_utils import sample_from_diffusion, sample_from_dict
from gen_multi_report import gen_multi_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--working_dir', type=str)
    parser.add_argument('--synthesizer', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--diffusion_steps', type=int, default=100000)
    parser.add_argument('--classifier_steps', type=int, default=10000)
    parser.add_argument('--config_path', type=str, default='')

    args = parser.parse_args()

    if args.synthesizer == 'clavaDDPM':
        configs = json.load(open(args.config_path, 'r'))
        args.data_dir = configs['general']['data_dir']

    save_dir = os.path.join(args.working_dir, args.synthesizer)
    os.makedirs(save_dir, exist_ok=True)

    single_save_dir = os.path.join(save_dir, 'single')
    final_save_dir = os.path.join(save_dir, 'final')

    os.makedirs(single_save_dir, exist_ok=True)
    os.makedirs(final_save_dir, exist_ok=True)

    dataset_meta = json.load(open(os.path.join(args.data_dir, 'dataset_meta.json'), 'r'))

    relation_order = dataset_meta['relation_order']
    relation_order_reversed = relation_order[::-1]

    tables = {}

    for table, meta in dataset_meta['tables'].items():
        tables[table] = {
            'df': pd.read_csv(os.path.join(args.data_dir, f'{table}.csv')),
            'domain': json.load(open(os.path.join(args.data_dir, f'{table}_domain.json'))),
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

    start_time = time.time()

    for table_name, val in tables.items():
        print(f'Training and synthesizing {table_name}...')
        table_save_path = os.path.join(
            single_save_dir, 
            f'{table_name}.csv'
        )
        if args.synthesizer == 'ctgan':
            if not os.path.exists(table_save_path):
                synthetic_data = train_ctgan(
                    val['df'],
                    val['domain'],
                    args.batch_size
                )
                synthetic_data.to_csv(table_save_path, index=False)
            else:
                print(f'{table_name} already synthesized, loading...')
                synthetic_data = pd.read_csv(table_save_path)

            tables[table_name]['synthetic_df'] = synthetic_data
        elif args.synthesizer == 'smote':
            if not os.path.exists(table_save_path):
                synthetic_data, _, _, _ = get_smote_res(
                    val['df'],
                    val['domain']
                )
                synthetic_data.to_csv(table_save_path, index=False)
            else:
                print(f'{table_name} already synthesized, loading...')
                synthetic_data = pd.read_csv(table_save_path)

            tables[table_name]['synthetic_df'] = synthetic_data
        elif args.synthesizer == 'tabDDPM':
            if not os.path.exists(table_save_path):
                print(f'Synthesizing {table_name}...')
                id_cols = [col for col in val['df'].columns if '_id' in col]
                df_without_ids = val['df'].drop(columns=id_cols)
                n_iter = math.ceil(args.diffusion_steps * min(len(df_without_ids), args.batch_size) / len(df_without_ids))
                print(f'n_epochs: {n_iter}')
                plugin = Plugins().get(
                    "ddpm",
                    n_iter=n_iter,
                    is_classification=False,
                    batch_size=args.batch_size,
                    num_timesteps=2000
                )
                domain_dict = val['domain']
                cat_cols = []
                for col in df_without_ids.columns:
                    if domain_dict[col]['type'] == 'discrete':
                        cat_cols.append(col)
                plugin.fit(df_without_ids, discrete_columns=cat_cols)
                synthetic_data = plugin.generate(len(df_without_ids)).data
                synthetic_data.to_csv(table_save_path, index=False)
            else:
                print(f'{table_name} already synthesized, loading...')
                synthetic_data = pd.read_csv(table_save_path)
            tables[table_name]['synthetic_df'] = synthetic_data
        elif args.synthesizer == 'clavaDDPM':
            if not os.path.exists(table_save_path):
                print(f'Synthesizing {table_name}...')
                df = tables[table_name]['df']
                id_cols = [col for col in df.columns if '_id' in col]
                df_without_id = df.drop(columns=id_cols)
                result = child_training(
                    df_without_id,
                    tables[table_name]['domain'],
                    None,
                    table_name,
                    configs
                )
                sample_scale=1 if not 'debug' in configs else configs['debug']['sample_scale']
                _, child_generated = sample_from_diffusion(
                    df=df_without_id, 
                    df_info=result['df_info'], 
                    diffusion=result['diffusion'],
                    dataset=result['dataset'],
                    label_encoders=result['label_encoders'],
                    sample_size=int(sample_scale * len(df_without_id)),
                    model_params=result['model_params'],
                    T_dict=result['T_dict'],
                )
                generated_df = pd.DataFrame(
                    child_generated.to_numpy(),
                    columns=result['df_info']['num_cols'] + result['df_info']['cat_cols'] + [result['df_info']['y_col']]
                )
                
                generated_df = generated_df[df_without_id.columns]
                generated_df = generated_df.drop(columns=result['df_info']['y_col'])
                generated_df.to_csv(table_save_path, index=False)
            else:
                print(f'{table_name} already synthesized, loading...')
                generated_df = pd.read_csv(table_save_path)

            tables[table_name]['synthetic_df'] = generated_df

            print('synthesized')
    
    # Compute group sizes
    group_probs = {}
    for table_name, val in tables.items():
        for parent in val['parents']:
            group_size_dict = get_group_sizes(
                val['df'],
                f'{parent}_id'
            )
            group_probs[(parent, table_name)] = get_group_size_prob(group_size_dict)

    # Generate synthetic foreign keys according to group probs
    for table_name, val in tables.items():
        for parent in val['parents']:
            synthetic_foreign_keys = []
            curr_key = 0
            while len(synthetic_foreign_keys) < len(tables[table_name]['synthetic_df']):
                sampled_group_size = sample_from_dict(group_probs[(parent, table_name)])
                synthetic_foreign_keys += [curr_key] * sampled_group_size
                curr_key += 1

            synthetic_foreign_keys = random.sample(synthetic_foreign_keys, len(tables[table_name]['synthetic_df']))

            foreign_key = f'{parent}_id'
            tables[table_name]['synthetic_df'][foreign_key] = synthetic_foreign_keys

    # Generate primary keys for each table
    for table_name, val in tables.items():
        tables[table_name]['synthetic_df'][f'{table_name}_id'] = range(len(tables[table_name]['synthetic_df']))
        # reassign columns to original order
        tables[table_name]['synthetic_df'] = tables[table_name]['synthetic_df'][tables[table_name]['original_cols']]

    # Save final synthetic data
    for table_name, val in tables.items():
        finale_save_path = os.path.join(
            final_save_dir,
            f'{table_name}_synthetic.csv'
        )
        tables[table_name]['synthetic_df'].to_csv(
            finale_save_path,
            index=False
        )
        print(f'Synthetic data for {table_name} saved to {final_save_dir}')

    end_time = time.time()
    time_spent = end_time - start_time

    real_data = {}
    synthetic_data = {}
    for table_name, df in tables.items():
        real_data[table_name] = df['original_df']
        synthetic_data[table_name] = tables[table_name]['synthetic_df']

    multi_meta = get_multi_metadata(tables, relation_order)
    
    gen_multi_report(
        args.data_dir,
        save_dir,
        'baseline'
    )

    with open(os.path.join(save_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(multi_meta, f)

    print('Time spent:', time_spent)
