import argparse
import math
import os
import random
import time

import json
import pandas as pd
import pickle
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

    unmatched_save_dir = os.path.join(save_dir, 'single')
    final_save_dir = os.path.join(save_dir, 'final')

    os.makedirs(unmatched_save_dir, exist_ok=True)
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

    synthesized = {}
    for parent, child in relation_order:
        if os.path.exists(os.path.join(unmatched_save_dir, f'{parent}_{child}.csv')):
            print(f'{parent}_{child} already synthesized')
            synthesized[f'{parent}_{child}'] = pd.read_csv(os.path.join(unmatched_save_dir, f'{parent}_{child}.csv'))
            continue
        if parent is not None:
            parent_df = tables[parent]['df']
            child_df = tables[child]['df']
            parent_domain = tables[parent]['domain']
            child_domain = tables[child]['domain']
            
            foreign_key = f'{parent}_id'
            # left join
            merged_df = pd.merge(
                child_df,
                parent_df,
                on=foreign_key,
                how='left',
            )

            id_cols = [col for col in merged_df.columns if '_id' in col]
            merged_df_no_id = merged_df.drop(columns=id_cols)

            merged_metadata = get_merged_metadata(
                merged_df_no_id,
                parent_domain,
                child_domain,
            )
            merged_domain = tables[child]['domain'].copy()
            merged_domain.update(tables[parent]['domain'])

            print(f'Training and synthesizing {parent}_{child}...')
            if args.synthesizer == 'ctgan':
                synthesizer = CTGANSynthesizer(
                    merged_metadata, 
                    batch_size=args.batch_size, 
                    verbose=True
                )
                synthesizer.fit(merged_df_no_id)
                synthesizer.save(os.path.join(unmatched_save_dir, f'{parent}_{child}.pkl'))
                synthetic_data = synthesizer.sample(num_rows=len(tables[child]['df']))
                synthesized[f'{parent}_{child}'] = synthetic_data
                synthetic_data.to_csv(os.path.join(unmatched_save_dir, f'{parent}_{child}.csv'), index=False)
            elif args.synthesizer == 'smote':
                synthetic_data, _, _, _ = get_smote_res(
                    merged_df_no_id,
                    merged_domain
                )
                synthesized[f'{parent}_{child}'] = synthetic_data
                synthetic_data.to_csv(os.path.join(unmatched_save_dir, f'{parent}_{child}.csv'), index=False)
            elif args.synthesizer == 'tabDDPM':
                n_iter = math.ceil(args.diffusion_steps * min(len(merged_df_no_id), args.batch_size) / len(merged_df_no_id))
                print(f'n_epochs: {n_iter}')
                plugin = Plugins().get(
                    "ddpm",
                    n_iter=n_iter,
                    is_classification=False,
                    batch_size=args.batch_size,
                    num_timesteps=2000
                )
                plugin.fit(merged_df_no_id)
                synthetic_data = plugin.generate(len(merged_df_no_id)).data
                synthesized[f'{parent}_{child}'] = synthetic_data
                synthetic_data.to_csv(os.path.join(unmatched_save_dir, f'{parent}_{child}.csv'), index=False)
            elif args.synthesizer == 'clavaDDPM':
                result = child_training(
                    merged_df_no_id,
                    merged_domain,
                    None,
                    child,
                    configs
                )
                sample_scale=1 if not 'debug' in configs else configs['debug']['sample_scale']
                _, child_generated = sample_from_diffusion(
                    df=merged_df_no_id, 
                    df_info=result['df_info'], 
                    diffusion=result['diffusion'],
                    dataset=result['dataset'],
                    label_encoders=result['label_encoders'],
                    sample_size=int(sample_scale * len(merged_df_no_id)),
                    model_params=result['model_params'],
                    T_dict=result['T_dict'],
                )
                generated_df = pd.DataFrame(
                    child_generated.to_numpy(),
                    columns=result['df_info']['num_cols'] + result['df_info']['cat_cols'] + [result['df_info']['y_col']]
                )
                
                generated_df = generated_df[merged_df_no_id.columns]
                generated_df = generated_df.drop(columns=result['df_info']['y_col'])
                synthesized[f'{parent}_{child}'] = generated_df
                generated_df.to_csv(os.path.join(unmatched_save_dir, f'{parent}_{child}.csv'), index=False)


    final_synthesized = {}
    for parent, child in relation_order:
        parent_synthesized = False
        child_synthesized = False
        if os.path.exists(
            os.path.join(final_save_dir, f'{parent}_synthetic.csv')
        ):
            print(f'{parent} already synthesized')
            parent_synthesized = True
            final_synthesized[parent] = pd.read_csv(os.path.join(final_save_dir, f'{parent}_synthetic.csv'))
            
        if os.path.exists(
            os.path.join(final_save_dir, f'{child}_synthetic.csv')
        ):
            print(f'{child} already synthesized')
            child_synthesized = True
            final_synthesized[child] = pd.read_csv(os.path.join(final_save_dir, f'{child}_synthetic.csv'))

        if parent_synthesized and child_synthesized:
            continue

        if parent is not None:
            synthetic_data_sorted = synthesized[f'{parent}_{child}']
            # sort by all the columns corresponding to parents
            synthetic_data_sorted = synthetic_data_sorted.sort_values(
                by=[col for col in tables[parent]['original_cols'] if col in synthetic_data_sorted.columns]
            ).reset_index(drop=True)
            synthetic_data_sorted[f'{child}_id'] = range(1, len(synthetic_data_sorted) + 1)
            group_size_dict = get_group_sizes(tables[child]['df'], f'{parent}_id')
            group_prob_dict = get_group_size_prob(group_size_dict)

            parent_ids = []
            curr_id = 0
            while len(parent_ids) < len(synthetic_data_sorted):
                group_size = sample_from_dict(group_prob_dict)
                parent_ids += [curr_id] * group_size
                curr_id += 1
            parent_ids = random.sample(parent_ids, len(synthetic_data_sorted))

            synthetic_data_sorted[f'{parent}_id'] = parent_ids

            parent_cols = [col for col in tables[parent]['original_cols'] if col in synthetic_data_sorted.columns]
            child_cols = [col for col in tables[child]['original_cols'] if col in synthetic_data_sorted.columns]
            synthetic_parent_df = synthetic_data_sorted[parent_cols]
            synthetic_child_df = synthetic_data_sorted[child_cols]
            synthetic_parent_df = synthetic_parent_df.drop_duplicates(subset=[f'{parent}_id']).reset_index(drop=True)

            if not parent in final_synthesized:
                synthetic_parent_df.to_csv(os.path.join(final_save_dir, f'{parent}_synthetic.csv'), index=False)
                final_synthesized[parent] = synthetic_parent_df

            if not child in final_synthesized:
                final_synthesized[child] = synthetic_child_df
            else:
                # Merge the old synthetic child with the new synthetic child
                # Common columns don't need to be merged
                # Newly introduced columns need to be added
                old_synthetic_child = final_synthesized[child]
                new_synthetic_child = synthetic_child_df
                common_cols = list(set(old_synthetic_child.columns) & set(new_synthetic_child.columns))
                new_cols = list(set(new_synthetic_child.columns) - set(old_synthetic_child.columns))
                
                # the new cols are directly added to the old synthetic child
                assert len(old_synthetic_child) == len(new_synthetic_child)
                old_synthetic_child = pd.concat([old_synthetic_child, new_synthetic_child[new_cols]], axis=1)
                final_synthesized[child] = old_synthetic_child
            # check if the final_synthesized child has the same number of columns as real child
            if len(final_synthesized[child].columns) == len(tables[child]['original_cols']):
                # reorder child columns to be the same as the original child
                final_synthesized[child] = final_synthesized[child][tables[child]['original_cols']]
                final_synthesized[child].to_csv(os.path.join(final_save_dir, f'{child}_synthetic.csv'), index=False)
                print(f'{child} synthesized')

    end_time = time.time()
    time_spent = end_time - start_time
    
    # save to tables
    for table_name, val in final_synthesized.items():
        tables[table_name]['synthetic_df'] = val

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
