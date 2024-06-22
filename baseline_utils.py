import pandas as pd
import os
import random
import numpy as np

from sdv.metadata import SingleTableMetadata
from sdv.metadata import MultiTableMetadata
from sdv.single_table import CTGANSynthesizer
from smote.sample_smote import sample_smote_baseline

def get_group_sizes(child_df, foreign_key):
    group_sizes = {}
    for group, group_df in child_df.groupby(foreign_key):
        group_sizes[group] = len(group_df)
    return group_sizes

def get_group_size_prob(group_size_dict):
    freqs = {}
    for _, freq in group_size_dict.items():
        if freq not in freqs:
            freqs[freq] = 0
        freqs[freq] += 1

    probs = {}
    for freq, count in freqs.items():
        probs[freq] = count / len(group_size_dict)

    return probs

def get_multi_metadata(tables, relation_order):
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

    return metadata

def get_merged_metadata(merged_df, parent_domain_dict, child_domain_dict):
    metadata = SingleTableMetadata()
    df_without_ids = merged_df.drop(columns=[col for col in merged_df.columns if '_id' in col])
    metadata.detect_from_dataframe(df_without_ids)
    for col in df_without_ids.columns:
        domain_dict = None
        if col in parent_domain_dict:
            domain_dict = parent_domain_dict
        elif col in child_domain_dict:
            domain_dict = child_domain_dict
        
        if domain_dict is not None:
            if domain_dict[col]['type'] == 'discrete':
                if domain_dict[col]['size'] < 1000:
                    metadata.update_column(
                        column_name=col,
                        sdtype='categorical',
                    )
                else:
                    metadata.update_column(
                        column_name=col,
                        sdtype='numerical',
                    )
            else:
                metadata.update_column(
                    column_name=col,
                    sdtype='numerical',
                )

    metadata.remove_primary_key()

    return metadata

def get_metadata(df, domain_dict=None):
    metadata = SingleTableMetadata()
    df_without_ids = df.drop(columns=[col for col in df.columns if '_id' in col])
    metadata.detect_from_dataframe(df_without_ids)
    if domain_dict is not None:
        for col in df_without_ids.columns:
            if domain_dict[col]['type'] == 'discrete':
                if domain_dict[col]['size'] < 1000:
                    metadata.update_column(
                        column_name=col,
                        sdtype='categorical',
                    )
                else:
                    metadata.update_column(
                        column_name=col,
                        sdtype='numerical',
                    )
            else:
                metadata.update_column(
                    column_name=col,
                    sdtype='numerical',
                )

    metadata.remove_primary_key()

    return metadata, df_without_ids

def train_ctgan(df, domain_dict, batch_size):
    metadata, df_without_ids = get_metadata(df, domain_dict)
    synthesizer = CTGANSynthesizer(metadata, batch_size=batch_size, verbose=True)
    synthesizer.fit(df_without_ids)

    synthetic_data = synthesizer.sample(num_rows=len(df_without_ids))

    return synthetic_data

def baseline_load_synthetic_data(path, tables):
    syn = {}
    for table, val in tables.items():
        syn[table] = {}
        syn[table]['df'] = pd.read_csv(os.path.join(
            path, 
            'final',
            f'{table}_synthetic.csv'
        ))
        syn[table]['domain'] = val['domain']
    return syn

def lava_load_synthetic_data(path, tables):
    syn = {}
    for table, val in tables.items():
        syn[table] = {}
        syn[table]['df'] = pd.read_csv(os.path.join(
            path, 
            table,
            '_final',
            f'{table}_synthetic.csv'
        ))
        syn[table]['domain'] = val['domain']
    return syn

def sdv_load_synthetic_data(path, tables):
    syn = {}
    for table, val in tables.items():
        syn[table] = {}
        syn[table]['df'] = pd.read_csv(os.path.join(
            path,
            f'{table}.csv'
        ))
        syn[table]['domain'] = val['domain']
    return syn


def get_smote_res(df, domain_dict):
    id_cols = [col for col in df.columns if '_id' in col]
    df_no_id = df.drop(columns=id_cols)
    num_cols = []
    cat_cols = []
    for col, val in domain_dict.items():
        if val['type'] == 'discrete':
            cat_cols.append(col)
        else:
            num_cols.append(col)

    all_cols = num_cols + cat_cols
    y_col = random.choice(all_cols)
    if y_col in num_cols:
        num_cols.remove(y_col)
        is_regression = True
    else:
        cat_cols.remove(y_col)
        is_regression = False

    X_num = {}
    X_num['train'] = df[num_cols].values
    X_cat = {}
    X_cat['train'] = df[cat_cols].values
    y = {}
    y['train'] = df[y_col].values

    syn_x_num, syn_x_cat, res_y = sample_smote_baseline(
        'smote_res',
        X_num,
        X_cat,
        y,
        eval_type = "synthetic",
        k_neighbours = 5,
        frac_samples = 1.0,
        frac_lam_del = 0.0,
        change_val = False,
        save = False,
        seed = 0,
        is_regression=is_regression
    )

    res = np.concatenate((syn_x_num, syn_x_cat, res_y.reshape((-1, 1))), axis=1)
    res_df = pd.DataFrame(res, columns=num_cols + cat_cols + [y_col])
    res_df = res_df[df_no_id.columns]

    return res_df, cat_cols, num_cols, y_col
