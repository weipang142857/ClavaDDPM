import os
import argparse
import pandas as pd
import numpy as np
import pickle

from pipeline_utils import load_multi_table
from report_utils import get_multi_metadata, baseline_load_synthetic_data, clava_load_synthetic_data, sdv_load_synthetic_data
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality as single_eval_quality
from sdv.evaluation.multi_table import evaluate_quality as multi_eval_quality
from collections import defaultdict

def get_avg_long_range_scores(res):
    avg_scores = {}
    for hop, scores in res.items():
        avg_scores[hop] = np.mean(list(scores.values()))
    return avg_scores

def find_paths_with_length_greater_than_one(dataset):
    # Build adjacency list while skipping edges that start with None
    graph = defaultdict(list)
    for parent, child in dataset:
        if parent is not None:
            graph[parent].append(child)

    # This will store all paths with at least two edges
    results = []

    # Helper function to perform DFS
    def dfs(node, path):
        if len(path) > 2:  # path contains at least two edges
            results.append(path[:])  # copy of the current path
        for neighbor in graph[node]:
            path.append(neighbor)
            dfs(neighbor, path)
            path.pop()

    # Start DFS from each node that has children, making sure we're not modifying the graph
    starting_nodes = list(graph.keys())
    for node in starting_nodes:
        dfs(node, [node])

    return results

def recursive_merge(dataframes, keys):
    # Start with the top table, which is the last in the list if we are going top to bottom
    result_df = dataframes[-1]
    for i in range(len(dataframes) - 2, -1, -1):  # Iterate backwards, excluding the last already used
        result_df = pd.merge(
            left=result_df,
            right=dataframes[i],
            on=keys[i],
            how='left'
        )
    return result_df

def get_joint_table(long_path, tables):
    path_tables = [tables[table]['df'] for table in long_path]
    path_keys = [f'{table}_id' for table in long_path]
    long_path_joined = recursive_merge(path_tables, path_keys)
    domains_merged = tables[long_path[0]]['domain'].copy()
    for table in long_path[1:]:
        domains_merged.update(tables[table]['domain'])

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(long_path_joined)
    for col, val in domains_merged.items():
        if val['type'] == 'discrete':
            metadata.update_column(
                column_name=col,
                sdtype='categorical'
            )
        elif val['type'] == 'continuous':
            metadata.update_column(
                column_name=col,
                sdtype='numerical'
            )
        else:
            raise ValueError(f'Unknown domain type: {val["type"]}')

    return long_path_joined, metadata

def evaluate_long_path(real_joined, syn_joined, metadata, top_table_cols, bottom_table_cols, top_table, bottom_table):
    quality = single_eval_quality(real_joined, syn_joined, metadata)

    column_pair_quality = quality.get_details('Column Pair Trends')

    top_table_cols = set(top_table_cols)
    bottom_table_cols = set(bottom_table_cols)

    res = {}

    for _, row in column_pair_quality.iterrows():
        col_1 = row['Column 1']
        col_2 = row['Column 2']

        if col_1 in top_table_cols and col_2 in bottom_table_cols or \
            col_1 in bottom_table_cols and col_2 in top_table_cols:
            res[(top_table, bottom_table, col_1, col_2)] = row['Score']
    return res

def get_long_range(real_tables, syn_tables, dataset_meta):
    long_paths = find_paths_with_length_greater_than_one(dataset_meta['relation_order'])
    res = {}
    for long_path in long_paths:
        hop = len(long_path) - 1
        real_joined, metadata = get_joint_table(long_path, real_tables)
        syn_joined_1, _ = get_joint_table(long_path, syn_tables)
        top_table = long_path[0]
        bottom_table = long_path[-1]
        top_table_cols = list(real_tables[top_table]['domain'].keys())
        bottom_table_cols = list(real_tables[bottom_table]['domain'].keys())

        scores = evaluate_long_path(
            real_joined, 
            syn_joined_1, 
            metadata, 
            top_table_cols,
            bottom_table_cols,
            top_table,
            bottom_table
        )
        if not hop in res:
            res[hop] = {}
        res[hop].update(scores)

    return res


def gen_multi_report(real_data_path, syn_data_path, syn_data_type):
    print(f'generating multi-table report for {syn_data_path}')

    tables, relation_order, dataset_meta = load_multi_table(real_data_path)
    multi_metadata = get_multi_metadata(tables, relation_order)
    
    if syn_data_type == 'baseline':
        syn = baseline_load_synthetic_data(syn_data_path, tables)
    elif syn_data_type == 'clava':
        syn = clava_load_synthetic_data(syn_data_path, tables)
    elif syn_data_type == 'sdv':
        syn = sdv_load_synthetic_data(syn_data_path, tables)

    for table_name, _ in syn.items():
        domain = tables[table_name]['domain']
        for col in domain.keys():
            if domain[col]['type'] == 'discrete':
                syn[table_name]['df'][col] = syn[table_name]['df'][col].astype(int)

    hop_relation = get_long_range(tables, syn, dataset_meta)

    real_tables = {}
    syn_tables ={}
    for table, val in tables.items():
        real_tables[table] = val['df']
        syn_tables[table] = syn[table]['df']

    multi_report = multi_eval_quality(
        real_tables,
        syn_tables,
        multi_metadata
    )

    one_hop = multi_report.get_details('Intertable Trends').dropna(subset=['Score'])
    one_hop_dict = {}
    for _, row in one_hop.iterrows():
        one_hop_dict[(row['Parent Table'], row['Child Table'], row['Column 1'], row['Column 2'])] = row['Score']

    hop_relation[1] = one_hop_dict

    zero_hop = multi_report.get_details('Column Pair Trends').dropna(subset=['Score'])
    zero_hop_dict = {}
    for _, row in zero_hop.iterrows():
        zero_hop_dict[(row['Table'], row['Table'], row['Column 1'], row['Column 2'])] = row['Score']
    hop_relation[0] = zero_hop_dict

    avg_scores = get_avg_long_range_scores(hop_relation)
    
    # avg scores for all hops:
    all_avg_score = 0
    num_scores = 0
    for hop, score in hop_relation.items():
        all_avg_score += np.sum(list(score.values()))
        num_scores += len(score)

    all_avg_score /= num_scores

    print('Long Range Scores:', avg_scores)
    print('All avg scores: ', all_avg_score)
    
    result = {}
    result['hop_relation'] = hop_relation
    result['avg_scores'] = avg_scores
    result['all_avg_score'] = all_avg_score
    result['report'] = multi_report
    result['multi_meta'] = multi_metadata

    pickle.dump(
        result, 
        open(os.path.join(syn_data_path, 'multi_quality.pkl'), 'wb')
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_data_path', type=str)
    parser.add_argument('--syn_data_path', type=str)
    parser.add_argument('--syn_data_type', type=str, choices=['baseline', 'clava'])

    args = parser.parse_args()

    gen_multi_report(
        args.real_data_path,
        args.syn_data_path,
        args.syn_data_type
    )
