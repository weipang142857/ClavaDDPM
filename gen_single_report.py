import os
import pickle
import argparse
from pipeline_utils import load_multi_table
from report_utils import baseline_load_synthetic_data, clava_load_synthetic_data

from eval.eval_quality import eval_metrics
from eval.eval_mle import eval_mle
from eval.eval_detection import eval_detection

def get_avg(mles):
    def mean_std(l):
        mean = sum(l) / len(l)
        std = (sum([(x - mean) ** 2 for x in l]) / len(l)) ** 0.5
        return mean, std
    f1_res = []
    r2_res = []
    for key, val in mles.items():
        if 'best_f1_scores' in val:
            res_res = val['best_f1_scores']['XGBClassifier']
            if 'binary_f1' in res_res:
                f1_res.append(res_res['binary_f1'])
            elif 'macro_f1' in res_res:
                f1_res.append(res_res['macro_f1'])
        else:
            res_res = val['best_r2_scores']['XGBRegressor']
            r2_res.append(res_res['r2'])
    
    if len(f1_res) > 0:
        f1_res = mean_std(f1_res)[0]
    else:
        f1_res = None
    
    if len(r2_res) > 0:
        r2_res = mean_std(r2_res)[0]
    else:
        r2_res = None
    return f1_res, r2_res

def get_info(syn_df, domain_dict, target_col):
    num_col_index = []
    cat_col_index = []
    target_col_index = []
    info = {}
    table_cols = list(syn_df.columns)
    for i in range(len(table_cols)):
        col = table_cols[i]
        if col in domain_dict and col != target_col:
            if domain_dict[col]['type'] == 'discrete':
                cat_col_index.append(i)
            else:
                num_col_index.append(i)
        if col == target_col:
            target_col_index.append(i)
            if col in domain_dict:
                if domain_dict[col]['type'] == 'discrete':
                    if domain_dict[col]['size'] == 2:
                        info['task_type'] = 'binclass'
                    else:
                        info['task_type'] = 'multiclass'
                else:
                    info['task_type'] = 'regression'

    info['num_col_idx'] = num_col_index
    info['cat_col_idx'] = cat_col_index
    info['target_col_idx'] = target_col_index
    if not 'task_type' in info:
        info['task_type'] = 'None'

    return info

def compute_alpha_beta(real_df, syn_df, domain_dict, sample_size=200000):
    # drop id cols
    all_columns = list(real_df.columns)
    id_cols = [col for col in all_columns if '_id' in col]
    real_df = real_df.drop(columns=id_cols)
    syn_df = syn_df.drop(columns=id_cols)

    info = get_info(syn_df, domain_dict, '')
    syn_df = syn_df.dropna()

    sample_size = min(sample_size, len(syn_df), len(real_df))

    syn_df = syn_df.sample(sample_size)
    real_df = real_df.sample(sample_size)

    if len(real_df) > len(syn_df):
        real_df = real_df.sample(len(syn_df))
    elif len(real_df) < len(syn_df):
        syn_df = syn_df.sample(len(real_df))

    alpha, beta = eval_metrics(syn_df, real_df, info)
    return alpha, beta

def compute_all_mle(syn_df, test_df, domain_dict):
    mles = {}
    for col, _ in domain_dict.items():
        print('Computing MLE for column:', col)
        mle = compute_mle(syn_df, test_df, domain_dict, col)
        mles[col] = mle
        print(f'MLE for column {col}: {mle}')
    return mles

def compute_mle(syn_df, test_df, domain_dict, target_col):
    # drop id cols
    all_columns = list(syn_df.columns)
    id_cols = [col for col in all_columns if '_id' in col]

    test_df = test_df.drop(columns=id_cols)
    syn_df = syn_df.drop(columns=id_cols)

    info = get_info(syn_df, domain_dict, target_col)
    syn_df = syn_df.dropna()

    mle = eval_mle(syn_df.values, test_df.values, info)
    return mle

def compute_detection(syn_df, real_df, domain_dict):
    # drop id cols
    all_columns = list(syn_df.columns)
    id_cols = [col for col in all_columns if '_id' in col]
    real_df = real_df.drop(columns=id_cols)
    syn_df = syn_df.drop(columns=id_cols)

    detection_score = eval_detection(syn_df, real_df, domain_dict)
    return detection_score

def gen_single_report(
        real_data, 
        syn_data,
        domain_dict,
        table_name,
        save_path,
        alpha_beta_sample_size=200000,
        test_data=None
    ):
    os.makedirs(save_path, exist_ok=True)

    if len(domain_dict) > 1:
        if not os.path.exists(os.path.join(save_path, f'{table_name}_alpha.pkl')) or\
            not os.path.exists(os.path.join(save_path, f'{table_name}_beta.pkl')):
            alpha, beta = compute_alpha_beta(real_data, syn_data, domain_dict, sample_size=alpha_beta_sample_size)
            with open(os.path.join(save_path, f'{table_name}_alpha.pkl'), 'wb') as f:
                pickle.dump(alpha, f)

            with open(os.path.join(save_path, f'{table_name}_beta.pkl'), 'wb') as f:
                pickle.dump(beta, f)

        else:
            alpha = pickle.load(open(os.path.join(save_path, f'{table_name}_alpha.pkl'), 'rb'))
            beta = pickle.load(open(os.path.join(save_path, f'{table_name}_beta.pkl'), 'rb'))

        print(f'alpha: {alpha}, beta: {beta}')
        print()
    

    if len(domain_dict) > 1:
        if test_data is not None and len(domain_dict) > 1 and len(test_data) > 0:
            if not os.path.exists(os.path.join(save_path, f'{table_name}_mles.pkl')):
                mles = compute_all_mle(syn_data, test_data, domain_dict)
                with open(os.path.join(save_path, f'{table_name}_mles.pkl'), 'wb') as f:
                    pickle.dump(mles, f)
            else:
                mles = pickle.load(open(os.path.join(save_path, f'{table_name}_mles.pkl'), 'rb'))
                print(get_avg(mles))
                print()
    
    if not os.path.exists(os.path.join(save_path, f'{table_name}_detection.pkl')):
        detection_score = compute_detection(syn_data, real_data, domain_dict)
        with open(os.path.join(save_path, f'{table_name}_detection.pkl'), 'wb') as f:
            pickle.dump(detection_score, f)
    else:
        detection_score = pickle.load(open(os.path.join(save_path, f'{table_name}_detection.pkl'), 'rb'))

    print(f'Detection Score: {detection_score}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_data_path', type=str)
    parser.add_argument('--syn_data_path', type=str)
    parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--save_path')
    parser.add_argument('--syn_data_type', type=str, default='lava')
    parser.add_argument('--alpha_beta_sample_size', type=int, default=200000)

    args = parser.parse_args()

    tables, relation_order, dataset_meta = load_multi_table(args.real_data_path)

    if args.syn_data_type == 'baseline':
        syn = baseline_load_synthetic_data(args.syn_data_path, tables)
    elif args.syn_data_type == 'clava':
        syn = clava_load_synthetic_data(args.syn_data_path, tables)

    if args.test_data_path is not None:
        test_tables, _, _ = load_multi_table(args.test_data_path)
    
    for table_name in tables.keys():
        print(f'Generating report for {table_name}')
        real_data = tables[table_name]['df']
        syn_data = syn[table_name]['df']
        domain_dict = tables[table_name]['domain']

        if args.test_data_path is not None:
            test_data = test_tables[table_name]['df']
        else:
            test_data = None

        gen_single_report(
            real_data, 
            syn_data,
            domain_dict,
            table_name,
            args.save_path,
            alpha_beta_sample_size=args.alpha_beta_sample_size,
            test_data=test_data
        )
