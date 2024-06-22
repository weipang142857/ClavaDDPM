from pipeline_utils import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from collections import defaultdict
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def aggregate_and_sample(cluster_probabilities, child_group_lengths):
    group_cluster_labels = []
    curr_index = 0
    agree_rates = []
    
    for group_length in child_group_lengths:
        # Aggregate the probability distributions by taking the mean
        group_probability_distribution = np.mean(cluster_probabilities[curr_index: curr_index + group_length], axis=0)
        
        # Sample the label from the aggregated distribution
        group_cluster_label = np.random.choice(range(len(group_probability_distribution)), p=group_probability_distribution)
        group_cluster_labels.append(group_cluster_label)
        
        # Compute the max probability as the agree rate
        max_probability = np.max(group_probability_distribution)
        agree_rates.append(max_probability)
        
        # Update the curr_index for the next iteration
        curr_index += group_length

    return group_cluster_labels, agree_rates

# Meant to be hard-coded, do not change
def get_table_info(df, domain_dict, y_col):
    cat_cols = []
    num_cols = []
    for col in df.columns:
        if col in domain_dict and col != y_col:
            if domain_dict[col]['type'] == 'discrete':
                cat_cols.append(col)
            else:
                num_cols.append(col)

    df_info = {}
    df_info['cat_cols'] = cat_cols
    df_info['num_cols'] = num_cols
    df_info['y_col'] = y_col
    df_info['n_classes'] = 0
    df_info['task_type'] = 'multiclass'

    return df_info

def get_model_params(rtdl_params=None):
    return {
        'num_classes': 0,
        'is_y_cond': 'none',
        'rtdl_params': {
            'd_layers': [
                512,
                1024,
                1024,
                1024,
                1024,
                512
            ],
            'dropout': 0.0
        } if rtdl_params is None else rtdl_params
    }

def get_T_dict():
    return {
        'seed': 0,
        'normalization': "quantile",
        'num_nan_policy': None,
        'cat_nan_policy': None,
        'cat_min_frequency': None,
        'cat_encoding': None,
        'y_policy': "default"
    }

def child_training(
        child_df_with_cluster,
        child_domain_dict,
        parent_name,
        child_name,
        configs
    ):
    if parent_name is None:
        y_col = 'placeholder'
        child_df_with_cluster['placeholder'] = list(range(len(child_df_with_cluster)))
    else:
        y_col = f'{parent_name}_{child_name}_cluster'
    child_info = get_table_info(child_df_with_cluster, child_domain_dict, y_col)
    child_model_params = get_model_params({
        'd_layers': configs['diffusion']['d_layers'],
        'dropout': configs['diffusion']['dropout']
    })
    child_T_dict = get_T_dict()
    
    child_result = train_model(
        child_df_with_cluster,
        child_info,
        child_model_params,
        child_T_dict,
        configs['diffusion']['iterations'],
        configs['diffusion']['batch_size'],
        configs['diffusion']['model_type'],
        configs['diffusion']['gaussian_loss_type'],
        configs['diffusion']['num_timesteps'],
        configs['diffusion']['scheduler'],
        configs['diffusion']['lr'],
        configs['diffusion']['weight_decay'],
    )

    if parent_name is None:
        child_result['classifier'] = None
    elif configs['classifier']['iterations'] > 0:
        child_classifier = train_classifier(
            child_df_with_cluster,
            child_info,
            child_model_params,
            child_T_dict,
            configs['classifier']['iterations'],
            configs['classifier']['batch_size'],
            configs['diffusion']['gaussian_loss_type'],
            configs['diffusion']['num_timesteps'],
            configs['diffusion']['scheduler'],
            cluster_col=y_col,
            d_layers=configs['classifier']['d_layers'],
            dim_t=configs['classifier']['dim_t'],
            lr=configs['classifier']['lr']
        )
        child_result['classifier'] = child_classifier

    child_result['df_info'] = child_info
    child_result['model_params'] = child_model_params
    child_result['T_dict'] = child_T_dict
    return child_result


def pair_clustering_keep_id(
        child_df, 
        child_domain_dict, 
        parent_df,
        parent_domain_dict,
        child_primary_key,
        parent_primary_key,
        num_clusters,
        parent_scale,
        key_scale,
        parent_name,
        child_name,
        clustering_method='kmeans'
    ):
    original_child_cols = list(child_df.columns)
    original_parent_cols = list(parent_df.columns)

    relation_cluster_name = f'{parent_name}_{child_name}_cluster'

    child_data = child_df.to_numpy()
    parent_data = parent_df.to_numpy()

    child_num_cols = []
    child_cat_cols = []

    parent_num_cols = []
    parent_cat_cols = []

    for col_index, col in enumerate(original_child_cols):
        if col in child_domain_dict:
            if child_domain_dict[col]['type'] == 'discrete':
                child_cat_cols.append((col_index, col))
            else:
                child_num_cols.append((col_index, col))

    for col_index, col in enumerate(original_parent_cols):
        if col in parent_domain_dict:
            if parent_domain_dict[col]['type'] == 'discrete':
                parent_cat_cols.append((col_index, col))
            else:
                parent_num_cols.append((col_index, col))
    
    parent_primary_key_index = original_parent_cols.index(parent_primary_key)
    foreing_key_index = original_child_cols.index(parent_primary_key)

    # sort child data by foreign key
    sorted_child_data = child_data[np.argsort(child_data[:, foreing_key_index])]
    child_group_data_dict = get_group_data_dict(sorted_child_data, [foreing_key_index,])

    # sort parent data by primary key
    sorted_parent_data = parent_data[np.argsort(parent_data[:, parent_primary_key_index])]

    group_lengths = []
    unique_group_ids = sorted_parent_data[:, parent_primary_key_index]
    for group_id in unique_group_ids:
        group_id = tuple([group_id])
        if not group_id in child_group_data_dict:
            group_lengths.append(0)
        else:
            group_lengths.append(len(child_group_data_dict[group_id]))

    group_lengths = np.array(group_lengths, dtype=int)

    sorted_parent_data_repeated = np.repeat(sorted_parent_data, group_lengths, axis=0)
    assert((sorted_parent_data_repeated[:, parent_primary_key_index] == sorted_child_data[:, foreing_key_index]).all())

    child_group_data = get_group_data(sorted_child_data, [foreing_key_index,])

    sorted_child_num_data = sorted_child_data[:, [col_index for col_index, col in child_num_cols]]
    sorted_child_cat_data = sorted_child_data[:, [col_index for col_index, col in child_cat_cols]]
    sorted_parent_num_data = sorted_parent_data_repeated[:, [col_index for col_index, col in parent_num_cols]]
    sorted_parent_cat_data = sorted_parent_data_repeated[:, [col_index for col_index, col in parent_cat_cols]]

    joint_num_matrix = np.concatenate([sorted_child_num_data, sorted_parent_num_data], axis=1)
    joint_cat_matrix = np.concatenate([sorted_child_cat_data, sorted_parent_cat_data], axis=1)

    if joint_cat_matrix.shape[1] > 0:

        joint_cat_matrix_p_index = sorted_child_cat_data.shape[1]
        joint_num_matrix_p_index = sorted_child_num_data.shape[1]

        cat_converted = []
        label_encoders = []
        for i in range(joint_cat_matrix.shape[1]):
            # A threshold of 1000 unique values is used to prevent the one-hot encoding of large categorical columns
            if len(np.unique(joint_cat_matrix[:, i])) > 1000:
                continue
            label_encoder = LabelEncoder()
            cat_converted.append(label_encoder.fit_transform(joint_cat_matrix[:, i]).astype(float))
            label_encoders.append(label_encoder)

        cat_converted = np.vstack(cat_converted).T

        # Initialize an empty array to store the encoded values
        cat_one_hot = np.empty((cat_converted.shape[0], 0))

        # Loop through each column in the data and encode it
        for col in range(cat_converted.shape[1]):
            encoder = OneHotEncoder(sparse_output=False)
            column = cat_converted[:, col].reshape(-1, 1)
            encoded_column = encoder.fit_transform(column)
            cat_one_hot = np.concatenate((cat_one_hot, encoded_column), axis=1)

        cat_one_hot[:, joint_cat_matrix_p_index:] = parent_scale * cat_one_hot[:, joint_cat_matrix_p_index:]

    # Perform quantile normalization using QuantileTransformer
    num_quantile = quantile_normalize_sklearn(joint_num_matrix)
    num_min_max = min_max_normalize_sklearn(joint_num_matrix)

    key_quantile = quantile_normalize_sklearn(sorted_parent_data_repeated[:, parent_primary_key_index].reshape(-1, 1))
    key_min_max = min_max_normalize_sklearn(sorted_parent_data_repeated[:, parent_primary_key_index].reshape(-1, 1))

    # key_scaled = key_scaler * key_quantile
    key_scaled = key_scale * key_min_max

    num_quantile[:, joint_num_matrix_p_index:] = parent_scale * num_quantile[:, joint_num_matrix_p_index:]
    num_min_max[:, joint_num_matrix_p_index:] = parent_scale * num_min_max[:, joint_num_matrix_p_index:]

    if joint_cat_matrix.shape[1] > 0:
        cluster_data = np.concatenate((num_min_max, cat_one_hot, key_scaled), axis=1)
    else:
        cluster_data = np.concatenate((num_min_max, key_scaled), axis=1)
    
    child_group_lengths = np.array([len(group) for group in child_group_data], dtype=int)
    num_clusters = min(num_clusters, len(cluster_data))

    print('clustering')
    if clustering_method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto', init='k-means++')
        kmeans.fit(cluster_data)
        cluster_labels = kmeans.labels_
    elif clustering_method == 'both':
        gmm = GaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type='diag',
            init_params='k-means++',
            tol=0.0001
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)
    elif clustering_method == 'variational':
        gmm = BayesianGaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type='diag',
            init_params='k-means++',
            tol=0.0001
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict_proba(cluster_data)
    elif clustering_method == 'gmm':
        gmm = GaussianMixture(
            n_components=num_clusters,
            verbose=1,
            covariance_type='diag',
        )
        gmm.fit(cluster_data)
        cluster_labels = gmm.predict(cluster_data)

    if clustering_method == 'variational':
        group_cluster_labels, agree_rates = aggregate_and_sample(cluster_labels, child_group_lengths)
    else:
        # voting to determine the cluster label for each parent
        group_cluster_labels = []
        curr_index = 0
        agree_rates = []
        for group_length in child_group_lengths:
            # First, determine the most common label in the current group
            most_common_label_count = np.max(np.bincount(cluster_labels[curr_index: curr_index + group_length]))
            group_cluster_label = np.argmax(np.bincount(cluster_labels[curr_index: curr_index + group_length]))
            group_cluster_labels.append(group_cluster_label)
            
            # Compute agree rate using the most common label count
            agree_rate = most_common_label_count / group_length
            agree_rates.append(agree_rate)
            
            # Then, update the curr_index for the next iteration
            curr_index += group_length

    # Compute the average agree rate across all groups
    average_agree_rate = np.mean(agree_rates)
    print('average agree rate: ', average_agree_rate)

    group_assignment = np.repeat(group_cluster_labels, child_group_lengths, axis=0).reshape((-1, 1))

    # obtain the child data with clustering
    sorted_child_data_with_cluster = np.concatenate(
        [
            sorted_child_data,
            group_assignment
        ],
        axis=1
    )

    group_labels_list = group_cluster_labels
    group_lengths_list = child_group_lengths.tolist()

    group_lengths_dict = {}
    for i in range(len(group_labels_list)):
        group_label = group_labels_list[i]
        if not group_label in group_lengths_dict:
            group_lengths_dict[group_label] = defaultdict(int)
        group_lengths_dict[group_label][group_lengths_list[i]] += 1

    group_lengths_prob_dicts = {}
    for group_label, freq_dict in group_lengths_dict.items():
        group_lengths_prob_dicts[group_label] = freq_to_prob(freq_dict)

    # recover the preprocessed data back to dataframe
    child_df_with_cluster = pd.DataFrame(
        sorted_child_data_with_cluster,
        columns=original_child_cols + [relation_cluster_name]
    )

    # recover child df order
    child_df_with_cluster = pd.merge(
        child_df[[child_primary_key]],
        child_df_with_cluster,
        on=child_primary_key,
        how='left',
    )

    parent_id_to_cluster = {}
    for i in range(len(sorted_child_data)):
        parent_id = sorted_child_data[i, foreing_key_index]
        if parent_id in parent_id_to_cluster:
            assert(parent_id_to_cluster[parent_id] == sorted_child_data_with_cluster[i, -1])
            continue
        parent_id_to_cluster[parent_id] = sorted_child_data_with_cluster[i, -1]

    max_cluster_label = max(parent_id_to_cluster.values())

    parent_data_clusters = []
    for i in range(len(parent_data)):
        if parent_data[i, parent_primary_key_index] in parent_id_to_cluster:
            parent_data_clusters.append(parent_id_to_cluster[parent_data[i, parent_primary_key_index]])
        else:
            parent_data_clusters.append(max_cluster_label + 1)

    parent_data_clusters = np.array(parent_data_clusters).reshape(-1, 1)
    parent_data_with_cluster = np.concatenate(
        [
            parent_data,
            parent_data_clusters
        ],
        axis=1
    )
    parent_df_with_cluster = pd.DataFrame(
        parent_data_with_cluster,
        columns=original_parent_cols + [relation_cluster_name]
    )

    new_col_entry = {
        'type': 'discrete',
        'size': len(set(parent_data_clusters.flatten()))
    }

    print('num clusters: ', len(set(parent_data_clusters.flatten())))

    parent_domain_dict[relation_cluster_name] = new_col_entry.copy()
    child_domain_dict[relation_cluster_name] = new_col_entry.copy()

    return parent_df_with_cluster, child_df_with_cluster, group_lengths_prob_dicts

