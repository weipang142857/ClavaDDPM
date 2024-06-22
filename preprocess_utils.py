import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import os

def calculate_days_since_earliest_date(dates):
    date_objects = [datetime.strptime(date, '%y%m%d') for date in dates]
    earliest_date = min(date_objects)
    days_since = [(date - earliest_date).days for date in date_objects]
    return days_since, earliest_date.strftime('%y%m%d')

def reconstruct_dates(days_since, earliest_date_str):
    earliest_date = datetime.strptime(earliest_date_str, '%y%m%d')
    original_dates = [(earliest_date + timedelta(days=days)).strftime('%y%m%d') for days in days_since]
    return original_dates

def birth_number_split(birth_numbers):
    years = [int(bn[:2]) for bn in birth_numbers]
    months = [int(bn[2:4]) for bn in birth_numbers]
    days = [int(bn[4:6]) for bn in birth_numbers]
    genders = []
    for i in range(len(months)):
        if months[i] >= 50:
            months[i] -= 50
            genders.append(1)
        else:
            genders.append(0)
    return years, months, days, genders

def table_label_encode(df, discrete_cols):
    df = df.copy()
    label_encoders = {}
    for col in discrete_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def table_label_decode(df, label_encoders):
    df = df.copy()
    for col, le in label_encoders.items():
        df[col] = le.inverse_transform(df[col])
    return df

def get_domain(df, id_cols, discrete_cols):
    domain = {}
    for col in df.columns:
        if col in discrete_cols:
            domain[col] = {
                'size': len(df[col].unique()),
                'type': 'discrete'
            }
        elif col not in id_cols:
            domain[col] = {
                'size': len(df[col].unique()),
                'type': 'continuous'
            }
    return domain

def encode_and_save(df, discrete_cols, keys, save_dir, table_name):
    df_encoded, df_label_encoders = table_label_encode(df, discrete_cols)
    df_encoded = df_encoded.astype('str')
    df_encoded.to_csv(os.path.join(save_dir, f'{table_name}.csv'), index=False)
    with open(os.path.join(save_dir, f'{table_name}_label_encoders.pkl'), 'wb') as f:
        pickle.dump(df_label_encoders, f)
    df_domain = get_domain(df_encoded, keys, discrete_cols)
    with open(os.path.join(save_dir, f'{table_name}_domain.json'), 'w') as f:
        json.dump(df_domain, f)


def topological_sort(graph):
    # Initialize the indegree map and output
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for child in graph[node]['children']:
            in_degree[child] += 1
    
    # Queue for nodes with no incoming edges
    zero_in_degree = [node for node, degree in in_degree.items() if degree == 0]
    
    # Output list for storing the order
    sorted_order = []

    # Start with root nodes and format them with None as parent
    for node in zero_in_degree:
        sorted_order.append([None, node])

    # Using a queue to maintain nodes to process
    queue = zero_in_degree[:]
    
    while queue:
        current = queue.pop(0)
        for child in graph[current]['children']:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
            # Add each parent-child relationship as we process them
            sorted_order.append([current, child])

    return sorted_order
