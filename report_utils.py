import pandas as pd
import os
from sdv.metadata import MultiTableMetadata

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

def clava_load_synthetic_data(path, tables):
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
