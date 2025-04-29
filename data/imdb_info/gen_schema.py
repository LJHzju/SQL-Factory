import os
import sys
import json
import numpy as np
import pandas as pd

schema_save_path = ''
raw_schema_path = ''    # such as 'imdb_schema.json'
raw_data_path = ''      # such as 'imdb_data.csv'


with open(raw_schema_path) as f:
    lines = f.read()

total_tables_info = {}
tables = lines.split('\n\n')
for table in tables:
    table_name = table.split(' TABLE ')[1].split(' (\n')[0]
    attrs = table.split('\n')[1:-1]
    table_info = {}
    for attr in attrs:
        attr_name, attr_type = attr.split()[0], attr.split()[1].strip(',')
        table_info[attr_name] = {"type": attr_type}
    total_tables_info[table_name] = table_info

for table, info in total_tables_info.items():
    print(f'table {table} start')
    table_path = os.path.join(raw_data_path, f'{table}.csv')
    table_columns = [column for column in info.keys()]
    df = pd.read_csv(table_path, names=table_columns, quotechar='"', escapechar='\\')
    for column in table_columns:
        col_type = info[column]['type']
        print(f'column {column}, type: {col_type}')
        col_value = df[column].dropna()
        if not len(col_value):
            continue
        if col_type == 'integer':
            col_min, col_max = np.min(col_value), np.max(col_value)
            col_min, col_max = int(col_min), int(col_max)
            total_tables_info[table][column]['max'] = col_max
            total_tables_info[table][column]['min'] = col_min
        if col_type == 'character' or col_type == 'text':
            unique_value = list(set(col_value))
            if len(unique_value) <= 5:
                total_tables_info[table][column]['samples'] = unique_value
            else:
                total_tables_info[table][column]['samples'] = random.sample(col_value, 3)
    print(f'table {table} finished\n\n')


print(total_tables_info)
with open(schema_save_path, 'w') as f:
    json.dump(total_tables_info, f)
