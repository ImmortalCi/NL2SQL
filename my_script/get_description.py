"""
此脚本用于获取数据库中的表格描述信息，将其保存为json文件
"""
import csv
import os
import json


def get_column_from_csv(csv_path, column_index):  # 读取csv文件的某一列，并保存元组（column_name，column_description）列表
    column_values = []
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as csv_file:
        reader = csv.reader(csv_file)
        for i,row in enumerate(reader):
            if i >0 and len(row) > column_index:
                column_values.append((row[0],row[column_index]))
    return column_values


def get_all_databases(root_path):
    res = {}
    res['database_name'] = [database for database in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, database))]
    res['database_path'] = [os.path.join(root_path, database) for database in res['database_name']]
    return res

def get_all_tables(database_path):
    # import pdb;pdb.set_trace()
    database_description_path = os.path.join(database_path, 'database_description')
    res = {}
    res['table_name'] = [table for table in os.listdir(database_description_path) if os.path.isfile(os.path.join(database_description_path, table)) and '.DS_Store' not in table]
    res['table_path'] = [os.path.join(database_description_path, table) for table in res['table_name']]
    return res

def get_all_descriptions(databases_path, save_path):
    all_descriptions = {}
    databases = get_all_databases(databases_path)
    # import pdb;pdb.set_trace()
    for i in range(len(databases['database_name'])):
        database_name = databases['database_name'][i]
        database_path = databases['database_path'][i]
        all_descriptions[database_name] = {}
        tables = get_all_tables(database_path)
        for j in range(len(tables['table_name'])):
            table_name = tables['table_name'][j]
            # table_name去掉.csv
            if table_name.endswith('.csv'):
                table_name = table_name[:-4]
            table_path = tables['table_path'][j]
            all_descriptions[database_name][table_name] = {}
            column_values = get_column_from_csv(table_path, 2)
            for column_name, column_description in column_values:
                all_descriptions[database_name][table_name][column_name] = column_description
    with open(save_path, 'w') as f:
        json.dump(all_descriptions, f, indent=4)
    # import pdb;pdb.set_trace()


get_all_descriptions('/opt/data/private/wtc_beifen/bird/data/dev_databases', '/opt/data/private/wtc_beifen/bird/data/description/eval_all_descriptions.json')
get_all_descriptions('/opt/data/private/wtc_beifen/bird/data/train/train_databases', '/opt/data/private/wtc_beifen/bird/data/description/train_all_databases.json')
# Save the descriptions to a JSON file
# output_path = '/opt/data/private/wtc_beifen/bird/data/description/all_descriptions.json'
# with open(output_path, 'w') as f:
#     json.dump(all_descriptions, f, indent=4)
