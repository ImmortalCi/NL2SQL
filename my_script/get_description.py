"""
此脚本用于获取数据库中的表格描述信息，将其保存为json文件。保持数据库描述信息的格式为{database_name: {table_name: {column_name: column_description}}}，大小是否保留可根据需求更改。
"""
import csv
import os
import json


def get_column_from_csv(csv_path, column_index):  # 读取csv文件的某一列，并保存元组（column_name，column_description）列表
    column_values = []
    with open(csv_path, 'r', encoding='UTF-8-sig', errors='ignore') as csv_file:
        data = []
        reader = csv.reader(csv_file)
        column_names = next(reader)  # 读取第一行作为列名
        # 去除列名中的空格、换行符等
        column_names = [column_name.strip() for column_name in column_names]
        for row in reader:
            # 将每行数据与列名打包成字典
            data_dict = dict(zip(column_names, row))
            data.append(data_dict)
        for row in data:
            try:
                if not row['original_column_name']=="":
                    column_values.append((row['original_column_name'], row['column_description']))
            except KeyError:
                if len(row) == 0:
                    continue
                else:
                    import pdb;pdb.set_trace()
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

def get_all_descriptions(databases_path, save_path, lower=False):
    all_descriptions = {}
    databases = get_all_databases(databases_path)
    # import pdb;pdb.set_trace()
    if not lower:
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
    else:
        for i in range(len(databases['database_name'])):
            database_name = databases['database_name'][i].lower()
            database_path = databases['database_path'][i]
            all_descriptions[database_name] = {}
            tables = get_all_tables(database_path)
            for j in range(len(tables['table_name'])):
                table_name = tables['table_name'][j].lower()
                # table_name去掉.csv
                if table_name.endswith('.csv'):
                    table_name = table_name[:-4]
                table_path = tables['table_path'][j]
                all_descriptions[database_name][table_name] = {}
                column_values = get_column_from_csv(table_path, 2)
                for column_name, column_description in column_values:
                    column_name = column_name.lower()
                    all_descriptions[database_name][table_name][column_name] = column_description
    return all_descriptions

    # with open(save_path, 'w') as f:
    #     json.dump(all_descriptions, f, indent=4)
    # import pdb;pdb.set_trace()


get_all_descriptions('/opt/data/private/wtc_beifen/bird/data/dev_databases', '/opt/data/private/wtc_beifen/bird/data/description/eval_all_descriptions_lower.json', lower=True)
get_all_descriptions('/opt/data/private/wtc_beifen/bird/data/train/train_databases', '/opt/data/private/wtc_beifen/bird/data/description/train_all_descriptions_lower.json', lower=True)
# Save the descriptions to a JSON file
# output_path = '/opt/data/private/wtc_beifen/bird/data/description/all_descriptions.json'
# with open(output_path, 'w') as f:
#     json.dump(all_descriptions, f, indent=4)
