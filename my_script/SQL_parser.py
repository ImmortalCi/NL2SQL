"""
解析SQL语句，提取列和表信息，并保存到json文件中（全部为小写）
"""

import sqlglot
from sqlglot import exp
import json
from tqdm import tqdm

# 从json文件读取SQL语句
def read_sqls_from_json(json_path):
    all_sqls = []
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# 解析SQL语句，并输出列和表信息
def sql_parser(all_data):
    res = []
    for sample in tqdm(all_data):
        sql = sample['SQL']
        try:
            parsed = sqlglot.parse(sql,dialect='sqlite')
            columns = []
            tables = []
            for ast in parsed:
                if ast is not None:
                    for node in ast.find_all(exp.Column):
                        columns.append(node.name.lower())
                    for node in ast.find_all(exp.Table):
                        tables.append(node.name.lower())
                    # 对columns和tables去重
                    columns = list(set(columns))
                    tables = list(set(tables))
            sample['columns'] = columns
            sample['tables'] = tables
            res.append(sample)
        except Exception as e:
            print(e)
            print(sql)
            continue
    return res

if __name__ == '__main__':
    json_path = '/opt/data/private/wtc_beifen/bird/data/dev.json'
    all_data = read_sqls_from_json(json_path)
    res = sql_parser(all_data)
    with open('/opt/data/private/wtc_beifen/bird/data/dev_with_columns_tables.json', 'w') as f:
        json.dump(res, f, indent=4)

# # 输入SQL语句
# sql = "SELECT `Date received` FROM callcenterlogs WHERE ser_time = ( SELECT MAX(ser_time) FROM callcenterlogs )"

# # 解析SQL语句
# parsed = sqlglot.parse(sql)

# for ast in parsed:
#     if ast is not None:
#     # 遍历AST以提取列和表信息
#         columns = []
#         tables = []

#         for node in ast.find_all(exp.Column):
#             columns.append(node.name)
#         for node in ast.find_all(exp.Table):
#             tables.append(node.name)

# # 输出列和表信息
# print("Columns:", columns)
# print("Tables:", tables)
