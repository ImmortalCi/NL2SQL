import sqlite3
import json

from networkx import all_triplets
from sqlglot import column

# def execute_sql(sql, db_path):
#     # Connect to the database
#     conn = sqlite3.connect(db_path)
#     # Create a cursor object
#     cursor = conn.cursor()
#     cursor.execute(sql)
#     results = cursor.fetchall()
    


# sql = "SELECT max_temperature_f, date FROM weather WHERE max_temperature_f = ( SELECT MAX(max_temperature_f) FROM weather WHERE max_temperature_f IS NOT NULL AND max_temperature_f IS NOT '' )"

# db_path = '/opt/data/private/wtc_beifen/bird/data/train/train_databases/bike_share_1/bike_share_1.sqlite'
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()
# import pdb; pdb.set_trace()

# 读取all_descriptions.json，分数据库进行存储（因为主要是数据库内涉及到schema-linking）
def merge_description(description_path):
    """
    合并描述信息

    参数:
    description_path (str): 描述信息文件的路径

    返回:
    dict: 合并后的描述信息字典
    """
    with open(description_path, 'r') as f:
        res = {}
        all_descriptions = json.load(f)
        for db_name in all_descriptions.keys():
            #判断db_name是否是res中的key
            if db_name not in res:
                res[db_name] = []
            db = all_descriptions[db_name]
            # 遍历db的所有key
            for table_name in db.keys():
                columns = db[table_name]
                for column_name in columns.keys():
                    column_description = columns[column_name]
                    prompt = db_name + '|' + table_name + '|' + column_name + '|' + column_description
                    res[db_name].append(prompt)
    return res

# from rank_bm25 import BM25Okapi
# from tqdm import tqdm
# import copy
# # 读取包含label的json文件
# all_descriptions = merge_description('/opt/data/private/wtc_beifen/bird/data/description/train_all_databases.json')
# with open('/opt/data/private/wtc_beifen/bird/data/train/train_with_columns_tables.json', 'r') as f:
#     data = json.load(f)
#     all_triplets_list = []
#     for sql_metadata in tqdm(data):
#         db_name = sql_metadata['db_id']
#         query = sql_metadata['question']
#         columns = sql_metadata['columns']
#         tables = sql_metadata['tables']
#         all_columns = copy.deepcopy(all_descriptions[db_name])
#         joins = []
#         # 遍历所有可能的column、table组合
#         for col in columns:
#             for tab in tables:
#                 joins.append('|' + tab + '|' + col + '|')
#         gold_label = []
#         for j in joins:
#             for col in all_columns:
#                 if j in col:
#                     gold_label.append(col)
#         couple_list = []
#         for label in gold_label:
#             couple_list.append((query, label))# query, label 暂时没有负样本加入
#         #在all_columns中剔除gold_label
#         for label in gold_label:
#             all_columns.remove(label)
#         # 构造bm25模型
#         tokenized_corpus = [doc.split(" ") for doc in all_columns]
#         try:
#             bm25 = BM25Okapi(tokenized_corpus)
#         except ZeroDivisionError:
#             print(db_name)
#             print(query)
#             print(gold_label)
#             print(all_columns)
#             exit()

#         triplet_list = []
#         for couple in couple_list:
#             query = couple[0]
#             label = couple[1]
#             tokenized_query = query.split(" ")
#             scores = bm25.get_scores(tokenized_query)
#             column_score_pair = []
#             for cor, score in zip(all_columns, scores):
#                 column_score_pair.append((cor, score))
#             column_score_pair_sorted = sorted(column_score_pair, key=lambda x: x[1], reverse=True)
#             if len(column_score_pair_sorted) > 10:
#                 column_score_pair_sorted = column_score_pair_sorted[:10]
#             for pair in column_score_pair_sorted:
#                 triplet_list.append((query, label, pair))
#         all_triplets_list.extend(triplet_list)

# with open('/opt/data/private/wtc_beifen/bird/data/train/debug.json', 'w') as f:
#     json.dump(all_triplets_list[:200], f, indent=4)
# from sklearn.feature_extraction.text import TfidfVectorizer
# from rank_bm25 import BM25Okapi
# import numpy as np

# # 示例语料库
# corpus = res

# # 查询语句
# query = "List the different director IDs of the movies whose user rating is more than 4."

# # 使用 BM25 进行向量化
# import pdb; pdb.set_trace()
# tokenized_corpus = [doc.split(" ") for doc in corpus]
# bm25 = BM25Okapi(tokenized_corpus)
# tokenized_query = query.split(" ")
# scores = bm25.get_scores(tokenized_query)

# res = []

# for cor, score in zip(corpus, scores):
#     res.append((cor, score))
# for r in sorted(res, key=lambda x: x[1], reverse=True)[:20]:
#     print(r)

from transformers import LlamaTokenizer

tokenize = LlamaTokenizer.from_pretrained('/opt/data/private/hf_models/Llama-2-7b-hf')

# 生成长度为10的全1向量
import torch
query = "SELECT CAST(T1.Free Meal Count (K-12) AS REAL) / T1.Enrollment (K-12) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.cdscode = T2.cdscode WHERE T2.county = 'Alameda' ORDER BY T1.Free Meal Count (K-12) DESC LIMIT 1, 10000000000000000000000000000000000"
tokenize.pad_token = tokenize.eos_token
tokenized_query = tokenize(query)
import pdb; pdb.set_trace()
input_ids = torch.ones(10, dtype=torch.long)
input_ids[9] = -100
print(input_ids)
print(tokenize.decode(input_ids, skip_special_tokens=True))