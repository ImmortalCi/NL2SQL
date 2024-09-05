"""
此脚本用于生成训练数据，生成的数据格式为(query, label, pair)，其中query是问题，label是正确的列描述，pair是其他列描述
"""
import json
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import copy

def merge_description(description_path):
    """
    合并描述信息

    参数:
    description_path (str): 描述信息文件的路径，由get_description脚本生成的字典。

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


# 读取包含label的json文件
def get_triplet_bm25(all_descriptions, metadata_path, save_path):
    """
    使用BM25，从元数据中获取三元组并保存到文件中。

    参数：
    all_descriptions (dict): 包含数据库描述的字典，键为数据库名称，值为描述。
    metadata_path (str): 元数据文件的路径，由SQL_parser脚本生成的每个SQL查询用到的列和表。
    save_path (str): 保存三元组的文件路径。

    返回：
    无返回值。

    Raises:
    无异常抛出。

    """
    with open(metadata_path, 'r') as f:
        data = json.load(f)
        all_triplets_list = []
        for sql_metadata in tqdm(data):
            db_name = sql_metadata['db_id']
            query = sql_metadata['question']
            columns = sql_metadata['columns']
            tables = sql_metadata['tables']
            all_columns = copy.deepcopy(all_descriptions[db_name])
            joins = []
            # 遍历所有可能的column、table组合
            for col in columns:
                for tab in tables:
                    joins.append('|' + tab + '|' + col + '|')
            gold_label = []
            for j in joins:
                for col in all_columns:
                    if j in col:
                        gold_label.append(col)
            couple_list = []
            for label in gold_label:
                couple_list.append((query, label))# query, label 暂时没有负样本加入
            #在all_columns中剔除gold_label
            for label in gold_label:
                all_columns.remove(label)
            # 构造bm25模型
            tokenized_corpus = [doc.split(" ") for doc in all_columns]
            try:
                bm25 = BM25Okapi(tokenized_corpus)
            except ZeroDivisionError:
                print(db_name)
                print(query)
                print(gold_label)
                print(all_columns)
                continue

            triplet_list = []
            for couple in couple_list:
                query = couple[0]
                label = couple[1]
                tokenized_query = query.split(" ")
                scores = bm25.get_scores(tokenized_query)
                column_score_pair = []
                for cor, score in zip(all_columns, scores):
                    column_score_pair.append((cor, score))
                column_score_pair_sorted = sorted(column_score_pair, key=lambda x: x[1], reverse=True)
                if len(column_score_pair_sorted) > 10:
                    column_score_pair_sorted = column_score_pair_sorted[:10]
                for pair in column_score_pair_sorted:
                    triplet_list.append((query, label, pair))
            all_triplets_list.extend(triplet_list)

    with open(save_path, 'w') as f:
        json.dump(all_triplets_list, f, indent=4)


if __name__ == '__main__':
    # test = merge_description('/opt/data/private/wtc_beifen/bird/data/description/train_all_databases.json')
    # get_triplet_bm25(test, '/opt/data/private/wtc_beifen/bird/data/train/train_with_columns_tables.json', '/opt/data/private/wtc_beifen/bird/data/train/train_triplets.json')