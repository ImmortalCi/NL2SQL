import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from sentence_transformers import SentenceTransformer
from get_triplet import merge_description
import json
from tqdm import tqdm


# 先给query按照db_id分类，减少embedding的计算量

# all_query_with_gold_label,all_description
all_descriptions = merge_description('/opt/data/private/wtc_beifen/bird/data/description/train_all_descriptions.json')  # 所有数据库列的描述
model = SentenceTransformer("/opt/data/private/wtc_beifen/moe/all-mpnet-base-v2") # 加载模型

with open('/opt/data/private/wtc_beifen/bird/data/train/train_gold_label.json', 'r') as f:  # 读取Schema-Linking的gold label
    data = json.load(f)
    eval_res = []
    for emp in tqdm(data):
        query = emp['question'] + ' ' + emp['evidence']
        label = emp['gold_label']
        if label == []:  # 如果gold label为空，跳过
            continue
        else:
            corpus_name = emp['db_id']
            corpus = all_descriptions[corpus_name]

            doc_embeddings = model.encode(corpus, convert_to_tensor=True)
            query_embedding = model.encode(query, convert_to_tensor=True)
            sim = model.similarity(query_embedding, doc_embeddings)
            res = []
            for i in range(len(corpus)):
                res.append((corpus[i], sim[0][i]))
            res = sorted(res, key=lambda x: x[1], reverse=True)

            scores = []
            for index, i in enumerate(res):
                if i[0] in label:
                    scores.append((index, res[index][1].tolist()))
            try:
                avg_index = sum([i[0] for i in scores]) / len(scores)
                avg_sim = sum([i[1] for i in scores]) / len(scores)
                eval_res.append((query, avg_index, avg_sim))
            except ZeroDivisionError:
                import pdb; pdb.set_trace()
        
    # print eval_res的均值
    avg_index = sum([i[1] for i in eval_res]) / len(eval_res)
    avg_sim = sum([i[2] for i in eval_res]) / len(eval_res)


    print(f"avg_index: {avg_index}, avg_sim{avg_sim}")

    eval_res.append(('avg', avg_index, avg_sim))

    # 保存文本结果
    with open('/opt/data/private/wtc_beifen/bird/exp_res/schema-linking/mpnet_raw_with_evi.json', 'w') as f:
        json.dump(eval_res, f, indent=4)