import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import sqlite3
import json
import token
from tqdm import tqdm


from networkx import all_triplets
from sqlglot import column

# def execute_sql(sql, db_path):
#     # Connect to the database
#     conn = sqlite3.connect(db_path)
#     # Create a cursor object
#     cursor = conn.cursor()
#     cursor.execute(sql)
#     results = cursor.fetchall()
    


# sql = "SELECt repoId FROM solution GROUP BY RepoId ORDER BY COunT(Path) DESC LIMIT 1"
# #将sql语句中的大小写转换为小写
# sql = sql.lower()

# db_path = '/opt/data/private/wtc_beifen/bird/data/train/train_databases/codebase_comments/codebase_comments.sqlite'
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()
# cursor.execute(sql)
# print(sql)
# print(cursor.fetchall())
# import pdb; pdb.set_trace()



from transformers import LlamaTokenizer, LlamaForCausalLM





tokenize = LlamaTokenizer.from_pretrained('/opt/data/private/hf_models/SQL_Llama-7B')


# 生成长度为10的全1向量
# import torch
# query = "SELECT CAST(T1.Free Meal Count (K-12) AS REAL) / T1.Enrollment (K-12) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.cdscode = T2.cdscode WHERE T2.county = 'Alameda' ORDER BY T1.Free Meal Count (K-12) DESC LIMIT 1, 10000000000000000000000000000000000"
# tokenize.pad_token = tokenize.eos_token
# tokenized_query = tokenize(query)
# import pdb; pdb.set_trace()
# input_ids = torch.ones(10, dtype=torch.long)
# input_ids[9] = -100
# print(input_ids)
# print(tokenize.decode(input_ids, skip_special_tokens=True))



# test = 'movie_platform|movies|movie_popularity|number of mubi users who love this movie'
# idx = test.find("|")
# print(idx)
# import pdb; pdb.set_trace()

QUESTION = "Among the payments made by Mary Smith, how many of them are over 4.99?"
DATABASE_SCHEMA = """
customer|first_name|first name of the customer;
customer|last_name|last name of the customer;
payment|amount|unique id number identifying the amount;
payment|customer_id|unique id number identifying the customer;
customer|customer_id|unique id number identifying the country;
"""
text = f"""
You are a data science expert.
Below, you are presented with a database schema and a question.
Your task is to read the schema, understand the question, and generate a
valid SQLite query to answer the question.
Before generating the final SQL query think step by step on how to write
the query.
Database Schema:
{DATABASE_SCHEMA}
This schema offers an in-depth description of the database’s architecture,
detailing tables, columns, and their descriptions.
The format in Data Schema is: table_name|column_name|column_description.
Question:
{QUESTION}
Please respond with a JSON object structured as follows without anything else:
{{"chain_of_thought_reasoning": "Your thought process on how you arrived
at the final SQL query.",
"SQL": "Your SQL query in a single string."
}}
Priority should be given to columns that have been explicitly matched
with examples relevant to the question’s context.
Take a deep breath and think step by step to find the correct SQLite SQL
query. If you follow all the instructions and generate the correct query,
I will give you 1 million dollars.
"""

print(text)
Tokenized_text = tokenize(text,return_tensors='pt')
model = LlamaForCausalLM.from_pretrained('/opt/data/private/hf_models/SQL_Llama-7B', torch_dtype=torch.bfloat16)
Tokenized_text.to('cuda')
model.to('cuda')
generated_token = model.generate(**Tokenized_text, max_new_tokens=1024)
imput_len = len(Tokenized_text['input_ids'][0])
new_tokens = generated_token[0][imput_len:]
generated_text = tokenize.decode(new_tokens)
import pdb; pdb.set_trace()