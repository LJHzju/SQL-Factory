import copy
import json
import os
import re
from typing import List

import numpy as np
import psycopg2
import sqlite3
import sqlparse
import torch
from sql_metadata import Parser
from transformers import AutoModel, AutoTokenizer
from sqlglot import parse_one
from sqlglot.optimizer.qualify import qualify

from config import sql_embedding_dim, pg_config, database_path

keyword_as_table_name = ["TRANSACTION", "PRIMARY"]
tokenizer = None
model = None


class PostgreSQLConnection:
    def __init__(self, config):
        self.host = config["host"]
        self.dbname = config["dbname"]
        self.user = config["user"]
        self.password = config["password"]
        self.port = config["port"]
        self.timeout = config["timeout"]
        self.connection = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                port=self.port,
                options=f"-c statement_timeout={self.timeout}",
            )
            # print(f"Successfully connected to PostgreSQL! DB_name: {self.dbname}")
            return self.connection
        except Exception as e:
            # print(f"Error occurred while connecting to PostgreSQL: {e}")
            return None

    def close(self):
        if self.connection:
            self.connection.close()
            # print("Connection closed.")

    def execute_query(self, query, params=None):
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return True
        except Exception as e:
            # print(f"Error executing query: {e}")
            self.connection.rollback()
            return False

    def execute_multiple_queries(self, queries, params=None):
        results = []
        for query in queries:
            query = "EXPLAIN " + query
            if params:
                results.append(self.execute_query(query, params))
                # results += self.execute_query(query, params)
            else:
                results.append(self.execute_query(query))
                # results += self.execute_query(query)
        return results


def encode_sql(query: str) -> List[float]:
    global tokenizer
    global model
    if tokenizer is None:
        checkpoint = "google-bert/bert-large-uncased"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

    inputs = tokenizer(
        query, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()

    return cls_embedding.tolist(), inputs.input_ids.shape[1]


def gen_query_embedding(query: str):
    query = normalize_sql(query)
    embedding, token_len = encode_sql(query)
    norm = np.linalg.norm(embedding)

    if norm != 0:
        embedding = [x / norm for x in embedding]
    else:
        embedding = [0.0] * len(embedding)

    return embedding, token_len


def record_query_tables(query: str):
    query = normalize_sql(query)
    parser = Parser(query)
    table_list = parser.tables
    query_tables = " "
    for table in table_list:
        query_tables += f"{table} "
    return query_tables


def format_query(query):
    if isinstance(query, str):
        return f"`{query}`"
    elif isinstance(query, list):
        query_str = ""
        for q in query:
            query_str += f"`{q}`\n"
        return query_str[:-1]


def remove_sql_comment(sql: str):
    sql = re.sub(r"--.*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def normalize_sql(sql: str):
    sql = remove_sql_comment(sql)
    sql = sql.replace("```sql", "").replace("```", "")
    # parsed = qualify(parse_one(sql)).sql(dialect="sqlite")
    parsed = sqlparse.format(sql, reindent=True, keyword_case="upper")
    normalized_sql = " ".join(parsed.split())
    suffix = ";" if normalized_sql[-1] != ";" else ""
    return normalized_sql + suffix


def fetch_executability(queries: List[str], benchmark: str, database: str, verbose=False) -> List[bool]:
    config = copy.deepcopy(pg_config)
    exe_per_query = []
    if benchmark in ["tpcds", "imdb", "tpch"]:
        config["dbname"] = benchmark
        conn = PostgreSQLConnection(config)
        conn.connect()
        exe_per_query = conn.execute_multiple_queries(queries)
        conn.close()
    elif benchmark in ["spider", "spider-dev", "bird", "bird-dev"]:
        benchmark_path = database_path[benchmark]
        # database = '.'.join(database.split('.')[1:])
        conn = sqlite3.connect(
            os.path.join(benchmark_path, database, f"{database}.sqlite")
        )
        cursor = conn.cursor()
        for idx, query in enumerate(queries):
            exe = False
            try:
                cursor.execute("EXPLAIN " + query)
                exe |= True
            except Exception as e:
                exe |= False

            if exe is False:
                try:
                    new_query = parse_one(query).sql(dialect="sqlite")
                    for table_name in keyword_as_table_name:
                        wrap_table_name = f'"{table_name.lower()}"'
                        new_query = re.sub(rf'\b{table_name}\b', wrap_table_name, new_query)
                    new_query = new_query.replace(".group", '."group"')
                        
                    cursor.execute("EXPLAIN " + new_query)
                    queries[idx] = new_query
                    exe |= True
                except Exception as e:
                    exe |= False
                    if verbose:    
                        print(e)

            exe_per_query.append(exe)
        cursor.close()
        conn.close()
    # executable_rate = 1.0 * executable_number / len(queries)
    return exe_per_query


def parse_sql(s: str):
    sqls = re.findall(r"<start-sql>\s*(.*?)\s*<end-sql>", s, re.DOTALL)
    return sqls
