from collections import defaultdict
import operator
import os
import sqlite3
from sql_metadata import Parser
from typing import Annotated, List, Literal

import numpy as np
from typing_extensions import TypedDict

from config import sql_embedding_dim
from util.sql import normalize_sql, gen_query_embedding
from rapidfuzz.fuzz import token_sort_ratio
from sqlglot import exp, diff, parse_one
from sqlglot.optimizer import optimize
from sqlglot.diff import Keep
import logging


embedding_columns = [f"query_embedding_{i}" for i in range(sql_embedding_dim)]
embedding_columns_str = ", ".join(embedding_columns)
create_instructions_str = """CREATE TABLE IF NOT EXISTS instructions (
    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `instruction` TEXT NOT NULL,
    `output` TEXT NOT NULL
)
"""
create_queries_str = (
    """CREATE TABLE IF NOT EXISTS queries (
    `id` INTEGER PRIMARY KEY AUTOINCREMENT,
    `query` TEXT UNIQUE NOT NULL,
    `type` TEXT NOT NULL,
    `table_list` TEXT NOT NULL,
    `db_name` TEXT NOT NULL,
    `token_len` DOUBLE NOT NULL,
"""
    + ",\n".join(
        [f"    `query_embedding_{i}` DOUBLE" for i in range(sql_embedding_dim)]
    )
    + """
)
"""
)


class DataPool:
    def __init__(self, schema, db_name=None):
        if db_name is None:
            db_name = "sqlgen"

        self.conn = sqlite3.connect(db_name + ".db", check_same_thread=False)

        cursor = self.conn.cursor()
        cursor.execute(create_instructions_str)
        cursor.execute(create_queries_str)
        self.conn.commit()

        self.database_dict = {}
        self.table_dict = {}
        for db_name, db_schema in schema.items():
            self.database_dict[db_name] = 0
            for table in db_schema.keys():
                self.table_dict[f"{db_name}.{table}"] = 0
        self.database_dict_key_lower = {_.lower(): _ for _ in self.database_dict}
        self.table_dict_key_lower = {_.lower(): _ for _ in self.table_dict}

        cursor.execute("SELECT db_name, table_list from queries;")
        for row in cursor.fetchall():
            db_name, table_list = row[0], row[1]

            self.database_dict[db_name] += 1
            for table in table_list.split(",")[1:-1]:
                self.table_dict[
                    self.table_dict_key_lower[f"{db_name}.{table}".lower()]
                ] += 1
        
        cursor.execute(f"SELECT {','.join([f'query_embedding_{i}' for i in range(sql_embedding_dim)])} FROM queries ORDER BY id")
        self.embeddings = cursor.fetchall()
        
        cursor.execute(f"SELECT query FROM queries ORDER BY id")
        self.queries = [row[0] for row in cursor.fetchall()] 

        cursor.close()

        

    def insert_instructions(self, instruction: str, output: str):
        cursor = self.conn.cursor()

        cursor.execute(
            "INSERT INTO instructions (instruction, output) VALUES (?, ?)",
            (instruction, output),
        )

        self.conn.commit()
        cursor.close()

    def insert_queries(
        self,
        queries: List[str],
        db_name: str,
        type: Literal["generation", "expansion"],
    ):
        cursor = self.conn.cursor()
        data = []
        for query in queries:
            parser = Parser(query)
            table_list = [
                table
                for table in parser.tables
                if f"{db_name}.{table}".lower() in self.table_dict_key_lower
            ]
            table_list.sort()
            table_list = "," + ",".join(table_list) + ","
            embed, token_len = gen_query_embedding(query)
            data.append((query, type, table_list, db_name, token_len, *embed))

        insert_success = [False] * len(data)
        for i, row in enumerate(data):
            res = cursor.execute(
                f"INSERT OR IGNORE INTO queries (query, type, table_list, db_name, token_len, {embedding_columns_str}) VALUES ({'?, ' * (sql_embedding_dim + 5 - 1)}?)",
                row,
            )
            if res.rowcount > 0:
                insert_success[i] = True
                self.embeddings.append(row[5:])
                self.queries.append(row[0])

        self.conn.commit()
        cursor.close()

        self.database_dict[db_name] += sum(insert_success)
        for row, success in zip(data, insert_success):
            if not success:
                continue
            for table in row[2].split(",")[1:-1]:
                self.table_dict[
                    self.table_dict_key_lower[f"{db_name}.{table}".lower()]
                ] += 1
        

        return insert_success

    def fetch_all_query_number(self):
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(id) FROM queries")
        row = cursor.fetchall()[0]

        cursor.close()

        return row[0]

    def fetch_db_query_number(self, db_name: str):
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(id) FROM queries WHERE db_name=?", (db_name,))
        row = cursor.fetchall()[0]

        cursor.close()

        return row[0]

    def fetch_expand_sample(self, db_name: str, table: str, count: int):

        cursor = self.conn.cursor()

        cursor.execute(
            f"SELECT query FROM queries WHERE type=? AND db_name=? AND table_list LIKE ? ORDER BY RANDOM() LIMIT {count}",
            ("generation", db_name, f"%,{table},%"),
        )
        rows = cursor.fetchall()

        queries = [row[0] for row in rows]

        cursor.close()

        return queries

    def fetch_similarity(self, queries: List[str]):
        return self.fetch_similarity_top_k(queries)

    def fetch_similar_queries_top_k(self, queries: List[str], k=10):
        cursor = self.conn.cursor()
        similar_queries = []
        queries_similarity = []
        for query in queries:
            try:
                query_embedding, token_len = gen_query_embedding(query)
                all_sim = np.dot(self.embeddings, query_embedding)
                top_indices = np.argsort(all_sim)[-k * 20:]
                top_sql = [self.queries[i] for i in top_indices]
                
                final_sim = []
                parsed_query = parse_one(query, dialect="sqlite")
                for idx, sql in enumerate(top_sql):
                    sim_fuzz = token_sort_ratio(query.lower(), sql.lower()) * 0.01
                    total_node = diff(parsed_query, parse_one(sql, dialect="sqlite"))
                    sim_diff = (
                        sum(1 if isinstance(e, Keep) else 0 for e in total_node)
                        * 1.0
                        / len(total_node)
                    )
                    final_sim.append(sim_fuzz * 0.6 + sim_diff * 0.3 + all_sim[idx] * 0.1)
                top_k_indices = np.argsort(final_sim)
                top_k_sql = [top_sql[i] for i in top_k_indices[-k:]]
                top_k_value = [final_sim[i] for i in top_k_indices[-k:]]
                similar_queries.append(top_k_sql)
                queries_similarity.append(top_k_value)
            except Exception as e:
                logging.warning(e)

        cursor.close()
        return similar_queries, queries_similarity

    def fetch_similarity_top_k(self, queries: List[str], k=10):
        cursor = self.conn.cursor()
        sim_per_query = []
        for query in queries:
            try:
                query_embedding, token_len = gen_query_embedding(query)
                all_sim = np.dot(self.embeddings, query_embedding)
                top_indices = np.argsort(all_sim)[-k * 20:]
                top_sql = [self.queries[i] for i in top_indices]
                
                final_sim = []
                parsed_query = parse_one(query, dialect="sqlite")
                for idx, sql in enumerate(top_sql):
                    sim_fuzz = token_sort_ratio(query.lower(), sql.lower()) * 0.01
                    total_node = diff(parsed_query, parse_one(sql, dialect="sqlite"))
                    sim_diff = (
                        sum(1 if isinstance(e, Keep) else 0 for e in total_node)
                        * 1.0
                        / len(total_node)
                    )
                    final_sim.append(sim_fuzz * 0.6 + sim_diff * 0.3 + all_sim[idx] * 0.1)
                sim_per_query.append(np.mean(np.sort(final_sim)[-k:]))
            except Exception as e:
                logging.warning(e)

        cursor.close()
        return sim_per_query

    def fetch_train_dataset(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT instruction, output FROM instructions")

        dataset = cursor.fetchall()
        cursor.close()

        return dataset
