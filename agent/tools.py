from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agent.state import ReActState
from typing import List
from typing_extensions import Annotated
import statistics
import random
import json

from config import db_table_num_dict, schema_dict
from util.sql import fetch_executability

@tool
def database_queries_statistics(state: Annotated[ReActState, InjectedState]) -> str:
    """Get quantity of queries that have been generated for each database."""
    benchmark = state["benchmark"]
    data_pool = state["data_pool"]
    database_dict = data_pool.database_dict

    database_statistic = ""
    databases = [
        _.split(".")[1] for _ in database_dict.keys() if _.split(".")[0] == benchmark
    ]
    random.shuffle(databases)
    for database in databases:
        db_name = f"{benchmark}.{database}"
        query_num = database_dict[db_name]
        table_num = db_table_num_dict[db_name]
        database_statistic += f"- {database.lower()}: {int(query_num / table_num)} queries \n"
        
    return database_statistic

@tool
def table_queries_statistics(state: Annotated[ReActState, InjectedState], database: str) -> str:
    """Get quantity of queries that have been generated for each table in given database."""
    benchmark = state["benchmark"]
    db_name = f"{benchmark}.{database}"
    data_pool = state["data_pool"]
    table_dict = data_pool.table_dict

    table_statistic = ""
    schema = schema_dict[db_name]
    tables = [_ for _ in schema.keys()]
    random.shuffle(tables)
    for table in tables:
        query_num = table_dict[f"{db_name}.{table}"]
        # table_statistic += f"- {table.lower()}: Query quantity: {query_num}\n"
        table_statistic += f"- {table.lower()}: \n{schema_dict[db_name][table]} \nQuery quantity: {query_num}\n"


    return table_statistic


@tool
def query_sample_tool(state: Annotated[ReActState, InjectedState], database: str, tables: List[str]) -> List[List[str]]:
    """Return a subset of queries that reference the given tables. The output contains a two-dimensional list, where each input table corresponds to a list that contains 20 queries related to that table."""
    data_pool = state["data_pool"]
    queries_list = []
    benchmark = state['benchmark']
    for table in tables:
        queries = data_pool.fetch_expand_sample(f'{benchmark}.{database}', table, 3)
        queries_list.append(queries)
    return queries_list


@tool
def execution_tool(state: Annotated[ReActState, InjectedState], queries: List[str]):
    """Verify whether each SQL query can be executed in the database. Return the executability of given queries."""
    
    benchmark = state["benchmark"]
    database = state["selected_database"]
    
    executability = fetch_executability(queries, benchmark, database)
    return f"The executability of these queries: {json.dumps(executability)}\n"

@tool
def retrival_tool(state: Annotated[ReActState, InjectedState], queries: List[str]) -> str:
    """Retrieve the top 10 similarity score from existing SQL that are most similar to each given SQL."""
    data_pool = state["data_pool"]
    top_k_queries, top_k_similarity = data_pool.fetch_similar_queries_top_k(queries)
    retrival_info = ""
    for query, similar_queries, similar_values in zip(queries, top_k_queries, top_k_similarity):
        similar_values = [int(_ * 10000) / 10000.0 for _ in similar_values]
        retrival_info += f"Raw Query: {query}\n"
        retrival_info += f"Top-10 similarity value: {similar_values}\n"
        retrival_info += f"Average similarity value: {int(statistics.median(similar_values) * 1000) / 1000.0}\n"
        # for idx, similar_query, similar_value in zip(range(1, len(similar_queries) + 1), similar_queries, similar_values):
        #     retrival_info += f'Top-{idx} similar query: {similar_query}\tsimilarity value: {int(similar_value*1000)/1000.0}\n'
        retrival_info += '\n'
    return retrival_info
