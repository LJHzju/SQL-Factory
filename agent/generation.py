import os
import sys
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import llm_config
from util.benchmark import *

llm = ChatOpenAI(**llm_config["generation"])
prompt = ChatPromptTemplate(
    [
        (
            "user",
            "You are a database expert. Below, you are provided with some tables and their corresponding database schemas. Your task is to generate ten complex and diverse SQL queries for these tables.\n"
            "The query you generate needs to meet the following conditions.\n"
            "1. The generated query needs to cover as many tables as possible.\n"
            "2. It is better if the generated query contains very complex operators and window functions, such as `WHERE`, `CASE WHEN`, `RANK`, `DENSE_RANK` and others.\n"
            "3. The generated queries should not be too similar to each other.\n"
            "4. You should ensure that the generated queries can be directly executed in sqlite.\n"
            "5. You'd better think carefully before giving an answer. You are very good at this and will definitely do well.\n\n"
            "Table List: {table_list}\n\n"
            "Table Schema: \n{schema_info}\n\n"
            "Foreign Key Dependency: {foreign_info}\n\n"
            "Each generated query must be wrapped in <start-sql> and <end-sql>, such as following format:\n"
            "<start-sql>\n"
            "[Query1 here]\n"
            "<end-sql>\n\n"
            "<start-sql>\n"
            "[Query2 here]\n"
            "<end-sql>\n",
        )
    ]
)

runnable = prompt | llm


def generation_agent(table_list: List[str], schema_info: str, foreign_info: str):
    return runnable.invoke(
        {
            "table_list": table_list,
            "schema_info": schema_info,
            "foreign_info": foreign_info,
        }
    )
