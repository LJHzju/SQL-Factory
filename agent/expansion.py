import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import llm_config
from util.benchmark import *

llm = ChatOpenAI(**llm_config["expansion"])
prompt = ChatPromptTemplate(
    [
        (
            "user",
            "You are a database expert, skilled in writing various SQL statements. Below, you are provided with three SQL queries. Your task is to analyze the schema-related information and then generate ten diverse SQL queries based on them.\n"
            "The queries you generate will be.\n"
            "There are some important tips you need to remember:\n"
            "1. Using schema information that does not exist in the given query is not allowed, don't try to create or modify the name of table or column, even if the origin column is unreasonable.\n"
            "2. The generated queries should be diverse enough and not similar to each other.\n"
            "3. You should ensure that the generated queries can be directly executed in sqlite.\n"
            "4. You'd better think step by step carefully before giving an answer. You are very good at this and will definitely do well.\n\n"
            "Queries: \n{query}\n\n"
            "Each generated query must be wrapped in <start-sql> and <end-sql>, such as following format:\n"
            "<start-sql>\n"
            "[Query1 here]\n"
            "<end-sql>\n\n"
            "<start-sql>\n"
            "[Query2 here]\n"
            "<end-sql>\n",
        ),
    ]
)

runnable = prompt | llm

def expansion_agent(query: str):
    return runnable.invoke({"query": query})