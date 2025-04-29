import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agent.state import ReActState
from agent.tools import execution_tool, retrival_tool
from config import llm_config
from util.benchmark import *

llm = ChatOpenAI(**llm_config["selection"])
prompt = ChatPromptTemplate(
    [
        (
            "user",
            "You are a database expert. You will be given a list of SQL queries.\n"
            "{task_desc}\n"
            "### Important tip\n"
            "You can only call **one tool at a time**, which means there can only be one `Action` and `Action Input` pair in one output.\n"
            "The Action Input must be a strict JSON object rather than a string. In other words, the Action Input dictionary cannot be enclosed in double quotes (\"). This is crucial to avoid syntax errors. Please always ensure JSON formatting is correct.\n"
            "Example:\n"
            "```\n"
            "Action: retrival_tool\n"
            'Action Input: {{"queries": [queries list]}}\n'
            "```"
        )
    ]
)

agent = create_react_agent(
    llm, [execution_tool, retrival_tool], state_schema=ReActState
)


def critical_agent(state, queries_list, next_strategy):
    queries_list_json = json.dumps(queries_list)
    if next_strategy == "generation":
        task_desc = (
            "Your task is to determine whether each query can be successfully executed on the database. Specifically, use the execution_tool to output a list of booleans for the given queries, indicating whether each query can be executed.\n"
            f"The input queries are {queries_list_json}\n"
            "Please return a JSON object containing a list of boolean values, where each value indicates whether the corresponding query is executable.\n"
            "Format your response exactly like this:\n"
            '{"executability": [true, false, true, true, false, ...]}\n'
            "or\n"
            '{"executability": [false, false, true, true, false, ...]}\n'
        )
    elif next_strategy == "expansion":
        task_desc = (
            "Your task is to evaluate the overall quality of these queries based on two criteria: executability and similarity.\n"
            "- **Executability** refers to whether each query can be successfully executed on the database. Specifically, use the execution_tool to to output a list of booleans for the given queries_list, indicating whether each query can be executed.\n"
            "- **Similarity** refers to the average similarity between each query and its top-10 most similar existing queries. Specifically, use the retrival_tool to get the similarity score for each queries in queries_list.\n"
            "Only queries with high executability (e.g., above 0.75) and low average similarity (e.g., below 0.8) in average are considered to meet the quality standard.\n"
            f"The input queries are: {queries_list_json}\n"
            "Start by providing a brief **analysis** explaining your final evaluation. "
            "Then output the results in JSON format, including three fields:\n"
            "1. 'executability': a list of booleans representing whether each query is executable.\n"
            "2. 'similarity': a list of floats representing average similarity scores for each query.\n"
            "3. 'final': a boolean indicating whether the overall set meets the quality threshold (high executability and low average similarity).\n"
            "Example outputs:\n"
            '{"executability": [true, false, true, true, false, ...], "similarity": [0.6, 0.3, 0.17, 0.45, 0.62, ...], "final": true}\n'
            "or\n"
            '{"executability": [false, false, true, true, false, ...], "similarity": [0.8, 0.75, 0.67, 0.95, 0.89, ...], "final": false}\n'
        )

    prompt_message = prompt.invoke({"task_desc": task_desc})
    for s in agent.stream(ReActState(messages=prompt_message.to_messages(), **state)):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            print(message)
            print("-" * 10)

    return s["messages"][-1]
