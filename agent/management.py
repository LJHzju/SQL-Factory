import os
import sys

from typing import Annotated, Dict, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import llm_config, GENERATE_EPOCHS
from util.benchmark import *

llm = ChatOpenAI(**llm_config["selection"])

prompt = ChatPromptTemplate(
    [
        (
            "user",
            "You are a database expert acting as the manager in a multi-agent SQL query generation system. Your role is to determine the next step in the agent workflow.\n"
            "There are three possible next-hop decisions: 'generation', 'expansion', or 'end'. Please follow the rules below to make your decision:\n"
            "1. If this is the initial stage (i.e., the number of synthesized SQL queries is 0), the next step must be 'generation'.\n"
            "2. If the current number of synthesized SQL queries has reached the target number, the next step should be 'end'.\n"
            "3. In all other cases, your decision should be based on the previous strategy and the current context:\n"
            "   - If the previous strategy was 'generation', you should only switch to 'expansion' after a predefined number of epochs (e.g., 100 epochs) have been completed. On the contrary, if the **Current Epoch Number** has not yet reached **Target Epoch Number**, then `generation` must continue.\n"
            "   - If the previous strategy was 'expansion', you must evaluate the overall quality of the generated queries based on two criteria: executability and similarity.\n"
            "     A query set is considered to meet the quality standard only if it achieves **high executability** (e.g., greater than 0.75) and **low average similarity** (e.g., less than 0.8) in average.\n"
            "     If the evaluation not meet these standards, the system should switch back to 'generation'; otherwise, it should continue with 'expansion'.\n"
            "Target number of SQL queries: {target_sql_num}\n"
            "Current number of synthesized SQL queries: {current_sql_num}\n"
            "Previous strategy: {prev_strategy}\n"
            "{extra_info}"
            "Please output your decision for the next hop as one of the following options: ['generation', 'expansion', 'end'].\n"
            "Your output should follow this exact format: <strategy>[your chosen next hop destination]<strategy>\n"
            "Example output: <strategy>generation<strategy>",
        )
    ]
)

management_runnable = prompt | llm

def management_agent(target_sql_num: int, current_sql_num: int, prev_strategy: Literal["generation", "expansion"], current_epoch: int, critical_score: Dict):
    if prev_strategy == 'generation':
        extra_info = f"Predefined Target Epoch Number per generation round: {GENERATE_EPOCHS}\nCurrent Epoch Number: {current_epoch}\n\n"
    elif prev_strategy == 'expansion':
        extra_info = f"Current quality assessment result (executability and similarity scores): {critical_score}\n\n"
    return management_runnable.invoke({"target_sql_num": target_sql_num, 'current_sql_num': current_sql_num, 'prev_strategy': prev_strategy, 'extra_info': extra_info})
