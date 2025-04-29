import json
import random
import re

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from tqdm import tqdm
from collections import defaultdict

from agent.expansion import expansion_agent
from agent.generation import generation_agent
from agent.management import management_agent
from agent.state import State
from agent.table_selection import table_selection_agent
from agent.seed_selection import seed_selection_agent
from agent.critical import critical_agent
from config import *
from util.datapool import DataPool
from util.parse import parse_state
from util.sql import format_query, normalize_sql, parse_sql

pbar = None


def entry_node(state: State):
    global pbar
    benchmark = state["benchmark"]

    schema_info_path = os.path.join(data_dir, f"{benchmark}_info")
    for file_name in os.listdir(schema_info_path):
        file_path = os.path.join(schema_info_path, file_name)
        prefix, suffix = file_name.rsplit("_", 1)
        prefix = f"{benchmark}." + prefix
        if suffix == "ri.csv":
            with open(file_path, "r") as f:
                foreign_key = pd.read_csv(f)
            foreign_key_dict[prefix] = foreign_key
        elif suffix == "schema.json":
            with open(file_path, "r") as f:
                schema = json.load(f)
            schema_dict[prefix] = schema
            db_table_num_dict[prefix] = len(schema)

    if state.get("epoch") is None:
        state["epoch"] = 0
    if state.get("data_pool") is None:
        state["data_pool"] = DataPool(schema_dict, db_name=f"sqlgen-{benchmark}")
    state["prev_strategy"] = "generation"
    state["selected_database"] = ""
    state["selected_tables"] = []
    state["selected_seed_queries"] = []
    state["each_synthesized_queries"] = []

    pbar = tqdm(total=state["target_sql_num"])
    num_queries = state["data_pool"].fetch_all_query_number()
    if num_queries > 0:
        pbar.update(num_queries)

    return state


def management_node(state: State):
    target_sql_number = state["target_sql_num"]
    current_sql_number = state["data_pool"].fetch_all_query_number()
    prev_strategy = state["prev_strategy"]
    current_epoch = state["epoch"]
    if len(state["critical_score"]) == 0:
        critical_score = {}
    else:
        critical_score = state["critical_score"][0]
    for _ in range(3):
        try:
            response = management_agent(
                target_sql_number,
                current_sql_number,
                prev_strategy,
                current_epoch,
                critical_score,
            )
            next_strategy = response.content.split("<strategy>")[1].strip().lower()

            if next_strategy in ["generation", "expansion", "end"]:
                break
        except Exception as e:
            continue
    state["critical_score"].clear()
    
    if prev_strategy == "expansion" and next_strategy == "generation":
        return State(next_strategy=next_strategy, epoch=0)
    else:
        return State(next_strategy=next_strategy)


def critical_node(state: State):
    queries_list = state["synthesized_queries"][0]
    next_strategy = state["next_strategy"]
    data_pool = state["data_pool"]
    db_name = f'{state["benchmark"]}.{state["selected_database"]}'

    # response = critical_agent(state, queries_list, next_strategy)
    # json_match = re.search(r"{.*?}", response.content, re.DOTALL)
    # if json_match:
    #     json_str = json_match.group(0)
    # # else:
    # #     continue
    # critical_score = json.loads(json_str.lower())

    for _ in range(3):
        try:
            print("********** Retry times: ", _, " **********")
            response = critical_agent(state, queries_list, next_strategy)
            json_match = re.search(r"{.*?}", response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                continue
            print(json_str)
            critical_score = json.loads(json_str)
            break
        except Exception as e:
            continue
        
    flag = 1
    if next_strategy == "expansion" and (critical_score["final"] == False or (isinstance(critical_score["final"], str) and critical_score["final"].lower() == "false")):
        flag = 0
    if flag:
        filtered_sqls = []
        for query, exe in zip(queries_list, critical_score["executability"]):
            if exe:
                filtered_sqls.append(query)
        insert_success = data_pool.insert_queries(filtered_sqls, db_name, next_strategy)
        pbar.update(sum(insert_success))

    # for _ in range(3):
    #     try:
    #         response = critical_agent(state, queries_list, prev_strategy)
    #         critical_score = parse_critical_score(response.content, prev_strategy)

    #         flag = 1
    #         if prev_strategy == 'expansion' and critical_score['final'].lower() == 'false':
    #             flag = 0
    #         if flag:
    #             filtered_sqls = []
    #             for query, exe in zip(queries_list, critical_score['executability']):
    #                 if exe:
    #                     filtered_sqls.append(query)
    #             insert_success = data_pool.insert_queries(filtered_sqls, db_name, prev_strategy)
    #             pbar.update(sum(insert_success))
    #         break
    #     except:
    #         continue
    # state["synthesized_queries"].clear()
    return State(critical_score=[critical_score])


def table_selection_node(state: State):
    for _ in range(3):
        try:
            response = table_selection_agent(state)
            json_match = re.search(r"{.*?}", response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                continue
            result = json.loads(json_str)
            break
        except Exception as e:
            continue
    return State(selected_database=result["database"], selected_tables=result["tables"])


def generation_node(state: State):
    benchmark = state["benchmark"]
    database = state["selected_database"]
    db_name = f"{benchmark}.{database}"
    table_list = state["selected_tables"]

    schemas = schema_dict[db_name]
    schema_info = {table: schemas[table] for table in table_list}

    foreign_keys = foreign_key_dict[db_name]
    foreign_key_info = []
    for _, row in foreign_keys.iterrows():
        if row["table_1"] in table_list and row["table_2"] in table_list:
            foreign_key_info.append(
                f"{row['table_1']}({row['attr_1']}) - {row['table_2']}({row['attr_2']})"
            )
    if foreign_key_info == []:
        foreign_key_info = "There are no foreign key dependencies for these tables."

    response = generation_agent(table_list, schema_info, foreign_key_info)
    sqls = parse_sql(response.content)
    sqls = [normalize_sql(sql) for sql in sqls]

    return State(synthesized_queries=[sqls], epoch=state["epoch"] + 1)


def seed_selection_node(state: State):
    response = seed_selection_agent(state)
    if "FINISH" in response.content:
        return State(next_strategy="generation")
    json_match = re.search(r"{.*?}", response.content, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    result = json.loads(json_str)
    database = result["database"]
    queries = result["queries"]
    
    # for _ in range(3):
    #     try:
    #         response = seed_selection_agent(state)
    #         if "FINISH" in response.content:
    #             return State(next_strategy="generation")
    #         json_match = re.search(r"{.*?}", response.content, re.DOTALL)
    #         if json_match:
    #             json_str = json_match.group(0)
    #         result = json.loads(json_str)
    #         database = result["database"]
    #         queries = result["queries"]
    #         break
    #     except:
    #         continue
    
    return State(selected_database=database,selected_seed_queries=queries)


def expansion_node(state: State):
    queries = state["selected_seed_queries"]
    queries_str = format_query(queries)

    for _ in range(3):
        response = expansion_agent(queries_str)
        sqls = parse_sql(response.content)
        if len(sqls) == 10:
            break
    sqls = [normalize_sql(sql) for sql in sqls]

    return State(synthesized_queries=[sqls])


def route_on_next_strategy(state: State):
    return state["next_strategy"]


def route_and_map_on_expansion_quantity(state: State):
    if len(state["synthesized_queries"]) * 10 < EVAL_COUNT:
        return "continue"
    else:
        synthesized_queries = state["synthesized_queries"]
        return [
            Send("critical_agent", {**state, "each_synthesized_queries": _})
            for _ in synthesized_queries
        ]


def map_expansion(state: State):
    if state["next_strategy"] == "generation":
        return "generation"
    
    
    return [
        Send("expansion_agent", {"selected_seed_queries": _})
        for _ in state["selected_seed_queries"]
    ]


def reduce_expansion(state: State):
    state["selected_seed_queries"].clear()
    return


def reduce_critical(state: State):
    com_critical_score_list = defaultdict(list)
    for item in state["critical_score"]:
        for k, v in item.items():
            com_critical_score_list[k].append(v)

    com_critical_score = {}
    if "similarity" in com_critical_score_list:
        com_critical_score["similarity"] = np.mean(
            com_critical_score_list["similarity"]
        )
    if "executability" in com_critical_score_list:
        com_critical_score["executability"] = np.median(
            com_critical_score_list["executability"]
        )

    state["synthesized_queries"].clear()
    state["critical_score"].clear()

    return State(
        prev_strategy=state["next_strategy"],
        next_strategy=None,
        critical_score=[com_critical_score],
    )


def init_graph():
    graph = StateGraph(State)
    graph.add_node("entry", entry_node)
    graph.add_node("management_agent", management_node)
    graph.add_node("critical_agent", critical_node)
    graph.add_node("table_selection_agent", table_selection_node)
    graph.add_node("generation_agent", generation_node)
    graph.add_node("seed_selection_agent", seed_selection_node)
    graph.add_node("expansion_agent", expansion_node)
    graph.add_node("reduce_expansion", reduce_expansion)
    graph.add_node("reduce_critical", reduce_critical)

    graph.add_edge(START, "entry")
    graph.add_edge("entry", "management_agent")
    graph.add_conditional_edges(
        "management_agent",
        route_on_next_strategy,
        {
            "end": END,
            "generation": "table_selection_agent",
            "expansion": "seed_selection_agent",
        },
    )
    graph.add_edge("table_selection_agent", "generation_agent")
    graph.add_conditional_edges(
        "seed_selection_agent", map_expansion, {"expansion_agent": "expansion_agent", "generation": "table_selection_agent"}
    )
    graph.add_edge("generation_agent", "critical_agent")
    graph.add_edge("expansion_agent", "reduce_expansion")
    graph.add_conditional_edges(
        "reduce_expansion",
        route_and_map_on_expansion_quantity,
        {"continue": "seed_selection_agent", "critical_agent": "critical_agent"},
    )
    graph.add_edge("critical_agent", "reduce_critical")
    graph.add_edge("reduce_critical", "management_agent")

    return graph.compile()
