import operator
from typing import Annotated, Dict, List, Literal

from typing_extensions import TypedDict
from langgraph.prebuilt.chat_agent_executor import AgentState

from util.datapool import DataPool


class State(TypedDict):
    # user_input: str
    benchmark: str
    target_sql_num: int
    data_pool: DataPool
    
    epoch: int
    prev_strategy: Literal["generation", "expansion"]
    next_strategy: Literal["generation", "expansion", "end"]

    selected_database: str
    selected_tables: List[str]
    selected_seed_queries: List[str] | List[List[str]]
    each_synthesized_queries: List[str]
    synthesized_queries: Annotated[List[List[str]], operator.add]
    critical_score: Annotated[List[dict], operator.add]
    
class ReActState(State, AgentState):
    pass

# class State(TypedDict):
#     data_pool: DataPool
#     target_num: int
#     benchmark: str
#     database: str
#     skip_train: bool
#     table_list: List[str]  # selected table
#     table_dict: Dict[str, int]  # key is benchmark.database.table, indicate the number of queries have generated for each table
#     database_dict: Dict[str, int]  # key is benchmark.database, indicate the number of queries have generated for each database
#     epoch: int  # epoch for generator
#     prev_strategy: Literal["generator", "expander"]
#     next_strategy: Literal["generator", "expander"]
#     sampled_query: List[str]  # sampled query for expander
#     expanded_query: Annotated[List[str], operator.add]  # query from expander
