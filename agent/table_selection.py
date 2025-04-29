import os
import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph.prebuilt import create_react_agent

from agent.state import ReActState
from agent.tools import database_queries_statistics, table_queries_statistics
from config import GENERATE_TABLE_MAX_NUM, llm_config
from util.benchmark import *

llm = ChatOpenAI(**llm_config["selection"])
prompt = ChatPromptTemplate(
    [
        (
            "user",
            """You are part of a SQL generation team responsible for producing high-quality SQL queries in multiple iterative rounds.

In this round, your task consists of **two steps**:

### Step 1: Select a suitable database

Use the `database_queries_statistics` tool to check the **total number of queries** already generated for each database under the current benchmark.  
Choose a database that has received relatively **fewer queries** so far, in order to help balance the overall query distribution across all databases.

### Step 2: Select {table_count} appropriate tables from the selected database

Once you've chosen the database, use the `table_queries_statistics` tool to get the **query count and schema information** for each table in that database.  
Select {table_count} tables that are either underutilized or structurally diverse to maximize the potential for high-quality and varied query synthesis.

---

Start by providing a brief **analysis** explaining your choice of database and tables.  
Then, return the final selection in the following **structured JSON format** for automatic parsing:

```json
{{
  "database": "<database_name>",
  "tables": ["<table_1>", "<table_2>", "   "]
}}
```

Example:

```json
{{
  "database": "sakila",
  "tables": ["actor", "film", "rental"]
}}
```

### Important tip
You can only call **one tool at a time**, which means there can only be one `Action` and `Action Input` pair in one output.
The Action Input must be a strict JSON object rather than a string.
Example:
```
Action: tool_name
Action Input: {{"param1": "value1", "param2": "value2"}}
```

""",
        )
    ]
)

agent = create_react_agent(
    llm,
    [database_queries_statistics, table_queries_statistics],
    state_schema=ReActState,
)


def table_selection_agent(state):
    prompt_message = prompt.invoke({"table_count": GENERATE_TABLE_MAX_NUM})
    # for s in agent.stream(ReActState(messages=prompt_message.to_messages(), **state)):
    #     message = s["messages"][-1]
    #     if isinstance(message, tuple):
    #         print(message)
    #     else:
    #         print(message)
    #         print("-" * 10)
    s = agent.invoke(ReActState(messages=prompt_message.to_messages(), **state))
    return s["messages"][-1]