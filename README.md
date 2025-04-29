# SQL-Factory
Code for 'SQL-Factory: A Multi-Agent Framework for High-Quality and Large-Scale SQL Generation'


# Implementation of SQL-Factory using LangGraph
```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        entry(entry)
        management_agent(management_agent)
        critical_agent(critical_agent)
        table_selection_agent(table_selection_agent)
        generation_agent(generation_agent)
        seed_selection_agent(seed_selection_agent)
        expansion_agent(expansion_agent)
        reduce_expansion(reduce_expansion)
        reduce_critical(reduce_critical)
        __end__([<p>__end__</p>]):::last
        __start__ --> entry;
        critical_agent --> reduce_critical;
        entry --> management_agent;
        expansion_agent --> reduce_expansion;
        generation_agent --> critical_agent;
        management_agent -. &nbsp;expansion&nbsp; .-> seed_selection_agent;
        management_agent -. &nbsp;generation&nbsp; .-> table_selection_agent;
        reduce_critical --> management_agent;
        reduce_expansion -.-> critical_agent;
        reduce_expansion -. &nbsp;continue&nbsp; .-> seed_selection_agent;
        seed_selection_agent -.-> expansion_agent;
        seed_selection_agent -. &nbsp;generation&nbsp; .-> table_selection_agent;
        table_selection_agent --> generation_agent;
        management_agent -.-> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
