import os

CWD = os.path.dirname(os.path.dirname(__file__))

data_dir = os.path.join(CWD, "data")
database_path = {
    "bird": os.path.join(data_dir, "bird", "train_databases"),
    "bird-dev": os.path.join(data_dir, "bird", "dev_databases"),
    "spider": os.path.join(data_dir, "spider", "database"),
    "spider-dev": os.path.join(data_dir, "spider", "database"),
}

if os.environ.get("OPENAI_API_KEY") is None:
    os.environ["OPENAI_API_KEY"] = "???"

sql_embedding_dim = 1024
GENERATE_EPOCHS = 100
EXPAND_PARALLEL_NUM = 3
GENERATE_TABLE_MAX_NUM = 8
EXPAND_TABLE_MAX_NUM = EXPAND_PARALLEL_NUM
EXPAND_SAMPLE_NUM = 3
EVAL_COUNT = 90

pg_config = {
    "host": "localhost",
    "dbname": "imdb",
    "user": "xxx",
    "password": "xxx",
    "timeout": 5000,
    "port": 5432,
}
