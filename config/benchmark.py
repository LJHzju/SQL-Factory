import os
from collections import defaultdict
from config.common import CWD

# ================= benchmark path =================
data_dir = os.path.join(CWD, 'data')

imdb_info_dir = os.path.join(data_dir, "imdb_info")
imdb_schema_save_path = os.path.join(imdb_info_dir, "imdb_schema.json")
imdb_link_save_path = os.path.join(imdb_info_dir, "imdb_ri.csv")

tpcds_info_dir = os.path.join(data_dir, "tpcds_info")
tpcds_schema_save_path = os.path.join(tpcds_info_dir, "tpcds_schema.json")
tpcds_link_save_path = os.path.join(tpcds_info_dir, "tpcds_ri.csv")

tpch_info_dir = os.path.join(data_dir, "tpch_info")
tpch_schema_save_path = os.path.join(tpch_info_dir, "tpch_schema.json")
tpch_link_save_path = os.path.join(tpch_info_dir, "tpch_ri.csv")

bird_info_dir = os.path.join(data_dir, "bird_info")
bird_info_save_path = os.path.join(bird_info_dir, "train-database")

bird_dev_info_dir = bird_info_dir
bird_dev_info_save_path = os.path.join(bird_dev_info_dir, "dev-database")

spider_info_dir = os.path.join(data_dir, "spider_info")
spider_info_save_path = os.path.join(spider_info_dir, "train-database")

spider_dev_info_dir = spider_info_dir
spider_dev_info_save_path = os.path.join(spider_info_dir, "dev-database")

bird_base_dir = os.path.join(data_dir, "bird")
bird_raw_data_path = os.path.join(bird_base_dir, "train_databases")
bird_dev_raw_data_path = os.path.join(bird_base_dir, "dev_databases")

spider_base_dir = os.path.join(data_dir, "spider")
spider_raw_data_path = os.path.join(spider_base_dir, "database")
spider_dev_raw_data_path = os.path.join(spider_base_dir, "database")

