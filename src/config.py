from datetime import datetime
import logging

class ServerConfig:
    mode = "collect"
    collect_strategy = "always_first"

class TestConfig:
    method = "runtime_cost"
    database = "imdb_10x"
    benchmark = "test_17a"
    timeout = 300
    max_retry = 3
    repeats = 1
    conf_path = f'/home/hejiahao/RuntimeCost/conf/{method}.conf/'
    benchmark_path = f'/home/hejiahao/AQORA/benchmarks/{benchmark}/'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'/home/hejiahao/RuntimeCost/data/{benchmark}_{method}_{current_time}'


class EnvironmentConfig:
    database = "imdb"
    table_file = f'/home/hejiahao/AQORA/benchmarks/{database}_tables.csv'
    if database == "tpch":
        table_num = 8
        col_num = 16
    if database == "imdb":
        table_num = 17
        col_num = 34
    if database == "stack":
        table_num = 11
        col_num = 25
    if database == "tpcds":
        table_num = 25

class LoggingConfig:
    log_level = logging.INFO