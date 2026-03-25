from datetime import datetime
import logging

PREFIX = "/home/hejiahao/Calibra"

class ServerConfig:
    data_path = f"{PREFIX}/data/STACK/STACK_40it.pt"
    save_error = True

class TestConfig:
    def __init__(self, method, database, benchmark, max_retry, repeats, save_latency) -> None:
        self.method = method
        self.database = database
        self.benchmark = benchmark
        self.timeout = 300
        self.max_retry = max_retry
        self.repeats = repeats
        self.explain_only = True if method == 'bootstrap' else False    
        self.conf_path = f'{PREFIX}/conf/{method}.conf'
        self.benchmark_path = f'{PREFIX}/benchmark/{benchmark}/'
        self.save_latency = save_latency
        self.save_path = f'{PREFIX}/results/{benchmark}_40its.json'

class TrainConfig:
    inference_only = False
    patience = 10
    epochs = 100
    batch_size = 64
    bootstrap_sample_size = 2000
    bce_loss_weight = 10
    save_bootstrap_samples = False
    bootstrap_samples_save_path = f'{PREFIX}/data/JOB_l4_bs_{bootstrap_sample_size}.pt'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_save_path = f'{PREFIX}/results/STACK_40its'
    model_save_path = f'{PREFIX}/data/STACK/models/STACK_40its.pt'
    # bs_model_save_path = f'{PREFIX}/data/l4k4rm3/models/JOB_10its.pt'
    bs_model_save_path = f''
    

class EnvironmentConfig:
    database = "stack"
    table_file = f'{PREFIX}/benchmark/{database}_tables.csv'
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
