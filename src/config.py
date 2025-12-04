from datetime import datetime
import logging

PREFIX = "/home/hejiahao/Calibra"

class ServerConfig:
    data_path = f"{PREFIX}/data/JOB_l4_40it.pt"

class TestConfig:
    method = "online_train"
    explain_only = False
    database = "imdb_10x"
    benchmark = "JOB"
    timeout = 300
    max_retry = 1
    repeats = 10
    conf_path = f'{PREFIX}/conf/{method}.conf'
    benchmark_path = f'{PREFIX}/benchmark/{benchmark}/'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_latency = False
    # save_path = f'{PREFIX}/results/{benchmark}_{current_time}'
    save_path = f'{PREFIX}/results/JOB_l4_40it.json'

class TrainConfig:
    inference_only = False
    patience = 10
    epochs = 100
    batch_size = 64
    bootstrap_sample_size = 2000
    bce_loss_weight = 10
    save_bootstrap_samples = True
    bootstrap_samples_save_path = f'{PREFIX}/data/{TestConfig.benchmark}_l4_bs_{bootstrap_sample_size}.pt'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_save_path = f'{PREFIX}/results/{TestConfig.benchmark}_l4_30it'
    # model_save_path = f'{PREFIX}/data/models/JOB_20251121_132234.pt'
    model_save_path = f'{PREFIX}/data/models/JOB_l4_40it.pt'
    bs_model_save_path = f'/home/hejiahao/Calibra/data/models/JOB_l4_30it.pt'

class EnvironmentConfig:
    database = "imdb"
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
