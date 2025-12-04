import os, signal, tempfile
from utils.logger import setup_custom_logger
from config import TestConfig, ServerConfig
import json
import numpy as np
import re
import subprocess
import time
import random

random.seed(3407)

logger = setup_custom_logger("TEST")

def execute(query_path):
    cur_time = int(round(time.time() * 1000))
    if TestConfig.explain_only:
        with open(query_path, 'r') as f:
            original_sql = f.read().strip()
        explain_sql = f"EXPLAIN EXTENDED {original_sql}"
        # 写入临时文件（避免覆盖原 SQL）
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sql", prefix="explain_")
        with os.fdopen(tmp_fd, "w") as tmp_file:
            tmp_file.write(explain_sql)
        sql_path = tmp_path
        logger.debug(f"[Explain Mode] Generated temp file: {sql_path}")
    else:
        sql_path = query_path
    cmd = f'''
        spark-sql \
        --database {TestConfig.database} \
        -f {sql_path} \
        --name {cur_time}-{query_path.split('/')[-1]} \
        --properties-file {TestConfig.conf_path} \
    '''
    elapsed_time_usecs = -1
    for i in range(TestConfig.max_retry):
        try:
            # 启动子进程执行命令
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                start_new_session=True,
                universal_newlines=True)
            output,_ = process.communicate(timeout=TestConfig.timeout)
            logger.debug(output)
            m = re.search("Time taken: (\d*.\d*) seconds", output)
            if m is None:
                logger.warning('Execution finished but no elapsed time found, try again...')
                elapsed_time_usecs = TestConfig.timeout
            else:
                elapsed_time_usecs = eval(m.group(1))
                break
        except Exception as e:
            logger.warning(f'Execution Timeout.')
            os.killpg(process.pid, signal.SIGKILL)
            elapsed_time_usecs = TestConfig.timeout
            break
    return elapsed_time_usecs

def test(f_list):
    data = {}
    time_sum_list = []
    for sql in f_list:
        data[sql] = []
    for i in range(TestConfig.repeats):
        time_sum = 0
        for f_name in f_list:
            logger.info(f'Running {f_name}...')
            f_path = os.path.join(f'{TestConfig.benchmark_path}', f_name)
            t = execute(f_path)
            if t == -1:
                logger.error(f'Execution failed for {f_name}, skipping this one...')
                continue
            data[f_name].append(t)
            logger.info(f'{i + 1}th execution time of {f_name}: {t}s')
            time_sum += t
        logger.info(f'{i + 1}th execution time: {time_sum}s')
        time_sum_list.append(time_sum)
    mean = np.mean(time_sum_list)
    std = np.std(time_sum_list)
    logger.info(f'Mean: {mean}, Std: {std}')
    for k, v in data.items():
        data[k] = sum(data[k]) / len(data[k])
    return data

def save(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        logger.info(f'Saved data to {save_path}')

org_f_list = sorted(os.listdir(TestConfig.benchmark_path))
logger.info('Found the following SQL files: %s', org_f_list)
if TestConfig.explain_only:
    logger.info("Running on EXPLAIN only mode")
org_data = test(org_f_list)
if TestConfig.save_latency:
    save(org_data, TestConfig.save_path)
