import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import time

import numpy as np

from config import DEFAULT_BENCHMARK, TestConfig, ensure_dir, ensure_parent_dir, update_manifest
from utils.logger import setup_custom_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="leap_train")
    parser.add_argument("--database", default="stack")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--save-latency", action="store_true", default=False)
    parser.add_argument("--save-path")
    parser.add_argument("--conf-path")
    parser.add_argument("--run-id")
    parser.add_argument("--sqldir")
    return parser.parse_args()


random.seed(3407)
args = parse_args()
logger = setup_custom_logger("TEST")
cfg = TestConfig(**vars(args))


def stage_log_path():
    return getattr(logger, "log_path", cfg.artifacts.log_artifact_path(f"{cfg.method}.log"))


def prepare_run_conf():
    conf_name = f"{cfg.method}.conf"
    run_conf_path = cfg.artifacts.conf_artifact_path(conf_name)
    ensure_dir(str(cfg.artifacts.conf_dir))
    ensure_parent_dir(run_conf_path)
    if os.path.abspath(cfg.conf_path) != os.path.abspath(run_conf_path):
        shutil.copyfile(cfg.conf_path, run_conf_path)
    return run_conf_path


cfg.conf_path = prepare_run_conf()

update_manifest(
    cfg.artifacts.manifest_path,
    {
        **cfg.artifacts.manifest_defaults(),
        "evaluation": {
            cfg.method: {
                "method": cfg.method,
                "database": cfg.database,
                "benchmark": cfg.benchmark,
                "run_id": cfg.run_id,
                "max_retry": cfg.max_retry,
                "repeats": cfg.repeats,
                "sqldir": cfg.benchmark_path,
                "conf_path": cfg.conf_path,
                "log_path": stage_log_path(),
            },
        },
    },
)


def execute(query_path):
    cur_time = int(round(time.time() * 1000))
    tmp_path = None

    if cfg.explain_only:
        with open(query_path, "r") as f:
            original_sql = f.read().strip()
        explain_sql = f"EXPLAIN EXTENDED {original_sql}"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sql", prefix="explain_")
        with os.fdopen(tmp_fd, "w") as tmp_file:
            tmp_file.write(explain_sql)
        sql_path = tmp_path
        logger.debug(f"[Explain Mode] Generated temp file: {sql_path}")
    else:
        sql_path = query_path

    cmd = [
        "spark-sql",
        "--database",
        cfg.database,
        "-f",
        sql_path,
        "--name",
        f"{cur_time}-{os.path.basename(query_path)}",
        "--properties-file",
        cfg.conf_path,
    ]
    env = os.environ.copy()

    elapsed_time_usecs = -1
    try:
        for _ in range(cfg.max_retry):
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    universal_newlines=True,
                    cwd=cfg.working_dir,
                    env=env,
                )
                output, _ = process.communicate(timeout=cfg.timeout)
                logger.debug(output)
                m = re.search(r"Time taken: (\d*.\d*) seconds", output)
                if m is None:
                    logger.warning("Execution finished but no elapsed time found, try again...")
                    elapsed_time_usecs = cfg.timeout
                else:
                    elapsed_time_usecs = float(m.group(1))
                    break
            except Exception:
                logger.warning("Execution Timeout.")
                os.killpg(process.pid, signal.SIGKILL)
                elapsed_time_usecs = cfg.timeout
                break
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return elapsed_time_usecs


def test(f_list):
    data = {}
    time_sum_list = []
    for sql in f_list:
        data[sql] = []
    for i in range(cfg.repeats):
        time_sum = 0
        for f_name in f_list:
            logger.info(f"Running {f_name}...")
            f_path = os.path.join(f"{cfg.benchmark_path}", f_name)
            t = execute(f_path)
            if t == -1:
                logger.error(f"Execution failed for {f_name}, skipping this one...")
                continue
            data[f_name].append(t)
            logger.info(f"{i + 1}th execution time of {f_name}: {t}s")
            time_sum += t
        logger.info(f"{i + 1}th execution time: {time_sum}s")
        time_sum_list.append(time_sum)
    mean = np.mean(time_sum_list)
    std = np.std(time_sum_list)
    logger.info(f"Mean: {mean}, Std: {std}")
    for k, v in data.items():
        data[k] = sum(v) / len(v)
    return data


def save(data, save_path):
    ensure_parent_dir(save_path)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
        logger.info(f"Saved data to {save_path}")


if __name__ == "__main__":
    org_f_list = sorted(
        f_name for f_name in os.listdir(cfg.benchmark_path) if f_name.endswith('.sql')
    )
    logger.info("Found the following SQL files in %s: %s", cfg.benchmark_path, org_f_list)
    if cfg.explain_only:
        logger.info("Running on EXPLAIN only mode")
    org_data = test(org_f_list)
    evaluation_updates = {
        "log_path": stage_log_path(),
        "conf_path": cfg.conf_path,
    }
    if cfg.save_latency:
        save(org_data, cfg.save_path)
        evaluation_updates.update({
            "save_path": cfg.save_path,
            "query_count": len(org_data),
        })

    update_manifest(
        cfg.artifacts.manifest_path,
        {
            "evaluation": {
                cfg.method: evaluation_updates,
            },
        },
    )
