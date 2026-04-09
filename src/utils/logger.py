import datetime
import logging
import os
import re
import sys

from config import DEFAULT_BENCHMARK, PROJECT_ROOT, LoggingConfig, ensure_dir, get_run_artifacts


LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RUN_LOG_TARGETS = {
    "offline_train_server.py": "offline_train_server",
    "test_server.py": "test_server",
    "train.py": "train",
    "bootstrap.py": "bootstrap",
    "bootstrap_server.py": "bootstrap_server",
}


def _arg_value(flag):
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _resolve_stage_name():
    script_name = os.path.basename(sys.argv[0])
    if script_name == "test.py":
        stage_name = _arg_value("--method") or "test"
    else:
        stage_name = RUN_LOG_TARGETS.get(script_name, os.path.splitext(script_name)[0] or "session")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stage_name)


def _resolve_run_log_path():
    run_id = os.getenv("CALIBRA_RUN_ID") or _arg_value("--run-id")
    if not run_id:
        return None

    benchmark = os.getenv("CALIBRA_BENCHMARK") or _arg_value("--benchmark") or DEFAULT_BENCHMARK
    try:
        artifacts = get_run_artifacts(benchmark, run_id)
    except ValueError:
        return None

    ensure_dir(str(artifacts.log_dir))
    return artifacts.log_artifact_path(f"{_resolve_stage_name()}.log")


def _resolve_fallback_log_path():
    log_dir = ensure_dir(str(PROJECT_ROOT / "logs" / datetime.datetime.now().strftime("%Y-%m-%d")))
    return os.path.join(log_dir, f"{datetime.datetime.now().strftime('%H-%M-%S')}.log")


def _resolve_log_path():
    return _resolve_run_log_path() or _resolve_fallback_log_path()


def setup_custom_logger(name):
    custom_logger = logging.getLogger(name)
    if getattr(custom_logger, "_calibra_configured", False):
        return custom_logger

    log_file = _resolve_log_path()
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    screen_handler.setLevel(LoggingConfig.log_level)

    custom_logger.setLevel(LoggingConfig.log_level)
    custom_logger.addHandler(file_handler)
    custom_logger.addHandler(screen_handler)
    custom_logger.propagate = False
    custom_logger.log_path = log_file
    custom_logger._calibra_configured = True
    return custom_logger
