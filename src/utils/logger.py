import sys
import logging
import datetime
import os
from config import LoggingConfig

# 确保日志目录存在
log_dir = f"/home/hejiahao/RuntimeCost/logs/{datetime.datetime.now().strftime('%Y-%m-%d')}"
os.makedirs(log_dir, exist_ok=True)

# 生成日志文件路径
log_file = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%H-%M-%S')}.log")

# 创建 formatter
formatter = logging.Formatter(fmt="%(asctime)s [%(name)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# 创建文件 handler
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)  # 确保 DEBUG 级别的日志可以写入文件

# 创建控制台 handler
screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
screen_handler.setLevel(logging.INFO)

# 统一 logger 设置
def setup_custom_logger(name):
    custom_logger = logging.getLogger(name)
    if not custom_logger.handlers:  # 避免重复添加 handler
        custom_logger.setLevel(LoggingConfig.log_level)  # 设置 logger 级别，确保 INFO 及以下日志能进入文件
        custom_logger.addHandler(file_handler)
        custom_logger.addHandler(screen_handler)
    return custom_logger