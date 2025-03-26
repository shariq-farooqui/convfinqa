import os
from convfinqa.loggers import get_singleton_logger

file_path = "/app/logs" if os.path.isdir("/app/logs") else "."
logger = get_singleton_logger(name=__name__, log_file_path=file_path)
