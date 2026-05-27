import logging
import sys
from pathlib import Path


def setup_logger(name: str, level="INFO", log_file: str = "logs/app.log") -> logging.Logger:

    LOGFILE_PATH = Path(log_file)
    LOGFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(LOGFILE_PATH)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
