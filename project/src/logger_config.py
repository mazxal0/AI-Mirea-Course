import logging
import sys
from src.config import config


def setup_logger(name: str) -> logging.Logger:
    """Set settings and return logger with name"""

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(
        config.LOGGER_PATH or "./logs/app.log", encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
