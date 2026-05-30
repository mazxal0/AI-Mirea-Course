import os
from dotenv import load_dotenv


class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "./artifacts/cross_encoder_finetuned")
    LOGGER_PATH = os.getenv("LOGGER_PATH", "./logs/app.log")
    DEVICE = os.getenv("DEVICE", "cpu")
    THRESHOLD = float(os.getenv("THRESHOLD", 0.55))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


config = Config()
