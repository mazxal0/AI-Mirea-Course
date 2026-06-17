from src.config import config


def test_check_types_of_env_variables():
    assert isinstance(config.DEVICE, str)
    assert isinstance(config.LOG_LEVEL, str)
    assert isinstance(config.THRESHOLD, float)
    assert isinstance(config.MODEL_PATH, str)
    assert isinstance(config.LOGGER_PATH, str)
