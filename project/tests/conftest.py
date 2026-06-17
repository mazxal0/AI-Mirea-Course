import pytest
from fastapi.testclient import TestClient
from src.service import app
from src.config import config
from src.models import PredictResponse


@pytest.fixture(scope="session")
def client():
    """Test client FastApi"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_same_pair():
    return {
        "text_a": "Как научиться программировать на Python?",
        "text_b": "С чего начать изучение языка Python?",
    }


@pytest.fixture
def sample_different_pair():
    return {"text_a": "Сегодня отличная погода", "text_b": "Нейросети покоряют мир"}


@pytest.fixture
def sample_invalid_pair():
    return {"text_a": "только один текст"}
