import pytest
from src.models import PredictResponse


def test_predict_valid_response_structure(client, sample_same_pair):
    response = client.post("/api/v1/predict", json=sample_same_pair)
    assert response.status_code == 200
    PredictResponse(**response.json())


@pytest.mark.parametrize(
    "pair,expected_label",
    [
        ({"text_a": "Привет", "text_b": "Привет"}, 1),
        ({"text_a": "Кошка любит мышь", "text_b": "Кошка любит мышь"}, 1),
        ({"text_a": "Сегодня хорошая погода", "text_b": "Завтра будет дождь"}, 0),
    ],
)
def test_predict_known_labels(client, pair, expected_label):
    response = client.post("/api/v1/predict", json=pair)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == expected_label
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_empty_strings(client):
    response = client.post("/api/v1/predict", json={"text_a": "", "text_b": ""})

    assert response.status_code in (200, 422)


def test_predict_long_texts(client):
    long_text = "слово" * 1000
    response = client.post(
        "/api/v1/predict", json={"text_a": long_text, "text_b": long_text}
    )

    assert response.status_code in (200, 413, 422)
