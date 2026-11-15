import pytest
from fastapi.testclient import TestClient
from app.main import app, APP_VERSION


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------- /health ----------
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------- /predict: setosa ----------
def test_predict_setosa(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["predicted_class"] == 0
    assert data["predicted_label"] == "setosa"
    assert len(data["probabilities"]) == 3
    assert sum(data["probabilities"]) == pytest.approx(1.0, abs=1e-3)


# ---------- /predict: virginica ----------
def test_predict_virginica(client):
    payload = {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predicted_label"] == "virginica"


def test_predict_invalid_input(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": -1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field(client):
    payload = {"sepal_length": 5.1, "sepal_width": 3.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    
# ---------- /version ----------
def test_version(client):
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": APP_VERSION}