"""
test_api.py — Tests unitaires de l'API Brain Tumor
Nadia Zemani — PFA 2024/2025
"""
import pytest
import numpy as np
import io
import sys
import os
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from api import app

client = TestClient(app)

# ── Fixture : image IRM de test ───────────────────────────────────────────────
@pytest.fixture
def test_image():
    """Crée une image JPEG valide en mémoire."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8), 'RGB'
    )
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

# ── Tests /health ─────────────────────────────────────────────────────────────
def test_health_status_200():
    r = client.get("/health")
    assert r.status_code == 200

def test_health_returns_model_name():
    r = client.get("/health")
    assert r.json()["model"] == "EfficientNet-B3"

def test_health_returns_classes():
    r = client.get("/health")
    data = r.json()
    assert "classes" in data
    assert len(data["classes"]) == 4

def test_health_contains_glioma():
    r = client.get("/health")
    assert "glioma" in r.json()["classes"]

# ── Tests /predict ────────────────────────────────────────────────────────────
def test_predict_status_200(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    assert r.status_code == 200

def test_predict_returns_prediction(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    assert "prediction" in r.json()

def test_predict_valid_class(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    assert r.json()["prediction"] in ["glioma", "meningioma", "notumor", "pituitary"]

def test_predict_confidence_range(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    conf = r.json()["confidence"]
    assert 0.0 <= conf <= 1.0

def test_predict_uncertainty_range(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    unc = r.json()["uncertainty"]
    assert 0.0 <= unc <= 1.0

def test_predict_returns_probabilities(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    assert "probabilities" in r.json()

def test_predict_probabilities_sum_to_one(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    probs = r.json()["probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01

def test_predict_returns_severity(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    assert "severity" in r.json()

def test_predict_returns_author(test_image):
    r = client.post("/predict", files={"file": ("mri.jpg", test_image, "image/jpeg")})
    assert "Nadia Zemani" in r.json()["author"]

def test_predict_no_file_returns_error():
    r = client.post("/predict")
    assert r.status_code == 422

# ── Tests /metrics ────────────────────────────────────────────────────────────
def test_metrics_status_200():
    r = client.get("/metrics")
    assert r.status_code == 200

def test_metrics_contains_predictions_total():
    r = client.get("/metrics")
    assert "predictions_total" in r.text

def test_metrics_contains_latency():
    r = client.get("/metrics")
    assert "prediction_latency_seconds" in r.text

def test_metrics_contains_confidence():
    r = client.get("/metrics")
    assert "prediction_confidence_last" in r.text

# ── Tests /monitoring ─────────────────────────────────────────────────────────
def test_monitoring_status_200():
    r = client.get("/monitoring")
    assert r.status_code == 200

def test_monitoring_returns_total():
    r = client.get("/monitoring")
    data = r.json()
    assert "total_predictions" in data or "error" in data

def test_monitoring_returns_confidence():
    r = client.get("/monitoring")
    data = r.json()
    assert "avg_confidence" in data or "error" in data

def test_monitoring_drift_field():
    r = client.get("/monitoring")
    data = r.json()
    assert "drift_detected" in data or "error" in data

# ── Tests /model/info ─────────────────────────────────────────────────────────
def test_model_info_status_200():
    r = client.get("/model/info")
    assert r.status_code == 200

def test_model_info_returns_model_name():
    r = client.get("/model/info")
    assert "model" in r.json()

def test_model_info_returns_version():
    r = client.get("/model/info")
    assert "version" in r.json()
