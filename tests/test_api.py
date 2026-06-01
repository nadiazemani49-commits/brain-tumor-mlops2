from fastapi.testclient import TestClient
import sys
sys.path.insert(0, '.')
from src.api import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Brain Tumor" in r.json()["message"]

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
