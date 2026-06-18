from fastapi.testclient import TestClient
from main import app
import json
import pytest

# Initialize a FastAPI client
client = TestClient(app)

# Test your root endpoint
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello stranger! This API allow you to evaluate the quality of red wine. Go to the /docs for more details."}

# Test your predict endpoint
def test_predict():
    response = client.post(
        "/predict",
        headers={'accept': 'application/json', 'Content-Type': 'application/json'},
        json={'alcohol': 9.4, 'volatile_acidity': 0.7},
    )
    assert response.status_code == 200
    result = json.loads(response.json())
    assert result['prediction'] == 0
    assert result['probability'] == pytest.approx([0.7759, 0.2241], abs=1e-3)