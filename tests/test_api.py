import sys
import os

# add the src folder to python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../src")
    )
)

from app import app
def test_home():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "Running"

def test_prediction():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={
            "study_hours": 7,
            "attendance": 85,
            "previous_marks": 75,
            "assignment_score": 8
        }
    )
    assert response.status_code == 200
    data = response.get_json()

    assert "prediction" in data

    assert data["prediction"] in ["Pass", "Fail"]
    
