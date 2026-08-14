#Test is created to validate if the api is running and currently
#two tests are working first is to check if the api is alive
#the second test is: does the prediction work on sample data

#Our test passed both examsssss!!!!

#GET /     →  passed
#POST /predict  →  passed

from fastapi.testclient import TestClient

from app.main import app


# Create a test client for our FastAPI application
client = TestClient(app)


# --------------------------------------------------
# 1. Test the home endpoint
# --------------------------------------------------

def test_home():

    response = client.get("/")

    # API should respond successfully
    assert response.status_code == 200

    # Check that the expected message is returned
    assert response.json()["message"] == (
        "Customer Churn Prediction API is running"
    )


# --------------------------------------------------
# 2. Test the prediction endpoint
# --------------------------------------------------

def test_predict():

    # Example customer data
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    response = client.post("/predict", json=customer)

    # API should respond successfully
    assert response.status_code == 200

    # Convert response into a Python dictionary
    result = response.json()

    # Check that the API returned the expected information
    assert "prediction" in result
    assert "churn_probability" in result
    assert "threshold" in result

    # Churn probability should be between 0 and 1
    assert 0 <= result["churn_probability"] <= 1

    # Our threshold should be 0.20
    assert result["threshold"] == 0.20