import joblib
import pandas as pd


# --------------------------------------------------
# 1. Load trained model and threshold
# --------------------------------------------------

MODEL_PATH = "models/churn_model.pkl"
THRESHOLD_PATH = "models/threshold.pkl"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# --------------------------------------------------
# 2. Function to predict churn
# --------------------------------------------------

def predict_churn(customer_data):
    """
    Predict whether a customer is likely to churn.

    Returns:
        churn_probability: probability that the customer will churn
        prediction: 1 for churn, 0 for no churn
    """

    # Convert dictionary into a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Get probability of churn
    churn_probability = model.predict_proba(customer_df)[0, 1]

    # Apply our chosen threshold
    prediction = int(churn_probability >= threshold)

    return churn_probability, prediction


# --------------------------------------------------
# 3. Example customer
# creating a new customer manually...
# gave the model original values because the pipeline has the preprocessor and it will do all the task..
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Make prediction
# --------------------------------------------------

probability, prediction = predict_churn(customer)


if prediction == 1:
    result = "Customer is likely to churn"
else:
    result = "Customer is unlikely to churn"


print("Prediction:", result)
print("Churn Probability:", probability)
print("Threshold:", threshold)