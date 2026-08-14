#FastAPI is a Python class that gives us the tools to create an API.
from fastapi import FastAPI

#Pydantic is a Python library that helps us check and organize incoming data.
from pydantic import BaseModel  #Gets the tool for validating input

#It gives us access to information and settings related to the Python environment.
#Helps Python find predict.py
import sys  
import os

# Allow Python to find our predict.py file inside src/
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

#Brings our prediction function into the API
from predict import predict_churn


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API by Amina Fayyaz",
    description="API for predicting customer churn",
    version="1.0.0"
)


# --------------------------------------------------
# 2. Define customer input
# --------------------------------------------------
#Customer is our class and it inherits from BaseModel. It defines the structure of the input data that we expect to receive in our API.
class Customer(BaseModel):

    gender: str  #gender should be in text form and vice versa
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# 3. Home endpoint
# --------------------------------------------------

@app.get("/") #HTTP method = GET and URL = /
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


# --------------------------------------------------
# 4. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")  #HTTP method = POST and URL = /predict
def predict(customer: Customer):  #  customer : ustomer->Tells FastAPI to validate incoming customer data

    # Convert Pydantic object into dictionary
    #customer is currently a Pydantic object.
    #Our predict_churn() function expects a normal Python dictionary.
    #model)dump() method converts the Pydantic object into a dictionary.
    customer_data = customer.model_dump()

    # Make prediction using our ML model
    probability, prediction = predict_churn(customer_data)  #Sends data to our ML model

    if prediction == 1:
        result = "Customer is likely to churn"
    else:
        result = "Customer is likely to stay"

    return { #Sends the prediction back as JSON
        "prediction": result,
        "churn_probability": round(probability, 4),
        "threshold": 0.20
    }