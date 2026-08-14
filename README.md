## Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn. The project includes data preprocessing, model training, evaluation, threshold selection based on business cost, a FastAPI prediction API, automated tests, and Docker deployment.

## Project Overview

Customer churn prediction helps a company identify customers who are likely to leave its service.

This project uses the Telco Customer Churn dataset and compares:

---Logistic Regression
---XGBoost

The final prediction system uses a trained machine learning pipeline containing preprocessing and the selected model.

Machine Learning Pipeline

The project uses a Scikit-learn Pipeline and ColumnTransformer.

The pipeline performs:

Numerical feature scaling using StandardScaler
Categorical feature encoding using OneHotEncoder
Prediction using the trained machine learning model

This allows raw customer information to be passed directly to the model without manually preprocessing it during prediction.

## Model Evaluation

The model was evaluated using:

ROC-AUC
Precision
Recall
F1 Score
Confusion Matrix
Business cost

A prediction threshold of 0.20 was selected because missing a customer who is actually going to churn was assigned a much higher business cost than incorrectly targeting a customer who would stay.

## Final evaluation:

Metric	Score
Threshold	  0.20
ROC-AUC	      0.8359
Precision	  0.4577
Recall	      0.8690
F1 Score	  0.5996

## Confusion Matrix:

[[648 385]
 [ 49 325]]


## Project Structure

Customer-Churn-Prediction/
│
├── app/
│   └── main.py
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── threshold.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md

## API

The trained model is exposed through a FastAPI application.

## Prerequisites

Before running the Dockerized application, make sure you have:

- Python 3.12+
- Docker Desktop
- Git

## Docker

Make sure Docker Desktop is installed and running.

Build and start the application:

```bash
docker compose up --build

## Start the API locally
uvicorn app.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Interactive Swagger documentation:

http://127.0.0.1:8000/docs
Prediction Endpoint
POST /predict

The endpoint accepts customer information and returns:

{
  "prediction": "Customer is likely to churn",
  "churn_probability": 0.6227,
  "threshold": 0.2
}

"Using a 0.20 threshold, the model identifies 325 actual churners while missing 49 churners, achieving 86.9% recall. The lower threshold prioritizes identifying potential churners because the assumed cost of missing a churner is substantially higher than the cost of targeting a customer who would otherwise stay."

## Testing

The API contains automated tests using pytest.

Run:

python -m pytest

Current tests cover:

Home endpoint
Prediction endpoint
API response structure
Churn probability range
Prediction threshold
Docker

The project includes Docker configuration for containerized deployment.

Build and run with Docker Compose
docker compose up --build

The API can then be accessed at:

http://127.0.0.1:8000/docs

## Technologies

Python
Pandas
NumPy
Scikit-learn
XGBoost
FastAPI
Pydantic
Uvicorn
Pytest
Docker
GitHub Actions
Important Files
churn_analysis.ipynb

The notebook contains the exploratory data analysis, feature preparation, model experimentation, evaluation, threshold analysis, and model development process.

src/train.py

Responsible for training the machine learning model.

src/evaluate.py

Responsible for evaluating the trained model and calculating performance metrics.

src/predict.py

Loads the trained model and threshold and performs predictions for new customers.

app/main.py

Creates the FastAPI application and exposes the prediction endpoint.

models/churn_model.pkl

Contains the trained machine learning pipeline.

models/threshold.pkl

Contains the selected prediction threshold.

tests/test_api.py

Contains automated tests for the API.

Dockerfile

Contains instructions for building the Docker image.

docker-compose.yml

Defines how the Docker container should be built and run.

.github/workflows/ci.yml

Runs automated tests through GitHub Actions.

## Workflow

Customer Data
     ↓
Data Preprocessing
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Threshold Selection
     ↓
Save Model + Threshold
     ↓
FastAPI Application
     ↓
Docker Container
     ↓
Prediction API