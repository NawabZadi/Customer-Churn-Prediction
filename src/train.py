import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocessor

#inside the churn_analysis.ipynb I have compared two models one is Xgboost and the other is Logistic regression and I have found that Logistic regression is performing better than Xgboost so I have used Logistic regression in this file.

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 2. Prepare data
# --------------------------------------------------

# Remove customer ID because it is only an identifier
# and does not help the model predict churn.

df = df.drop(columns=["customerID"])


# Convert TotalCharges from text to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Remove rows where TotalCharges is missing
df = df.dropna(subset=["TotalCharges"])


# Convert target:
# No  -> 0
# Yes -> 1

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})


# --------------------------------------------------
# 3. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Create Logistic Regression pipeline
# --------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

model_pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 6. Save trained model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model_pipeline,
    "models/churn_model.pkl"
)


print("Model trained successfully.")
print("Model saved to models/churn_model.pkl")