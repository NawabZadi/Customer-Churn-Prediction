import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 2. Prepare data
# --------------------------------------------------

df = df.drop(columns=["customerID"])

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})


# --------------------------------------------------
# 3. Create the same final train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Load trained model
# --------------------------------------------------

model = joblib.load("models/churn_model.pkl")


# --------------------------------------------------
# 5. Get churn probabilities
# --------------------------------------------------

y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 6. Load chosen threshold
# --------------------------------------------------

threshold = joblib.load("models/threshold.pkl")


# Convert probabilities into predictions
y_pred = (y_prob >= threshold).astype(int)


# --------------------------------------------------
# 7. Calculate evaluation metrics
# --------------------------------------------------

roc_auc = roc_auc_score(y_test, y_prob)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("Model Evaluation")
print("----------------")

print(f"Threshold: {threshold}")

print(f"ROC-AUC: {roc_auc:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall: {recall:.4f}")

print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)