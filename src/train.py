# src/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow, mlflow.sklearn
import pickle, os
from mlflow.models.signature import infer_signature

# Set MLflow experiment
mlflow.set_experiment("GRC_Risk_Model")

# Ensure relative paths (works on Linux, Windows, CI)
DATA_PATH = os.path.join("data", "processed", "train.csv")
MODEL_DIR = os.path.join("models")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")

# Load data
train = pd.read_csv(DATA_PATH)
X = train[["severity", "likelihood", "impact"]]
y = train["risk_level"]

# Initialize model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Start MLflow run
with mlflow.start_run():
    model.fit(X, y)

    # Log params and metrics
    mlflow.log_param("n_estimators", 10)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    mlflow.log_metric("train_accuracy", acc)

    # Create an input example and infer signature
    input_example = X.iloc[:1]
    signature = infer_signature(X, model.predict(X))

    # Log model with signature and input example
    mlflow.sklearn.log_model(
        sk_model=model,
        name="risk_model",
        input_example=input_example,
        signature=signature
    )

    # Save local copy for DVC or CI artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

print(f" Model trained and saved to {MODEL_PATH}")
