# batch_predict.py

import pandas as pd
import json
import pickle
import os
import logging
from src.preprocess import preprocess_data

ARTIFACT_DIR = "artifacts"
MODEL_PATH = f"{ARTIFACT_DIR}/final_model.pkl"
SCALER_PATH = f"{ARTIFACT_DIR}/scaler.pkl"
FEATURE_COLS_PATH = f"{ARTIFACT_DIR}/feature_columns.json"
THRESHOLD_PATH = f"{ARTIFACT_DIR}/best_threshold.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]
    return model, scaler, feature_cols, threshold

def predict_batch(csv_path, output_path="predictions.csv"):
    logging.info(f"Loading input data from {csv_path}")
    df = pd.read_csv(csv_path)
    model, scaler, feature_cols, threshold = load_artifacts()

    X_scaled, _, _ = preprocess_data(df, scaler=scaler, fit_scaler=False)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    df['Churn_Probability'] = probabilities
    df['Churn_Prediction'] = predictions

    df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    predict_batch("new_customers.csv", "batch_predictions.csv")
