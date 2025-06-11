import pandas as pd
import logging
import time
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def preprocess_data(input_df: pd.DataFrame, scaler=None, fit_scaler=True):
    start_time = time.time()
    logging.info(f"Starting preprocessing for dataframe with shape: {input_df.shape}")

    if 'ChurnStatus' not in input_df.columns:
        raise ValueError("Target column 'ChurnStatus' not found in the dataset.")

    X = input_df.drop(columns=['ChurnStatus'])
    y = input_df['ChurnStatus']

    ordinal_cols = ['IncomeLevelEncoded']
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.difference(ordinal_cols)
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    if not fit_scaler:
        with open(os.path.join(ARTIFACT_DIR, "feature_columns.json")) as f:
            expected_cols = json.load(f)
        for col in expected_cols:
            if col not in X_encoded:
                X_encoded[col] = 0
        X_encoded = X_encoded[expected_cols]

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        with open(os.path.join(ARTIFACT_DIR, "feature_columns.json"), "w") as f:
            json.dump(X_encoded.columns.tolist(), f, indent=2)
        with open(os.path.join(ARTIFACT_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        logging.info("Saved feature columns and scaler.")
    else:
        X_scaled = scaler.transform(X_encoded)

    elapsed = time.time() - start_time
    logging.info(f"Preprocessing completed in {elapsed:.2f} seconds.")

    return X_scaled, y, scaler

if __name__ == "__main__":
    df = pd.read_csv("cleaned_customer_data.csv")
    X_train_scaled, y_train, scaler = preprocess_data(df, fit_scaler=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")