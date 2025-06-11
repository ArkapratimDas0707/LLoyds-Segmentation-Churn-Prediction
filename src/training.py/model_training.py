# train_model.py

import logging
import time
import json
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_recall_curve, classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data  # Reuse preprocessing function

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def train_and_evaluate(input_csv: str):
    start_time = time.time()
    logging.info("Starting training pipeline...")

    # Step 1: Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(input_csv)

    # Step 2: Resample using SMOTETomek
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    logging.info(f"After resampling: {X_res.shape}, Target dist: {np.bincount(y_res)}")

    # Step 3: Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid=param_grid,
        scoring='average_precision',
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_res, y_res)
    best_rf = grid.best_estimator_
    logging.info(f"Best params: {grid.best_params_}")

    # Step 4: Calibration
    calibrator = CalibratedClassifierCV(estimator=best_rf, method='sigmoid', cv=3)
    calibrator.fit(X_res, y_res)

    # Step 5: Probability prediction
    y_proba = calibrator.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Step 6: Find threshold (recall ≥ 0.80 + best F1)
    best_thresh = 0.5
    best_f1 = 0
    for p, r, t in zip(precision, recall, thresholds):
        if r >= 0.80:
            f1 = 2 * p * r / (p + r + 1e-10)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t

    logging.info(f"Chosen threshold: {best_thresh:.2f} | Recall ≥ 0.80 | F1 = {best_f1:.3f}")

    # Step 7: Final prediction
    y_pred = (y_proba >= best_thresh).astype(int)

    # Step 8: Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info("Classification report:")
    logging.info(json.dumps(report, indent=2))

    # Step 9: Save model, threshold, metrics
    with open(os.path.join(ARTIFACT_DIR, "model.pkl"), "wb") as f:
        pickle.dump(calibrator, f)
    with open(os.path.join(ARTIFACT_DIR, "threshold.json"), "w") as f:
        json.dump({"threshold": float(best_thresh)}, f)
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "best_params": grid.best_params_,
            "chosen_threshold": float(best_thresh),
            "target_recall_f1": best_f1,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": report
        }, f, indent=2)

    elapsed = time.time() - start_time
    logging.info(f"Training pipeline completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    train_and_evaluate("cleaned_customer_data.csv")
