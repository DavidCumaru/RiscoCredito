"""Model training with XGBoost, credit-risk metrics, and MLflow tracking."""

import os
import joblib
import numpy as np
import mlflow
import mlflow.xgboost

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from src.config import MODEL_PATH, ARTIFACTS_DIR, MLFLOW_TRACKING_URI, FEATURE_COLUMNS, TARGET
from src.data.generator import load_dataset
from src.features.pipeline import build_pipeline, save_pipeline


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic for credit risk discrimination."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini coefficient (2*AUC - 1)."""
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def train(test_size: float = 0.2, seed: int = 42) -> dict:
    """Train the credit risk model and log to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("credit_risk")

    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    pipeline = build_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=seed,
        eval_metric="logloss",
    )

    with mlflow.start_run():
        model.fit(X_train_processed, y_train)

        y_prob = model.predict_proba(X_test_processed)[:, 1]
        y_pred = model.predict(X_test_processed)

        auc = roc_auc_score(y_test, y_prob)
        ks = ks_statistic(y_test.values, y_prob)
        gini = gini_coefficient(y_test, y_prob)

        metrics = {"auc": auc, "ks_statistic": ks, "gini": gini}

        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "test_size": test_size,
            "n_samples": len(df),
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print(f"AUC: {auc:.4f} | KS: {ks:.4f} | Gini: {gini:.4f}")
        print(classification_report(y_test, y_pred, target_names=["adimplente", "inadimplente"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_pipeline(pipeline)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Pipeline saved to {ARTIFACTS_DIR / 'pipeline.joblib'}")

    return metrics


def load_model() -> XGBClassifier:
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train()
