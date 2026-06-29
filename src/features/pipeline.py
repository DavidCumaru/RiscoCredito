"""Scikit-learn feature engineering pipeline for credit risk."""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, PIPELINE_PATH


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific derived features."""
    X = X.copy()
    X["razao_divida_renda"] = X["divida_existente"] / (X["renda_mensal"] + 1)
    X["razao_valor_renda"] = X["valor_solicitado"] / (X["renda_mensal"] + 1)
    X["razao_valor_divida"] = X["valor_solicitado"] / (X["divida_existente"] + 1)
    X["renda_per_capita"] = X["renda_mensal"] / (X["num_dependentes"] + 1)
    return X


DERIVED_NUMERIC = [
    "razao_divida_renda",
    "razao_valor_renda",
    "razao_valor_divida",
    "renda_per_capita",
]

ALL_NUMERIC = NUMERIC_FEATURES + DERIVED_NUMERIC


def build_pipeline() -> Pipeline:
    """Build the full preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_NUMERIC),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[
        ("derived", FunctionTransformer(add_derived_features, validate=False)),
        ("preprocessor", preprocessor),
    ])

    return pipeline


def save_pipeline(pipeline: Pipeline) -> None:
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)


def load_pipeline() -> Pipeline:
    return joblib.load(PIPELINE_PATH)


def get_feature_names() -> list[str]:
    return ALL_NUMERIC + CATEGORICAL_FEATURES
