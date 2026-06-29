"""Tests for feature engineering pipeline."""

import pytest
import numpy as np
import pandas as pd
from src.features.pipeline import build_pipeline, add_derived_features
from src.data.generator import generate_dataset
from src.config import FEATURE_COLUMNS


@pytest.fixture
def sample_data():
    return generate_dataset(n_samples=200, seed=42)[FEATURE_COLUMNS]


def test_derived_features():
    df = pd.DataFrame([{
        "idade": 30,
        "renda_mensal": 5000,
        "tempo_emprego_meses": 24,
        "num_dependentes": 2,
        "valor_solicitado": 10000,
        "divida_existente": 2000,
        "atrasos_12m": 0,
        "score_externo": 700,
        "tipo_residencia": "propria",
        "finalidade_emprestimo": "pessoal",
    }])
    result = add_derived_features(df)

    assert "razao_divida_renda" in result.columns
    assert "razao_valor_renda" in result.columns
    assert "razao_valor_divida" in result.columns
    assert "renda_per_capita" in result.columns

    assert result["razao_divida_renda"].iloc[0] == pytest.approx(2000 / 5001, rel=1e-3)
    assert result["renda_per_capita"].iloc[0] == pytest.approx(5000 / 3, rel=1e-3)


def test_pipeline_output_shape(sample_data):
    pipeline = build_pipeline()
    X_transformed = pipeline.fit_transform(sample_data)
    assert X_transformed.shape[0] == len(sample_data)
    assert X_transformed.shape[1] == 14  # 8 numeric + 4 derived + 2 categorical


def test_pipeline_no_nans(sample_data):
    pipeline = build_pipeline()
    X_transformed = pipeline.fit_transform(sample_data)
    assert not np.isnan(X_transformed).any()


def test_pipeline_handles_missing():
    df = pd.DataFrame([{
        "idade": 30,
        "renda_mensal": None,
        "tempo_emprego_meses": 24,
        "num_dependentes": 2,
        "valor_solicitado": 10000,
        "divida_existente": 2000,
        "atrasos_12m": 0,
        "score_externo": 700,
        "tipo_residencia": "propria",
        "finalidade_emprestimo": "pessoal",
    }])
    train_data = generate_dataset(n_samples=200, seed=42)[FEATURE_COLUMNS]

    pipeline = build_pipeline()
    pipeline.fit(train_data)
    result = pipeline.transform(df)
    assert not np.isnan(result).any()
