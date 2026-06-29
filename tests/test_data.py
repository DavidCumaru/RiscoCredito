"""Tests for synthetic dataset generation."""

import pytest
from src.data.generator import generate_dataset
from src.config import FEATURE_COLUMNS, TARGET


@pytest.fixture
def dataset():
    return generate_dataset(n_samples=1000, seed=42)


def test_dataset_shape(dataset):
    assert len(dataset) == 1000
    assert set(FEATURE_COLUMNS + [TARGET]).issubset(set(dataset.columns))


def test_dataset_no_nulls(dataset):
    assert dataset[FEATURE_COLUMNS].isnull().sum().sum() == 0


def test_target_is_binary(dataset):
    assert set(dataset[TARGET].unique()).issubset({0, 1})


def test_default_rate_is_realistic(dataset):
    rate = dataset[TARGET].mean()
    assert 0.05 < rate < 0.60, f"Default rate {rate:.2%} is unrealistic"


def test_age_range(dataset):
    assert dataset["idade"].min() >= 18
    assert dataset["idade"].max() <= 70


def test_score_range(dataset):
    assert dataset["score_externo"].min() >= 200
    assert dataset["score_externo"].max() <= 900


def test_categorical_values(dataset):
    valid_residencia = {"propria", "alugada", "familiar", "financiada"}
    assert set(dataset["tipo_residencia"].unique()).issubset(valid_residencia)

    valid_finalidade = {"pessoal", "veiculo", "imovel", "educacao", "negocios"}
    assert set(dataset["finalidade_emprestimo"].unique()).issubset(valid_finalidade)


def test_reproducibility():
    df1 = generate_dataset(n_samples=100, seed=42)
    df2 = generate_dataset(n_samples=100, seed=42)
    assert df1.equals(df2)
