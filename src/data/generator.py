"""Synthetic credit-risk dataset generator with realistic logistic relationships."""

import numpy as np
import pandas as pd
from src.config import DATA_PATH, TARGET


def generate_dataset(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic credit applicant dataset.

    The default probability follows a logistic function of weighted features,
    producing realistic, learnable patterns rather than pure noise.
    """
    rng = np.random.default_rng(seed)

    idade = rng.integers(18, 70, size=n_samples).astype(float)
    renda_mensal = np.round(rng.lognormal(mean=8.5, sigma=0.6, size=n_samples), 2)
    tempo_emprego_meses = rng.integers(0, 360, size=n_samples).astype(float)
    num_dependentes = rng.poisson(lam=1.2, size=n_samples).astype(float)
    valor_solicitado = np.round(rng.lognormal(mean=9.0, sigma=0.8, size=n_samples), 2)
    divida_existente = np.round(rng.exponential(scale=5000, size=n_samples), 2)
    atrasos_12m = rng.poisson(lam=0.8, size=n_samples).astype(float)
    score_externo = np.clip(rng.normal(loc=600, scale=120, size=n_samples), 200, 900).astype(float)

    tipos_residencia = ["propria", "alugada", "familiar", "financiada"]
    tipo_residencia = rng.choice(tipos_residencia, size=n_samples, p=[0.35, 0.30, 0.20, 0.15])

    finalidades = ["pessoal", "veiculo", "imovel", "educacao", "negocios"]
    finalidade_emprestimo = rng.choice(finalidades, size=n_samples, p=[0.30, 0.25, 0.15, 0.15, 0.15])

    razao_divida_renda = divida_existente / (renda_mensal + 1)
    razao_valor_renda = valor_solicitado / (renda_mensal + 1)

    residencia_score = np.where(tipo_residencia == "propria", -0.3,
                       np.where(tipo_residencia == "financiada", 0.2,
                       np.where(tipo_residencia == "alugada", 0.1, 0.0)))

    finalidade_score = np.where(finalidade_emprestimo == "imovel", -0.2,
                      np.where(finalidade_emprestimo == "educacao", -0.1,
                      np.where(finalidade_emprestimo == "negocios", 0.3,
                      np.where(finalidade_emprestimo == "pessoal", 0.2, 0.0))))

    logit = (
        1.5
        + 0.02 * (50 - idade)
        + 0.8 * np.log1p(razao_divida_renda)
        + 0.6 * np.log1p(razao_valor_renda)
        - 0.008 * score_externo
        + 0.5 * atrasos_12m
        - 0.004 * tempo_emprego_meses
        + 0.15 * num_dependentes
        + residencia_score
        + finalidade_score
        + rng.normal(0, 0.5, size=n_samples)
    )

    prob_inadimplencia = 1 / (1 + np.exp(-logit))
    inadimplente = rng.binomial(1, prob_inadimplencia).astype(int)

    df = pd.DataFrame({
        "idade": idade,
        "renda_mensal": renda_mensal,
        "tempo_emprego_meses": tempo_emprego_meses,
        "num_dependentes": num_dependentes,
        "valor_solicitado": valor_solicitado,
        "divida_existente": divida_existente,
        "atrasos_12m": atrasos_12m,
        "score_externo": score_externo,
        "tipo_residencia": tipo_residencia,
        "finalidade_emprestimo": finalidade_emprestimo,
        "inadimplente": inadimplente,
    })

    return df


def save_dataset(df: pd.DataFrame) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH} ({len(df)} rows)")
    print(f"Default rate: {df[TARGET].mean():.2%}")


def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    df = generate_dataset()
    save_dataset(df)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    save_dataset(df)
