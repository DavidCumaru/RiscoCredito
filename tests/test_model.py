"""Tests for model training and prediction."""

import pytest
import json
from src.models.trainer import train
from src.config import MODEL_PATH, PIPELINE_PATH


@pytest.fixture(scope="module")
def trained_metrics():
    return train(test_size=0.2, seed=42)


def test_model_artifacts_exist(trained_metrics):
    assert MODEL_PATH.exists(), "Model file not saved"
    assert PIPELINE_PATH.exists(), "Pipeline file not saved"


def test_auc_above_threshold(trained_metrics):
    assert trained_metrics["auc"] > 0.70, f"AUC {trained_metrics['auc']:.4f} too low"


def test_ks_above_threshold(trained_metrics):
    assert trained_metrics["ks_statistic"] > 0.30, f"KS {trained_metrics['ks_statistic']:.4f} too low"


def test_gini_above_threshold(trained_metrics):
    assert trained_metrics["gini"] > 0.40, f"Gini {trained_metrics['gini']:.4f} too low"


def test_prediction_tool(trained_metrics):
    from src.agent.tools import predict_credit_risk

    result_str = predict_credit_risk.invoke({
        "idade": 35,
        "renda_mensal": 5000,
        "tempo_emprego_meses": 36,
        "num_dependentes": 1,
        "valor_solicitado": 15000,
        "divida_existente": 3000,
        "atrasos_12m": 0,
        "score_externo": 700,
        "tipo_residencia": "propria",
        "finalidade_emprestimo": "pessoal",
    })
    result = json.loads(result_str)

    assert "probabilidade_inadimplencia" in result
    assert 0 <= result["probabilidade_inadimplencia"] <= 1
    assert result["faixa_risco"] in ("baixo", "medio", "alto")
    assert result["decisao_recomendada"] in ("APROVAR", "REVISAR", "NEGAR")


def test_explanation_tool(trained_metrics):
    from src.agent.tools import explain_credit_decision

    result_str = explain_credit_decision.invoke({
        "idade": 25,
        "renda_mensal": 2000,
        "tempo_emprego_meses": 6,
        "num_dependentes": 3,
        "valor_solicitado": 30000,
        "divida_existente": 15000,
        "atrasos_12m": 3,
        "score_externo": 400,
        "tipo_residencia": "alugada",
        "finalidade_emprestimo": "negocios",
    })
    result = json.loads(result_str)

    assert "principais_fatores" in result
    assert len(result["principais_fatores"]) > 0
    assert "fator" in result["principais_fatores"][0]


def test_simulation_tool(trained_metrics):
    from src.agent.tools import simulate_scenario

    result_str = simulate_scenario.invoke({
        "idade": 35,
        "renda_mensal": 5000,
        "tempo_emprego_meses": 36,
        "num_dependentes": 1,
        "valor_solicitado": 15000,
        "divida_existente": 3000,
        "atrasos_12m": 0,
        "score_externo": 700,
        "tipo_residencia": "propria",
        "finalidade_emprestimo": "pessoal",
        "campo_alterado": "renda_mensal",
        "novo_valor": 10000,
    })
    result = json.loads(result_str)

    assert "cenario_original" in result
    assert "cenario_simulado" in result
    assert "variacao_probabilidade" in result
