"""Agent tools for credit risk prediction, explanation, simulation, and RAG."""

import json
import numpy as np
import pandas as pd
import shap
from langchain_core.tools import tool

from src.config import NUMERIC_FEATURES
from src.models.trainer import load_model
from src.features.pipeline import load_pipeline, get_feature_names


def _get_model_and_pipeline():
    return load_model(), load_pipeline()


def _classify_risk(prob: float) -> tuple[str, str]:
    if prob < 0.2:
        return "baixo", "APROVAR"
    elif prob < 0.5:
        return "medio", "REVISAR"
    else:
        return "alto", "NEGAR"


def _build_applicant_df(
    idade: float,
    renda_mensal: float,
    tempo_emprego_meses: float,
    num_dependentes: float,
    valor_solicitado: float,
    divida_existente: float,
    atrasos_12m: float,
    score_externo: float,
    tipo_residencia: str,
    finalidade_emprestimo: str,
) -> pd.DataFrame:
    return pd.DataFrame([{
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
    }])


@tool
def predict_credit_risk(
    idade: float,
    renda_mensal: float,
    tempo_emprego_meses: float,
    num_dependentes: float,
    valor_solicitado: float,
    divida_existente: float,
    atrasos_12m: float,
    score_externo: float,
    tipo_residencia: str,
    finalidade_emprestimo: str,
) -> str:
    """Predict credit default probability for a loan applicant.

    Args:
        idade: Age in years (18-70)
        renda_mensal: Monthly income in BRL
        tempo_emprego_meses: Employment duration in months
        num_dependentes: Number of dependents
        valor_solicitado: Requested loan amount in BRL
        divida_existente: Existing debt in BRL
        atrasos_12m: Number of late payments in last 12 months
        score_externo: External credit score (200-900)
        tipo_residencia: Housing type (propria/alugada/familiar/financiada)
        finalidade_emprestimo: Loan purpose (pessoal/veiculo/imovel/educacao/negocios)
    """
    model, pipeline = _get_model_and_pipeline()
    df = _build_applicant_df(
        idade, renda_mensal, tempo_emprego_meses, num_dependentes,
        valor_solicitado, divida_existente, atrasos_12m, score_externo,
        tipo_residencia, finalidade_emprestimo,
    )
    X = pipeline.transform(df)
    prob = float(model.predict_proba(X)[:, 1][0])
    risk_band, decision = _classify_risk(prob)

    return json.dumps({
        "probabilidade_inadimplencia": round(prob, 4),
        "faixa_risco": risk_band,
        "decisao_recomendada": decision,
        "score_risco": round((1 - prob) * 1000),
    }, ensure_ascii=False)


@tool
def explain_credit_decision(
    idade: float,
    renda_mensal: float,
    tempo_emprego_meses: float,
    num_dependentes: float,
    valor_solicitado: float,
    divida_existente: float,
    atrasos_12m: float,
    score_externo: float,
    tipo_residencia: str,
    finalidade_emprestimo: str,
) -> str:
    """Explain the main factors behind a credit risk prediction using SHAP.

    Same parameters as predict_credit_risk.
    """
    model, pipeline = _get_model_and_pipeline()
    df = _build_applicant_df(
        idade, renda_mensal, tempo_emprego_meses, num_dependentes,
        valor_solicitado, divida_existente, atrasos_12m, score_externo,
        tipo_residencia, finalidade_emprestimo,
    )
    X = pipeline.transform(df)
    prob = float(model.predict_proba(X)[:, 1][0])

    feature_names = get_feature_names()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap_vals = shap_values[0] if len(shap_values.shape) == 2 else shap_values
    except (ValueError, TypeError):
        importances = model.feature_importances_
        x_row = np.array(X[0]).flatten() if hasattr(X, 'toarray') else np.array(X).flatten()
        shap_vals = importances * x_row

    indices = np.argsort(np.abs(shap_vals))[::-1][:5]
    factors = []
    for idx in indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        val = float(shap_vals[idx])
        direction = "aumenta" if val > 0 else "diminui"
        factors.append({
            "fator": name,
            "impacto_shap": round(val, 4),
            "direcao": f"{direction} o risco",
        })

    return json.dumps({
        "probabilidade_inadimplencia": round(prob, 4),
        "principais_fatores": factors,
    }, ensure_ascii=False)


@tool
def simulate_scenario(
    idade: float,
    renda_mensal: float,
    tempo_emprego_meses: float,
    num_dependentes: float,
    valor_solicitado: float,
    divida_existente: float,
    atrasos_12m: float,
    score_externo: float,
    tipo_residencia: str,
    finalidade_emprestimo: str,
    campo_alterado: str,
    novo_valor: float,
) -> str:
    """Simulate a what-if scenario by changing one field and comparing results.

    Args:
        campo_alterado: Name of the field to change (e.g. 'renda_mensal')
        novo_valor: New value for the field
        (other args same as predict_credit_risk)
    """
    model, pipeline = _get_model_and_pipeline()

    original = _build_applicant_df(
        idade, renda_mensal, tempo_emprego_meses, num_dependentes,
        valor_solicitado, divida_existente, atrasos_12m, score_externo,
        tipo_residencia, finalidade_emprestimo,
    )

    if campo_alterado not in original.columns:
        return json.dumps({"erro": f"Campo '{campo_alterado}' não encontrado. Campos válidos: {list(original.columns)}"})

    modified = original.copy()
    modified[campo_alterado] = novo_valor

    X_orig = pipeline.transform(original)
    X_mod = pipeline.transform(modified)

    prob_orig = float(model.predict_proba(X_orig)[:, 1][0])
    prob_mod = float(model.predict_proba(X_mod)[:, 1][0])

    risk_orig, dec_orig = _classify_risk(prob_orig)
    risk_mod, dec_mod = _classify_risk(prob_mod)

    return json.dumps({
        "cenario_original": {
            "probabilidade": round(prob_orig, 4),
            "faixa_risco": risk_orig,
            "decisao": dec_orig,
        },
        "cenario_simulado": {
            "campo_alterado": campo_alterado,
            "valor_anterior": float(original[campo_alterado].iloc[0]) if campo_alterado in NUMERIC_FEATURES else str(original[campo_alterado].iloc[0]),
            "novo_valor": novo_valor,
            "probabilidade": round(prob_mod, 4),
            "faixa_risco": risk_mod,
            "decisao": dec_mod,
        },
        "variacao_probabilidade": round(prob_mod - prob_orig, 4),
    }, ensure_ascii=False)


ALL_TOOLS = [predict_credit_risk, explain_credit_decision, simulate_scenario]
