"""Streamlit frontend: credit score dashboard + agent chat."""

import os
import uuid
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Risco de Crédito AI", page_icon="🏦", layout="wide")
st.title("🏦 Sistema de Risco de Crédito com Agente de IA")

tab_score, tab_chat = st.tabs(["📊 Score de Crédito", "💬 Chat com Agente"])

with tab_score:
    st.header("Avaliação de Risco de Crédito")

    col1, col2 = st.columns(2)

    with col1:
        idade = st.number_input("Idade", min_value=18, max_value=100, value=35)
        renda_mensal = st.number_input("Renda Mensal (R$)", min_value=0.0, value=5000.0, step=500.0)
        tempo_emprego = st.number_input("Tempo de Emprego (meses)", min_value=0, value=36)
        num_dependentes = st.number_input("Nº de Dependentes", min_value=0, value=1)
        tipo_residencia = st.selectbox("Tipo de Residência", ["propria", "alugada", "familiar", "financiada"])

    with col2:
        valor_solicitado = st.number_input("Valor Solicitado (R$)", min_value=1.0, value=15000.0, step=1000.0)
        divida_existente = st.number_input("Dívida Existente (R$)", min_value=0.0, value=3000.0, step=500.0)
        atrasos_12m = st.number_input("Atrasos nos últimos 12 meses", min_value=0, value=0)
        score_externo = st.number_input("Score Externo (200-900)", min_value=200, max_value=900, value=650)
        finalidade = st.selectbox("Finalidade do Empréstimo", ["pessoal", "veiculo", "imovel", "educacao", "negocios"])

    applicant_data = {
        "idade": float(idade),
        "renda_mensal": float(renda_mensal),
        "tempo_emprego_meses": float(tempo_emprego),
        "num_dependentes": float(num_dependentes),
        "valor_solicitado": float(valor_solicitado),
        "divida_existente": float(divida_existente),
        "atrasos_12m": float(atrasos_12m),
        "score_externo": float(score_externo),
        "tipo_residencia": tipo_residencia,
        "finalidade_emprestimo": finalidade,
    }

    col_pred, col_explain = st.columns(2)

    with col_pred:
        if st.button("🔍 Calcular Risco", type="primary", use_container_width=True):
            try:
                resp = requests.post(f"{API_URL}/predict", json=applicant_data, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    prob = result["probabilidade_inadimplencia"]
                    risk = result["faixa_risco"]
                    decision = result["decisao_recomendada"]
                    score = result["score_risco"]

                    color = {"baixo": "green", "medio": "orange", "alto": "red"}.get(risk, "gray")

                    st.metric("Score de Risco", f"{score}/1000")
                    st.metric("Probabilidade de Inadimplência", f"{prob:.1%}")
                    st.markdown(f"**Faixa de Risco:** :{color}[{risk.upper()}]")
                    st.markdown(f"**Decisão Recomendada:** **{decision}**")
                else:
                    st.error(f"Erro na API: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("API não disponível. Inicie com `make api`.")

    with col_explain:
        if st.button("🧠 Explicar Decisão", use_container_width=True):
            try:
                resp = requests.post(f"{API_URL}/explain", json=applicant_data, timeout=60)
                if resp.status_code == 200:
                    result = resp.json()
                    st.subheader("Principais Fatores")
                    for factor in result.get("principais_fatores", []):
                        icon = "🔴" if "aumenta" in factor["direcao"] else "🟢"
                        st.markdown(f"{icon} **{factor['fator']}**: {factor['direcao']} (SHAP: {factor['impacto_shap']:.4f})")
                else:
                    st.error(f"Erro na API: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("API não disponível. Inicie com `make api`.")


with tab_chat:
    st.header("Chat com Agente de IA")
    st.caption("Converse com o agente para avaliar solicitantes, simular cenários ou consultar a política de crédito.")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Faça uma pergunta sobre risco de crédito..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"session_id": st.session_state.session_id, "message": prompt},
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        answer = resp.json()["response"]
                        st.session_state.session_id = resp.json()["session_id"]
                    else:
                        answer = f"Erro: {resp.text}"
                except requests.exceptions.ConnectionError:
                    answer = "API não disponível. Inicie com `make api`."
                except requests.exceptions.ReadTimeout:
                    answer = "O modelo LLM demorou demais para responder. Tente uma pergunta mais curta ou verifique se o Ollama está rodando."

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
