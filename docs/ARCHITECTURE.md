# Arquitetura do Sistema

## Visão Geral

Este sistema combina um modelo de Machine Learning para risco de crédito com um agente de IA conversacional que utiliza ferramentas (function calling) para tomar decisões autônomas.

## Diagrama de Arquitetura

```mermaid
graph TB
    subgraph Frontend
        ST[Streamlit App]
        ST --> |Score Tab| FORM[Formulário de Score]
        ST --> |Chat Tab| CHAT[Interface de Chat]
    end

    subgraph API["FastAPI REST API"]
        HEALTH[GET /health]
        PREDICT[POST /predict]
        EXPLAIN[POST /explain]
        CHATEP[POST /chat]
    end

    subgraph Agent["Agente de IA - LangGraph"]
        LLM[LLM Local - Ollama]
        ROUTER{Router de Ferramentas}
        LLM --> ROUTER
        ROUTER --> T1[predict_credit_risk]
        ROUTER --> T2[explain_credit_decision]
        ROUTER --> T3[simulate_scenario]
        ROUTER --> T4[query_credit_policy]
    end

    subgraph ML["Pipeline de ML"]
        DATA[Gerador de Dados Sintéticos]
        FE[Feature Engineering<br/>scikit-learn Pipeline]
        MODEL[XGBoost Classifier]
        SHAP[SHAP Explainer]
        MLFLOW[MLflow Tracking]
    end

    subgraph Storage["Armazenamento"]
        SQLITE[(SQLite<br/>Previsões + Chat)]
        CHROMA[(ChromaDB<br/>Política de Crédito)]
        ARTIFACTS[Artefatos<br/>model.joblib + pipeline.joblib]
    end

    FORM --> PREDICT
    FORM --> EXPLAIN
    CHAT --> CHATEP

    PREDICT --> T1
    EXPLAIN --> T2
    CHATEP --> LLM

    T1 --> MODEL
    T2 --> MODEL
    T2 --> SHAP
    T3 --> MODEL
    T4 --> CHROMA

    DATA --> FE
    FE --> MODEL
    MODEL --> MLFLOW
    MODEL --> ARTIFACTS

    PREDICT --> SQLITE
    CHATEP --> SQLITE

    style Frontend fill:#4CAF50,color:white
    style API fill:#2196F3,color:white
    style Agent fill:#FF9800,color:white
    style ML fill:#9C27B0,color:white
    style Storage fill:#607D8B,color:white
```

## Fluxo de Decisão do Agente

```mermaid
flowchart TD
    START([Mensagem do Usuário]) --> LLM[LLM Analisa Intenção]
    LLM --> DECIDE{Qual ferramenta usar?}

    DECIDE -->|Dados de solicitante| PRED[predict_credit_risk]
    DECIDE -->|"Por que?"| EXPL[explain_credit_decision]
    DECIDE -->|"E se...?"| SIM[simulate_scenario]
    DECIDE -->|Política/regras| RAG[query_credit_policy]
    DECIDE -->|Conversa geral| RESP[Resposta Direta]

    PRED --> RESULT[Resultado da Ferramenta]
    EXPL --> RESULT
    SIM --> RESULT
    RAG --> RESULT

    RESULT --> FORMAT[LLM Formata Resposta]
    FORMAT --> USER([Resposta ao Usuário])
    RESP --> USER

    style START fill:#4CAF50,color:white
    style USER fill:#4CAF50,color:white
    style DECIDE fill:#FF9800,color:white
    style LLM fill:#2196F3,color:white
    style FORMAT fill:#2196F3,color:white
```

## Fluxo de Dados do Pipeline de ML

```mermaid
flowchart LR
    GEN[Gerador Sintético<br/>10.000 amostras] --> RAW[Dataset CSV]
    RAW --> SPLIT[Train/Test Split<br/>80/20 estratificado]
    SPLIT --> FE[Feature Engineering]
    FE --> |Variáveis derivadas| DER[razão dívida/renda<br/>razão valor/renda<br/>renda per capita]
    DER --> SCALE[StandardScaler +<br/>OrdinalEncoder]
    SCALE --> XGB[XGBoost Training]
    XGB --> EVAL[Avaliação<br/>AUC / KS / Gini]
    EVAL --> MLFLOW[MLflow Logging]
    XGB --> SAVE[Salvar Artefatos]

    style GEN fill:#9C27B0,color:white
    style XGB fill:#F44336,color:white
    style MLFLOW fill:#2196F3,color:white
```

## Componentes

| Componente | Tecnologia | Responsabilidade |
|---|---|---|
| Frontend | Streamlit | Dashboard de score + Chat |
| API | FastAPI + Uvicorn | Endpoints REST |
| Agente | LangGraph + Ollama | Orquestração de ferramentas via LLM |
| Modelo | XGBoost + scikit-learn | Previsão de inadimplência |
| Explicabilidade | SHAP | Interpretação por previsão |
| RAG | ChromaDB + sentence-transformers | Consulta à política de crédito |
| Banco de Dados | SQLite + SQLAlchemy | Persistência de previsões e chat |
| Tracking | MLflow | Métricas e artefatos de experimento |
| Containerização | Docker + docker-compose | Deploy completo |
