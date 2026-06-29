# 🏦 Risco de Crédito com Agente de IA

[![CI](https://github.com/davidcumaru/RiscoCredito/actions/workflows/ci.yml/badge.svg)](https://github.com/davidcumaru/RiscoCredito/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![100% Gratuito](https://img.shields.io/badge/custo-100%25%20gratuito-brightgreen.svg)]()

Sistema completo de **análise de risco de crédito** que combina um modelo de Machine Learning (XGBoost) com um **agente de IA conversacional** capaz de prever inadimplência, explicar decisões, simular cenários e consultar políticas de crédito via RAG.

**100% gratuito e open-source** — roda localmente sem nenhuma API key.

---

## 📋 Stack 100% Gratuita

| Componente | Tecnologia | Custo |
|---|---|---|
| Linguagem | Python 3.11+ | Gratuito |
| Modelo ML | scikit-learn + XGBoost | Gratuito |
| Explicabilidade | SHAP | Gratuito |
| Agente de IA | LangGraph (tool-calling) | Gratuito |
| LLM | Ollama (Llama 3.1) | Gratuito, local |
| RAG / Vector DB | ChromaDB + sentence-transformers | Gratuito |
| API | FastAPI + Uvicorn | Gratuito |
| Frontend | Streamlit | Gratuito |
| Banco de Dados | SQLite + SQLAlchemy | Gratuito |
| Tracking ML | MLflow | Gratuito |
| Containerização | Docker + docker-compose | Gratuito |
| CI/CD | GitHub Actions | Gratuito |
| Testes | pytest | Gratuito |

---

## 📊 Métricas do Modelo

O modelo treinado atinge as seguintes métricas de risco de crédito:

| Métrica | Valor | Threshold mínimo |
|---|---|---|
| **AUC** (Area Under ROC Curve) | 0.7964 | > 0.70 |
| **KS Statistic** (Kolmogorov-Smirnov) | 0.4546 | > 0.30 |
| **Gini Coefficient** | 0.5928 | > 0.40 |

Taxa de inadimplência do dataset: **17.67%** (10.000 amostras sintéticas com relação logística realista).

Todas as métricas e parâmetros são logados automaticamente no **MLflow** (backend SQLite).

---

## 🚀 Início Rápido

### Pré-requisitos

- Docker e docker-compose instalados
- ~6GB de espaço livre (para o modelo Llama 3.1)

### Setup com Docker (recomendado)

```bash
# 1. Clone e configure
git clone https://github.com/davidcumaru/RiscoCredito.git
cd RiscoCredito
cp .env.example .env

# 2. Suba todos os serviços (API + Streamlit + Ollama)
docker-compose up --build -d

# 3. Baixe o modelo LLM dentro do container Ollama (~4.7GB)
docker exec -it riscocredito-ollama ollama pull llama3.1
```

Acesse:
- **Frontend Streamlit:** http://localhost:8501
- **API docs (Swagger):** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### Setup Local (sem Docker)

```bash
# 1. Clone o repositório
git clone https://github.com/davidcumaru/RiscoCredito.git
cd RiscoCredito

# 2. Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o ambiente
cp .env.example .env

# 5. Instale o Ollama (https://ollama.ai) e baixe o modelo
ollama pull llama3.1

# 6. Gere os dados e treine o modelo
make data
make train

# 7. Inicie a API (terminal 1)
make api

# 8. Inicie o frontend (terminal 2)
make app
```

> **Nota:** O modelo `llama3.1` é necessário para o agente funcionar (suporta tool-calling). Modelos menores como `phi3` não suportam chamada de ferramentas.

---

## 🏗️ Estrutura do Projeto

```
RiscoCredito/
├── src/
│   ├── config.py           # Configuração centralizada
│   ├── data/               # Geração de dataset sintético
│   │   └── generator.py    # Gerador com relação logística realista
│   ├── features/           # Pipeline de engenharia de atributos
│   │   └── pipeline.py     # Imputação, encoding, scaling, variáveis derivadas
│   ├── models/             # Treino e métricas
│   │   └── trainer.py      # XGBoost + MLflow tracking
│   ├── agent/              # Agente de IA conversacional
│   │   ├── tools.py        # 4 ferramentas (predict, explain, simulate, RAG)
│   │   ├── rag.py          # ChromaDB + sentence-transformers
│   │   └── graph.py        # LangGraph agent com memória de sessão
│   ├── db/                 # Persistência
│   │   └── models.py       # SQLAlchemy (Prediction + ChatHistory)
│   ├── api/                # REST API
│   │   └── main.py         # FastAPI (/predict, /explain, /chat, /health)
│   └── app/                # Interface visual
│       └── streamlit_app.py # Dashboard de score + chat com agente
├── tests/                  # 19 testes automatizados (pytest)
├── docs/
│   ├── ARCHITECTURE.md     # Diagramas Mermaid
│   └── credit_policy_sample.md  # Política de crédito fictícia (RAG)
├── .github/workflows/ci.yml    # CI: lint + treino + testes
├── Dockerfile              # Container da API
├── Dockerfile.streamlit    # Container do frontend
├── docker-compose.yml      # API + Streamlit + Ollama
├── Makefile                # Atalhos: make data, train, api, app, test
├── requirements.txt
├── .env.example
└── LICENSE (MIT)
```

---

## 🤖 Agente de IA — Ferramentas

O agente decide **autonomamente** qual ferramenta chamar baseado na mensagem do usuário:

| Ferramenta | Descrição |
|---|---|
| `predict_credit_risk` | Prevê probabilidade de inadimplência, faixa de risco e decisão (APROVAR/REVISAR/NEGAR) |
| `explain_credit_decision` | Retorna os principais fatores da decisão (SHAP) em linguagem natural |
| `simulate_scenario` | Roda simulações "e se" alterando uma variável e comparando resultados |
| `query_credit_policy` | Consulta a política de crédito interna via RAG (ChromaDB) |

### Exemplos de conversa com o agente

```
Usuário: Avalie um solicitante de 30 anos, renda R$4000, 24 meses de emprego,
         sem dependentes, pedindo R$10000, sem dívida, sem atrasos, score 650,
         moradia própria, empréstimo pessoal.

Agente:  O solicitante tem uma probabilidade de inadimplência de 7,5% e é
         classificado como baixo risco. A decisão recomendada é aprovar o
         empréstimo. Os principais fatores foram: renda mensal de R$4.000,
         tempo de emprego de 24 meses, score externo de 650.

Usuário: E se a renda dele fosse R$8000, mudaria a decisão?

Agente:  Se a renda fosse de R$8.000, a probabilidade diminuiria para 4,84%.
         A decisão continua sendo aprovar. A renda aumentou em 100% e a
         probabilidade de inadimplência diminuiu em 6,7%.

Usuário: Qual o limite máximo de crédito para risco baixo?

Agente:  O limite máximo para clientes de risco baixo é até R$50.000,
         podendo chegar a 10x a renda mensal. Taxa a partir de 1,5% ao mês,
         prazo máximo de 60 meses.
```

---

## 🔌 API REST

| Endpoint | Método | Descrição |
|---|---|---|
| `/health` | GET | Status da API e se o modelo está carregado |
| `/predict` | POST | Previsão de risco para um solicitante |
| `/explain` | POST | Explicação SHAP dos fatores da decisão |
| `/chat` | POST | Chat com o agente (com memória de sessão) |

Documentação interativa (Swagger): http://localhost:8000/docs

---

## 🔄 Providers de LLM Alternativos

O sistema usa Ollama por padrão (local, gratuito), mas suporta providers com camadas gratuitas via variável de ambiente:

```bash
# Groq (free tier — https://console.groq.com)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_sua_chave

# Google Gemini (free tier — https://aistudio.google.com)
LLM_PROVIDER=gemini
GEMINI_API_KEY=sua_chave
```

---

## 📂 Usando Datasets Públicos Reais

O sistema gera dados sintéticos por padrão, mas pode ser adaptado para datasets reais:

1. **UCI German Credit Data:** Baixe de [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)), renomeie as colunas para o padrão do projeto e salve como `artifacts/credit_data.csv`.

2. **Kaggle "Give Me Some Credit":** Baixe de [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit), mapeie as colunas (`MonthlyIncome` → `renda_mensal`, etc.) e ajuste `src/config.py`.

3. **Home Credit Default Risk:** Dataset maior e mais complexo, disponível no [Kaggle](https://www.kaggle.com/c/home-credit-default-risk). Requer engenharia de features adicional.

---

## 🧪 Testes

```bash
# Rodar todos os testes (19 testes)
make test

# Com detalhes
pytest tests/ -v --tb=short

# Lint
make lint
```

Os testes cobrem:
- **test_data.py** — shape, nulos, faixas de valores, reprodutibilidade do dataset
- **test_features.py** — variáveis derivadas, pipeline de transformação, tratamento de nulos
- **test_model.py** — métricas (AUC/KS/Gini), predict, explain (SHAP), simulate

---

## 📐 Arquitetura

Veja o diagrama completo com fluxo de dados e fluxo de decisão do agente em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---
## 📄 Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).
