import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
DATA_PATH = ARTIFACTS_DIR / "credit_data.csv"
POLICY_DOC_PATH = ROOT_DIR / "docs" / "credit_policy_sample.md"
CHROMA_DIR = ROOT_DIR / "chroma_db"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'credit_risk.db'}")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")

FEATURE_COLUMNS = [
    "idade",
    "renda_mensal",
    "tempo_emprego_meses",
    "num_dependentes",
    "valor_solicitado",
    "divida_existente",
    "atrasos_12m",
    "score_externo",
    "tipo_residencia",
    "finalidade_emprestimo",
]

NUMERIC_FEATURES = [
    "idade",
    "renda_mensal",
    "tempo_emprego_meses",
    "num_dependentes",
    "valor_solicitado",
    "divida_existente",
    "atrasos_12m",
    "score_externo",
]

CATEGORICAL_FEATURES = [
    "tipo_residencia",
    "finalidade_emprestimo",
]

TARGET = "inadimplente"
