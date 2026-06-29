"""Integration tests for the FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.generator import generate_dataset, save_dataset
from src.models.trainer import train
from src.config import MODEL_PATH, PIPELINE_PATH


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    """Ensure model artifacts and database tables exist before API tests."""
    if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
        save_dataset(generate_dataset(n_samples=1000, seed=42))
        train(test_size=0.2, seed=42)
    from src.db.models import init_db
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


VALID_APPLICANT = {
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
}


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredict:
    def test_predict_valid_applicant(self, client):
        resp = client.post("/predict", json=VALID_APPLICANT)
        assert resp.status_code == 200
        data = resp.json()
        assert "probabilidade_inadimplencia" in data
        assert 0 <= data["probabilidade_inadimplencia"] <= 1
        assert data["faixa_risco"] in ("baixo", "medio", "alto")
        assert data["decisao_recomendada"] in ("APROVAR", "REVISAR", "NEGAR")
        assert 0 <= data["score_risco"] <= 1000

    def test_predict_high_risk_applicant(self, client):
        high_risk = {
            "idade": 20,
            "renda_mensal": 1500,
            "tempo_emprego_meses": 3,
            "num_dependentes": 4,
            "valor_solicitado": 50000,
            "divida_existente": 30000,
            "atrasos_12m": 5,
            "score_externo": 300,
            "tipo_residencia": "alugada",
            "finalidade_emprestimo": "negocios",
        }
        resp = client.post("/predict", json=high_risk)
        assert resp.status_code == 200
        data = resp.json()
        assert data["probabilidade_inadimplencia"] > 0.5
        assert data["decisao_recomendada"] == "NEGAR"

    def test_predict_low_risk_applicant(self, client):
        low_risk = {
            "idade": 45,
            "renda_mensal": 15000,
            "tempo_emprego_meses": 120,
            "num_dependentes": 0,
            "valor_solicitado": 5000,
            "divida_existente": 0,
            "atrasos_12m": 0,
            "score_externo": 850,
            "tipo_residencia": "propria",
            "finalidade_emprestimo": "educacao",
        }
        resp = client.post("/predict", json=low_risk)
        assert resp.status_code == 200
        data = resp.json()
        assert data["probabilidade_inadimplencia"] < 0.2
        assert data["decisao_recomendada"] == "APROVAR"

    def test_predict_missing_field(self, client):
        incomplete = {k: v for k, v in VALID_APPLICANT.items() if k != "idade"}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422

    def test_predict_invalid_age(self, client):
        invalid = {**VALID_APPLICANT, "idade": 10}
        resp = client.post("/predict", json=invalid)
        assert resp.status_code == 422

    def test_predict_invalid_score(self, client):
        invalid = {**VALID_APPLICANT, "score_externo": 1000}
        resp = client.post("/predict", json=invalid)
        assert resp.status_code == 422

    def test_predict_invalid_residencia(self, client):
        invalid = {**VALID_APPLICANT, "tipo_residencia": "invalido"}
        resp = client.post("/predict", json=invalid)
        assert resp.status_code == 422

    def test_predict_invalid_finalidade(self, client):
        invalid = {**VALID_APPLICANT, "finalidade_emprestimo": "viagem"}
        resp = client.post("/predict", json=invalid)
        assert resp.status_code == 422


class TestExplain:
    def test_explain_returns_factors(self, client):
        resp = client.post("/explain", json=VALID_APPLICANT)
        assert resp.status_code == 200
        data = resp.json()
        assert "probabilidade_inadimplencia" in data
        assert "principais_fatores" in data
        assert len(data["principais_fatores"]) > 0
        factor = data["principais_fatores"][0]
        assert "fator" in factor
        assert "impacto_shap" in factor
        assert "direcao" in factor

    def test_explain_invalid_input(self, client):
        resp = client.post("/explain", json={"idade": 35})
        assert resp.status_code == 422


def _ollama_available() -> bool:
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
class TestChat:
    def test_chat_creates_session(self, client):
        resp = client.post("/chat", json={"message": "oi"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "response" in data
        assert len(data["session_id"]) > 0
        assert len(data["response"]) > 0

    def test_chat_with_session_id(self, client):
        resp1 = client.post("/chat", json={"message": "oi"})
        session_id = resp1.json()["session_id"]

        resp2 = client.post("/chat", json={
            "session_id": session_id,
            "message": "obrigado",
        })
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id

    def test_chat_empty_message(self, client):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 200
