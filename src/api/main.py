"""FastAPI REST API for credit risk prediction and agent chat."""

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.config import MODEL_PATH, PIPELINE_PATH
from src.db.models import init_db, get_session, Prediction, ChatHistory


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Credit Risk AI API",
    description="API de risco de crédito com agente de IA conversacional",
    version="1.0.0",
    lifespan=lifespan,
)


class ApplicantInput(BaseModel):
    idade: float = Field(..., ge=18, le=100)
    renda_mensal: float = Field(..., ge=0)
    tempo_emprego_meses: float = Field(..., ge=0)
    num_dependentes: float = Field(..., ge=0)
    valor_solicitado: float = Field(..., gt=0)
    divida_existente: float = Field(..., ge=0)
    atrasos_12m: float = Field(..., ge=0)
    score_externo: float = Field(..., ge=200, le=900)
    tipo_residencia: str = Field(..., pattern="^(propria|alugada|familiar|financiada)$")
    finalidade_emprestimo: str = Field(..., pattern="^(pessoal|veiculo|imovel|educacao|negocios)$")


class ChatInput(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists() and PIPELINE_PATH.exists()
    return {"status": "healthy", "model_loaded": model_ready}


@app.post("/predict")
def predict(applicant: ApplicantInput, db: Session = Depends(get_session)):
    import json
    from src.agent.tools import predict_credit_risk

    result_str = predict_credit_risk.invoke(applicant.model_dump())
    result = json.loads(result_str)

    record = Prediction(
        **applicant.model_dump(),
        probability=result["probabilidade_inadimplencia"],
        risk_band=result["faixa_risco"],
        decision=result["decisao_recomendada"],
    )
    db.add(record)
    db.commit()

    return result


@app.post("/explain")
def explain(applicant: ApplicantInput):
    import json
    from src.agent.tools import explain_credit_decision

    result_str = explain_credit_decision.invoke(applicant.model_dump())
    return json.loads(result_str)


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(chat_input: ChatInput, db: Session = Depends(get_session)):
    from src.agent.graph import chat as agent_chat

    session_id = chat_input.session_id or str(uuid.uuid4())

    db.add(ChatHistory(session_id=session_id, role="user", content=chat_input.message))
    db.commit()

    response = agent_chat(session_id, chat_input.message)

    db.add(ChatHistory(session_id=session_id, role="assistant", content=response))
    db.commit()

    return ChatResponse(session_id=session_id, response=response)
