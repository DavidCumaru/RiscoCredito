"""SQLAlchemy models and database session management."""

from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    idade = Column(Float)
    renda_mensal = Column(Float)
    tempo_emprego_meses = Column(Float)
    num_dependentes = Column(Float)
    valor_solicitado = Column(Float)
    divida_existente = Column(Float)
    atrasos_12m = Column(Float)
    score_externo = Column(Float)
    tipo_residencia = Column(String(50))
    finalidade_emprestimo = Column(String(50))
    probability = Column(Float)
    risk_band = Column(String(20))
    decision = Column(String(20))


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), index=True)
    role = Column(String(20))
    content = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
