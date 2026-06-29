"""RAG module for credit policy document retrieval."""

import json
from langchain_core.tools import tool

from src.config import POLICY_DOC_PATH, CHROMA_DIR

_collection = None


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    collection_name = "credit_policy"
    existing = [c.name for c in client.list_collections()]

    if collection_name in existing:
        _collection = client.get_collection(
            name=collection_name,
            embedding_function=_get_embedding_fn(),
        )
        return _collection

    _collection = client.create_collection(
        name=collection_name,
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )

    _index_policy_document(_collection)
    return _collection


def _get_embedding_fn():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _index_policy_document(collection):
    if not POLICY_DOC_PATH.exists():
        return

    text = POLICY_DOC_PATH.read_text(encoding="utf-8")
    chunks = _split_by_sections(text)

    ids = [f"policy_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)


def _split_by_sections(text: str, max_chunk_size: int = 500) -> list[str]:
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        if line.startswith("## ") and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = len(line)
        else:
            current_chunk.append(line)
            current_size += len(line)
            if current_size >= max_chunk_size and not line.startswith("#"):
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


@tool
def query_credit_policy(query: str) -> str:
    """Search the internal credit policy document to answer questions about approval rules, limits, and guidelines.

    Args:
        query: Natural language question about credit policy
    """
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=3)

    if not results["documents"] or not results["documents"][0]:
        return json.dumps({"resposta": "Nenhuma informação encontrada na política de crédito."})

    passages = results["documents"][0]
    distances = results["distances"][0] if results["distances"] else []

    context_parts = []
    for i, passage in enumerate(passages):
        dist = distances[i] if i < len(distances) else None
        context_parts.append({
            "trecho": passage,
            "relevancia": round(1 - dist, 3) if dist is not None else None,
        })

    return json.dumps({
        "consulta": query,
        "trechos_relevantes": context_parts,
    }, ensure_ascii=False)
