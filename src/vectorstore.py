from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import (
    FAISS_INDEX_PATH,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_LOCAL_PATH,
    QDRANT_URL,
)


@lru_cache(maxsize=1)
def _get_qdrant_client():
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(path=str(QDRANT_LOCAL_PATH))


def _get_qdrant_connection_kwargs():
    if QDRANT_URL:
        return {"url": QDRANT_URL, "api_key": QDRANT_API_KEY}
    return {"path": str(QDRANT_LOCAL_PATH)}


def create_faiss_vectorstore(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)


def create_qdrant_vectorstore(documents, embeddings, collection_name=QDRANT_COLLECTION):
    return QdrantVectorStore.from_documents(
        documents,
        embeddings,
        collection_name=collection_name,
        force_recreate=True,
        **_get_qdrant_connection_kwargs(),
    )


def create_vectorstore(documents, embeddings, backend="faiss", collection_name=QDRANT_COLLECTION):
    if backend == "faiss":
        return create_faiss_vectorstore(documents, embeddings)
    if backend == "qdrant":
        return create_qdrant_vectorstore(documents, embeddings, collection_name=collection_name)
    raise ValueError(f"Unsupported backend: {backend}")


def save_vectorstore(vectorstore, path=FAISS_INDEX_PATH):
    vectorstore.save_local(path)


def load_faiss_vectorstore(path, embeddings):
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_qdrant_vectorstore(embeddings, collection_name=QDRANT_COLLECTION):
    client = _get_qdrant_client()
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def load_vectorstore(path, embeddings, backend="faiss", collection_name=QDRANT_COLLECTION):
    if backend == "faiss":
        return load_faiss_vectorstore(path, embeddings)
    if backend == "qdrant":
        return load_qdrant_vectorstore(embeddings, collection_name=collection_name)
    raise ValueError(f"Unsupported backend: {backend}")
