from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


class BatchedGoogleEmbeddings(Embeddings):
    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE) -> None:
        self.batch_size = batch_size
        self._client = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            vertexai=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(
                self._client.embed_documents(
                    batch,
                    batch_size=self.batch_size,
                    task_type="RETRIEVAL_DOCUMENT",
                )
            )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text, task_type="RETRIEVAL_QUERY")


def get_embeddings() -> Embeddings:
    return BatchedGoogleEmbeddings()
