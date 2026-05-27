import json
import math
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_PATH,
    SEMANTIC_CACHE_THRESHOLD,
    SEMANTIC_CACHE_VERSION,
)


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    return sum(left_value * right_value for left_value, right_value in zip(left_norm, right_norm))


def _serialize_documents(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]


def _deserialize_documents(payload: list[dict[str, Any]]) -> list[Document]:
    return [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {}),
        )
        for item in payload
    ]


class SemanticCache:
    def __init__(
        self,
        embeddings,
        cache_path: Path = SEMANTIC_CACHE_PATH,
        similarity_threshold: float = SEMANTIC_CACHE_THRESHOLD,
        enabled: bool = SEMANTIC_CACHE_ENABLED,
        cache_version: str = SEMANTIC_CACHE_VERSION,
    ) -> None:
        self.embeddings = embeddings
        self.cache_path = cache_path
        self.similarity_threshold = similarity_threshold
        self.enabled = enabled
        self.cache_version = cache_version
        self.entries = self._load_entries()

    def _load_entries(self) -> list[dict[str, Any]]:
        if not self.enabled or not self.cache_path.exists():
            return []
        with self.cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _persist(self) -> None:
        if not self.enabled:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")

    def lookup(self, query: str, backend: str) -> dict[str, Any] | None:
        if not self.enabled or not self.entries:
            return None

        query_vector = self.embeddings.embed_query(query)
        candidates = [
            entry
            for entry in self.entries
            if entry.get("backend") == backend
            and entry.get("cache_version") == self.cache_version
        ]
        if not candidates:
            return None

        best_entry: dict[str, Any] | None = None
        best_score = -1.0
        for entry in candidates:
            score = _cosine_similarity(query_vector, entry["query_embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self.similarity_threshold:
            return None

        return {
            "query": best_entry["query"],
            "result": best_entry["result"],
            "source_documents": _deserialize_documents(best_entry.get("source_documents", [])),
            "cache_hit": True,
            "similarity": round(best_score, 4),
        }

    def store(self, query: str, backend: str, result: str, source_documents: list[Document]) -> None:
        if not self.enabled:
            return

        entry = {
            "backend": backend,
            "cache_version": self.cache_version,
            "query": query,
            "query_embedding": self.embeddings.embed_query(query),
            "result": result,
            "source_documents": _serialize_documents(source_documents),
        }

        self.entries = [
            existing
            for existing in self.entries
            if not (
                existing.get("backend") == backend
                and existing.get("query") == query
                and existing.get("cache_version") == self.cache_version
            )
        ]
        self.entries.append(entry)
        self._persist()
