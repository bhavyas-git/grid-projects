import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.config import (
    PHASE6_FAISS_INDEX_PATH,
    PHASE6_PATCH_EMBEDDINGS_PATH,
    PHASE6_PATCH_STORE_PATH,
    PHASE6_PATCH_TOKEN_EMBEDDINGS_PATH,
    PHASE6_QDRANT_COLLECTION,
    PHASE6_RETRIEVAL_CANDIDATES,
    PHASE6_TOP_K,
)
from src.llm import get_chat_model
from src.vectorstore import load_vectorstore


def _document_key(document: Document) -> tuple[str, tuple]:
    metadata_items = tuple(sorted((key, str(value)) for key, value in document.metadata.items()))
    return document.page_content, metadata_items


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _query_facets(query: str) -> list[str]:
    facets = [query.strip()]
    lowered = query.lower()
    if " and " in lowered:
        facets.extend(part.strip() for part in query.split(" and ") if part.strip())
    if "," in query:
        facets.extend(part.strip() for part in query.split(",") if part.strip())
    seen: list[str] = []
    for facet in facets:
        if facet and facet not in seen:
            seen.append(facet)
    return seen


@dataclass
class Phase6Answer:
    result: str
    source_documents: list[Document]
    comparison: dict | None = None


class Phase6PatchRetriever:
    def __init__(
        self,
        embeddings,
        backend: str = "faiss",
        top_k: int = PHASE6_TOP_K,
        candidate_k: int = PHASE6_RETRIEVAL_CANDIDATES,
        page_range: tuple[int | None, int | None] | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.backend = backend
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.page_range = page_range
        self.vectorstore = load_vectorstore(
            PHASE6_FAISS_INDEX_PATH,
            embeddings,
            backend=backend,
            collection_name=PHASE6_QDRANT_COLLECTION,
        )
        self.documents, self.embedding_lookup, self.interaction_lookup = self._load_patch_store()

    def _load_patch_store(
        self,
    ) -> tuple[list[Document], dict[tuple[str, tuple], np.ndarray], dict[tuple[str, tuple], list[np.ndarray]]]:
        records = json.loads(PHASE6_PATCH_STORE_PATH.read_text(encoding="utf-8"))
        matrix = np.load(PHASE6_PATCH_EMBEDDINGS_PATH)
        token_records = []
        if PHASE6_PATCH_TOKEN_EMBEDDINGS_PATH.exists():
            token_records = json.loads(PHASE6_PATCH_TOKEN_EMBEDDINGS_PATH.read_text(encoding="utf-8"))
        documents: list[Document] = []
        lookup: dict[tuple[str, tuple], np.ndarray] = {}
        interaction_lookup: dict[tuple[str, tuple], list[np.ndarray]] = {}
        for index, record in enumerate(records):
            document = Document(
                page_content=record["page_content"],
                metadata=record["metadata"],
            )
            documents.append(document)
            key = _document_key(document)
            lookup[key] = _normalize(matrix[index])
            if index < len(token_records):
                interaction_lookup[key] = [
                    _normalize(np.array(vector, dtype=np.float32))
                    for vector in token_records[index].get("vectors", [])
                ]
        return documents, lookup, interaction_lookup

    def _facet_vectors(self, query: str) -> list[np.ndarray]:
        vectors = [np.array(self.embeddings.embed_query(facet), dtype=np.float32) for facet in _query_facets(query)]
        return [_normalize(vector) for vector in vectors]

    def _page_allowed(self, document: Document) -> bool:
        if not self.page_range:
            return True
        start_page, end_page = self.page_range
        page_number = document.metadata.get("page_number")
        if page_number is None:
            return False
        if start_page is not None and page_number < start_page:
            return False
        if end_page is not None and page_number > end_page:
            return False
        return True

    @staticmethod
    def _maxsim_score(query_vectors: list[np.ndarray], patch_vectors: list[np.ndarray]) -> float:
        if not query_vectors or not patch_vectors:
            return 0.0
        score = 0.0
        for query_vector in query_vectors:
            score += max(float(np.dot(query_vector, patch_vector)) for patch_vector in patch_vectors)
        return score

    def get_relevant_documents(self, query: str) -> list[Document]:
        dense_candidates = self.vectorstore.similarity_search(query, k=self.candidate_k)
        dense_candidates = [document for document in dense_candidates if self._page_allowed(document)]
        facet_vectors = self._facet_vectors(query)
        ranked: list[tuple[float, Document]] = []

        for dense_rank, document in enumerate(dense_candidates):
            document_vector = self.embedding_lookup.get(_document_key(document))
            if document_vector is None:
                continue
            interaction_vectors = self.interaction_lookup.get(_document_key(document), [])
            if interaction_vectors:
                facet_score = self._maxsim_score(facet_vectors, interaction_vectors)
            else:
                facet_score = sum(float(np.dot(facet_vector, document_vector)) for facet_vector in facet_vectors)
            text = document.page_content.lower()
            metadata = document.metadata
            heuristic = 0.0
            if any(token in query.lower() for token in ("figure", "chart", "image", "graph")):
                heuristic += 0.2 * any(token in text for token in ("chart", "graph", "figure"))
            if any(token in query.lower() for token in ("fy24", "fy23", "2024", "2023", "income", "assets", "net income")) and any(character.isdigit() for character in text):
                heuristic += 0.15
            if metadata.get("patch_role") == "image_patch":
                heuristic += 0.05
            score = facet_score + heuristic + (1 / (60 + dense_rank))
            ranked.append((score, document))

        ranked.sort(key=lambda item: (item[0], -item[1].metadata.get("page_number", 0)), reverse=True)
        return [document for _, document in ranked[: self.top_k]]

    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


class Phase6MultimodalQA:
    def __init__(self, retriever: Phase6PatchRetriever) -> None:
        self.retriever = retriever
        self.llm = get_chat_model()

    @staticmethod
    def _context_block(documents: list[Document]) -> str:
        blocks = []
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            blocks.append(
                (
                    f"[Patch {index}] page={metadata.get('page_number')} patch={metadata.get('patch_index')} "
                    f"bbox={metadata.get('bbox')} role={metadata.get('patch_role')}\n{document.page_content}"
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _image_message_parts(documents: list[Document]) -> list[dict]:
        parts: list[dict] = []
        for document in documents:
            patch_path = Path(document.metadata["patch_path"])
            if not patch_path.exists():
                continue
            encoded = base64.b64encode(patch_path.read_bytes()).decode("utf-8")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        return parts

    def invoke(self, inputs: dict, config: dict | None = None) -> dict:
        query = inputs["query"]
        documents = self.retriever.get_relevant_documents(query)
        message = self._build_message(query, documents)
        result = self.llm.invoke([message], config=config)
        return {
            "result": getattr(result, "content", str(result)),
            "source_documents": documents,
        }

    def stream(self, inputs: dict, config: dict | None = None):
        query = inputs["query"]
        documents = self.retriever.get_relevant_documents(query)
        message = self._build_message(query, documents)
        for chunk in self.llm.stream([message], config=config):
            yield getattr(chunk, "content", str(chunk)), documents

    def _build_message(self, query: str, documents: list[Document]) -> HumanMessage:
        context = self._context_block(documents)
        return HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You are answering questions about the IFC 2024 annual report.\n"
                        "Use both the textual patch summaries and the attached document patch images.\n"
                        "Answer only from the provided evidence. If the evidence is incomplete, say so plainly.\n"
                        "When citing a value from a chart, table, or visual patch, state that it comes from the retrieved visual context.\n"
                        "If the question compares values, use a compact markdown table.\n\n"
                        f"Question: {query}\n\n"
                        f"Patch context:\n{context}"
                    ),
                },
                *self._image_message_parts(documents),
            ]
        )
