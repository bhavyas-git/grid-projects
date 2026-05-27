import math
import re
from collections import Counter

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from src.config import (
    CROSS_ENCODER_MODEL,
    HYBRID_CANDIDATES,
    HYBRID_RERANK_LIMIT,
    RETRIEVAL_CANDIDATES,
    TOP_K,
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _query_terms(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9']+", query.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _query_intent(query: str) -> dict[str, bool]:
    lowered = query.lower()
    return {
        "asks_numeric": any(token in lowered for token in ("what was", "value", "amount", "income", "assets", "net income", "designation", "fy24", "fy23", "2024", "2023")),
        "asks_explanation": any(token in lowered for token in ("why", "update", "explanation", "drivers", "reason", "main drivers", "what update")),
        "asks_visual": any(token in lowered for token in ("figure", "chart", "graph", "trend", "image")),
        "asks_table": any(token in lowered for token in ("table", "columns", "row")),
    }


def _parse_query_page_range(query: str) -> tuple[int | None, int | None]:
    match = re.search(r"pages?\s+(\d+)\s*[-to]+\s*(\d+)", query.lower())
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return min(start, end), max(start, end)
    exact_match = re.search(r"page\s+(\d+)", query.lower())
    if exact_match:
        page = int(exact_match.group(1))
        return page, page
    return None, None


def _document_key(document: Document) -> tuple[str, tuple]:
    metadata_items = tuple(sorted((key, str(value)) for key, value in document.metadata.items()))
    return document.page_content, metadata_items


def _metadata_matches(
    document: Document,
    page_range: tuple[int | None, int | None] | None,
    content_type: str = "all",
) -> bool:
    source_type = document.metadata.get("source_type", document.metadata.get("content_type", "text"))
    if content_type != "all":
        if content_type == "image":
            text = document.page_content.lower()
            is_figure_text = source_type == "text" and any(token in text for token in ("figure ", "chart ", "income measures", "graph "))
            if source_type != "image" and not is_figure_text:
                return False
        elif source_type != content_type:
            return False

    if page_range:
        start_page, end_page = page_range
        page_number = document.metadata.get("page_number", document.metadata.get("page"))
        if page_number is None:
            return False
        if start_page is not None and page_number < start_page:
            return False
        if end_page is not None and page_number > end_page:
            return False
    return True


def _heuristic_rerank_score(query: str, document: Document) -> tuple[float, float]:
    query_terms = _query_terms(query)
    intent = _query_intent(query)
    text = document.page_content.lower()
    overlap = sum(term in text for term in query_terms)
    phrase_bonus = 0.0

    key_phrases = [
        "total assets",
        "net income",
        "income available",
        "disbursed investment portfolio",
        "borrowings outstanding",
        "charges on borrowings",
    ]
    for phrase in key_phrases:
        if phrase in query.lower() and phrase in text:
            phrase_bonus += 2.0

    year_matches = re.findall(r"\b20\d{2}\b|\bfy\d{2}\b", query.lower())
    if year_matches:
        phrase_bonus += 0.5 * sum(match in text for match in year_matches)

    source_type = document.metadata.get("source_type", "text")
    source_bonus = 0.0
    if intent["asks_numeric"] and source_type in {"table", "image"}:
        source_bonus += 1.0
    if intent["asks_numeric"] and source_type == "text" and any(token in text for token in ("figure ", "chart ", "income measures")):
        source_bonus += 1.5
    if intent["asks_explanation"] and source_type == "text":
        source_bonus += 1.5
    if intent["asks_table"] and source_type == "table":
        source_bonus += 2.0
    if intent["asks_visual"] and source_type == "image":
        source_bonus += 2.0
    if source_type == "image" and any(token in text for token in ("signature", "president", "auditor", "deloitte", "logo")):
        source_bonus -= 3.0
    if source_type == "text" and any(token in text for token in ("independent auditor", "management's report regarding effectiveness")):
        source_bonus -= 2.0
    if any(character.isdigit() for character in query) and any(character.isdigit() for character in document.page_content):
        source_bonus += 0.5

    return overlap + phrase_bonus + source_bonus, -document.metadata.get("page_number", 0)


def _diversify_documents(query: str, documents: list[Document], top_k: int) -> list[Document]:
    if not documents:
        return []

    intent = _query_intent(query)
    if intent["asks_visual"]:
        image_first = sorted(
            documents,
            key=lambda document: (
                document.metadata.get("source_type") != "image",
                -_heuristic_rerank_score(query, document)[0],
            ),
        )
        return image_first[:top_k]

    if not (intent["asks_numeric"] and intent["asks_explanation"]):
        return documents[:top_k]

    selected: list[Document] = []
    seen = set()

    for preferred_type in ("table", "image", "text"):
        for document in documents:
            key = _document_key(document)
            if key in seen:
                continue
            if document.metadata.get("source_type") == preferred_type:
                selected.append(document)
                seen.add(key)
                break

    for document in documents:
        key = _document_key(document)
        if key in seen:
            continue
        selected.append(document)
        seen.add(key)
        if len(selected) >= top_k:
            break

    return selected[:top_k]


def _combine_page_ranges(
    user_page_range: tuple[int | None, int | None] | None,
    query_page_range: tuple[int | None, int | None],
) -> tuple[int | None, int | None] | None:
    if user_page_range and any(value is not None for value in user_page_range):
        return user_page_range
    if any(value is not None for value in query_page_range):
        return query_page_range
    return None


class SparseRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.doc_term_counts = [Counter(self._tokenize(document.page_content)) for document in documents]
        self.doc_lengths = [sum(counter.values()) for counter in self.doc_term_counts]
        self.average_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for term_counts in self.doc_term_counts:
            document_frequency.update(term_counts.keys())
        self.idf = {
            term: math.log(1 + ((len(documents) - frequency + 0.5) / (frequency + 0.5)))
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[A-Za-z0-9']+", text.lower())
            if len(token) > 2 and token not in STOPWORDS
        ]

    def score(self, query: str, index: int, k1: float = 1.5, b: float = 0.75) -> float:
        query_terms = self._tokenize(query)
        term_counts = self.doc_term_counts[index]
        document_length = self.doc_lengths[index] or 1
        score = 0.0
        for term in query_terms:
            if term not in term_counts:
                continue
            frequency = term_counts[term]
            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (1 - b + b * document_length / max(self.average_doc_length, 1))
            score += self.idf.get(term, 0.0) * (numerator / denominator)
        return score

    def search(
        self,
        query: str,
        limit: int,
        page_range: tuple[int | None, int | None] | None,
        content_type: str,
    ) -> list[tuple[Document, float]]:
        scored: list[tuple[Document, float]] = []
        for index, document in enumerate(self.documents):
            if not _metadata_matches(document, page_range=page_range, content_type=content_type):
                continue
            score = self.score(query, index)
            if score > 0:
                scored.append((document, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self.available = False

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self.available = True
        except Exception:
            self._model = None
            self.available = False
        return self._model

    def rerank(self, query: str, documents: list[Document], limit: int) -> list[Document]:
        model = self._load()
        if model is None or not documents:
            return documents[:limit]
        pairs = [(query, document.page_content) for document in documents]
        scores = model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
        return [document for document, _ in ranked[:limit]]


def _load_all_documents(vectorstore, backend: str) -> list[Document]:
    if backend == "faiss":
        docstore = getattr(vectorstore, "docstore", None)
        if docstore is None:
            return []
        return list(getattr(docstore, "_dict", {}).values())

    if backend == "qdrant":
        client = getattr(vectorstore, "client", None)
        collection_name = getattr(vectorstore, "collection_name", None)
        if client is None or collection_name is None:
            return []

        documents: list[Document] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                content = payload.get("page_content") or payload.get("text") or ""
                metadata = payload.get("metadata") or {}
                documents.append(Document(page_content=content, metadata=metadata))
            if offset is None:
                break
        return documents

    return []


class BasePhaseRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: object
    backend: str = "faiss"
    top_k: int = TOP_K
    candidate_k: int = RETRIEVAL_CANDIDATES
    content_type: str = "all"
    page_range: tuple[int | None, int | None] | None = None

    def _effective_page_range(self, query: str) -> tuple[int | None, int | None] | None:
        return _combine_page_ranges(self.page_range, _parse_query_page_range(query))


class DenseMetadataRetriever(BasePhaseRetriever):
    def _get_relevant_documents(self, query: str) -> list[Document]:
        page_range = self._effective_page_range(query)
        candidates = self.vectorstore.similarity_search(query, k=max(self.candidate_k, self.top_k * 3))
        filtered = [
            document
            for document in candidates
            if _metadata_matches(document, page_range=page_range, content_type=self.content_type)
        ]
        reranked = sorted(filtered, key=lambda document: _heuristic_rerank_score(query, document), reverse=True)
        return _diversify_documents(query, reranked, self.top_k)


class HybridRerankRetriever(BasePhaseRetriever):
    all_documents: list[Document]
    hybrid_candidates: int = HYBRID_CANDIDATES
    rerank_limit: int = HYBRID_RERANK_LIMIT
    sparse_retriever: SparseRetriever | None = None
    cross_encoder: CrossEncoderReranker | None = None

    def model_post_init(self, __context) -> None:
        if self.sparse_retriever is None:
            self.sparse_retriever = SparseRetriever(self.all_documents)
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoderReranker()

    def _get_relevant_documents(self, query: str) -> list[Document]:
        page_range = self._effective_page_range(query)
        dense_docs = self.vectorstore.similarity_search(query, k=self.hybrid_candidates)
        sparse_results = self.sparse_retriever.search(
            query=query,
            limit=self.hybrid_candidates,
            page_range=page_range,
            content_type=self.content_type,
        )

        dense_rank = {
            _document_key(document): rank
            for rank, document in enumerate(
                document
                for document in dense_docs
                if _metadata_matches(document, page_range=page_range, content_type=self.content_type)
            )
        }
        sparse_rank = {
            _document_key(document): rank
            for rank, (document, _) in enumerate(sparse_results)
        }

        merged: dict[tuple[str, tuple], Document] = {}
        for document in dense_docs:
            if _metadata_matches(document, page_range=page_range, content_type=self.content_type):
                merged[_document_key(document)] = document
        for document, _ in sparse_results:
            merged[_document_key(document)] = document

        def reciprocal_rank_score(key: tuple[str, tuple]) -> float:
            dense_component = 1 / (60 + dense_rank[key]) if key in dense_rank else 0.0
            sparse_component = 1 / (60 + sparse_rank[key]) if key in sparse_rank else 0.0
            return dense_component + sparse_component

        fused = sorted(
            merged.items(),
            key=lambda item: (
                reciprocal_rank_score(item[0]),
                _heuristic_rerank_score(query, item[1]),
            ),
            reverse=True,
        )
        rerank_input = [document for _, document in fused[: self.rerank_limit]]
        reranked = self.cross_encoder.rerank(query, rerank_input, self.top_k)
        if len(reranked) < self.top_k:
            seen = {_document_key(document) for document in reranked}
            for _, document in fused:
                key = _document_key(document)
                if key in seen:
                    continue
                reranked.append(document)
                seen.add(key)
                if len(reranked) == self.top_k:
                    break
        return _diversify_documents(query, reranked, self.top_k)


def get_retriever(
    vectorstore,
    *,
    backend: str = "faiss",
    phase: str = "default",
    content_type: str = "all",
    page_range: tuple[int | None, int | None] | None = None,
):
    if phase == "phase1":
        return DenseMetadataRetriever(
            vectorstore=vectorstore,
            backend=backend,
            content_type="text",
            page_range=page_range,
        )

    if phase in {"phase3", "phase4", "phase5"}:
        return HybridRerankRetriever(
            vectorstore=vectorstore,
            backend=backend,
            content_type=content_type,
            page_range=page_range,
            all_documents=_load_all_documents(vectorstore, backend),
        )

    return DenseMetadataRetriever(
        vectorstore=vectorstore,
        backend=backend,
        content_type=content_type,
        page_range=page_range,
    )
