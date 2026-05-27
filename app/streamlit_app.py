import json
import re
import sys
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVAL_RESULTS_DIR, FAISS_INDEX_PATH, PATCH_ATTRIBUTION_DIR, PDF_PATH
from src.embeddings import get_embeddings
from src.observability import get_callbacks
from src.phase6_pipeline import Phase6MultimodalQA, Phase6PatchRetriever
from src.query_planner import plan_retrieval_filters
from src.retriever import get_retriever
from src.rag_chain import build_rag_chain, stream_rag_response
from src.semantic_cache import SemanticCache
from src.vectorstore import load_vectorstore

PHASE_OPTIONS = {
    "phase1": "Phase 1: Foundational Text-Based RAG System",
    "phase2": "Phase 2: RAG Evaluation",
    "phase3": "Phase 3: Hybrid Search & Re-ranking",
    "phase4": "Phase 4: Advanced RAG Techniques",
    "phase5": "Phase 5: Multimodal RAG",
    "phase6": "Phase 6: Multimodal RAG with ColPali-like Approach",
}

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


def _content_units(text: str) -> list[str]:
    paragraph = " ".join(line.strip() for line in text.splitlines() if line.strip())
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if not paragraph:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph)
        if sentence.strip()
    ]


def _is_sentence_like(text: str) -> bool:
    cleaned = text.strip()
    return len(cleaned) >= 40 and cleaned[-1:] in ".!?"


def _best_snippet(text: str, query: str, limit: int = 280) -> str:
    units = _content_units(text)
    if not units:
        return ""

    query_terms = _query_terms(query)
    candidates: list[str] = []
    for index, unit in enumerate(units):
        if _is_sentence_like(unit):
            candidates.append(unit)
        if index + 1 < len(units):
            combined = f"{unit} {units[index + 1]}".strip()
            if _is_sentence_like(combined):
                candidates.append(combined)

    if not candidates:
        candidates = units

    def _score(unit: str) -> tuple[int, int]:
        lowered = unit.lower()
        return (
            sum(term in lowered for term in query_terms),
            -abs(len(unit) - (limit // 2)),
        )

    if query_terms:
        snippet = max(candidates, key=_score)
    else:
        snippet = max(candidates, key=lambda unit: _score(unit)[1])

    snippet = " ".join(snippet.split())
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 1].rstrip() + "…"


def _source_title(index: int, metadata: dict) -> str:
    source_type = metadata.get("source_type", "text").title()
    page_number = metadata.get("page_number", metadata.get("page"))
    parts = [f"Source {index}", source_type]
    if page_number is not None:
        parts.append(f"Page {page_number}")
    approximate_section = metadata.get("approximate_section")
    if approximate_section:
        parts.append(approximate_section[:40])
    return " | ".join(parts)


def _source_annotation(metadata: dict, snippet: str) -> str:
    source_type = metadata.get("source_type", "text")
    if source_type == "table":
        columns = metadata.get("columns") or []
        if columns:
            return f"Relevant columns: {', '.join(columns[:4])}"
    if source_type == "image":
        entities = metadata.get("key_entities") or []
        if entities:
            return f"Relevant entities: {', '.join(entities[:4])}"
    if source_type == "visual_patch":
        signals = metadata.get("visual_signals") or []
        if signals:
            return f"Relevant visual signals: {', '.join(signals[:4])}"
        bbox = metadata.get("bbox")
        if bbox:
            return f"Patch region: {bbox}"
    return snippet


def _phase_caption(phase_key: str) -> str:
    if phase_key == "phase1":
        return "Text-only retrieval over the IFC 2024 annual report."
    if phase_key == "phase2":
        return "Evaluation summaries for the IFC RAG pipeline."
    if phase_key == "phase4":
        return "Advanced RAG view using the existing semantic cache with hybrid retrieval and reranking."
    if phase_key == "phase5":
        return "Multimodal retrieval across text, tables, and image-derived evidence from the IFC 2024 annual report."
    if phase_key == "phase6":
        return "Patch-based multimodal retrieval over page regions with visual-context generation and source attribution."
    return "Hybrid dense + sparse retrieval with metadata filtering and cross-encoder reranking."


def _phase_prompt_label(phase_key: str) -> str:
    if phase_key == "phase1":
        return "Ask a question for the text-only RAG system:"
    if phase_key == "phase4":
        return "Ask a question for the advanced RAG system:"
    if phase_key == "phase5":
        return "Ask a question for the multimodal RAG system:"
    if phase_key == "phase6":
        return "Ask a question for the patch-based multimodal RAG system:"
    return "Ask a question for the hybrid retrieval system:"


def _load_summary(backend: str) -> dict | None:
    summary_path = EVAL_RESULTS_DIR / f"phase2_summary_{backend}.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _looks_figure_related(document_text: str, metadata: dict) -> bool:
    lowered = document_text.lower()
    if metadata.get("source_type") == "image":
        return True
    if metadata.get("source_type") == "visual_patch":
        return True
    return any(token in lowered for token in ("figure ", "chart ", "graph ", "income measures"))


@st.cache_resource(show_spinner=False)
def load_page_preview(page_number: int) -> str | None:
    if page_number is None or page_number <= 0:
        return None

    preview_dir = PROJECT_ROOT / "outputs" / "page_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"page_{page_number:03d}.png"
    if preview_path.exists():
        return str(preview_path)

    pdf = fitz.open(PDF_PATH)
    try:
        if page_number - 1 >= len(pdf):
            return None
        page = pdf[page_number - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        pixmap.save(preview_path)
        return str(preview_path)
    finally:
        pdf.close()


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def load_vector_backend(selected_backend: str):
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(FAISS_INDEX_PATH, embeddings, backend=selected_backend)
    cache = SemanticCache(embeddings)
    return embeddings, vectorstore, cache


@st.cache_resource(show_spinner=False)
def load_phase6_backend(selected_backend: str, start_page: int, end_page: int):
    embeddings = load_embeddings()
    cache = SemanticCache(embeddings)
    page_range = None if start_page <= 0 or end_page <= 0 else (min(start_page, end_page), max(start_page, end_page))
    retriever = Phase6PatchRetriever(
        embeddings=embeddings,
        backend=selected_backend,
        page_range=page_range,
    )
    return retriever, Phase6MultimodalQA(retriever), cache


@st.cache_data(show_spinner=False)
def load_patch_attribution(page_image_path: str, bbox: list[float], patch_index: int) -> str | None:
    source_path = Path(page_image_path)
    if not source_path.exists() or len(bbox) != 4:
        return None
    PATCH_ATTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PATCH_ATTRIBUTION_DIR / f"{source_path.stem}_patch_{patch_index:03d}.png"
    if output_path.exists():
        return str(output_path)

    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    pdf = fitz.open(PDF_PATH)
    try:
        page = pdf[int(source_path.stem.split("_")[-1]) - 1]
        scale_x = width / page.rect.width
        scale_y = height / page.rect.height
    finally:
        pdf.close()

    x0, y0, x1, y1 = bbox
    draw.rectangle(
        [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
        outline=(220, 20, 60),
        width=5,
    )
    image.save(output_path)
    return str(output_path)


def load_backend(
    selected_backend: str,
    phase_key: str,
    content_type: str,
    start_page: int,
    end_page: int,
):
    embeddings, vectorstore, cache = load_vector_backend(selected_backend)
    page_range = None if start_page <= 0 or end_page <= 0 else (min(start_page, end_page), max(start_page, end_page))
    retriever = get_retriever(
        vectorstore,
        backend=selected_backend,
        phase=phase_key,
        content_type=content_type,
        page_range=page_range,
    )
    reranker_configured = phase_key in {"phase3", "phase4", "phase5"} and hasattr(retriever, "cross_encoder")
    answer_style = "default"
    if phase_key == "phase5":
        answer_style = "multimodal"
    elif phase_key in {"phase3", "phase4"}:
        if content_type == "table":
            answer_style = "table"
        elif content_type == "image":
            answer_style = "image"
        else:
            answer_style = "hybrid"
    return retriever, build_rag_chain(retriever, answer_style=answer_style), cache, reranker_configured, answer_style


def _stream_response(qa_chain, query: str, callbacks: list | None, phase_key: str, answer_style: str = "default") -> dict:
    config = {"callbacks": callbacks} if callbacks else None
    chunks: list[str] = []
    source_documents: list = []
    placeholder = st.empty()

    if phase_key == "phase6":
        stream = qa_chain.stream({"query": query}, config=config)
    else:
        stream = stream_rag_response(
            qa_chain.retriever,
            query,
            answer_style=answer_style,
            config=config,
        )

    for chunk, documents in stream:
        chunks.append(chunk)
        source_documents = documents
        placeholder.markdown("".join(chunks))

    result = "".join(chunks).strip()
    if not result:
        response = qa_chain.invoke({"query": query}, config=config)
        return response

    return {
        "result": result,
        "source_documents": source_documents,
    }


def _render_source_documents(documents: list, query: str, phase_key: str) -> None:
    with st.expander("Source Citations"):
        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            snippet = _best_snippet(document.page_content, query)
            annotation = _source_annotation(metadata, snippet)
            st.markdown(f"**{_source_title(index, metadata)}**")
            st.caption(annotation)
            if snippet and annotation != snippet:
                st.write(snippet)

            image_path = metadata.get("image_path")
            patch_path = metadata.get("patch_path")
            if patch_path and phase_key == "phase6":
                attribution_path = load_patch_attribution(
                    metadata.get("page_image_path", ""),
                    metadata.get("bbox") or [],
                    int(metadata.get("patch_index", index)),
                )
                if attribution_path:
                    st.image(attribution_path, caption=f"Highlighted patch on page {metadata.get('page_number')}")
                st.image(patch_path, caption=Path(patch_path).name)
            elif image_path:
                st.image(image_path, caption=Path(image_path).name)
            elif _looks_figure_related(document.page_content, metadata):
                preview_path = load_page_preview(metadata.get("page_number", metadata.get("page")))
                if preview_path:
                    st.image(preview_path, caption=f"Page {metadata.get('page_number', metadata.get('page'))} preview")

            table_path = metadata.get("table_markdown_path")
            if table_path:
                st.caption(f"Table file: {Path(table_path).name}")
            if index < len(documents):
                st.divider()


st.title("📊 IFC RAG Chatbot")

phase_key = st.sidebar.selectbox(
    "Project phase",
    options=list(PHASE_OPTIONS.keys()),
    format_func=lambda key: PHASE_OPTIONS[key],
)
backend = st.sidebar.selectbox("Vector store", ["faiss", "qdrant"])
show_sources = st.sidebar.checkbox("Show retrieved sources", value=True)

content_type = "all"
start_page = 0
end_page = 0
if phase_key in {"phase1", "phase3", "phase4", "phase5", "phase6"}:
    st.sidebar.markdown("**Metadata filters**")
    if phase_key in {"phase3", "phase4", "phase5"}:
        content_type = st.sidebar.selectbox("Content type", ["all", "text", "table", "image"])
    start_page = st.sidebar.number_input("Start page", min_value=0, value=0, step=1)
    end_page = st.sidebar.number_input("End page", min_value=0, value=0, step=1)
compare_with_phase5 = phase_key == "phase6" and st.sidebar.checkbox("Compare with Phase 5", value=True)

st.header(PHASE_OPTIONS[phase_key])
st.caption(_phase_caption(phase_key))

if phase_key == "phase2":
    st.write(
        "This phase uses the synthetic Q&A evaluation dataset, RAGAS metrics, and an LLM-as-judge workflow."
    )
    for selected_backend in ("faiss", "qdrant"):
        summary = _load_summary(selected_backend)
        st.subheader(selected_backend.upper())
        if summary is None:
            st.info(
                f"No evaluation summary found for `{selected_backend}`. Run `python evaluation/run_evaluation.py --backend {selected_backend}`."
            )
            continue
        st.json(summary)
        st.caption(f"Raw results: {summary['raw_results_path']}")
else:
    callbacks = get_callbacks()
    if phase_key == "phase6":
        retriever, qa_chain, semantic_cache = load_phase6_backend(
            backend,
            int(start_page),
            int(end_page),
        )
        reranker_configured = True
        answer_style = "multimodal"
    else:
        retriever, qa_chain, semantic_cache, reranker_configured, answer_style = load_backend(
            backend,
            phase_key,
            content_type,
            int(start_page),
            int(end_page),
        )

    if phase_key in {"phase3", "phase4", "phase5"}:
        if reranker_configured:
            st.caption("Hybrid retrieval is active with metadata filtering and cross-encoder reranking support.")
        else:
            st.caption("Hybrid retrieval is active.")
    if phase_key == "phase4":
        st.caption("Semantic caching is active in this phase and will reuse answers for similar queries when possible.")
    if phase_key == "phase5":
        st.caption("This phase is designed for multimodal questions spanning text, tables, and image-derived chart evidence.")
    if phase_key == "phase6":
        st.caption("This phase retrieves page patches, provides them directly to the multimodal model, and highlights the contributing patch regions.")

    query = st.text_input(_phase_prompt_label(phase_key))

    if query:
        if phase_key in {"phase3", "phase5"} and content_type == "all" and int(start_page) == 0 and int(end_page) == 0:
            try:
                planned_filters = plan_retrieval_filters(query)
                planned_content_type = planned_filters.get("content_type", "all")
                if planned_content_type in {"text", "table", "image"}:
                    retriever, qa_chain, semantic_cache, reranker_configured, answer_style = load_backend(
                        backend,
                        phase_key,
                        planned_content_type,
                        int(planned_filters.get("start_page", 0)),
                        int(planned_filters.get("end_page", 0)),
                    )
                    content_type = planned_content_type
                    start_page = int(planned_filters.get("start_page", 0))
                    end_page = int(planned_filters.get("end_page", 0))
            except Exception:
                pass
        cache_scope = f"{backend}:{phase_key}:{content_type}:{int(start_page)}:{int(end_page)}"
        response = semantic_cache.lookup(query, cache_scope)
        if response is None:
            response = _stream_response(qa_chain, query, callbacks, phase_key, answer_style)
            semantic_cache.store(query, cache_scope, response["result"], response["source_documents"])

        if response.get("cache_hit"):
            st.write(response["result"])
        if response.get("cache_hit"):
            st.caption(f"Retrieved from semantic cache (similarity: {response['similarity']}).")
        if show_sources:
            _render_source_documents(response["source_documents"], query, phase_key)
        if phase_key == "phase6" and compare_with_phase5:
            _, phase5_chain, _, _, phase5_answer_style = load_backend(
                backend,
                "phase5",
                "all",
                int(start_page),
                int(end_page),
            )
            st.subheader("Phase 6 vs Phase 5")
            st.markdown("**Phase 5 baseline answer**")
            comparison = _stream_response(phase5_chain, query, callbacks, "phase5", phase5_answer_style)
            if show_sources:
                _render_source_documents(comparison["source_documents"], query, "phase5")
