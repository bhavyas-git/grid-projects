import json
from pathlib import Path

from src.config import EVAL_RESULTS_DIR, FAISS_INDEX_PATH
from src.embeddings import get_embeddings
from src.retriever import get_retriever
from src.vectorstore import load_vectorstore


def _document_signature(document) -> str:
    metadata = document.metadata
    return f"{metadata.get('source_type', 'text')}:page-{metadata.get('page_number', metadata.get('page', 'unknown'))}:{document.page_content[:80]}"


def _retrieve(backend: str, queries: list[str]) -> dict:
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(FAISS_INDEX_PATH, embeddings, backend=backend)
    retriever = get_retriever(vectorstore, backend=backend, phase="phase1")
    results = {}
    for query in queries:
        documents = retriever.invoke(query)
        results[query] = [
            {
                "rank": index,
                "signature": _document_signature(document),
                "source_type": document.metadata.get("source_type", "text"),
                "page_number": document.metadata.get("page_number", document.metadata.get("page")),
                "snippet": " ".join(document.page_content.split())[:240],
            }
            for index, document in enumerate(documents, start=1)
        ]
    return results


def _quality_rows(faiss_results: dict, qdrant_results: dict) -> list[dict]:
    rows = []
    for query, faiss_docs in faiss_results.items():
        qdrant_docs = qdrant_results[query]
        faiss_signatures = {document["signature"] for document in faiss_docs}
        qdrant_signatures = {document["signature"] for document in qdrant_docs}
        overlap = len(faiss_signatures & qdrant_signatures)
        rows.append(
            {
                "query": query,
                "top_k_overlap": overlap,
                "faiss_pages": [document["page_number"] for document in faiss_docs],
                "qdrant_pages": [document["page_number"] for document in qdrant_docs],
                "faiss_source_types": [document["source_type"] for document in faiss_docs],
                "qdrant_source_types": [document["source_type"] for document in qdrant_docs],
            }
        )
    return rows


def summarize_use_case(faiss_metrics: dict, qdrant_metrics: dict) -> dict:
    faiss_latency = faiss_metrics["average_latency_seconds"]
    qdrant_latency = qdrant_metrics["average_latency_seconds"]
    faster_backend = "faiss" if faiss_latency <= qdrant_latency else "qdrant"
    return {
        "faster_backend": faster_backend,
        "recommended_use_cases": {
            "faiss": "Best for simple local experimentation and lightweight single-node usage.",
            "qdrant": "Best when you want a dedicated vector database interface and easier future scaling.",
        },
    }


def _write_markdown_report(path: Path, quality_rows: list[dict], faiss_results: dict, qdrant_results: dict) -> None:
    lines = [
        "# FAISS vs Qdrant Retrieval Comparison",
        "",
        "This report compares the Phase 1 text retriever over the same IFC Annual Report embeddings.",
        "",
        "| Query | Top-k overlap | FAISS pages | Qdrant pages |",
        "| --- | ---: | --- | --- |",
    ]
    for row in quality_rows:
        lines.append(
            f"| {row['query']} | {row['top_k_overlap']} | {row['faiss_pages']} | {row['qdrant_pages']} |"
        )
    lines.extend(["", "## Retrieved Evidence", ""])
    for query in faiss_results:
        lines.extend([f"### {query}", "", "FAISS:"])
        lines.extend(
            f"- rank {document['rank']}, page {document['page_number']}, {document['source_type']}: {document['snippet']}"
            for document in faiss_results[query]
        )
        lines.append("")
        lines.append("Qdrant:")
        lines.extend(
            f"- rank {document['rank']}, page {document['page_number']}, {document['source_type']}: {document['snippet']}"
            for document in qdrant_results[query]
        )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    queries = [
        "What is IFC's mission?",
        "What was the net income for FY24 and FY23?",
        "Show me the trend of IFC's net income from FY22 to FY24.",
    ]
    faiss_results = _retrieve("faiss", queries)
    qdrant_results = _retrieve("qdrant", queries)
    quality_rows = _quality_rows(faiss_results, qdrant_results)
    report_path = EVAL_RESULTS_DIR / "faiss_vs_qdrant_retrieval_comparison.md"
    _write_markdown_report(report_path, quality_rows, faiss_results, qdrant_results)
    print(json.dumps({"quality": quality_rows, "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
