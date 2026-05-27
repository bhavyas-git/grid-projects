# Module 2 RAG Capstone

This project implements a phased Retrieval-Augmented Generation system over the IFC Annual Report 2024 Financials PDF. The app starts with text-only RAG and extends the same pipeline to evaluation, hybrid retrieval, semantic caching, table/image-aware multimodal RAG, and a ColPali-like visual patch retrieval path.

## Canonical Resources

The assignment resources are available at the expected reviewer paths:

- PDF: `resources/documents/ifc-annual-report-2024-financials.pdf`
- Evaluation dataset: `resources/datasets/rag_evaluation_dataset.csv`

Legacy local copies under `data/` and `evaluation/` are kept for compatibility. `src/config.py` prefers the canonical `resources/` paths when they exist.

## Technical Requirements Map

| Requirement | Implementation |
| --- | --- |
| Gemini 2.0 Flash or newer | `gemini-2.0-flash` in `src/config.py`; used through `src/llm.py` and `src/rag_chain.py`. |
| Google GenAI SDK with Vertex AI auth | `google.genai.Client(vertexai=True, project=..., location=...)` in `src/llm.py`; LangChain Google integrations also use `vertexai=True`. |
| No Gemini API-key path | The runtime expects GCP/Vertex credentials, not API keys. |
| Streaming | Standard RAG streaming in `src/rag_chain.py`; Phase 6 multimodal streaming in `src/phase6_pipeline.py`; rendered in `app/streamlit_app.py`. |
| Function calling | Gemini tool binding in `src/query_planner.py` selects retrieval filters from a user query. |
| JSON / structured output | Pydantic structured outputs for image captions, patch descriptions, and LLM-as-judge scoring. |
| UI | Streamlit app in `app/streamlit_app.py`. |
| Vector DBs | FAISS and Qdrant in `src/vectorstore.py`; selectable in the UI. |
| PDF processing | PyMuPDF via `fitz`; LangChain `PyPDFLoader` for text. |
| RAG framework | LangChain retrievers, vector stores, prompts, chains, documents, and tools. |
| Containerization | `Dockerfile` and `docker-compose.yml`; compose includes Qdrant. |
| Observability | Langfuse callback handler in `src/observability.py`, wired into the app and evaluation. |
| Evaluation | RAGAS metrics in `src/metrics.py`; LLM-as-judge in `src/judge.py`. |

## Running Locally

Use Vertex AI credentials. The project is intentionally configured for Vertex/GCP auth to match the assignment requirement.

```bash
cd /Users/bsama/PycharmProjects/RAG
source .venv/bin/activate
export GOOGLE_CLOUD_PROJECT="<your-gcp-project>"
export GOOGLE_CLOUD_LOCATION="global"
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

If you have not authenticated locally:

```bash
gcloud auth application-default login
```

## Running With Docker

```bash
cd /Users/bsama/PycharmProjects/RAG
export GOOGLE_CLOUD_PROJECT="<your-gcp-project>"
export GOOGLE_CLOUD_LOCATION="global"
docker compose up --build
```

Docker Compose starts Qdrant and the Streamlit app on port `8501`. It mounts local Google Cloud credentials from `${HOME}/.config/gcloud`.

## Implemented Phases

### Phase 1: Naive Text RAG

- Extracts text with page metadata in `src/ingestion.py`.
- Chunks documents with LangChain text splitting.
- Embeds chunks with Gemini embeddings in `src/embeddings.py`.
- Builds FAISS and Qdrant indexes through `build_index.py`.
- Retrieves top-k chunks through `src/retriever.py`.
- Generates answers with Gemini through `src/rag_chain.py`.
- Exposes the pipeline in Streamlit.
- Wires Langfuse callbacks through `src/observability.py`.

### Phase 2: RAG Pipeline Evaluation

- Uses `resources/datasets/rag_evaluation_dataset.csv`.
- Runs LLM-as-judge scoring in `src/judge.py`.
- Runs RAGAS metrics: faithfulness, answer relevancy, context precision, and context recall.
- Supports phase-aware evaluation through `evaluation/run_evaluation.py`.

### Phase 3: Hybrid Search and Re-ranking

- Implements dense retrieval plus sparse BM25-style retrieval.
- Merges dense and sparse candidates with reciprocal-rank style fusion.
- Applies metadata filtering for page ranges and content type.
- Uses a cross-encoder reranker when available.

### Phase 4: Advanced RAG

- Implements semantic caching in `src/semantic_cache.py`.
- Cache scope includes backend, phase, content type, and page filter.

### Phase 5: Multimodal RAG

- Extracts images from the PDF and asks Gemini for finance-focused captions.
- Extracts tables and saves each table as `.json`, `.md`, and `.csv`.
- Indexes text chunks, table representations, and image-caption documents together.
- Adds prompts for table, image, hybrid, and multimodal answer styles.

### Phase 6: ColPali-like Multimodal RAG

- Renders pages and segments document blocks into visual patches.
- Generates Gemini structured descriptions for visual/text patches.
- Stores patch images and page-level source metadata.
- Builds a MaxSim-style late-interaction retrieval representation over patch interaction units, including summary, visible text, visual signals, page context, and patch role.
- Sends retrieved patch images plus patch summaries to Gemini for multimodal answer generation.
- Displays patch source attribution and highlighted page regions in the UI.
- Provides a Phase 6 vs Phase 5 comparison option in the sidebar.

Note: Phase 6 is intentionally ColPali-like rather than a direct ColPali model integration. It approximates the late-interaction retrieval behavior with Gemini-generated visual patch summaries and Gemini text embeddings over patch interaction units. This keeps the implementation compatible with the assignment's Gemini/Vertex stack while still providing patch-level multimodal retrieval, visual context generation, and attribution.

## Rebuilding Indexes

Full rebuild:

```bash
python build_index.py --backend both
```

Useful variants:

```bash
python build_index.py --backend faiss
python build_index.py --backend qdrant
python build_index.py --backend both --skip-images
python build_index.py --backend both --skip-tables
```

The full rebuild can take a long time because it extracts/captions images and builds Phase 6 patch artifacts. Existing generated artifacts are committed in the workspace under `db/`, `outputs/`, and `evaluation/results/`.

Generated Phase 6 late-interaction artifact:

```text
db/phase6_patch_token_embeddings.json
```

## FAISS vs Qdrant Comparison

Run:

```bash
python compare_vectorstores.py
```

On macOS, if local ML/vector libraries hit an OpenMP duplicate-runtime error, use:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python compare_vectorstores.py
```

The script compares retrieval quality on assignment-style sample queries by recording top-k overlap, retrieved pages, source types, and evidence snippets. It writes:

```text
evaluation/results/faiss_vs_qdrant_retrieval_comparison.md
```

Observed comparison from the latest run showed identical top-k overlap for the tested sample queries, with both backends returning the same retrieved page sets.

## Evaluation Commands

Baseline Phase 1 over both vector stores:

```bash
python evaluation/run_evaluation.py --backend both --phase phase1
```

Later-phase regression checks:

```bash
python evaluation/run_evaluation.py --backend faiss --phase phase3
python evaluation/run_evaluation.py --backend faiss --phase phase5
python evaluation/run_evaluation.py --backend faiss --phase phase6
python evaluation/run_evaluation.py --backend faiss --phase all
```

Use the OpenMP workaround if needed:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python evaluation/run_evaluation.py --backend faiss --phase phase5
```

Latest generated summaries:

- `evaluation/results/phase2_summary_phase1_faiss.json`
- `evaluation/results/phase2_summary_phase1_qdrant.json`
- `evaluation/results/phase2_summary_phase3_faiss.json`
- `evaluation/results/phase2_summary_phase5_faiss.json`
- `evaluation/results/phase2_summary_phase6_faiss.json`

Latest observed judge averages:

| Phase | Backend | Judge average |
| --- | --- | ---: |
| Phase 1 | FAISS | 8.7647 |
| Phase 1 | Qdrant | 8.7647 |
| Phase 3 | FAISS | 8.2647 |
| Phase 5 | FAISS | 8.3235 |
| Phase 6 | FAISS | 2.8824 |

The Phase 6 score is lower because the provided evaluation dataset is mostly text/table-answer oriented, while Phase 6 intentionally emphasizes visual patch retrieval. The pipeline runs end to end, but Phase 5 is the stronger route for the current evaluation set.

## Main Entrypoints

- App: `streamlit run app/streamlit_app.py`
- Index build: `python build_index.py --backend both`
- Vector store comparison: `python compare_vectorstores.py`
- Evaluation: `python evaluation/run_evaluation.py --backend both --phase phase1`

## Environment Variables

Required:

- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`, defaults to `global`

Optional:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST`
- `LANGFUSE_ENABLED`, defaults to `true`
- `SEMANTIC_CACHE_ENABLED`, defaults to `true`
