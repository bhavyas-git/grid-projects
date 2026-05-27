import json
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from src.config import (
    DEFAULT_EVAL_DATASET,
    EVAL_RESULTS_DIR,
    FAISS_INDEX_PATH,
    PHASE2_EVAL_DATASET,
    RAGAS_ANSWER_RELEVANCY_STRICTNESS,
    RAGAS_BATCH_SIZE,
    RAGAS_MAX_RETRIES,
    RAGAS_MAX_WAIT_SECONDS,
    RAGAS_MAX_WORKERS,
    RAGAS_TIMEOUT_SECONDS,
)
from src.embeddings import get_embeddings
from src.evaluator import run_rag_evaluation
from src.llm import get_chat_model
from src.metrics import evaluate_ragas as evaluate_ragas_metrics
from src.observability import get_callbacks
from src.phase6_pipeline import Phase6MultimodalQA, Phase6PatchRetriever
from src.rag_chain import build_rag_chain
from src.retriever import get_retriever
from src.vectorstore import load_vectorstore


def load_eval_samples(dataset_path: Path = DEFAULT_EVAL_DATASET) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_phase2_dataset(dataset_path: Path = PHASE2_EVAL_DATASET) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def benchmark_vectorstore(backend: str, queries: list[str]) -> dict:
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(FAISS_INDEX_PATH, embeddings, backend=backend)
    retriever = get_retriever(vectorstore)

    latencies: list[float] = []
    for query in queries:
        start = time.perf_counter()
        retriever.invoke(query)
        latencies.append(time.perf_counter() - start)

    average_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "backend": backend,
        "queries": len(queries),
        "average_latency_seconds": round(average_latency, 4),
    }


def evaluate_rag(dataset_path: Path = DEFAULT_EVAL_DATASET, backend: str = "faiss") -> dict:
    samples = load_eval_samples(dataset_path)
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(FAISS_INDEX_PATH, embeddings, backend=backend)
    retriever = get_retriever(vectorstore)
    qa_chain = build_rag_chain(retriever)
    callbacks = get_callbacks()

    rows = []
    for sample in samples:
        result = qa_chain.invoke(
            {"query": sample["question"]},
            config={"callbacks": callbacks} if callbacks else None,
        )
        rows.append(
            {
                "user_input": sample["question"],
                "response": result["result"],
                "retrieved_contexts": [
                    doc.page_content for doc in result["source_documents"]
                ],
                "reference": sample["ground_truth"],
            }
        )

    dataset = Dataset.from_list(rows)
    ragas_llm = LangchainLLMWrapper(qa_chain.combine_documents_chain.llm_chain.llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    run_config = RunConfig(
        timeout=RAGAS_TIMEOUT_SECONDS,
        max_retries=RAGAS_MAX_RETRIES,
        max_wait=RAGAS_MAX_WAIT_SECONDS,
        max_workers=RAGAS_MAX_WORKERS,
        log_tenacity=False,
    )
    evaluation = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                strictness=RAGAS_ANSWER_RELEVANCY_STRICTNESS,
            ),
            ContextPrecision(llm=ragas_llm),
            ContextRecall(llm=ragas_llm),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        batch_size=RAGAS_BATCH_SIZE,
    )
    return evaluation.to_pandas().mean(numeric_only=True).to_dict()


def run_phase2_evaluation(
    dataset_path: Path = PHASE2_EVAL_DATASET,
    backend: str = "faiss",
    phase: str = "phase1",
    results_dir: Path = EVAL_RESULTS_DIR,
) -> dict:
    dataset = load_phase2_dataset(dataset_path)
    embeddings = get_embeddings()
    if phase == "phase6":
        retriever = Phase6PatchRetriever(embeddings=embeddings, backend=backend)
        qa_chain = Phase6MultimodalQA(retriever)
    else:
        vectorstore = load_vectorstore(FAISS_INDEX_PATH, embeddings, backend=backend)
        retriever = get_retriever(vectorstore, backend=backend, phase=phase)
        answer_style = "multimodal" if phase == "phase5" else "hybrid" if phase in {"phase3", "phase4"} else "default"
        qa_chain = build_rag_chain(retriever, answer_style=answer_style)
    callbacks = get_callbacks()

    results_df = run_rag_evaluation(dataset, qa_chain, retriever, callbacks=callbacks)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = results_dir / f"phase2_results_{phase}_{backend}.csv"
    results_df.to_csv(raw_results_path, index=False)

    ragas_scores = evaluate_ragas_metrics(
        results_df=results_df,
        llm=get_chat_model(temperature=0),
        embeddings=embeddings,
    )
    summary = {
        "backend": backend,
        "phase": phase,
        "dataset_path": str(dataset_path),
        "result_rows": len(results_df),
        "judge_average_score": round(float(results_df["judge_score"].mean()), 4),
        "ragas": ragas_scores,
        "raw_results_path": str(raw_results_path),
    }
    summary_path = results_dir / f"phase2_summary_{phase}_{backend}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if phase == "phase1":
        legacy_summary_path = results_dir / f"phase2_summary_{backend}.json"
        legacy_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
