from typing import Any

import pandas as pd

from src.judge import judge_answer


def run_rag_evaluation(dataset: pd.DataFrame, qa_chain, retriever, callbacks: list | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in dataset.iterrows():
        question = row["Question"]
        docs = retriever.invoke(question)
        result = qa_chain.invoke(
            {"query": question},
            config={"callbacks": callbacks} if callbacks else None,
        )
        generated_answer = result["result"]
        judged = judge_answer(
            question=question,
            generated_answer=generated_answer,
            ground_truth=row["Ground_Truth_Answer"],
        )

        rows.append(
            {
                "question": question,
                "ground_truth_context": row.get("Ground_Truth_Context", ""),
                "ground_truth_answer": row["Ground_Truth_Answer"],
                "page_number": row.get("Page_Number", ""),
                "context_content_type": row.get("Context_Content_Type", ""),
                "generated_answer": generated_answer,
                "retrieved_contexts": [doc.page_content for doc in docs],
                "retrieved_source_types": [doc.metadata.get("source_type", "text") for doc in docs],
                "retrieved_pages": [doc.metadata.get("page_number", doc.metadata.get("page")) for doc in docs],
                "judge_score": judged["score"],
                "judge_rationale": judged["rationale"],
            }
        )

    return pd.DataFrame(rows)
