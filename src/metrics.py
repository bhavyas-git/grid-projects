from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from src.config import (
    RAGAS_ANSWER_RELEVANCY_STRICTNESS,
    RAGAS_BATCH_SIZE,
    RAGAS_MAX_RETRIES,
    RAGAS_MAX_WAIT_SECONDS,
    RAGAS_MAX_WORKERS,
    RAGAS_TIMEOUT_SECONDS,
)


def evaluate_ragas(results_df, llm, embeddings) -> dict:
    dataset = Dataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["generated_answer"],
                "retrieved_contexts": row["retrieved_contexts"],
                "reference": row["ground_truth_answer"],
            }
            for _, row in results_df.iterrows()
        ]
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    run_config = RunConfig(
        timeout=RAGAS_TIMEOUT_SECONDS,
        max_retries=RAGAS_MAX_RETRIES,
        max_wait=RAGAS_MAX_WAIT_SECONDS,
        max_workers=RAGAS_MAX_WORKERS,
        log_tenacity=False,
    )
    score = evaluate(
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
    metrics_df = score.to_pandas()
    return metrics_df.mean(numeric_only=True).to_dict()
