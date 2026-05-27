import argparse
import json
from pathlib import Path

from src.config import DEFAULT_EVAL_DATASET, PHASE2_EVAL_DATASET
from src.evaluation import benchmark_vectorstore, evaluate_rag, run_phase2_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark and evaluate the IFC RAG system.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="Path to a JSON evaluation dataset with question and ground_truth fields.",
    )
    parser.add_argument(
        "--backend",
        choices=["faiss", "qdrant", "both"],
        default="both",
        help="Vector store backend to benchmark and evaluate.",
    )
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "both"],
        default="both",
        help="Which evaluation set to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    backends = ["faiss", "qdrant"] if args.backend == "both" else [args.backend]
    queries = [
        "What is IFC's mission?",
        "What was IFC's net income for FY24?",
        "Which region had the highest disbursed investment portfolio in FY24?",
    ]

    for backend in backends:
        print(json.dumps(benchmark_vectorstore(backend, queries), indent=2))
        if args.phase in {"phase1", "both"}:
            print(
                json.dumps(
                    evaluate_rag(dataset_path=args.dataset, backend=backend),
                    indent=2,
                )
            )
        if args.phase in {"phase2", "both"}:
            print(
                json.dumps(
                    run_phase2_evaluation(dataset_path=PHASE2_EVAL_DATASET, backend=backend),
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
