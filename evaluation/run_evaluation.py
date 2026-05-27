import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_phase2_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 2 RAG evaluation on the IFC dataset.")
    parser.add_argument(
        "--backend",
        choices=["faiss", "qdrant", "both"],
        default="both",
        help="Vector store backend to evaluate.",
    )
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase3", "phase5", "phase6", "all"],
        default="phase1",
        help="Pipeline phase to evaluate with the same RAGAS and LLM-as-judge workflow.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    backends = ["faiss", "qdrant"] if args.backend == "both" else [args.backend]
    phases = ["phase1", "phase3", "phase5", "phase6"] if args.phase == "all" else [args.phase]
    for phase in phases:
        for backend in backends:
            print(json.dumps(run_phase2_evaluation(backend=backend, phase=phase), indent=2))


if __name__ == "__main__":
    main()
