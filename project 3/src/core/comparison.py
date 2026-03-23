from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = PROJECT_ROOT / "results/summaries"
OUTPUT_FILE = PROJECT_ROOT / "results/balancing_comparison.md"


def extract_metrics(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    val_acc = re.search(r"Val Acc:\s*([\d.]+)", text)
    f1 = re.search(r"Macro F1:\s*([\d.]+)", text)

    return (
        float(val_acc.group(1)) if val_acc else 0,
        float(f1.group(1)) if f1 else 0
    )


def generate():
    rows = []

    for file in SUMMARY_DIR.glob("*_best.md"):
        name = file.stem.replace("_best", "")
        strategy = name.split("_")[0]

        val_acc, f1 = extract_metrics(file)

        rows.append((strategy, val_acc, f1))

    rows.sort(key=lambda x: x[1], reverse=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("# Balancing Strategy Comparison\n\n")
        f.write("| Strategy | Val Accuracy | Macro F1 |\n")
        f.write("|----------|-------------|----------|\n")

        for s, acc, f1 in rows:
            f.write(f"| {s} | {acc:.2f}% | {f1:.4f} |\n")

    print(f"✅ Comparison saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate()