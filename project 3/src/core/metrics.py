# src/core/evaluate.py

import torch
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================
# Evaluate single model
# =============================
def evaluate_model(model, dataloader, device, class_names, strategy_name="model"):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # =============================
    # Metrics
    # =============================
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    micro_f1 = f1_score(all_labels, all_preds, average="micro")

    # Minority classes (customizable)
    minority_classes = ["disgust", "fear"]
    minority_f1 = np.mean([
        report[c]["f1-score"] for c in minority_classes if c in report
    ])

    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # =============================
    # Save confusion matrix
    # =============================
    results_dir = PROJECT_ROOT / "results"
    cm_dir = results_dir / "confusion_matrices"
    os.makedirs(cm_dir, exist_ok=True)

    np.save(cm_dir / f"{strategy_name}_cm.npy", cm)

    # =============================
    # Save per-class metrics
    # =============================
    metrics_path = results_dir / f"{strategy_name}_per_class.txt"

    with open(metrics_path, "w") as f:
        f.write(f"=== {strategy_name.upper()} ===\n\n")

        for i, cls in enumerate(class_names):
            f.write(f"{cls}:\n")
            f.write(f"  Accuracy: {per_class_acc[i]:.4f}\n")
            f.write(f"  Precision: {report[cls]['precision']:.4f}\n")
            f.write(f"  Recall: {report[cls]['recall']:.4f}\n")
            f.write(f"  F1-score: {report[cls]['f1-score']:.4f}\n\n")

    return {
        "strategy": strategy_name,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "minority_f1": minority_f1,
        "per_class_acc": per_class_acc,
        "report": report
    }


# =============================
# Generate comparison table
# =============================
def generate_comparison(results_list):
    results_dir = PROJECT_ROOT / "results"
    table_path = results_dir / "comparison_table.md"

    with open(table_path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write("| Strategy | Macro F1 | Micro F1 | Minority F1 |\n")
        f.write("|----------|----------|----------|--------------|\n")

        for res in results_list:
            f.write(
                f"| {res['strategy']} "
                f"| {res['macro_f1']:.4f} "
                f"| {res['micro_f1']:.4f} "
                f"| {res['minority_f1']:.4f} |\n"
            )

    print(f"✅ Comparison table saved to: {table_path}")