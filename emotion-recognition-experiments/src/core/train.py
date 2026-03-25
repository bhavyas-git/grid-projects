# src/core/train.py

import torch
import os
import datetime
import time
import psutil
from pathlib import Path
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)  # MB


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                epochs=10,
                model_name="mobilenet",
                hyperparams=None,
                #strategy="standard",
                training_step_fn=None  # 🔥 NEW (strategy-specific logic)
                ):

    # =============================
    # Directories
    # =============================
    results_dir = PROJECT_ROOT / "results"
    ckpt_dir = results_dir / "checkpoints"
    log_dir = results_dir / "training_logs"
    summary_dir = results_dir / "summaries"

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = log_dir / f"{model_name}_{timestamp}.log"
    best_md_path = summary_dir / f"{model_name}_best.md"
    best_ckpt_path = ckpt_dir / f"{model_name}_best.pth"

    best_val = 0
    best_summary = []

    log_file = open(log_path, "w")

    def log(text):
        print(text)
        log_file.write(text + "\n")
        log_file.flush()

    # =============================
    # START TIME + MEMORY
    # =============================
    start_time = time.time()
    start_mem = get_memory_usage()

    log("=" * 50)
    #log(f"Strategy: {strategy}")
    log(f"Model: {model_name}")
    log(f"Device: {device}")
    log(f"Start Memory: {start_mem:.2f} MB")
    log("=" * 50)

    # =============================
    # TRAIN LOOP
    # =============================
    for epoch in range(epochs):

        model.train()
        train_loss_total = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # 🔥 Strategy-specific step
            if training_step_fn:
                outputs, loss = training_step_fn(model, images, labels, criterion)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = 100 * train_correct / train_total
        train_loss = train_loss_total / len(train_loader)

        # =============================
        # VALIDATION
        # =============================
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        scheduler.step()

        log(f"\nEpoch {epoch+1}/{epochs}")
        log(f"Train Loss: {train_loss:.4f}")
        log(f"Train Acc: {train_acc:.2f}%")
        log(f"Val Acc:   {val_acc:.2f}%")
        log(f"Macro F1:  {macro_f1:.4f}")

        # =============================
        # BEST MODEL
        # =============================
        if val_acc > best_val:
            best_val = val_acc

            torch.save(model.state_dict(), best_ckpt_path)
            log("✅ New best model saved")

            best_summary = [
                f"# Training Summary ",
                f"**Model:** {model_name}",
                f"**Device:** {device}",
                f"## Best Epoch: {epoch+1}",
                f"- Train Loss: {train_loss:.4f}",
                f"- Train Acc: {train_acc:.2f}%",
                f"- Val Acc: {val_acc:.2f}%",
                f"- Macro F1: {macro_f1:.4f}",
                f"- Checkpoint: {best_ckpt_path}\n"
            ]

    # =============================
    # END TIME + MEMORY
    # =============================
    end_time = time.time()
    end_mem = get_memory_usage()

    total_time = end_time - start_time
    mem_used = end_mem - start_mem

    log("\n" + "=" * 50)
    log(f"Training Time: {total_time:.2f} sec")
    log(f"Memory Used: {mem_used:.2f} MB")
    log("=" * 50)

    # Add to summary
    best_summary += [
        "## Efficiency Metrics",
        f"- Training Time: {total_time:.2f} sec",
        f"- Memory Used: {mem_used:.2f} MB"
    ]

    # Write summary
    if best_summary:
        with open(best_md_path, "w") as f:
            for line in best_summary:
                f.write(line + "\n")

    log_file.close()

    return best_val