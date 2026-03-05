# src/train.py

import torch
import os
import datetime


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                epochs=10,
                model_name="efficient_net",
                hyperparams=None):

    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Unique run ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/{model_name}_loss_weights_{timestamp}.log"

    best_val = 0

    # Open log file
    with open(log_path, "w") as log_file:

        def log(message):
            print(message)
            log_file.write(message + "\n")
            log_file.flush()  # ensures writing even if crash

        log("=" * 60)
        log(f"<== Training Started ==>")
        log(f"Model: {model_name}")
        log(f"Timestamp: {timestamp}")
        log(f"Device: {device}")

        if hyperparams:
            log("Hyperparameters:")
            for k, v in hyperparams.items():
                log(f"  {k}: {v}")

        log("=" * 60)

        for epoch in range(epochs):

            model.train()
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_acc = 100 * train_correct / train_total

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)

                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = 100 * val_correct / val_total

            scheduler.step()

            log(f"\nEpoch {epoch+1}/{epochs}")
            log(f"Train Acc: {train_acc:.2f}%")
            log(f"Val Acc:   {val_acc:.2f}%")

            # Save best model of this run
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(),
                           f"checkpoints/{model_name}_best.pth")
                log("✅ New best model saved")

        log("=" * 60)
        log(f"Training Finished")
        log(f"Best Validation Accuracy: {best_val:.2f}%")
        log("=" * 60)

    return best_val
