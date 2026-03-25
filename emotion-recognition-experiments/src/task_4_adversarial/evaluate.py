import torch
from .attacks import fgsm_attack, pgd_attack


def evaluate(model, loader, device, attack=None, attack_kwargs=None):
    model.eval()
    correct = 0
    total = 0
    attack_kwargs = attack_kwargs or {}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if attack == "fgsm":
            images = fgsm_attack(model, images, labels, **attack_kwargs)

        elif attack == "pgd":
            images = pgd_attack(model, images, labels, **attack_kwargs)

        with torch.no_grad():
            outputs = model(images)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total if total else 0.0
