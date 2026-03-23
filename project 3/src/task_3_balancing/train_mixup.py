from core.train import train_model
import torch
import numpy as np

def mixup_step(model, images, labels, criterion, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).to(images.device)

    mixed_x = lam * images + (1 - lam) * images[index]
    y_a, y_b = labels, labels[index]

    outputs = model(mixed_x)
    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

    return outputs, loss


def run(model, train_loader, val_loader, optimizer, scheduler, device):
    return train_model(
        model,
        train_loader,
        val_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        strategy="mixup",
        training_step_fn=mixup_step
    )