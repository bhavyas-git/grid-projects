import torch
import torch.nn as nn
from .attacks import pgd_attack


def train_robust(model, train_loader, device, epochs=10, adv_fraction=0.1, attack_kwargs=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    attack_kwargs = attack_kwargs or {}

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Replace roughly adv_fraction of batches with PGD adversarial examples.
            if i % max(1, int(round(1 / adv_fraction))) == 0:
                images = pgd_attack(model, images, labels, **attack_kwargs)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Robust Train] Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f}")
