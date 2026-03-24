import torch
from core.train import train_model
import torch.nn as nn
from imblearn.over_sampling import ADASYN
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def apply_adasyn(train_loader):
    X = []
    y = []

    # Collect all data
    for images, labels in train_loader:
        X.append(images.view(images.size(0), -1).numpy())  # flatten
        y.append(labels.numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Apply ADASYN
    adasyn = ADASYN()
    X_res, y_res = adasyn.fit_resample(X, y)

    # Convert back to tensors
    X_res = torch.tensor(X_res).float().view(-1, 3, 224, 224)
    y_res = torch.tensor(y_res).long()

    dataset = TensorDataset(X_res, y_res)

    return DataLoader(dataset, batch_size=train_loader.batch_size, shuffle=True)


def run(model, train_loader, val_loader, optimizer, scheduler, device):

    # 🔥 APPLY ADASYN HERE
    train_loader = apply_adasyn(train_loader)

    return train_model(
        model,
        train_loader,
        val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        strategy="adasyn"
    )