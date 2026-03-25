from core.train import train_model
from core.losses import FocalLoss

def run(model, train_loader, val_loader, optimizer, scheduler, device):
    return train_model(
        model,
        train_loader,
        val_loader,
        criterion=FocalLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        strategy="focal"
    )