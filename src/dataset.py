# src/dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    return transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_dataloaders(train_dir, val_dir, batch_size=64):

    transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader