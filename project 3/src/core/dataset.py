# src/dataset.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =============================
# TRAIN TRANSFORMS
# =============================
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# =============================
# VAL TRANSFORMS (NO AUGMENTATION)
# =============================
def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# =============================
# STANDARD DATALOADER (FOCAL / BASELINE)
# =============================
def get_dataloaders(train_dir, val_dir, batch_size=64):

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# =============================
# DATASETS ONLY (NO SAMPLING LOGIC)
# =============================
def get_train_dataset(train_dir):
    return datasets.ImageFolder(train_dir, transform=get_train_transforms())

def get_val_dataset(val_dir):
    return datasets.ImageFolder(val_dir, transform=get_val_transforms())
