# src/augment.py

import torchvision.transforms as transforms


class AugmentationFactory:

    @staticmethod
    def get_train_transforms(config):
        image_size = config["data"]["image_size"]

        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    @staticmethod
    def get_val_transforms(config):
        image_size = config["data"]["image_size"]

        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])