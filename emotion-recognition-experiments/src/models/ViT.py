import timm
import torch.nn as nn


def get_vit_small(num_classes=7, pretrained=True):

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained
    )

    # Replace classifier head (timm uses model.head)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model