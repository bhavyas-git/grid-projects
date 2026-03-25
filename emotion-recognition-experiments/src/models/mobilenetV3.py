# src/mobilenetV3.py

import torchvision.models as models
import torch.nn as nn

def get_mobilenet_v3(num_classes=7, pretrained=True):

    model = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)

    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model