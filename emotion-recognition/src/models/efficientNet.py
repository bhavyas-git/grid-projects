
import torchvision.models as models
import torch.nn as nn

def get_efficient_net(num_classes=7, pretrained=True):

    model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    return model