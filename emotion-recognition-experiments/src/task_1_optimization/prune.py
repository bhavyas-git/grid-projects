# optimization/prune.py

import torch
import torch.nn.utils.prune as prune


def prune_model(model, amount=0.2):
    """
    Prune 20% smallest weights in FC layers
    """

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

    print("✅ Pruning applied (20% smallest weights removed)")
    return model