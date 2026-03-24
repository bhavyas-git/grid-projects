import torch


def _mask_heads(attention_probs, heads_to_mask):
    masked = attention_probs.clone()
    for head in heads_to_mask:
        masked[:, head] = 0
    return masked


class HeadAblation:
    def __init__(self, model):
        self.model = model

    def evaluate_accuracy(self, dataloader, device):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = self.model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total if total else 0.0

    def ablate(self, dataloader, device, heads_to_mask):
        self.model.eval()
        correct = 0
        total = 0

        def hook_fn(module, inputs, output):
            return _mask_heads(output, heads_to_mask)

        handle = self.model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = self.model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        handle.remove()
        return correct / total if total else 0.0

    def compare(self, dataloader, device, low_mi_heads, high_mi_heads):
        baseline = self.evaluate_accuracy(dataloader, device)
        low_mask = self.ablate(dataloader, device, low_mi_heads)
        high_mask = self.ablate(dataloader, device, high_mi_heads)

        return {
            "baseline_accuracy": baseline,
            "mask_low_mi_accuracy": low_mask,
            "mask_high_mi_accuracy": high_mask,
            "low_mi_drop": baseline - low_mask,
            "high_mi_drop": baseline - high_mask,
        }
