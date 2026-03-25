import math

import matplotlib.pyplot as plt
import torch


class AttentionExtractor:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.hook = None

    def _get_attention(self, module, inputs, output):
        x = inputs[0]
        batch_size, num_tokens, embed_dim = x.shape

        qkv = module.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            module.num_heads,
            embed_dim // module.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key = qkv[0], qkv[1]
        attention = (query @ key.transpose(-2, -1)) * module.scale
        attention = attention.softmax(dim=-1)
        self.attention_maps.append(attention.detach().cpu())

    def register_hook(self):
        self.hook = self.model.blocks[-1].attn.register_forward_hook(self._get_attention)

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def extract(self, dataloader, device):
        self.model.eval()
        self.attention_maps = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                _ = self.model(images)
                all_labels.append(labels.cpu())

        if not self.attention_maps:
            raise RuntimeError("No attention maps captured from the last transformer block.")

        return torch.cat(self.attention_maps, dim=0), torch.cat(all_labels, dim=0)


def compute_mean_attention(attn_maps, labels, num_classes=7):
    class_maps = {}

    for class_idx in range(num_classes):
        indices = (labels == class_idx).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        class_maps[class_idx] = attn_maps[indices].mean(dim=0)

    return class_maps


def reshape_to_spatial(attn_map):
    cls_attention = attn_map[:, 0, 1:]
    num_tokens = cls_attention.shape[-1]
    side = int(math.sqrt(num_tokens))
    return cls_attention.reshape(attn_map.shape[0], side, side)


def plot_attention_grid(spatial_maps, class_names, max_heads=None, cmap="jet"):
    available_classes = sorted(spatial_maps.keys())
    if not available_classes:
        raise ValueError("No class attention maps available to plot.")

    head_count = spatial_maps[available_classes[0]].shape[0]
    if max_heads is not None:
        head_count = min(head_count, max_heads)

    fig, axes = plt.subplots(
        len(available_classes),
        head_count,
        figsize=(2.4 * head_count, 2.4 * len(available_classes)),
        squeeze=False,
    )

    for row, class_idx in enumerate(available_classes):
        maps = spatial_maps[class_idx][:head_count]
        for head in range(head_count):
            ax = axes[row, head]
            ax.imshow(maps[head], cmap=cmap)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Head {head}")

        # Add a figure-level row label so each heatmap row is explicitly tied to its emotion.
        row_axis = axes[row, 0]
        row_y = row_axis.get_position().y0 + row_axis.get_position().height / 2
        fig.text(
            0.01,
            row_y,
            class_names[class_idx],
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        f"Per-emotion spatial attention maps from the last transformer block ({head_count} heads shown)",
        y=0.995,
    )
    fig.tight_layout(rect=[0.08, 0.0, 1.0, 0.97])
    return fig


def describe_attention_region(spatial_map):
    height, width = spatial_map.shape
    row_slices = [
        spatial_map[: max(1, height // 3), :],
        spatial_map[height // 3 : max(height // 3 + 1, 2 * height // 3), :],
        spatial_map[max(2 * height // 3, 1) :, :],
    ]
    col_slices = [
        spatial_map[:, : max(1, width // 3)],
        spatial_map[:, width // 3 : max(width // 3 + 1, 2 * width // 3)],
        spatial_map[:, max(2 * width // 3, 1) :],
    ]

    vertical_scores = [
        ("eyes/forehead", float(row_slices[0].mean())),
        ("nose/cheeks", float(row_slices[1].mean())),
        ("mouth/jaw", float(row_slices[2].mean())),
    ]
    horizontal_scores = [
        ("left face", float(col_slices[0].mean())),
        ("center face", float(col_slices[1].mean())),
        ("right face", float(col_slices[2].mean())),
    ]

    # Secondary region helps avoid repetitive summaries when scores are close.
    vertical_sorted = sorted(vertical_scores, key=lambda item: item[1], reverse=True)
    horizontal_sorted = sorted(horizontal_scores, key=lambda item: item[1], reverse=True)

    primary_vertical = vertical_sorted[0][0]
    secondary_vertical = vertical_sorted[1][0]
    primary_horizontal = horizontal_sorted[0][0]

    if primary_vertical == secondary_vertical:
        return f"{primary_vertical}, {primary_horizontal}"

    return f"{primary_vertical} (secondary: {secondary_vertical}), {primary_horizontal}"
