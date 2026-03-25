import numpy as np
from sklearn.metrics import mutual_info_score


def _to_numpy(array_like):
    if hasattr(array_like, "detach"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def summarize_head_responses(attn_maps):
    attn_np = _to_numpy(attn_maps)
    cls_attention = attn_np[:, :, 0, 1:]
    return cls_attention.mean(axis=-1)


def discretize_head_responses(head_responses, bins=10):
    head_responses = _to_numpy(head_responses)
    discretized = np.zeros_like(head_responses, dtype=np.int64)

    for head in range(head_responses.shape[1]):
        head_values = head_responses[:, head]
        bin_edges = np.histogram_bin_edges(head_values, bins=bins)
        if np.allclose(bin_edges[0], bin_edges[-1]):
            discretized[:, head] = 0
        else:
            discretized[:, head] = np.digitize(
                head_values,
                bin_edges[1:-1],
                right=False,
            )

    return discretized


def compute_mi(attn_maps, labels, bins=10):
    discrete = discretize_head_responses(summarize_head_responses(attn_maps), bins=bins)
    labels_np = _to_numpy(labels)

    return np.array(
        [mutual_info_score(discrete[:, head], labels_np) for head in range(discrete.shape[1])]
    )


def compute_classwise_mi(attn_maps, labels, num_classes=7, bins=10):
    discrete = discretize_head_responses(summarize_head_responses(attn_maps), bins=bins)
    labels_np = _to_numpy(labels)
    classwise = {}

    for class_idx in range(num_classes):
        one_vs_rest = (labels_np == class_idx).astype(int)
        classwise[class_idx] = np.array(
            [
                mutual_info_score(discrete[:, head], one_vs_rest)
                for head in range(discrete.shape[1])
            ]
        )

    return classwise


def get_high_mi_heads(mi_scores, threshold=None, top_k=3):
    mi_scores = np.asarray(mi_scores)
    if threshold is not None:
        return np.where(mi_scores >= threshold)[0]
    top_k = min(top_k, len(mi_scores))
    return np.argsort(mi_scores)[-top_k:][::-1]
