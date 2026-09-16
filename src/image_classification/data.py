"""Validated, domain-neutral image inputs."""
import numpy as np


def validate(images, labels, groups):
    if images.ndim != 4 or any(n < 1 for n in images.shape):
        raise ValueError("images must have nonempty shape (N, H, W, C)")
    if images.dtype.kind not in "uif" or not np.isfinite(images).all():
        raise ValueError("images must be finite numeric values")
    if labels.shape != (len(images),) or groups.shape != labels.shape:
        raise ValueError("labels and groups must have shape (N,)")
    if labels.dtype.kind not in "iu" or set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both integer classes 0 and 1")
    if groups.dtype.kind not in "US" or any(not str(g).strip() for g in groups.astype(str)):
        raise ValueError("groups must be nonempty strings")
    for group in np.unique(groups):
        if len(np.unique(labels[groups == group])) != 1:
            raise ValueError("Each group must have one consistent label")
    with np.errstate(over="ignore"):
        images = images.astype(np.float32)
    if not np.isfinite(images).all():
        raise ValueError("images exceed float32 range")
    return images, labels.astype(np.int32), groups.astype(str)


def load_data(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"images", "labels", "groups"}:
            raise ValueError("Expected exactly images, labels, and groups arrays")
        return validate(data["images"], data["labels"], data["groups"])


def channel_means(images):
    return images.mean(axis=(1, 2), dtype=np.float64)
