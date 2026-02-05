"""Pretty-print helpers for .oinf verification output."""
from __future__ import annotations

import numpy as np


def format_scalar(value: object) -> str:
    """Format a scalar value for display."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"\"{value}\""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_values_1d(values: np.ndarray) -> str:
    """Format a 1D array with truncation."""
    flat = values.flatten()
    total = flat.size
    if total <= 10:
        items = [format_scalar(v.item()) for v in flat]
    else:
        first = [format_scalar(v.item()) for v in flat[:5]]
        last = [format_scalar(v.item()) for v in flat[-5:]]
        items = first + ["..."] + last
    return "{ " + ", ".join(items) + " }"


def format_values(values: np.ndarray) -> str:
    """Format an array with truncation and row limits."""
    if values.ndim <= 1:
        return format_values_1d(values)
    lines = ["{ "]
    rows = min(values.shape[0], values.ndim, 5)
    for idx in range(rows):
        row = np.array(values[idx]).flatten()
        lines.append(f"{format_values_1d(row)} ,")
    if values.shape[0] > rows:
        lines.append("...")
    lines.append("}")
    return "\n".join(lines)


def histogram_string(values: np.ndarray) -> str:
    """Build a compact histogram string for numeric arrays."""
    if values.size == 0:
        return "{(empty)}"
    if values.dtype == np.bool_:
        counts = np.bincount(values.astype(np.uint8), minlength=2)
        return "{([0, 0], " + str(int(counts[0])) + "), ([1, 1], " + str(int(counts[1])) + ")}"
    numeric = values.astype(np.float64, copy=False)
    vmin = float(np.min(numeric))
    vmax = float(np.max(numeric))
    entries = []
    if np.issubdtype(values.dtype, np.integer) and vmax - vmin <= 20:
        for val in range(int(vmin), int(vmax) + 1):
            count = int(np.sum(values == val))
            if count:
                entries.append(f"([{val}, {val}], {count})")
    else:
        bins = 10
        if vmin == vmax:
            entries.append(f"([{vmin:.6g}, {vmax:.6g}], {values.size})")
        else:
            hist, edges = np.histogram(numeric, bins=bins, range=(vmin, vmax))
            for i in range(bins):
                count = int(hist[i])
                if count:
                    entries.append(
                        f"([{edges[i]:.6g}, {edges[i + 1]:.6g}], {count})"
                    )
    return "{" + ", ".join(entries) + "}"
