"""Shared helpers with no heavy ML imports."""

from __future__ import annotations

import cv2
import numpy as np


def resize_keep_aspect(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Resize frame so width <= max_width; return (image, scale factor)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / w
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale
