"""Facial emotion recognition for registered users (OpenCV Zoo ONNX)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
FER_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/"
    "facial_expression_recognition_mobilefacenet_2022july.onnx"
)
FER_LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
INPUT_SIZE = (112, 112)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


class EmotionDetector:
    """7-class emotion classifier on aligned face crops."""

    def __init__(self) -> None:
        model_path = MODELS_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx"
        _download(FER_URL, model_path)
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self._history: dict[str, list[str]] = {}

    def _smooth(self, person_key: str, label: str, window: int = 5) -> str:
        hist = self._history.setdefault(person_key, [])
        hist.append(label)
        if len(hist) > window:
            hist.pop(0)
        return max(set(hist), key=hist.count)

    def predict(self, aligned_bgr: np.ndarray, person_key: str = "") -> tuple[str, float]:
        if aligned_bgr is None or aligned_bgr.size == 0:
            return "neutral", 0.0

        blob = cv2.dnn.blobFromImage(
            aligned_bgr,
            scalefactor=1.0 / 128.0,
            size=INPUT_SIZE,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        scores = self.net.forward().flatten()
        idx = int(np.argmax(scores))
        conf = float(scores[idx])
        # Softmax-like scale if logits
        if conf > 1.0 or conf < 0:
            exp = np.exp(scores - scores.max())
            probs = exp / exp.sum()
            idx = int(np.argmax(probs))
            conf = float(probs[idx])

        label = FER_LABELS[idx] if idx < len(FER_LABELS) else "neutral"
        if person_key:
            label = self._smooth(person_key, label)
        return label, conf
