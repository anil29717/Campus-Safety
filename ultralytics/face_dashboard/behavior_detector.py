"""Pose-based behavior classification for registered users."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO

# COCO pose keypoint indices
NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16


def _visible(kpt: np.ndarray, idx: int, thresh: float = 0.3) -> bool:
    if idx >= len(kpt):
        return False
    return float(kpt[idx][2]) >= thresh


def _expand_face_bbox(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Expand face box to approximate upper-body region for pose."""
    x, y, w, h = bbox
    fh, fw = frame_shape[:2]
    pad_x = int(w * 0.6)
    pad_top = int(h * 0.4)
    pad_bottom = int(h * 2.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_top)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_bottom)
    return x1, y1, x2, y2


def _classify_keypoints(kpt: np.ndarray) -> str:
    """Map COCO keypoints to coarse behavior label."""
    if not (_visible(kpt, L_HIP) and _visible(kpt, R_HIP)):
        return "idle"

    hip_y = (kpt[L_HIP][1] + kpt[R_HIP][1]) / 2.0
    shoulder_y = None
    if _visible(kpt, L_SHOULDER) and _visible(kpt, R_SHOULDER):
        shoulder_y = (kpt[L_SHOULDER][1] + kpt[R_SHOULDER][1]) / 2.0

    knee_visible = _visible(kpt, L_KNEE) or _visible(kpt, R_KNEE)
    if knee_visible:
        knee_ys = []
        if _visible(kpt, L_KNEE):
            knee_ys.append(kpt[L_KNEE][1])
        if _visible(kpt, R_KNEE):
            knee_ys.append(kpt[R_KNEE][1])
        knee_y = sum(knee_ys) / len(knee_ys)
        # Sitting: knees near same level as hips (bent legs)
        if abs(knee_y - hip_y) < abs(hip_y * 0.15 + 20):
            return "sitting"
        if knee_y > hip_y + 30:
            return "standing"

    if shoulder_y is not None and hip_y > shoulder_y + 20:
        return "standing"
    return "idle"


class BehaviorDetector:
    """YOLO pose + simple rules; runs on person crop around identified face."""

    def __init__(self, pose_model: str = "yolo11n-pose.pt") -> None:
        self.pose_model_name = pose_model
        self._pose: YOLO | None = None
        self._positions: dict[str, list[tuple[float, float]]] = {}
        self._history: dict[str, list[str]] = {}

    def _get_pose(self) -> YOLO:
        if self._pose is None:
            from ultralytics import YOLO

            self._pose = YOLO(self.pose_model_name)
        return self._pose

    def _movement_label(self, person_key: str, cx: float, cy: float) -> str | None:
        hist = self._positions.setdefault(person_key, [])
        hist.append((cx, cy))
        if len(hist) > 8:
            hist.pop(0)
        if len(hist) < 4:
            return None
        dx = hist[-1][0] - hist[0][0]
        dy = hist[-1][1] - hist[0][1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 25:
            return "walking"
        return None

    def _smooth(self, person_key: str, label: str) -> str:
        hist = self._history.setdefault(person_key, [])
        hist.append(label)
        if len(hist) > 5:
            hist.pop(0)
        return max(set(hist), key=hist.count)

    def predict(
        self,
        frame: np.ndarray,
        face_bbox: tuple[int, int, int, int],
        person_key: str,
    ) -> tuple[str, float]:
        x1, y1, x2, y2 = _expand_face_bbox(face_bbox, frame.shape)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "idle", 0.0

        pose = self._get_pose()
        results = pose.predict(crop, verbose=False, imgsz=256)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return "idle", 0.0

        kpts = results[0].keypoints.data
        if kpts is None or len(kpts) == 0:
            return "idle", 0.0

        kpt = kpts[0].cpu().numpy()
        label = _classify_keypoints(kpt)

        # Movement overrides idle/standing when person moves across frames
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        move = self._movement_label(person_key, cx, cy)
        if move:
            label = move

        label = self._smooth(person_key, label)
        conf = 0.75 if label != "idle" else 0.5
        return label, conf
