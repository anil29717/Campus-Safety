"""Fire, smoke, and fight detection for campus safety."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "models"
FIRE_SMOKE_URL = (
    "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
)
FIRE_SMOKE_FILENAME = "fire_smoke_yolov8n.pt"

# COCO pose keypoints
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {dest.name} for fire/smoke detection...")
    urllib.request.urlretrieve(url, dest)


def _visible(kpt: np.ndarray, idx: int, thresh: float = 0.35) -> bool:
    if idx >= len(kpt):
        return False
    return float(kpt[idx][2]) >= thresh


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ca, cb = _box_center(a), _box_center(b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def _box_width(box: tuple[int, int, int, int]) -> float:
    return max(float(box[2] - box[0]), 1.0)


@dataclass
class DetectedHazard:
    """One fire, smoke, or fight detection."""

    hazard_type: str
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    message: str


@dataclass
class HazardSnapshot:
    detections: list[DetectedHazard] = field(default_factory=list)
    new_alerts: list[DetectedHazard] = field(default_factory=list)
    fighting_active: bool = False


class FireSmokeDetector:
    """YOLOv8n fine-tuned on D-Fire (smoke + fire)."""

    def __init__(self, conf: float = 0.4) -> None:
        self.conf = conf
        self._model: YOLO | None = None

    def _get_model(self) -> YOLO:
        if self._model is None:
            from ultralytics import YOLO

            model_path = MODELS_DIR / FIRE_SMOKE_FILENAME
            _download(FIRE_SMOKE_URL, model_path)
            self._model = YOLO(str(model_path))
        return self._model

    def detect(self, frame: np.ndarray, imgsz: int = 416) -> list[DetectedHazard]:
        model = self._get_model()
        results = model.predict(frame, conf=self.conf, imgsz=imgsz, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        names = results[0].names
        hazards: list[DetectedHazard] = []
        for box, cls_id, score in zip(
            results[0].boxes.xyxy.cpu().numpy(),
            results[0].boxes.cls.cpu().numpy().astype(int),
            results[0].boxes.conf.cpu().numpy(),
        ):
            label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else names[int(cls_id)]
            hazard_type = str(label).lower()
            if hazard_type not in {"fire", "smoke"}:
                continue
            x1, y1, x2, y2 = box.astype(int)
            bbox = (int(x1), int(y1), int(x2), int(y2))
            conf = float(score)
            msg = f"{hazard_type.title()} detected ({conf:.0%})"
            hazards.append(
                DetectedHazard(
                    hazard_type=hazard_type,
                    label=str(label),
                    confidence=conf,
                    bbox=bbox,
                    message=msg,
                )
            )
        return hazards


class FightDetector:
    """Pose-based fight detection when two or more persons are close with aggressive arm poses."""

    def __init__(self, conf: float = 0.35) -> None:
        self.conf = conf
        self._pose: YOLO | None = None
        self._streak = 0
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._wrist_history: list[tuple[float, float]] = []

    def _get_pose(self) -> YOLO:
        if self._pose is None:
            from ultralytics import YOLO

            self._pose = YOLO("yolo11n-pose.pt")
        return self._pose

    def reset(self) -> None:
        self._streak = 0
        self._last_bbox = None
        self._wrist_history.clear()

    def _arms_aggressive(self, kpt: np.ndarray) -> bool:
        if not (_visible(kpt, L_SHOULDER) and _visible(kpt, R_SHOULDER)):
            return False
        shoulder_y = (kpt[L_SHOULDER][1] + kpt[R_SHOULDER][1]) / 2.0
        raised = 0
        extended = 0
        for wrist, elbow in ((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)):
            if _visible(kpt, wrist) and kpt[wrist][1] < shoulder_y - 15:
                raised += 1
            if _visible(kpt, wrist) and _visible(kpt, elbow):
                arm_len = float(np.linalg.norm(kpt[wrist][:2] - kpt[elbow][:2]))
                if arm_len > 40:
                    extended += 1
        return raised >= 1 or extended >= 2

    def _bbox_from_keypoints(self, kpt: np.ndarray) -> tuple[int, int, int, int]:
        pts = kpt[kpt[:, 2] >= 0.3][:, :2]
        if len(pts) == 0:
            return (0, 0, 0, 0)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return (int(x1), int(y1), int(x2), int(y2))

    def _wrist_velocity(self, wrists: list[tuple[float, float]]) -> float:
        self._wrist_history.extend(wrists)
        if len(self._wrist_history) > 12:
            self._wrist_history = self._wrist_history[-12:]
        if len(self._wrist_history) < 4:
            return 0.0
        first = self._wrist_history[0]
        last = self._wrist_history[-1]
        return ((first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2) ** 0.5

    def detect(
        self,
        frame: np.ndarray,
        person_boxes: list[tuple[int, int, int, int]],
        imgsz: int = 416,
    ) -> DetectedHazard | None:
        if len(person_boxes) < 2:
            self._streak = 0
            return None

        pose = self._get_pose()
        results = pose.predict(frame, conf=self.conf, imgsz=imgsz, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            self._streak = 0
            return None

        persons: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        kpts = results[0].keypoints.data
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None
        for idx in range(len(kpts)):
            kpt = kpts[idx].cpu().numpy()
            if boxes is not None and idx < len(boxes):
                b = boxes[idx].astype(int)
                bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            else:
                bbox = self._bbox_from_keypoints(kpt)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            persons.append((bbox, kpt))

        if len(persons) < 2:
            self._streak = 0
            return None

        fight_pair: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
        all_wrists: list[tuple[float, float]] = []

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                box_a, kpt_a = persons[i]
                box_b, kpt_b = persons[j]
                avg_w = (_box_width(box_a) + _box_width(box_b)) / 2.0
                if _box_distance(box_a, box_b) > avg_w * 1.4:
                    continue
                aggressive = self._arms_aggressive(kpt_a) or self._arms_aggressive(kpt_b)
                if not aggressive:
                    continue
                fight_pair = (box_a, box_b)
                for kpt in (kpt_a, kpt_b):
                    for wrist in (L_WRIST, R_WRIST):
                        if _visible(kpt, wrist):
                            all_wrists.append((float(kpt[wrist][0]), float(kpt[wrist][1])))

        velocity = self._wrist_velocity(all_wrists)
        if fight_pair and (velocity > 30 or self._arms_aggressive(persons[0][1])):
            self._streak += 1
        else:
            self._streak = 0

        if self._streak < 2:
            return None

        box_a, box_b = fight_pair  # type: ignore[misc]
        x1 = min(box_a[0], box_b[0])
        y1 = min(box_a[1], box_b[1])
        x2 = max(box_a[2], box_b[2])
        y2 = max(box_a[3], box_b[3])
        bbox = (x1, y1, x2, y2)
        self._last_bbox = bbox
        return DetectedHazard(
            hazard_type="fighting",
            label="fighting",
            confidence=0.8,
            bbox=bbox,
            message="Possible fighting detected between persons",
        )


class HazardDetector:
    """Run fire/smoke YOLO and pose-based fight detection on a schedule."""

    def __init__(
        self,
        *,
        fire_conf: float = 0.4,
        fight_conf: float = 0.35,
    ) -> None:
        self.fire_smoke = FireSmokeDetector(conf=fire_conf)
        self.fight = FightDetector(conf=fight_conf)
        self._frame_idx = 0
        self._last_detections: list[DetectedHazard] = []
        self._seen_hazards: set[str] = set()

    def reset(self) -> None:
        self._frame_idx = 0
        self._last_detections.clear()
        self._seen_hazards.clear()
        self.fight.reset()

    def update(
        self,
        frame: np.ndarray,
        person_boxes: list[tuple[int, int, int, int]],
        *,
        enable_fire_smoke: bool = True,
        enable_fight: bool = True,
        hazard_every: int = 6,
        imgsz: int = 416,
    ) -> HazardSnapshot:
        self._frame_idx += 1
        run = self._frame_idx == 1 or self._frame_idx % max(hazard_every, 1) == 0
        if not run:
            fighting = any(h.hazard_type == "fighting" for h in self._last_detections)
            return HazardSnapshot(
                detections=list(self._last_detections),
                fighting_active=fighting,
            )

        detections: list[DetectedHazard] = []
        new_alerts: list[DetectedHazard] = []

        if enable_fire_smoke:
            for hazard in self.fire_smoke.detect(frame, imgsz=imgsz):
                detections.append(hazard)
                key = f"{hazard.hazard_type}:{hazard.bbox[0]}:{hazard.bbox[1]}"
                if key not in self._seen_hazards:
                    self._seen_hazards.add(key)
                    new_alerts.append(hazard)

        if enable_fight:
            fight = self.fight.detect(frame, person_boxes, imgsz=imgsz)
            if fight:
                detections.append(fight)
                if "fighting" not in self._seen_hazards:
                    new_alerts.append(fight)
                self._seen_hazards.add("fighting")
            elif "fighting" in self._seen_hazards:
                self._seen_hazards.discard("fighting")

        self._last_detections = detections
        return HazardSnapshot(
            detections=detections,
            new_alerts=new_alerts,
            fighting_active=any(h.hazard_type == "fighting" for h in detections),
        )
