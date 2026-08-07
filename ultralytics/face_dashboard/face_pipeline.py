"""YOLO person counting + OpenCV YuNet/SFace face detection and recognition."""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np
from ultralytics import YOLO

from face_dashboard.camera_config import is_rtsp_source
from face_dashboard.config import ProcessConfig
from face_dashboard.face_store import FaceStore, KnownFace
from face_dashboard.utils import resize_keep_aspect

if TYPE_CHECKING:
    from face_dashboard.behavior_detector import BehaviorDetector
    from face_dashboard.emotion_detector import EmotionDetector

__all__ = [
    "DetectedFace",
    "DetectedObject",
    "FacePipeline",
    "FrameResult",
    "ProcessConfig",
    "open_camera",
    "read_camera_frame",
    "warm_up_camera",
]

MODELS_DIR = Path(__file__).parent / "models"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
COSINE_THRESHOLD = 0.36
PERSON_CLASS = 0


@dataclass
class DetectedFace:
    """One face in the current frame."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    name: str
    score: float
    person_id: str | None = None
    emotion: str | None = None
    emotion_conf: float = 0.0
    behavior: str | None = None
    behavior_conf: float = 0.0
    aligned: np.ndarray | None = None
    feature: np.ndarray | None = None


@dataclass
class DetectedObject:
    """One non-person object in the current frame (YOLO / COCO)."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str
    class_id: int
    confidence: float


@dataclass
class FrameResult:
    """Annotated frame plus detection stats."""

    frame: np.ndarray
    person_count: int
    face_count: int
    known_count: int
    unknown_count: int
    faces: list[DetectedFace]
    objects: list[DetectedObject]
    object_count: int
    person_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    fps: float = 0.0


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def open_camera(
    source: Union[int, str] = 0,
    width: int = 640,
    height: int = 480,
) -> cv2.VideoCapture:
    """Open webcam by index or IP camera (RTSP URL, e.g. CP Plus)."""
    if is_rtsp_source(source):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    index = int(source)
    backends = []
    if sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
    else:
        backends = [0]

    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if cap.isOpened():
            break
        cap.release()
        cap = None

    if cap is None or not cap.isOpened():
        return cv2.VideoCapture(index)

    if sys.platform == "win32":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def _is_valid_frame(frame: np.ndarray) -> bool:
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return False
    if frame.shape[2] != 3:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 8.0:
        return False
    return True


def warm_up_camera(cap: cv2.VideoCapture, n: int = 20) -> None:
    import time

    # RTSP streams need more frames before the first valid image.
    for _ in range(n):
        cap.read()
        time.sleep(0.08 if n > 15 else 0.05)


def read_camera_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    for _ in range(5):
        ok, frame = cap.read()
        if ok and _is_valid_frame(frame):
            return True, frame
    return False, None


def _resize_keep_aspect(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    return resize_keep_aspect(frame, max_width)


def _scale_boxes(boxes: np.ndarray, inv_scale: float) -> np.ndarray:
    if inv_scale == 1.0:
        return boxes
    return (boxes * inv_scale).astype(int)


class FacePipeline:
    """Detect persons with YOLO, faces with YuNet, recognize with SFace."""

    def __init__(
        self,
        store: FaceStore,
        yolo_model: str = "yolo26n.pt",
        threshold: float = COSINE_THRESHOLD,
    ) -> None:
        self.store = store
        self.threshold = threshold
        self.yolo = YOLO(yolo_model)

        yunet_path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
        sface_path = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
        _download(YUNET_URL, yunet_path)
        _download(SFACE_URL, sface_path)

        self.detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")

        self._emotion: EmotionDetector | None = None
        self._behavior: BehaviorDetector | None = None
        self._last_emotion: dict[str, tuple[str, float]] = {}
        self._last_behavior: dict[str, tuple[str, float]] = {}

        self._frame_idx = 0
        self._last_person_count = 0
        self._last_person_boxes: np.ndarray | None = None
        self._last_objects: list[DetectedObject] = []

    def _get_emotion(self) -> EmotionDetector:
        if self._emotion is None:
            from face_dashboard.emotion_detector import EmotionDetector

            self._emotion = EmotionDetector()
        return self._emotion

    def _get_behavior(self) -> BehaviorDetector:
        if self._behavior is None:
            from face_dashboard.behavior_detector import BehaviorDetector

            self._behavior = BehaviorDetector()
        return self._behavior

    def reset_cache(self) -> None:
        self._frame_idx = 0
        self._last_person_count = 0
        self._last_person_boxes = None
        self._last_objects = []
        self._last_emotion.clear()
        self._last_behavior.clear()

    def _match_identity(self, feature: np.ndarray) -> tuple[str, float, str | None]:
        best_name = "Unknown"
        best_id: str | None = None
        best_score = self.threshold
        for known in self.store.faces:
            score = self.recognizer.match(
                feature, known.feature, cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best_score = float(score)
                best_name = known.name
                best_id = known.id
        return best_name, best_score, best_id

    def detect_faces(self, frame: np.ndarray, inv_scale: float = 1.0) -> list[DetectedFace]:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, raw = self.detector.detect(frame)
        if raw is None:
            return []

        faces: list[DetectedFace] = []
        for row in raw:
            x = int(row[0] * inv_scale)
            y = int(row[1] * inv_scale)
            fw = int(row[2] * inv_scale)
            fh = int(row[3] * inv_scale)
            aligned = self.recognizer.alignCrop(frame, row)
            feature = self.recognizer.feature(aligned)
            name, score, person_id = self._match_identity(feature)
            faces.append(
                DetectedFace(
                    bbox=(x, y, fw, fh),
                    name=name,
                    score=score,
                    person_id=person_id,
                    aligned=aligned,
                    feature=feature,
                )
            )
        return faces

    def _enrich_known_faces(
        self,
        faces: list[DetectedFace],
        work_frame: np.ndarray,
        full_frame: np.ndarray,
        cfg: ProcessConfig,
    ) -> None:
        run_emotion = cfg.enable_emotion and (
            self._frame_idx == 1 or self._frame_idx % cfg.emotion_every == 0
        )
        run_behavior = cfg.enable_behavior and (
            self._frame_idx == 1 or self._frame_idx % cfg.behavior_every == 0
        )

        for face in faces:
            if face.name == "Unknown" or not face.person_id:
                continue

            key = face.person_id

            if run_emotion and face.aligned is not None:
                try:
                    emo, e_conf = self._get_emotion().predict(face.aligned, person_key=key)
                    self._last_emotion[key] = (emo, e_conf)
                except Exception:
                    pass

            if key in self._last_emotion:
                face.emotion, face.emotion_conf = self._last_emotion[key]

            if run_behavior:
                try:
                    beh, b_conf = self._get_behavior().predict(full_frame, face.bbox, key)
                    self._last_behavior[key] = (beh, b_conf)
                except Exception:
                    pass

            if key in self._last_behavior:
                face.behavior, face.behavior_conf = self._last_behavior[key]

    def _run_yolo(
        self,
        frame: np.ndarray,
        conf: float,
        imgsz: int,
        *,
        detect_persons: bool,
        detect_objects: bool,
    ) -> tuple[int, np.ndarray | None, list[DetectedObject]]:
        results = self.yolo.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return 0, None, []

        names = results[0].names
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        person_mask = classes == PERSON_CLASS
        person_count = int(person_mask.sum()) if detect_persons else 0
        person_boxes = xyxy[person_mask] if detect_persons and person_count else None

        objects: list[DetectedObject] = []
        if detect_objects:
            for box, cls_id, score in zip(xyxy, classes, confs):
                if cls_id == PERSON_CLASS:
                    continue
                label = names.get(int(cls_id), str(int(cls_id))) if isinstance(names, dict) else names[int(cls_id)]
                x1, y1, x2, y2 = box.astype(int)
                objects.append(
                    DetectedObject(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        label=str(label),
                        class_id=int(cls_id),
                        confidence=float(score),
                    )
                )

        return person_count, person_boxes, objects

    def process_frame(self, frame: np.ndarray, cfg: ProcessConfig | None = None) -> FrameResult:
        import time

        if cfg is None:
            cfg = ProcessConfig()
        elif not isinstance(cfg, ProcessConfig):
            cfg = ProcessConfig(
                conf=getattr(cfg, "conf", 0.35),
                max_width=getattr(cfg, "max_width", 640),
                display_width=getattr(cfg, "display_width", 960),
                imgsz=getattr(cfg, "imgsz", 416),
                yolo_every=getattr(cfg, "yolo_every", 3),
                faces_only=getattr(cfg, "faces_only", False),
                jpeg_quality=getattr(cfg, "jpeg_quality", 72),
                emotion_every=getattr(cfg, "emotion_every", 3),
                behavior_every=getattr(cfg, "behavior_every", 5),
                enable_emotion=getattr(cfg, "enable_emotion", True),
                enable_behavior=getattr(cfg, "enable_behavior", True),
                enable_objects=getattr(cfg, "enable_objects", True),
                object_every=getattr(cfg, "object_every", 5),
                enable_safety=getattr(cfg, "enable_safety", True),
                unattended_dwell_sec=getattr(cfg, "unattended_dwell_sec", 30.0),
                enable_fire_smoke=getattr(cfg, "enable_fire_smoke", True),
                enable_fight=getattr(cfg, "enable_fight", True),
                hazard_every=getattr(cfg, "hazard_every", 6),
            )
        t0 = time.perf_counter()
        self._frame_idx += 1

        work_frame, scale = _resize_keep_aspect(frame, cfg.max_width)
        inv_scale = 1.0 / scale

        detect_persons = not cfg.faces_only
        detect_objects = cfg.enable_objects
        yolo_interval = cfg.object_every if detect_objects and not detect_persons else cfg.yolo_every
        run_yolo = (detect_persons or detect_objects) and (
            self._frame_idx == 1 or self._frame_idx % yolo_interval == 0
        )
        if run_yolo:
            count, boxes, objects = self._run_yolo(
                work_frame,
                cfg.conf,
                cfg.imgsz,
                detect_persons=detect_persons,
                detect_objects=detect_objects,
            )
            if detect_persons:
                self._last_person_count = count
                self._last_person_boxes = boxes
            if detect_objects:
                self._last_objects = objects

        person_count = self._last_person_count if detect_persons else 0
        person_boxes = self._last_person_boxes if detect_persons else None
        objects = list(self._last_objects) if detect_objects else []

        person_boxes_list: list[tuple[int, int, int, int]] = []

        if person_boxes is not None:
            person_boxes = _scale_boxes(person_boxes, inv_scale)
            for box in person_boxes:
                x1, y1, x2, y2 = box.astype(int)
                person_boxes_list.append((int(x1), int(y1), int(x2), int(y2)))

        if objects and inv_scale != 1.0:
            scaled_objects: list[DetectedObject] = []
            for obj in objects:
                x1, y1, x2, y2 = obj.bbox
                scaled_objects.append(
                    DetectedObject(
                        bbox=(
                            int(x1 * inv_scale),
                            int(y1 * inv_scale),
                            int(x2 * inv_scale),
                            int(y2 * inv_scale),
                        ),
                        label=obj.label,
                        class_id=obj.class_id,
                        confidence=obj.confidence,
                    )
                )
            objects = scaled_objects

        faces = self.detect_faces(work_frame, inv_scale)
        self._enrich_known_faces(faces, work_frame, frame, cfg)

        known_count = sum(1 for f in faces if f.name != "Unknown")
        unknown_count = len(faces) - known_count

        if cfg.faces_only:
            person_count = len(faces)

        display, dscale = _resize_keep_aspect(frame, cfg.display_width)

        if person_boxes is not None and detect_persons:
            for box in person_boxes:
                x1, y1, x2, y2 = (box * dscale).astype(int) if dscale != 1.0 else box.astype(int)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 128, 0), 2)
                cv2.putText(
                    display, "person", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2, cv2.LINE_AA,
                )

        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            if dscale != 1.0:
                x1, y1, x2, y2 = int(x1 * dscale), int(y1 * dscale), int(x2 * dscale), int(y2 * dscale)
            color = (255, 0, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.label} {obj.confidence:.0%}"
            cv2.putText(
                display, label, (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA,
            )

        for face in faces:
            x, y, fw, fh = face.bbox
            if dscale != 1.0:
                x, y, fw, fh = int(x * dscale), int(y * dscale), int(fw * dscale), int(fh * dscale)
            color = (0, 200, 0) if face.name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x + fw, y + fh), color, 2)

            if face.name != "Unknown":
                label = face.name
                if face.person_id:
                    label += f" ({face.person_id})"
                if face.emotion:
                    label += f" | {face.emotion}"
                if face.behavior:
                    label += f" | {face.behavior}"
            else:
                label = "Unknown"

            cv2.putText(
                display, label, (x, max(y - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

        cv2.putText(
            display,
            (
                f"Persons: {person_count}  Faces: {len(faces)}  Known: {known_count}  "
                f"Unknown: {unknown_count}  Objects: {len(objects)}"
            ),
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        elapsed = time.perf_counter() - t0
        fps = 1.0 / max(elapsed, 1e-6)

        return FrameResult(
            frame=display,
            person_count=person_count,
            face_count=len(faces),
            known_count=known_count,
            unknown_count=unknown_count,
            faces=faces,
            objects=objects,
            object_count=len(objects),
            person_boxes=person_boxes_list,
            fps=fps,
        )

    def register_face(
        self, frame: np.ndarray, name: str, face_index: int = 0
    ) -> KnownFace | None:
        work, scale = _resize_keep_aspect(frame, 640)
        faces = self.detect_faces(work, 1.0 / scale)
        if not faces:
            return None
        idx = min(face_index, len(faces) - 1)
        face = faces[idx]
        if face.feature is None or face.aligned is None:
            return None
        return self.store.add(name, face.feature, face.aligned)
