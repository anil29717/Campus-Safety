"""Background camera thread with feature orchestrator."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from face_dashboard.activity_log import ActivityLogger
from face_dashboard.camera_config import (
    CameraSource,
    describe_source,
    get_default_camera_source,
    is_rtsp_source,
    resolve_camera_source,
)
from face_dashboard.camera_registry import CameraRegistry
from face_dashboard.config import ProcessConfig
from face_dashboard.entry_tracker import EntryTracker
from face_dashboard.face_pipeline import FacePipeline, open_camera, read_camera_frame, warm_up_camera
from face_dashboard.face_store import FaceStore
from face_dashboard.feature_config import FeatureConfig
from face_dashboard.safety_monitor import SafetyMonitor
from face_dashboard.services.orchestrator import FeatureOrchestrator
from face_dashboard.unknown_face_store import UnknownFaceStore

ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = ROOT / "data" / "known_faces"
ENTRY_DIR = ROOT / "data" / "entries"
UNKNOWN_DIR = ROOT / "data" / "unknown_faces"
ACTIVITY_DIR = ROOT / "data" / "activity"
SAFETY_DIR = ROOT / "data" / "safety"


@dataclass
class LiveState:
    running: bool = False
    fps: float = 0.0
    person_count: int = 0
    face_count: int = 0
    known_count: int = 0
    unknown_count: int = 0
    object_count: int = 0
    objects: list[dict] = field(default_factory=list)
    faces: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    hazards: list[dict] = field(default_factory=list)
    live_activity: list[dict] = field(default_factory=list)
    jpeg: bytes | None = None
    error: str | None = None
    active_features: list[str] = field(default_factory=list)
    camera_source: str | None = None
    camera_id: str | None = None
    camera_name: str | None = None
    camera_role: str | None = None


class CameraService:
    """Singleton camera + feature orchestrator for HTTP/MJPEG clients."""

    _instance: CameraService | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.store = FaceStore(DB_DIR)
        self.unknown_store = UnknownFaceStore(UNKNOWN_DIR)
        self.pipeline = FacePipeline(self.store)
        self.entry_tracker = EntryTracker(ENTRY_DIR, unknown_store=self.unknown_store)
        self.activity_logger = ActivityLogger(ACTIVITY_DIR)
        self.safety = SafetyMonitor(SAFETY_DIR, dwell_sec=30.0)
        self.feature_config = FeatureConfig()
        self.process_config = ProcessConfig(
            max_width=480,
            display_width=720,
            imgsz=320,
            yolo_every=4,
            emotion_every=4,
            behavior_every=6,
            object_every=5,
            hazard_every=6,
            unattended_dwell_sec=30.0,
        )
        self.orchestrator = FeatureOrchestrator(
            pipeline=self.pipeline,
            features=self.feature_config,
            entry_tracker=self.entry_tracker,
            unknown_store=self.unknown_store,
            activity_logger=self.activity_logger,
            safety=self.safety,
            process_config=self.process_config,
        )
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state = LiveState()
        self._state_lock = threading.Lock()

    @classmethod
    def get(cls) -> CameraService:
        with cls._lock:
            if cls._instance is None:
                cls._instance = CameraService()
            return cls._instance

    def get_features(self) -> list[dict[str, Any]]:
        with self._state_lock:
            running = self.state.running
        return self.orchestrator.get_feature_list(running)

    def update_features(self, updates: dict[str, bool]) -> list[dict[str, Any]]:
        self.feature_config.update_bulk(updates)
        return self.get_features()

    def set_feature(self, feature_id: str, enabled: bool) -> list[dict[str, Any]]:
        self.feature_config.set_enabled(feature_id, enabled)
        return self.get_features()

    def start(
        self,
        source: CameraSource | None = None,
        *,
        camera_id: str | None = None,
    ) -> dict[str, Any]:
        registry = CameraRegistry.get()
        cam_rec = registry.get_camera(camera_id) if camera_id else None
        if cam_rec is not None:
            resolved = registry.resolve_source(cam_rec)
            cam_id = cam_rec.id
            cam_name = cam_rec.name
            cam_role = cam_rec.role
        else:
            resolved = resolve_camera_source(source)
            cam_id = camera_id
            cam_name = describe_source(resolved)
            cam_role = "none"

        with self._state_lock:
            if self.state.running:
                if self.state.camera_id and cam_id and self.state.camera_id == cam_id:
                    return {"ok": True, "message": "Camera already running"}
                if self.state.camera_source == describe_source(resolved) and not cam_id:
                    return {"ok": True, "message": "Camera already running"}
                self._stop.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._stop.clear()
        self._cap = open_camera(resolved, width=640, height=480)
        if not self._cap.isOpened():
            label = describe_source(resolved)
            return {"ok": False, "message": f"Cannot open camera: {label}"}
        warm_up = 30 if is_rtsp_source(resolved) else 15
        warm_up_camera(self._cap, n=warm_up)
        self.orchestrator.reset_cache()
        self.orchestrator.set_camera_context(cam_id, cam_name, cam_role)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        with self._state_lock:
            self.state.running = True
            self.state.error = None
            self.state.camera_source = describe_source(resolved)
            self.state.camera_id = cam_id
            self.state.camera_name = cam_name
            self.state.camera_role = cam_role
        role_note = f", role={cam_role}" if cam_role and cam_role != "none" else ""
        return {
            "ok": True,
            "message": f"Camera started ({cam_name}{role_note})",
            "camera_source": describe_source(resolved),
            "camera_id": cam_id,
            "camera_name": cam_name,
            "camera_role": cam_role,
        }

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.orchestrator.clear_camera_context()
        with self._state_lock:
            self.state.running = False
            self.state.jpeg = None
            self.state.active_features = []
            self.state.camera_source = None
            self.state.camera_id = None
            self.state.camera_name = None
            self.state.camera_role = None
        self.feature_config.set_active(set())
        return {"ok": True, "message": "Camera stopped"}

    def _loop(self) -> None:
        while not self._stop.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                with self._state_lock:
                    self.state.error = "Camera lost"
                    self.state.running = False
                break
            ok, frame = read_camera_frame(cap)
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            try:
                out = self.orchestrator.process_frame(frame)
                result = out.frame_result

                _, jpeg = cv2.imencode(
                    ".jpg", result.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                )

                faces_data = [
                    {
                        "name": f.name,
                        "person_id": f.person_id,
                        "known": f.name != "Unknown",
                        "emotion": f.emotion,
                        "behavior": f.behavior,
                        "score": round(f.score, 3),
                    }
                    for f in result.faces
                ]
                objects_data = [
                    {
                        "label": o.label,
                        "confidence": round(o.confidence, 3),
                        "bbox": list(o.bbox),
                    }
                    for o in result.objects
                ]

                with self._state_lock:
                    self.state.fps = round(result.fps, 1)
                    self.state.person_count = result.person_count
                    self.state.face_count = result.face_count
                    self.state.known_count = result.known_count
                    self.state.unknown_count = result.unknown_count
                    self.state.object_count = result.object_count
                    self.state.faces = faces_data
                    self.state.objects = objects_data
                    self.state.alerts = out.alerts
                    self.state.hazards = out.hazards
                    self.state.live_activity = out.live_activity
                    self.state.active_features = sorted(out.active_features)
                    self.state.jpeg = jpeg.tobytes()
                    self.state.error = None
                    self.state.running = True
            except Exception as exc:
                with self._state_lock:
                    self.state.error = str(exc)
            time.sleep(0.02)

    def get_status(self) -> dict[str, Any]:
        with self._state_lock:
            s = self.state
            running = s.running
            payload = {
                "running": s.running,
                "fps": s.fps,
                "person_count": s.person_count,
                "face_count": s.face_count,
                "known_count": s.known_count,
                "unknown_count": s.unknown_count,
                "object_count": s.object_count,
                "faces": list(s.faces),
                "objects": list(s.objects),
                "alerts": list(s.alerts),
                "hazards": list(s.hazards),
                "fire_detected": any(h.get("type") == "fire" for h in s.hazards),
                "smoke_detected": any(h.get("type") == "smoke" for h in s.hazards),
                "fighting_detected": any(h.get("type") == "fighting" for h in s.hazards),
                "bags_tracked": sum(
                    1
                    for o in s.objects
                    if o.get("label", "").lower() in {"backpack", "handbag", "suitcase"}
                ),
                "live_activity": list(s.live_activity),
                "active_features": list(s.active_features),
                "camera_source": s.camera_source,
                "camera_id": s.camera_id,
                "camera_name": s.camera_name,
                "camera_role": s.camera_role,
                "default_camera_source": describe_source(get_default_camera_source()),
                "error": s.error,
                "entries_total": len(self.entry_tracker.entries),
                "incidents_total": len(self.safety.incidents.incidents),
                "registered_persons": len(self.store.faces),
                "unknown_persons": self.unknown_store.count(),
            }
        payload["features"] = self.orchestrator.get_feature_list(running)
        return payload

    def get_jpeg(self) -> bytes | None:
        with self._state_lock:
            return self.state.jpeg

    def mjpeg_generator(self):
        while not self._stop.is_set():
            jpeg = self.get_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.05)
