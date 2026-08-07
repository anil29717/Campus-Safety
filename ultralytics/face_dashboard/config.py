"""Performance and detection settings (no heavy imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProcessConfig:
    """Tunable performance settings for live monitoring."""

    conf: float = 0.35
    max_width: int = 640
    display_width: int = 960
    imgsz: int = 416
    yolo_every: int = 3
    faces_only: bool = False
    jpeg_quality: int = 72
    emotion_every: int = 3
    behavior_every: int = 5
    enable_emotion: bool = True
    enable_behavior: bool = True
    enable_objects: bool = True
    object_every: int = 5
    enable_safety: bool = True
    unattended_dwell_sec: float = 30.0
    enable_fire_smoke: bool = True
    enable_fight: bool = True
    hazard_every: int = 6
    enable_person_detection: bool = True
    enable_face_recognition: bool = True
    enable_unknown_visitors: bool = True
    enable_bag_unattended: bool = True
    enable_safety_zones: bool = True
    enable_entry_tracking: bool = True


def reset_frame_cache(pipeline: object) -> None:
    """Reset YOLO skip-frame cache (safe for Streamlit-cached pipeline instances)."""
    if hasattr(pipeline, "reset_cache"):
        pipeline.reset_cache()
        return
    pipeline._frame_idx = 0
    pipeline._last_person_count = 0
    pipeline._last_person_boxes = None
    if hasattr(pipeline, "_last_objects"):
        pipeline._last_objects = []
