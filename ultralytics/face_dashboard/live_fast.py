"""
Fast OpenCV window version — much smoother than Streamlit for live video.

Run:
    python -m face_dashboard.live_fast
    python -m face_dashboard.live_fast --preset full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_dashboard.config import ProcessConfig
from face_dashboard.face_pipeline import FacePipeline, open_camera, warm_up_camera
from face_dashboard.face_store import FaceStore
from face_dashboard.safety_monitor import SafetyMonitor

WINDOW = "Campus Safety — name · objects · emotion · behavior · zones"
DB_DIR = ROOT / "data" / "known_faces"
SAFETY_DIR = ROOT / "data" / "safety"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast face + person monitor (OpenCV window)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--preset", choices=["full", "fast", "fastest", "balanced"], default="full")
    return p.parse_args()


def preset_config(name: str) -> ProcessConfig:
    if name == "fastest":
        return ProcessConfig(
            max_width=480, display_width=640, imgsz=320, yolo_every=99, faces_only=True,
            enable_emotion=True, enable_behavior=False, emotion_every=4, behavior_every=8,
            enable_objects=True, object_every=8,
        )
    if name == "fast":
        return ProcessConfig(
            max_width=480, display_width=720, imgsz=320, yolo_every=5,
            enable_emotion=True, enable_behavior=True, enable_objects=True, object_every=5,
        )
    # full / balanced — all features + campus safety
    return ProcessConfig(
        max_width=480, display_width=720, imgsz=320, yolo_every=4, faces_only=False,
        enable_emotion=True, enable_behavior=True, enable_objects=True,
        emotion_every=4, behavior_every=6, object_every=5,
        enable_safety=True, unattended_dwell_sec=30.0,
        enable_fire_smoke=True, enable_fight=True, hazard_every=6,
    )


def main() -> None:
    args = parse_args()
    cfg = preset_config(args.preset)
    store = FaceStore(DB_DIR)
    pipeline = FacePipeline(store)
    safety = SafetyMonitor(SAFETY_DIR, dwell_sec=cfg.unattended_dwell_sec)

    cap = open_camera(args.camera, width=640, height=480)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    warm_up_camera(cap, n=15)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Detection: bags (orange), fire/smoke/fighting (red), zones (orange/yellow)")
    print("Press Q/Esc to quit. Press N to register largest face (name in terminal).")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            result = pipeline.process_frame(frame, cfg)
            if cfg.enable_safety:
                snap = safety.update(result, frame, cfg=cfg)
                for alert in snap.alerts:
                    print(f"ALERT: {alert.message}")
            cv2.imshow(WINDOW, result.frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("n"), ord("N")) and result.faces:
                name = input("Enter name for largest face: ").strip()
                if name:
                    pipeline.register_face(frame, name, 0)
                    print(f"Registered {name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
