"""
Live webcam detection with a visible OpenCV window (boxes + labels).

Usage (from d:\\imp\\ultralytics):
    python live_camera.py
    python live_camera.py --camera 1
    python live_camera.py --persons-only

Press Q or Esc in the video window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
from ultralytics import YOLO

WINDOW_NAME = "YOLO Live Detection"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live camera with detection overlay window")
    p.add_argument("--model", default="yolo26n.pt", help="YOLO weights")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (0 = default)")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--persons-only", action="store_true", help="Only detect people (COCO class 0)")
    p.add_argument("--track", action="store_true", help="Enable tracking IDs (needs lap package)")
    p.add_argument("--width", type=int, default=1280, help="Requested capture width")
    p.add_argument("--height", type=int, default=720, help="Requested capture height")
    return p.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open webcam; CAP_DSHOW is often more stable on Windows."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {index}. Close Teams/Zoom/Camera app and try --camera 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> None:
    args = parse_args()
    classes = [0] if args.persons_only else None

    print(f"Loading {args.model} ...", flush=True)
    model = YOLO(args.model)

    print(f"Opening camera {args.camera} ...", flush=True)
    cap = open_camera(args.camera, args.width, args.height)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    print(f"Window '{WINDOW_NAME}' opened. Press Q or Esc in that window to quit.", flush=True)

    prev_t = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame — retrying ...")
                time.sleep(0.05)
                continue

            if args.track:
                results = model.track(
                    frame,
                    conf=args.conf,
                    classes=classes,
                    persist=True,
                    verbose=False,
                )
            else:
                results = model.predict(
                    frame,
                    conf=args.conf,
                    classes=classes,
                    verbose=False,
                )

            annotated = results[0].plot()
            n = len(results[0].boxes) if results[0].boxes is not None else 0

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            label = f"FPS: {fps:.1f}  |  Detections: {n}  |  Q = quit"
            cv2.putText(
                annotated,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
