import time
from typing import Generator

import cv2

from face_dashboard.camera_config import is_rtsp_source, resolve_camera_source


def raw_mjpeg_generator(source: str | int | None) -> Generator[bytes, None, None]:
    """MJPEG stream from cv2.VideoCapture (no AI). Supports RTSP / CP Plus."""
    resolved = resolve_camera_source(source)

    if isinstance(resolved, str) and resolved.isdigit():
        resolved = int(resolved)

    if is_rtsp_source(resolved):
        cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(resolved)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        yield b""
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            time.sleep(0.04)
    finally:
        cap.release()
