"""Camera source from environment (webcam index or CP Plus RTSP)."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_PATH)
    except ImportError:
        if ENV_PATH.exists():
            for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

CameraSource = str | int


def build_cpplus_rtsp_url(
    ip: str,
    username: str = "admin",
    password: str = "",
    channel: int = 1,
    subtype: int = 1,
    port: int = 554,
) -> str:
    """Build CP Plus / Dahua-style RTSP URL. subtype=1 is lower-res (faster)."""
    host = ip.replace("http://", "").replace("https://", "").strip("/").split("/")[0]
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    return (
        f"rtsp://{user}:{pwd}@{host}:{port}/cam/realmonitor"
        f"?channel={channel}&subtype={subtype}"
    )


def get_default_camera_source() -> CameraSource:
    """Default source: CAMERA_SOURCE, or CP Plus from CAMERA_IP, else webcam index."""
    if url := os.environ.get("CAMERA_SOURCE", "").strip():
        return url
    ip = os.environ.get("CAMERA_IP", "").strip()
    if ip:
        return build_cpplus_rtsp_url(
            ip=ip,
            username=os.environ.get("CAMERA_USER", "admin"),
            password=os.environ.get("CAMERA_PASSWORD", ""),
            channel=int(os.environ.get("CAMERA_CHANNEL", "1")),
            subtype=int(os.environ.get("CAMERA_SUBTYPE", "1")),
            port=int(os.environ.get("CAMERA_PORT", "554")),
        )
    return int(os.environ.get("CAMERA_INDEX", "0"))


def get_cpplus_env() -> dict | None:
    """CP Plus NVR settings from .env, or None if not configured."""
    ip = os.environ.get("CAMERA_IP", "").strip()
    if not ip:
        return None
    return {
        "ip": ip,
        "username": os.environ.get("CAMERA_USER", "admin"),
        "password": os.environ.get("CAMERA_PASSWORD", ""),
        "port": int(os.environ.get("CAMERA_PORT", "554")),
        "subtype": int(os.environ.get("CAMERA_SUBTYPE", "1")),
        "channels": int(os.environ.get("CAMERA_CHANNELS_COUNT", "4")),
    }


def get_env_cameras() -> list[dict]:
    """Return all camera channels based on .env config."""
    cfg = get_cpplus_env()
    if not cfg:
        return []
    cams = []
    for i in range(1, cfg["channels"] + 1):
        url = build_cpplus_rtsp_url(
            cfg["ip"],
            cfg["username"],
            cfg["password"],
            channel=i,
            subtype=cfg["subtype"],
            port=cfg["port"],
        )
        cams.append({
            "id": f"env-cam-{i}",
            "name": f"CP Plus CH {i}",
            "type": "rtsp",
            "source": url,
            "channel": i,
        })
    return cams


def probe_cpplus_channels(
    max_channels: int | None = None,
    read_timeout_sec: float = 1.5,
) -> list[int]:
    """Try opening each RTSP channel (parallel); return channels that return a frame."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import cv2

    cfg = get_cpplus_env()
    if not cfg:
        return []
    n = max_channels or cfg["channels"]
    n = max(1, min(int(n), 32))

    def _probe_one(i: int) -> int | None:
        url = build_cpplus_rtsp_url(
            cfg["ip"],
            cfg["username"],
            cfg["password"],
            channel=i,
            subtype=cfg["subtype"],
            port=cfg["port"],
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                return None
            import time

            deadline = time.time() + read_timeout_sec
            while time.time() < deadline:
                ret, frame = cap.read()
                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                    return i
                time.sleep(0.05)
            return None
        finally:
            cap.release()

    working: list[int] = []
    with ThreadPoolExecutor(max_workers=min(4, n)) as pool:
        futures = [pool.submit(_probe_one, i) for i in range(1, n + 1)]
        for fut in as_completed(futures):
            try:
                ch = fut.result()
            except Exception:
                ch = None
            if ch is not None:
                working.append(ch)
    return sorted(working)


def is_rtsp_source(source: CameraSource) -> bool:
    return isinstance(source, str) and source.lower().startswith("rtsp://")


def describe_source(source: CameraSource) -> str:
    """Human-readable label without credentials."""
    if isinstance(source, int):
        return f"Webcam (index {source})"
    if is_rtsp_source(source):
        parsed = urlparse(source)
        host = parsed.hostname or "unknown"
        query = parsed.query
        if "channel=" in query:
            import urllib.parse
            q_dict = urllib.parse.parse_qs(query)
            if "channel" in q_dict:
                return f"CP Plus RTSP ({host} CH {q_dict['channel'][0]})"
        return f"CP Plus RTSP ({host})"
    return str(source)


def resolve_camera_source(source: CameraSource | None = None) -> CameraSource:
    if source is None:
        return get_default_camera_source()
    if isinstance(source, str):
        s = source.strip()
        if not s:
            return get_default_camera_source()
        if s.isdigit():
            return int(s)
        return s
    return source
