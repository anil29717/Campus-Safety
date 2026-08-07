"""Persistent camera registry with entry / exit / restricted roles."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from face_dashboard.camera_config import (
    build_cpplus_rtsp_url,
    describe_source,
    get_env_cameras,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "config" / "cameras.json"

CAMERA_ROLES = ("none", "entry", "exit", "restricted")
ROLE_LABELS = {
    "none": "General",
    "entry": "Entry gate",
    "exit": "Exit gate",
    "restricted": "Restricted area",
}


@dataclass
class CameraRecord:
    id: str
    name: str
    type: str = "rtsp"  # rtsp | webcam | default
    source: str = ""
    role: str = "none"  # none | entry | exit | restricted
    channel: int | None = None
    enabled: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role_label"] = ROLE_LABELS.get(self.role, self.role)
        d["source_label"] = self._source_label()
        return d

    def _source_label(self) -> str:
        if self.type == "webcam":
            return f"Webcam {self.source or '0'}"
        if self.type == "default" or not self.source:
            return "System default (.env)"
        try:
            return describe_source(self.source)
        except Exception:
            return self.source[:48]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraRecord:
        role = str(data.get("role", "none")).lower()
        if role not in CAMERA_ROLES:
            role = "none"
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:10]),
            name=str(data.get("name", "Camera")),
            type=str(data.get("type", "rtsp")),
            source=str(data.get("source", "")),
            role=role,
            channel=data.get("channel"),
            enabled=bool(data.get("enabled", True)),
            notes=str(data.get("notes", "")),
        )


class CameraRegistry:
    """Load/save cameras.json and sync NVR channels from .env."""

    _instance: CameraRegistry | None = None
    _lock = threading.Lock()

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cameras: list[CameraRecord] = []
        self._file_lock = threading.Lock()
        self.load()
        self.sync_from_env()

    @classmethod
    def get(cls) -> CameraRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = CameraRegistry()
            return cls._instance

    def load(self) -> None:
        if not self.path.exists():
            self._cameras = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            raw = data.get("cameras", data if isinstance(data, list) else [])
            self._cameras = [CameraRecord.from_dict(c) for c in raw]
        except Exception:
            self._cameras = []

    def save(self) -> None:
        with self._file_lock:
            payload = {"cameras": [asdict(c) for c in self._cameras]}
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def sync_from_env(self) -> None:
        """Ensure CP Plus channels from .env appear in the registry (keep roles)."""
        env_cams = get_env_cameras()
        if not env_cams:
            if not self._cameras:
                self._cameras = [
                    CameraRecord(
                        id="default-1",
                        name="Default camera",
                        type="default",
                        source="",
                        role="none",
                    )
                ]
                self.save()
            return

        by_source = {c.source: c for c in self._cameras if c.source}
        by_id = {c.id: c for c in self._cameras}
        changed = False
        for ec in env_cams:
            existing = by_id.get(ec["id"]) or by_source.get(ec["source"])
            if existing:
                if existing.source != ec["source"] or existing.name != ec["name"]:
                    existing.source = ec["source"]
                    existing.type = "rtsp"
                    if existing.name.startswith("Channel ") or existing.id.startswith("env-cam-"):
                        existing.name = ec["name"]
                    ch = ec["id"].replace("env-cam-", "")
                    existing.channel = int(ch) if ch.isdigit() else existing.channel
                    changed = True
            else:
                ch = ec["id"].replace("env-cam-", "")
                self._cameras.append(
                    CameraRecord(
                        id=ec["id"],
                        name=ec["name"],
                        type="rtsp",
                        source=ec["source"],
                        role="none",
                        channel=int(ch) if ch.isdigit() else None,
                    )
                )
                changed = True
        if changed:
            self.save()

    def sync_cpplus_from_env(
        self,
        channels: int | None = None,
        probe: bool = False,
        max_probe: int = 16,
    ) -> dict[str, Any]:
        """Fetch all CP Plus NVR channels from .env and upsert into registry."""
        from face_dashboard.camera_config import (
            build_cpplus_rtsp_url,
            get_cpplus_env,
            probe_cpplus_channels,
        )

        cfg = get_cpplus_env()
        if not cfg:
            self.sync_from_env()
            return {
                "ok": False,
                "message": "CAMERA_IP not set in .env",
                "synced": 0,
                "probed": [],
                "cameras": self.list_all(),
            }

        if probe:
            # Always register .env channel count first; probe only annotates live ones.
            n = channels or cfg["channels"]
            channel_list = list(range(1, n + 1))
            probed = probe_cpplus_channels(max_channels=max_probe or n)
        else:
            n = channels or cfg["channels"]
            channel_list = list(range(1, n + 1))
            probed = []

        by_id = {c.id: c for c in self._cameras}
        synced = 0
        for i in channel_list:
            cam_id = f"env-cam-{i}"
            url = build_cpplus_rtsp_url(
                cfg["ip"],
                cfg["username"],
                cfg["password"],
                channel=i,
                subtype=cfg["subtype"],
                port=cfg["port"],
            )
            existing = by_id.get(cam_id)
            if existing:
                if existing.source != url or existing.channel != i:
                    existing.source = url
                    existing.type = "rtsp"
                    existing.channel = i
                    synced += 1
            else:
                self._cameras.append(
                    CameraRecord(
                        id=cam_id,
                        name=f"CP Plus CH {i}",
                        type="rtsp",
                        source=url,
                        role="none",
                        channel=i,
                    )
                )
                synced += 1
        self.save()
        return {
            "ok": True,
            "message": f"Synced {len(channel_list)} CP Plus channel(s)",
            "synced": synced,
            "channel_numbers": channel_list,
            "probed": probed,
            "cameras": self.list_all(),
        }

    def list_all(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self._cameras]

    def get_camera(self, camera_id: str) -> CameraRecord | None:
        for c in self._cameras:
            if c.id == camera_id:
                return c
        return None

    def get_by_role(self, role: str) -> list[CameraRecord]:
        return [c for c in self._cameras if c.role == role and c.enabled]

    def roles_summary(self) -> dict[str, Any]:
        summary = {r: [] for r in CAMERA_ROLES}
        for c in self._cameras:
            summary.setdefault(c.role, []).append(
                {"id": c.id, "name": c.name, "channel": c.channel}
            )
        return {
            "roles": summary,
            "labels": ROLE_LABELS,
            "entry": [c.to_dict() for c in self.get_by_role("entry")],
            "exit": [c.to_dict() for c in self.get_by_role("exit")],
            "restricted": [c.to_dict() for c in self.get_by_role("restricted")],
        }

    def add(self, data: dict[str, Any]) -> CameraRecord:
        cam = CameraRecord.from_dict(
            {
                **data,
                "id": data.get("id") or f"cam-{uuid.uuid4().hex[:8]}",
            }
        )
        self._cameras.append(cam)
        self.save()
        return cam

    def update(self, camera_id: str, updates: dict[str, Any]) -> CameraRecord | None:
        cam = self.get_camera(camera_id)
        if cam is None:
            return None
        if "name" in updates and updates["name"] is not None:
            cam.name = str(updates["name"]).strip() or cam.name
        if "type" in updates and updates["type"] is not None:
            cam.type = str(updates["type"])
        if "source" in updates and updates["source"] is not None:
            cam.source = str(updates["source"])
        if "role" in updates and updates["role"] is not None:
            role = str(updates["role"]).lower()
            if role not in CAMERA_ROLES:
                raise ValueError(f"Invalid role: {role}. Use {CAMERA_ROLES}")
            cam.role = role
        if "enabled" in updates and updates["enabled"] is not None:
            cam.enabled = bool(updates["enabled"])
        if "notes" in updates and updates["notes"] is not None:
            cam.notes = str(updates["notes"])
        if "channel" in updates:
            cam.channel = updates["channel"]
        self.save()
        return cam

    def set_role(self, camera_id: str, role: str) -> CameraRecord | None:
        return self.update(camera_id, {"role": role})

    def remove(self, camera_id: str) -> bool:
        before = len(self._cameras)
        self._cameras = [c for c in self._cameras if c.id != camera_id]
        if len(self._cameras) == before:
            return False
        self.save()
        return True

    def resolve_source(self, camera: CameraRecord):
        from face_dashboard.camera_config import get_default_camera_source, resolve_camera_source

        if camera.type == "default" or not camera.source:
            return get_default_camera_source()
        if camera.type == "webcam":
            try:
                return int(camera.source)
            except ValueError:
                return 0
        return resolve_camera_source(camera.source)

    def discover_nvr(
        self,
        ip: str,
        username: str = "admin",
        password: str = "",
        channels: int = 4,
        port: int = 554,
        subtype: int = 1,
    ) -> list[CameraRecord]:
        """Add or refresh NVR channels; keep existing roles."""
        added: list[CameraRecord] = []
        by_id = {c.id: c for c in self._cameras}
        host_key = ip.replace(".", "-")
        for i in range(1, channels + 1):
            url = build_cpplus_rtsp_url(ip, username, password, i, subtype, port)
            cam_id = f"env-cam-{i}"
            alt_id = f"nvr-{host_key}-ch{i}"
            existing = by_id.get(cam_id) or by_id.get(alt_id)
            if existing:
                existing.source = url
                existing.type = "rtsp"
                existing.channel = i
                if existing.name.startswith("Channel ") or existing.id.startswith("env-cam-"):
                    existing.name = f"CP Plus CH {i}"
                continue
            cam = CameraRecord(
                id=cam_id,
                name=f"CP Plus CH {i}",
                type="rtsp",
                source=url,
                role="none",
                channel=i,
            )
            self._cameras.append(cam)
            added.append(cam)
        self.save()
        return added
