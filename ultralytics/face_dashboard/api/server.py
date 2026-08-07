"""
Campus Safety Admin API.

Run:
    cd ultralytics
    pip install fastapi uvicorn python-multipart
    python -m face_dashboard.api.server
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_dashboard.api.camera_service import CameraService
from face_dashboard.camera_config import describe_source, get_default_camera_source
from face_dashboard.camera_registry import CAMERA_ROLES, ROLE_LABELS, CameraRegistry

app = FastAPI(title="Campus Safety API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def svc() -> CameraService:
    return CameraService.get()


class CameraStartBody(BaseModel):
    camera_index: int | None = None
    source: str | None = None  # "0" webcam, "rtsp://..." CP Plus, null = .env default
    camera_id: str | None = None  # preferred — uses registry + role


class CameraCreateBody(BaseModel):
    name: str
    type: str = "rtsp"
    source: str = ""
    role: str = "none"
    channel: int | None = None
    notes: str = ""


class CameraUpdateBody(BaseModel):
    name: str | None = None
    type: str | None = None
    source: str | None = None
    role: str | None = None
    enabled: bool | None = None
    notes: str | None = None
    channel: int | None = None


class CameraRoleBody(BaseModel):
    role: str


class NvrDiscoverBody(BaseModel):
    ip: str
    username: str = "admin"
    password: str = ""
    channels: int = 4
    port: int = 554
    subtype: int = 1


class SyncCpPlusBody(BaseModel):
    channels: int | None = None
    probe: bool = False
    max_probe: int = 16


class ZoneUpdateBody(BaseModel):
    enabled: bool | None = None
    max_persons: int | None = None
    allow_unknown: bool | None = None


class RegisterFromCameraBody(BaseModel):
    name: str
    face_index: int = 0


class FeatureToggleBody(BaseModel):
    enabled: bool


@app.get("/api/features")
def list_features() -> list[dict[str, Any]]:
    return svc().get_features()


@app.patch("/api/features")
def update_features_bulk(body: dict[str, bool]) -> list[dict[str, Any]]:
    return svc().update_features(body)


@app.patch("/api/features/{feature_id}")
def update_feature(feature_id: str, body: FeatureToggleBody) -> list[dict[str, Any]]:
    from face_dashboard.services.registry import FEATURE_IDS

    if feature_id not in FEATURE_IDS:
        raise HTTPException(404, "Unknown feature")
    return svc().set_feature(feature_id, body.enabled)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return svc().get_status()


@app.get("/api/camera/config")
def camera_config() -> dict[str, Any]:
    from face_dashboard.camera_config import (
        describe_source,
        get_cpplus_env,
        get_default_camera_source,
        get_env_cameras,
    )

    default = get_default_camera_source()
    cp = get_cpplus_env()
    return {
        "default_source": describe_source(default),
        "is_rtsp_default": isinstance(default, str) and default.startswith("rtsp://"),
        "env_cameras": get_env_cameras(),
        "roles": ROLE_LABELS,
        "cpplus": (
            {
                "ip": cp["ip"],
                "username": cp["username"],
                "port": cp["port"],
                "subtype": cp["subtype"],
                "channels": cp["channels"],
                "has_password": bool(cp["password"]),
            }
            if cp
            else None
        ),
    }


@app.get("/api/cameras")
def list_cameras() -> dict[str, Any]:
    reg = CameraRegistry.get()
    reg.sync_cpplus_from_env(probe=False)
    return {
        "cameras": reg.list_all(),
        "roles": ROLE_LABELS,
        "role_ids": list(CAMERA_ROLES),
        "summary": reg.roles_summary(),
    }


@app.post("/api/cameras/sync-cpplus")
async def sync_cpplus_cameras(body: SyncCpPlusBody | None = None) -> dict[str, Any]:
    """Fetch all CP Plus channels from .env. probe=true tests RTSP (slower)."""
    import asyncio

    reg = CameraRegistry.get()
    opts = body or SyncCpPlusBody()
    # Run in thread so RTSP probe does not block the whole API event loop.
    return await asyncio.to_thread(
        reg.sync_cpplus_from_env,
        opts.channels,
        opts.probe,
        opts.max_probe,
    )


@app.post("/api/cameras")
def create_camera(body: CameraCreateBody) -> dict[str, Any]:
    if body.role not in CAMERA_ROLES:
        raise HTTPException(400, f"Invalid role. Use one of: {CAMERA_ROLES}")
    cam = CameraRegistry.get().add(body.model_dump())
    return cam.to_dict()


@app.patch("/api/cameras/{camera_id}")
def update_camera(camera_id: str, body: CameraUpdateBody) -> dict[str, Any]:
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        cam = CameraRegistry.get().update(camera_id, updates)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if cam is None:
        raise HTTPException(404, "Camera not found")
    return cam.to_dict()


@app.patch("/api/cameras/{camera_id}/role")
def set_camera_role(camera_id: str, body: CameraRoleBody) -> dict[str, Any]:
    try:
        cam = CameraRegistry.get().set_role(camera_id, body.role)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if cam is None:
        raise HTTPException(404, "Camera not found")
    return cam.to_dict()


@app.delete("/api/cameras/{camera_id}")
def delete_camera(camera_id: str) -> dict[str, str]:
    if not CameraRegistry.get().remove(camera_id):
        raise HTTPException(404, "Camera not found")
    return {"message": "Deleted"}


@app.post("/api/cameras/discover-nvr")
def discover_nvr(body: NvrDiscoverBody) -> dict[str, Any]:
    added = CameraRegistry.get().discover_nvr(
        ip=body.ip,
        username=body.username,
        password=body.password,
        channels=body.channels,
        port=body.port,
        subtype=body.subtype,
    )
    return {
        "added": [c.to_dict() for c in added],
        "cameras": CameraRegistry.get().list_all(),
        "message": f"Added {len(added)} channel(s)",
    }


@app.post("/api/camera/start")
def camera_start(body: CameraStartBody) -> dict[str, Any]:
    if body.camera_id:
        return svc().start(camera_id=body.camera_id)
    if body.source is not None:
        return svc().start(body.source)
    if body.camera_index is not None:
        return svc().start(body.camera_index)
    return svc().start(None)


@app.post("/api/camera/stop")
def camera_stop() -> dict[str, Any]:
    return svc().stop()


@app.get("/api/camera/stream")
def camera_stream() -> StreamingResponse:
    return StreamingResponse(
        svc().mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/camera/raw_stream")
def raw_camera_stream(
    source: str | None = None,
    camera_id: str | None = None,
) -> StreamingResponse:
    from face_dashboard.api.raw_streamer import raw_mjpeg_generator

    resolved = source
    if camera_id:
        cam = CameraRegistry.get().get_camera(camera_id)
        if cam is None:
            raise HTTPException(404, "Camera not found")
        resolved = CameraRegistry.get().resolve_source(cam)
    if resolved is None:
        raise HTTPException(400, "Provide camera_id or source query parameter")

    return StreamingResponse(
        raw_mjpeg_generator(resolved),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/camera/snapshot")
def camera_snapshot() -> dict[str, Any]:
    jpeg = svc().get_jpeg()
    if not jpeg:
        raise HTTPException(404, "No frame available — start the camera first")
    return {"image_base64": base64.b64encode(jpeg).decode("ascii")}


@app.get("/api/unknown-persons")
def list_unknown_persons() -> list[dict[str, Any]]:
    rows = []
    for person in svc().unknown_store.list_all():
        img_b64 = None
        if person.image_path and Path(person.image_path).exists():
            img_b64 = base64.b64encode(Path(person.image_path).read_bytes()).decode("ascii")
        rows.append(
            {
                "id": person.id,
                "first_seen": person.first_seen,
                "last_seen": person.last_seen,
                "visit_count": person.visit_count,
                "image_base64": img_b64,
            }
        )
    return rows


@app.delete("/api/unknown-persons/{person_id}")
def delete_unknown_person(person_id: str) -> dict[str, str]:
    if not svc().unknown_store.remove(person_id):
        raise HTTPException(404, "Unknown person not found")
    return {"message": "Deleted"}


@app.get("/api/persons")
def list_persons() -> list[dict[str, Any]]:
    s = svc()
    rows = []
    for face in s.store.faces:
        img_b64 = None
        if face.image_path and Path(face.image_path).exists():
            img_b64 = base64.b64encode(Path(face.image_path).read_bytes()).decode("ascii")
        rows.append(
            {
                "id": face.id,
                "name": face.name,
                "created_at": face.created_at,
                "image_base64": img_b64,
            }
        )
    return rows


@app.delete("/api/persons/{person_id}")
def delete_person(person_id: str) -> dict[str, str]:
    s = svc()
    if not s.store.remove(person_id):
        raise HTTPException(404, "Person not found")
    s.store.load()
    return {"message": "Deleted"}


@app.post("/api/persons")
async def register_person(
    name: str = Form(...),
    file: UploadFile = File(...),
    face_index: int = Form(0),
) -> dict[str, Any]:
    if not name.strip():
        raise HTTPException(400, "Name is required")
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image file")
    s = svc()
    face = s.pipeline.register_face(frame, name.strip(), int(face_index))
    if face is None:
        raise HTTPException(400, "No face detected — use a clear front-facing photo")
    s.store.load()
    return {"id": face.id, "name": face.name, "message": f"Registered {face.name}"}


@app.post("/api/persons/from-camera")
def register_from_camera(body: RegisterFromCameraBody) -> dict[str, Any]:
    if not body.name.strip():
        raise HTTPException(400, "Name is required")
    jpeg = svc().get_jpeg()
    if not jpeg:
        raise HTTPException(400, "Start the live camera first (Live Camera page)")
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode camera frame")
    s = svc()
    face = s.pipeline.register_face(frame, body.name.strip(), int(body.face_index))
    if face is None:
        raise HTTPException(400, "No face detected in camera frame")
    s.store.load()
    return {"id": face.id, "name": face.name, "message": f"Registered {face.name}"}


@app.get("/api/entries")
def list_entries(limit: int = 100) -> dict[str, Any]:
    t = svc().entry_tracker
    visits = t.visits
    return {
        "total": len(visits),
        "identified": sum(1 for v in visits if v.entry_type == "identified"),
        "unknown": sum(1 for v in visits if v.entry_type == "unknown"),
        "active": len(t._active_visits),
        "summary": t.summary_rows(),
        "currently_in": t.currently_in_rows(),
        "recent": t.visit_rows(limit),
    }


@app.get("/api/activity")
def list_activity(limit: int = 50) -> dict[str, Any]:
    a = svc().activity_logger
    return {
        "summary": a.summary_by_person(),
        "recent": a.recent_rows(limit),
        "chart_series": a.chart_series(limit_per_person=min(limit, 60)),
    }


@app.get("/api/incidents")
def list_incidents(limit: int = 50) -> dict[str, Any]:
    inc = svc().safety.incidents
    return {
        "total": len(inc.incidents),
        "counts": inc.count_by_type(),
        "recent": inc.recent_rows(limit),
    }


@app.post("/api/incidents/reset")
def reset_incidents() -> dict[str, str]:
    svc().safety.reset_session()
    return {"message": "Incidents reset"}


@app.get("/api/zones")
def list_zones() -> list[dict[str, Any]]:
    return [z.to_dict() for z in svc().safety.zones.zones]


@app.patch("/api/zones/{zone_id}")
def update_zone(zone_id: str, body: ZoneUpdateBody) -> dict[str, Any]:
    zones = svc().safety.zones
    for zone in zones.zones:
        if zone.id == zone_id:
            if body.enabled is not None:
                zone.enabled = body.enabled
            if body.max_persons is not None:
                zone.max_persons = body.max_persons
            if body.allow_unknown is not None:
                zone.allow_unknown = body.allow_unknown
            zones.save()
            return zone.to_dict()
    raise HTTPException(404, "Zone not found")


@app.get("/api/hazards/live")
def live_hazards() -> list[dict[str, Any]]:
    return svc().get_status().get("hazards", [])


@app.get("/api/objects/live")
def live_objects() -> list[dict[str, Any]]:
    return svc().get_status().get("objects", [])


def main() -> None:
    import uvicorn

    uvicorn.run(
        "face_dashboard.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
