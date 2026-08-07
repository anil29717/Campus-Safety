"""Track person in/out times when entering or leaving the camera view."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

UNKNOWN_MATCH_THRESHOLD = 0.32


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _new_unknown_person_id() -> str:
    return f"UNK-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class VisitRecord:
    """One camera visit: in time when seen, out time when left."""

    id: str
    name: str
    entry_type: str  # "identified" | "unknown"
    identity_key: str
    person_id: str
    in_time: str
    out_time: str | None = None
    duration_sec: float | None = None
    camera_id: str | None = None
    camera_name: str | None = None
    gate_role: str | None = None  # entry | exit | restricted | none

    @property
    def time(self) -> str:
        """Backward compatibility with legacy entry rows."""
        return self.in_time


# Legacy alias
EntryRecord = VisitRecord


@dataclass
class EntryStats:
    """Current snapshot for dashboard."""

    in_frame_total: int = 0
    in_frame_identified: int = 0
    in_frame_unknown: int = 0
    total_entries: int = 0
    identified_entries: int = 0
    unknown_entries: int = 0
    active_visits: int = 0
    new_entries_this_frame: list[VisitRecord] = field(default_factory=list)
    closed_visits_this_frame: list[VisitRecord] = field(default_factory=list)


class EntryTracker:
    """Log in/out times each time a person enters or leaves the camera."""

    def __init__(
        self,
        log_dir: Path,
        exit_grace_sec: float = 2.5,
        unknown_store: object | None = None,
    ) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "entry_log.json"
        self.exit_grace_sec = exit_grace_sec
        self._unknown_store = unknown_store

        self.visits: list[VisitRecord] = []
        self.counts: dict[str, int] = {}
        self.name_counts: dict[str, int] = {}

        self._active_visits: dict[str, VisitRecord] = {}
        self._visit_started: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._unknown_prototypes: list[tuple[str, np.ndarray]] = []
        self._unknown_person_ids: dict[str, str] = {}
        self._unknown_counter = 0

        self._load()

    @property
    def entries(self) -> list[VisitRecord]:
        return self.visits

    def _visit_from_dict(self, raw: dict) -> VisitRecord:
        allowed = {f.name for f in fields(VisitRecord)}
        data = {k: v for k, v in raw.items() if k in allowed}

        if not data.get("in_time") and raw.get("time"):
            data["in_time"] = raw["time"]

        if not data.get("id"):
            data["id"] = raw.get("id") or uuid.uuid4().hex[:10]

        identity_key = data.get("identity_key", raw.get("identity_key", ""))
        entry_type = data.get("entry_type", raw.get("entry_type", ""))
        name = data.get("name", raw.get("name", ""))

        for key, value in (("name", name), ("entry_type", entry_type), ("identity_key", identity_key)):
            if key not in data and value:
                data[key] = value

        if not data.get("person_id"):
            data["person_id"] = self._derive_person_id(identity_key, entry_type, name)

        return VisitRecord(**data)

    def _derive_person_id(self, identity_key: str, entry_type: str, name: str) -> str:
        if identity_key.startswith("known:"):
            suffix = identity_key.split(":", 1)[1]
            if suffix.startswith("CSU-") or len(suffix) >= 10:
                return suffix
        if identity_key.startswith("unknown:"):
            uid = identity_key.split(":", 1)[1]
            return self._unknown_person_ids.get(uid, _new_unknown_person_id())
        if entry_type == "unknown":
            return _new_unknown_person_id()
        return f"CSU-LEGACY-{uuid.uuid4().hex[:6].upper()}"

    def _load(self) -> None:
        if not self.log_file.exists():
            return
        data = json.loads(self.log_file.read_text(encoding="utf-8"))
        raw = data.get("visits", data.get("entries", []))
        self.visits = [self._visit_from_dict(v) for v in raw]
        self.counts = data.get("counts", {})
        self.name_counts = data.get("name_counts", {})
        self._unknown_counter = data.get("unknown_counter", 0)
        self._unknown_person_ids = data.get("unknown_person_ids", {})
        for v in self.visits:
            if v.entry_type == "unknown" and v.identity_key.startswith("unknown:"):
                uid = v.identity_key.split(":", 1)[1]
                if uid not in self._unknown_person_ids:
                    self._unknown_person_ids[uid] = v.person_id
            if v.out_time is None:
                self._active_visits[v.identity_key] = v

    def _save(self) -> None:
        payload = {
            "visits": [v.__dict__ for v in self.visits[-500:]],
            "counts": self.counts,
            "name_counts": self.name_counts,
            "unknown_counter": self._unknown_counter,
            "unknown_person_ids": self._unknown_person_ids,
        }
        self.log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def reset_session(self) -> None:
        self.visits.clear()
        self.counts.clear()
        self.name_counts.clear()
        self._active_visits.clear()
        self._visit_started.clear()
        self._last_seen.clear()
        self._unknown_prototypes.clear()
        self._unknown_person_ids.clear()
        self._unknown_counter = 0
        if self.log_file.exists():
            self.log_file.unlink()

    def _match_unknown(self, feature: np.ndarray) -> str:
        best_id = None
        best_score = UNKNOWN_MATCH_THRESHOLD
        a = feature.flatten().astype(np.float32)
        for uid, proto in self._unknown_prototypes:
            b = proto.flatten().astype(np.float32)
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            if sim > best_score:
                best_score = sim
                best_id = uid
        if best_id is None:
            self._unknown_counter += 1
            best_id = f"unknown_{self._unknown_counter}"
            self._unknown_prototypes.append((best_id, feature.copy()))
            self._unknown_person_ids[best_id] = _new_unknown_person_id()
        return best_id

    def _identity(
        self,
        name: str,
        feature: np.ndarray | None,
        registered_person_id: str | None = None,
    ) -> tuple[str, str, str, str]:
        if name != "Unknown":
            person_id = registered_person_id or f"CSU-LEGACY-{name.replace(' ', '_')}"
            key = f"known:{person_id}"
            return key, name, "identified", person_id

        if feature is None:
            self._unknown_counter += 1
            uid = f"unknown_{self._unknown_counter}"
            person_id = _new_unknown_person_id()
            self._unknown_person_ids[uid] = person_id
            label = f"Unknown ({person_id})"
            return f"unknown:{uid}", label, "unknown", person_id

        uid = self._match_unknown(feature)
        person_id = self._unknown_person_ids[uid]
        label = f"Unknown ({person_id})"
        return f"unknown:{uid}", label, "unknown", person_id

    def _open_visit(
        self,
        key: str,
        display_name: str,
        entry_type: str,
        person_id: str,
        now: float,
        camera_id: str | None = None,
        camera_name: str | None = None,
        gate_role: str | None = None,
    ) -> VisitRecord:
        visit = VisitRecord(
            id=uuid.uuid4().hex[:10],
            name=display_name,
            entry_type=entry_type,
            identity_key=key,
            person_id=person_id,
            in_time=_now_str(),
            camera_id=camera_id,
            camera_name=camera_name,
            gate_role=gate_role,
        )
        self._active_visits[key] = visit
        self._visit_started[key] = now
        self.visits.append(visit)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.name_counts[display_name] = self.name_counts.get(display_name, 0) + 1
        return visit

    def _close_visit(
        self,
        key: str,
        now: float,
        camera_id: str | None = None,
        camera_name: str | None = None,
        gate_role: str | None = None,
    ) -> VisitRecord | None:
        visit = self._active_visits.pop(key, None)
        if visit is None:
            return None
        visit.out_time = _now_str()
        started = self._visit_started.pop(key, now)
        visit.duration_sec = round(max(now - started, 0.0), 1)
        if camera_id:
            visit.camera_id = camera_id
        if camera_name:
            visit.camera_name = camera_name
        if gate_role:
            visit.gate_role = gate_role
        return visit

    def update(
        self,
        faces: list,
        now: float | None = None,
        *,
        mode: str = "general",
        camera_id: str | None = None,
        camera_name: str | None = None,
    ) -> EntryStats:
        """Update visits.

        mode:
          - general: open on appear, close after leave grace (legacy)
          - entry: open on appear; do not close when leaving frame
          - exit: close active visit when face appears; do not open new
          - restricted: same as general (presence + zones handled elsewhere)
        """
        import time

        now = now or time.time()
        mode = (mode or "general").lower()
        if mode == "none":
            mode = "general"
        gate_role = mode if mode in {"entry", "exit", "restricted"} else None

        stats = EntryStats()
        stats.in_frame_total = len(faces)
        stats.in_frame_identified = sum(1 for f in faces if f.name != "Unknown")
        stats.in_frame_unknown = stats.in_frame_total - stats.in_frame_identified

        seen_now: set[str] = set()
        new_visits: list[VisitRecord] = []
        closed_visits: list[VisitRecord] = []

        for face in faces:
            feature = getattr(face, "feature", None)
            registered_id = getattr(face, "person_id", None)
            key, display_name, entry_type, person_id = self._identity(
                face.name, feature, registered_id
            )
            seen_now.add(key)
            self._last_seen[key] = now

            if mode == "exit":
                if key in self._active_visits:
                    closed = self._close_visit(
                        key, now, camera_id, camera_name, "exit"
                    )
                    if closed:
                        closed_visits.append(closed)
                continue

            # entry / general / restricted — open if new
            new_visit = key not in self._active_visits
            if new_visit:
                new_visits.append(
                    self._open_visit(
                        key,
                        display_name,
                        entry_type,
                        person_id,
                        now,
                        camera_id=camera_id,
                        camera_name=camera_name,
                        gate_role=("entry" if mode == "entry" else gate_role),
                    )
                )

            if entry_type == "unknown" and self._unknown_store is not None:
                aligned = getattr(face, "aligned", None)
                self._unknown_store.record(
                    person_id,
                    aligned,
                    new_visit=new_visit,
                )

        # Close on leave only for general / restricted — NOT for entry gate
        if mode in {"general", "restricted"}:
            for key in list(self._active_visits.keys()):
                if key in seen_now:
                    continue
                last = self._last_seen.get(key, 0.0)
                if now - last < self.exit_grace_sec:
                    continue
                closed = self._close_visit(key, now, camera_id, camera_name, gate_role)
                if closed:
                    closed_visits.append(closed)

        stats.new_entries_this_frame = new_visits
        stats.closed_visits_this_frame = closed_visits
        stats.active_visits = len(self._active_visits)
        stats.total_entries = len(self.visits)
        stats.identified_entries = sum(1 for v in self.visits if v.entry_type == "identified")
        stats.unknown_entries = sum(1 for v in self.visits if v.entry_type == "unknown")
        self._save()
        return stats

    def summary_rows(self) -> list[dict]:
        by_person: dict[str, dict] = {}
        for v in self.visits:
            pid = v.person_id
            if pid not in by_person:
                by_person[pid] = {"Person ID": pid, "Name": v.name, "Visits": 0}
            by_person[pid]["Visits"] += 1
            if v.name and not v.name.startswith("Unknown"):
                by_person[pid]["Name"] = v.name
        rows = list(by_person.values())
        rows.sort(key=lambda r: -r["Visits"])
        return rows

    def visit_rows(self, n: int = 50) -> list[dict]:
        recent = list(reversed(self.visits[-n:]))
        rows = []
        for v in recent:
            status = "Inside" if v.out_time is None else "Left"
            duration = f"{v.duration_sec:.0f}s" if v.duration_sec is not None else "—"
            rows.append(
                {
                    "Person ID": v.person_id,
                    "Person": v.name,
                    "Type": v.entry_type,
                    "In time": v.in_time,
                    "Out time": v.out_time or "—",
                    "Duration": duration,
                    "Status": status,
                    "Camera": v.camera_name or "—",
                    "Gate": v.gate_role or "—",
                }
            )
        return rows

    def currently_in_rows(self) -> list[dict]:
        rows = []
        for v in self._active_visits.values():
            rows.append(
                {
                    "Person ID": v.person_id,
                    "Person": v.name,
                    "Type": v.entry_type,
                    "In time": v.in_time,
                    "Status": "Inside",
                    "Camera": v.camera_name or "—",
                    "Gate": v.gate_role or "—",
                }
            )
        return sorted(rows, key=lambda r: r["Person ID"])

    def total_entries_display(self) -> int:
        return len(self.visits)

    def identified_entries_display(self) -> int:
        return sum(1 for v in self.visits if v.entry_type == "identified")

    def unknown_entries_display(self) -> int:
        return sum(1 for v in self.visits if v.entry_type == "unknown")
