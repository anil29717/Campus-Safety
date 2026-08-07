"""Log emotion and behavior events for registered (DB) users only."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ActivityRecord:
    time: str
    person_id: str
    name: str
    emotion: str | None = None
    emotion_conf: float = 0.0
    behavior: str | None = None
    behavior_conf: float = 0.0


@dataclass
class LiveActivity:
    """Current emotion/behavior per registered person in frame."""

    person_id: str
    name: str
    emotion: str
    emotion_conf: float
    behavior: str
    behavior_conf: float


@dataclass
class ActivitySnapshot:
    live: list[LiveActivity] = field(default_factory=list)
    new_records: list[ActivityRecord] = field(default_factory=list)


class ActivityLogger:
    """Track and persist emotion/behavior for known users."""

    def __init__(self, log_dir: Path, log_interval_sec: float = 4.0) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "activity_log.json"
        self.log_interval_sec = log_interval_sec

        self.records: list[ActivityRecord] = []
        self._last_log: dict[str, float] = {}
        self._last_emotion: dict[str, str] = {}
        self._last_behavior: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.log_file.exists():
            return
        data = json.loads(self.log_file.read_text(encoding="utf-8"))
        self.records = [ActivityRecord(**r) for r in data.get("records", [])]

    def _save(self) -> None:
        payload = {"records": [r.__dict__ for r in self.records[-500:]]}
        self.log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self.records.clear()
        self._last_log.clear()
        self._last_emotion.clear()
        self._last_behavior.clear()
        if self.log_file.exists():
            self.log_file.unlink()

    def update(self, faces: list, now: float | None = None) -> ActivitySnapshot:
        import time

        now = now or time.time()
        snap = ActivitySnapshot()

        for face in faces:
            if face.name == "Unknown" or not getattr(face, "person_id", None):
                continue

            pid = face.person_id
            emotion = getattr(face, "emotion", None) or "—"
            behavior = getattr(face, "behavior", None) or "—"
            e_conf = float(getattr(face, "emotion_conf", 0.0))
            b_conf = float(getattr(face, "behavior_conf", 0.0))

            snap.live.append(
                LiveActivity(
                    person_id=pid,
                    name=face.name,
                    emotion=emotion,
                    emotion_conf=e_conf,
                    behavior=behavior,
                    behavior_conf=b_conf,
                )
            )

            last_t = self._last_log.get(pid, 0.0)
            emotion_changed = self._last_emotion.get(pid) != emotion
            behavior_changed = self._last_behavior.get(pid) != behavior
            should_log = (now - last_t) >= self.log_interval_sec or emotion_changed or behavior_changed

            if should_log and emotion != "—":
                rec = ActivityRecord(
                    time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    person_id=pid,
                    name=face.name,
                    emotion=emotion,
                    emotion_conf=e_conf,
                    behavior=behavior if behavior != "—" else None,
                    behavior_conf=b_conf,
                )
                self.records.append(rec)
                snap.new_records.append(rec)
                self._last_log[pid] = now
                self._last_emotion[pid] = emotion
                self._last_behavior[pid] = behavior

        self._save()
        return snap

    def summary_by_person(self) -> list[dict]:
        """Latest emotion + dominant behavior per registered person."""
        latest: dict[str, ActivityRecord] = {}
        for rec in self.records:
            if rec.person_id not in latest or rec.time > latest[rec.person_id].time:
                latest[rec.person_id] = rec

        rows = []
        for rec in sorted(latest.values(), key=lambda r: r.name):
            rows.append(
                {
                    "Person": rec.name,
                    "Last emotion": rec.emotion or "—",
                    "Last behavior": rec.behavior or "—",
                    "Time": rec.time,
                }
            )
        return rows

    def recent_rows(self, n: int = 15) -> list[dict]:
        recent = list(reversed(self.records[-n:]))
        return [
            {
                "Time": r.time,
                "Person": r.name,
                "Emotion": r.emotion or "—",
                "Behavior": r.behavior or "—",
            }
            for r in recent
        ]

    def chart_series(self, limit_per_person: int = 40) -> list[dict]:
        """Time series per person for multi-line charts (emotion confidence 0–100)."""
        by_person: dict[str, list[ActivityRecord]] = {}
        for rec in self.records:
            by_person.setdefault(rec.name, []).append(rec)

        series = []
        for name in sorted(by_person.keys()):
            points = []
            for rec in by_person[name][-limit_per_person:]:
                points.append(
                    {
                        "time": rec.time,
                        "emotion": rec.emotion or "—",
                        "behavior": rec.behavior or "—",
                        "value": round(rec.emotion_conf * 100, 1),
                    }
                )
            if points:
                series.append({"person": name, "points": points})
        return series
