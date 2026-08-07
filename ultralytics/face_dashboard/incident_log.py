"""Persist safety incidents (unattended objects, zone violations, overcrowding)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Incident:
    time: str
    incident_type: str
    severity: str
    message: str
    zone_id: str | None = None
    details: dict = field(default_factory=dict)


class IncidentLogger:
    """Append-only incident log with de-duplication cooldown."""

    def __init__(self, log_dir: Path, cooldown_sec: float = 20.0) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "incidents.json"
        self.cooldown_sec = cooldown_sec
        self.incidents: list[Incident] = []
        self._last_report: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.log_file.exists():
            return
        data = json.loads(self.log_file.read_text(encoding="utf-8"))
        self.incidents = [Incident(**i) for i in data.get("incidents", [])]

    def _save(self) -> None:
        payload = {"incidents": [i.__dict__ for i in self.incidents[-300:]]}
        self.log_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self.incidents.clear()
        self._last_report.clear()
        if self.log_file.exists():
            self.log_file.unlink()

    def _should_report(self, key: str, now: float) -> bool:
        last = self._last_report.get(key, 0.0)
        if now - last < self.cooldown_sec:
            return False
        self._last_report[key] = now
        return True

    def report(
        self,
        incident_type: str,
        message: str,
        *,
        severity: str = "warning",
        zone_id: str | None = None,
        dedupe_key: str | None = None,
        details: dict | None = None,
    ) -> Incident | None:
        import time

        now = time.time()
        key = dedupe_key or f"{incident_type}:{message}"
        if not self._should_report(key, now):
            return None
        incident = Incident(
            time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            incident_type=incident_type,
            severity=severity,
            message=message,
            zone_id=zone_id,
            details=details or {},
        )
        self.incidents.append(incident)
        self._save()
        return incident

    def recent_rows(self, n: int = 15) -> list[dict]:
        rows = []
        for inc in reversed(self.incidents[-n:]):
            rows.append(
                {
                    "Time": inc.time,
                    "Type": inc.incident_type.replace("_", " ").title(),
                    "Severity": inc.severity,
                    "Message": inc.message,
                }
            )
        return rows

    def count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for inc in self.incidents:
            counts[inc.incident_type] = counts.get(inc.incident_type, 0) + 1
        return counts
