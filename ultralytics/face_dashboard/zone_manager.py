"""Camera zones: restricted areas and crowd limits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Zone:
    """Normalized rectangular zone (fractions of frame width/height)."""

    id: str
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    zone_type: str = "restricted"  # restricted | crowd
    allow_unknown: bool = False
    max_persons: int = 5
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "zone_type": self.zone_type,
            "allow_unknown": self.allow_unknown,
            "max_persons": self.max_persons,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Zone:
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
            zone_type=str(data.get("zone_type", "restricted")),
            allow_unknown=bool(data.get("allow_unknown", False)),
            max_persons=int(data.get("max_persons", 5)),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class ZoneAlert:
    zone_id: str
    zone_name: str
    alert_type: str  # unauthorized | overcrowding
    message: str
    count: int = 0


DEFAULT_ZONES = [
    Zone(
        id="restricted_lab",
        name="Restricted Lab",
        x1=0.2,
        y1=0.15,
        x2=0.8,
        y2=0.9,
        zone_type="restricted",
        allow_unknown=False,
    ),
    Zone(
        id="main_hall",
        name="Main Hall",
        x1=0.0,
        y1=0.0,
        x2=1.0,
        y2=1.0,
        zone_type="crowd",
        max_persons=8,
        allow_unknown=True,
    ),
]


def _box_center_xyxy(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_center_xywh(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


class ZoneManager:
    """Load zones and evaluate face/person positions against rules."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = config_dir / "zones.json"
        self.zones: list[Zone] = []
        self._load()

    def _load(self) -> None:
        if not self.config_file.exists():
            self.zones = list(DEFAULT_ZONES)
            self.save()
            return
        data = json.loads(self.config_file.read_text(encoding="utf-8"))
        self.zones = [Zone.from_dict(z) for z in data.get("zones", [])]
        if not self.zones:
            self.zones = list(DEFAULT_ZONES)
            self.save()

    def save(self) -> None:
        payload = {"zones": [z.to_dict() for z in self.zones]}
        self.config_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def pixel_rect(self, zone: Zone, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = int(zone.x1 * width)
        y1 = int(zone.y1 * height)
        x2 = int(zone.x2 * width)
        y2 = int(zone.y2 * height)
        return x1, y1, x2, y2

    def contains_point(self, zone: Zone, cx: float, cy: float, width: int, height: int) -> bool:
        x1, y1, x2, y2 = self.pixel_rect(zone, width, height)
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def evaluate(
        self,
        faces: list,
        person_boxes: list[tuple[int, int, int, int]],
        frame_width: int,
        frame_height: int,
    ) -> list[ZoneAlert]:
        alerts: list[ZoneAlert] = []
        for zone in self.zones:
            if not zone.enabled:
                continue

            if zone.zone_type == "restricted":
                for face in faces:
                    cx, cy = _box_center_xywh(face.bbox)
                    if not self.contains_point(zone, cx, cy, frame_width, frame_height):
                        continue
                    if face.name == "Unknown" and not zone.allow_unknown:
                        alerts.append(
                            ZoneAlert(
                                zone_id=zone.id,
                                zone_name=zone.name,
                                alert_type="unauthorized",
                                message=f"Unknown person in restricted zone: {zone.name}",
                            )
                        )
                        break

            elif zone.zone_type == "crowd":
                count = 0
                for box in person_boxes:
                    cx, cy = _box_center_xyxy(box)
                    if self.contains_point(zone, cx, cy, frame_width, frame_height):
                        count += 1
                if count == 0:
                    for face in faces:
                        cx, cy = _box_center_xywh(face.bbox)
                        if self.contains_point(zone, cx, cy, frame_width, frame_height):
                            count += 1
                if count > zone.max_persons:
                    alerts.append(
                        ZoneAlert(
                            zone_id=zone.id,
                            zone_name=zone.name,
                            alert_type="overcrowding",
                            message=f"Overcrowding in {zone.name}: {count}/{zone.max_persons} persons",
                            count=count,
                        )
                    )
        return alerts
