"""Campus safety: unattended objects, restricted zones, overcrowding, hazards."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from face_dashboard.hazard_detector import DetectedHazard, HazardDetector
from face_dashboard.incident_log import Incident, IncidentLogger
from face_dashboard.face_pipeline import FrameResult
from face_dashboard.unattended_object_tracker import UnattendedObjectTracker
from face_dashboard.zone_manager import ZoneManager

if TYPE_CHECKING:
    from face_dashboard.face_pipeline import DetectedFace, DetectedObject


@dataclass
class SafetyAlert:
    alert_type: str
    message: str
    severity: str = "warning"


@dataclass
class SafetySnapshot:
    alerts: list[SafetyAlert] = field(default_factory=list)
    new_incidents: list[Incident] = field(default_factory=list)
    unattended_count: int = 0
    zone_violations: int = 0
    hazards: list[DetectedHazard] = field(default_factory=list)
    fire_count: int = 0
    smoke_count: int = 0
    fighting_active: bool = False
    bag_count: int = 0


class SafetyMonitor:
    """Orchestrate zone checks, unattended bag tracking, hazards, and incident logging."""

    def __init__(
        self,
        data_dir: Path,
        *,
        dwell_sec: float = 30.0,
        enable_unattended: bool = True,
        enable_zones: bool = True,
        enable_fire_smoke: bool = True,
        enable_fight: bool = True,
        hazard_every: int = 6,
    ) -> None:
        self.data_dir = data_dir
        self.enable_unattended = enable_unattended
        self.enable_zones = enable_zones
        self.enable_fire_smoke = enable_fire_smoke
        self.enable_fight = enable_fight
        self.hazard_every = hazard_every
        self.dwell_sec = dwell_sec

        self.zones = ZoneManager(data_dir / "zones")
        self.unattended = UnattendedObjectTracker(dwell_sec=dwell_sec)
        self.hazards = HazardDetector()
        self.incidents = IncidentLogger(data_dir / "incidents")

    def set_dwell_sec(self, sec: float) -> None:
        self.dwell_sec = sec
        self.unattended.dwell_sec = sec

    def reset_session(self) -> None:
        self.unattended.reset()
        self.hazards.reset()
        self.incidents.reset()

    def _hazard_settings(self, cfg: Any | None) -> tuple[bool, bool, int, int]:
        if cfg is None:
            return self.enable_fire_smoke, self.enable_fight, self.hazard_every, 416
        return (
            getattr(cfg, "enable_fire_smoke", self.enable_fire_smoke),
            getattr(cfg, "enable_fight", self.enable_fight),
            getattr(cfg, "hazard_every", self.hazard_every),
            getattr(cfg, "imgsz", 416),
        )

    def update(
        self,
        result: FrameResult,
        source_frame: np.ndarray,
        cfg: Any | None = None,
    ) -> SafetySnapshot:
        fh, fw = source_frame.shape[:2]
        dscale = result.frame.shape[1] / fw if fw else 1.0

        faces: list[DetectedFace] = result.faces
        objects: list[DetectedObject] = result.objects
        person_boxes = list(result.person_boxes)

        alerts: list[SafetyAlert] = []
        new_incidents: list[Incident] = []
        hazard_list: list[DetectedHazard] = []

        enable_fire, enable_fight, hazard_every, imgsz = self._hazard_settings(cfg)
        if enable_fire or enable_fight:
            hazard_snap = self.hazards.update(
                source_frame,
                person_boxes,
                enable_fire_smoke=enable_fire and self.enable_fire_smoke,
                enable_fight=enable_fight and self.enable_fight,
                hazard_every=hazard_every,
                imgsz=imgsz,
            )
            hazard_list = hazard_snap.detections
            for hazard in hazard_snap.new_alerts:
                sev = "high" if hazard.hazard_type in {"fire", "fighting"} else "warning"
                alerts.append(
                    SafetyAlert(alert_type=hazard.hazard_type, message=hazard.message, severity=sev)
                )
                inc = self.incidents.report(
                    hazard.hazard_type,
                    hazard.message,
                    severity=sev,
                    dedupe_key=f"{hazard.hazard_type}:live",
                    details={
                        "confidence": hazard.confidence,
                        "bbox": list(hazard.bbox),
                    },
                )
                if inc:
                    new_incidents.append(inc)

        if self.enable_unattended:
            un_snap = self.unattended.update(objects, person_boxes)
            for alert in un_snap.new_alerts:
                alerts.append(
                    SafetyAlert(alert_type="unattended_bag", message=alert.message, severity="high")
                )
                inc = self.incidents.report(
                    "unattended_bag",
                    alert.message,
                    severity="high",
                    dedupe_key=f"unattended:{alert.track_id}",
                    details={"label": alert.label, "dwell_sec": alert.dwell_sec},
                )
                if inc:
                    new_incidents.append(inc)

        if self.enable_zones:
            zone_alerts = self.zones.evaluate(faces, person_boxes, fw, fh)
            for za in zone_alerts:
                sev = "high" if za.alert_type == "unauthorized" else "warning"
                alerts.append(SafetyAlert(alert_type=za.alert_type, message=za.message, severity=sev))
                inc = self.incidents.report(
                    za.alert_type,
                    za.message,
                    severity=sev,
                    zone_id=za.zone_id,
                    dedupe_key=f"{za.alert_type}:{za.zone_id}",
                    details={"count": za.count},
                )
                if inc:
                    new_incidents.append(inc)

        self._draw_overlay(result.frame, dscale, fw, fh, objects, hazard_list)

        unattended_n = sum(
            1 for t in self.unattended._tracks.values() if t.unattended
        )
        bag_n = len(self.unattended._tracks)
        zone_v = sum(1 for a in alerts if a.alert_type in ("unauthorized", "overcrowding"))
        fire_n = sum(1 for h in hazard_list if h.hazard_type == "fire")
        smoke_n = sum(1 for h in hazard_list if h.hazard_type == "smoke")
        fighting = any(h.hazard_type == "fighting" for h in hazard_list)

        return SafetySnapshot(
            alerts=alerts,
            new_incidents=new_incidents,
            unattended_count=unattended_n,
            zone_violations=zone_v,
            hazards=hazard_list,
            fire_count=fire_n,
            smoke_count=smoke_n,
            fighting_active=fighting,
            bag_count=bag_n,
        )

    def _draw_overlay(
        self,
        display: np.ndarray,
        dscale: float,
        fw: int,
        fh: int,
        objects: list,
        hazards: list[DetectedHazard],
    ) -> None:
        if self.enable_zones:
            for zone in self.zones.zones:
                if not zone.enabled:
                    continue
                x1, y1, x2, y2 = self.zones.pixel_rect(zone, fw, fh)
                if dscale != 1.0:
                    x1, y1, x2, y2 = int(x1 * dscale), int(y1 * dscale), int(x2 * dscale), int(y2 * dscale)
                color = (0, 165, 255) if zone.zone_type == "restricted" else (0, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label = f"{zone.name}"
                if zone.zone_type == "crowd":
                    label += f" (max {zone.max_persons})"
                cv2.putText(
                    display, label, (x1 + 4, max(y1 + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )

        hazard_colors = {
            "fire": (0, 0, 255),
            "smoke": (180, 180, 180),
            "fighting": (0, 0, 255),
        }
        for hazard in hazards:
            x1, y1, x2, y2 = hazard.bbox
            if dscale != 1.0:
                x1, y1, x2, y2 = int(x1 * dscale), int(y1 * dscale), int(x2 * dscale), int(y2 * dscale)
            color = hazard_colors.get(hazard.hazard_type, (0, 0, 255))
            thickness = 3 if hazard.hazard_type in {"fire", "fighting"} else 2
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                display,
                f"{hazard.label.upper()} {hazard.confidence:.0%}",
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if self.enable_unattended:
            for track in self.unattended._tracks.values():
                x1, y1, x2, y2 = track.bbox
                if dscale != 1.0:
                    x1, y1, x2, y2 = int(x1 * dscale), int(y1 * dscale), int(x2 * dscale), int(y2 * dscale)
                if track.unattended:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        display, f"UNATTENDED {track.label} {track.dwell_sec:.0f}s",
                        (x1, min(y2 + 16, display.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
                    )
                elif track.label.lower() in {"backpack", "handbag", "suitcase"}:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 140, 255), 2)
                    cv2.putText(
                        display, f"BAG: {track.label}",
                        (x1, min(y2 + 14, display.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA,
                    )

        active_alerts: list[str] = []
        for hazard in hazards:
            active_alerts.append(hazard.message)
        if self.enable_unattended:
            for t in self.unattended._tracks.values():
                if t.unattended:
                    active_alerts.append(f"UNATTENDED {t.label}")
        if active_alerts:
            banner = " | ".join(active_alerts[:3])
            cv2.rectangle(display, (0, display.shape[0] - 28), (display.shape[1], display.shape[0]), (0, 0, 180), -1)
            cv2.putText(
                display, banner, (8, display.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
