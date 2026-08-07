"""Pose-based fighting detection service."""

from __future__ import annotations

from face_dashboard.hazard_detector import FightDetector
from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class FightService(FeatureService):
    feature_id = "fighting"
    dependencies = ("person_detection",)

    def __init__(self) -> None:
        self._detector: FightDetector | None = None
        self._frame_idx = 0

    def _get(self) -> FightDetector:
        if self._detector is None:
            self._detector = FightDetector()
        return self._detector

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx) or not self.deps_ok(ctx):
            return result

        self._frame_idx += 1
        every = max(ctx.cfg.hazard_every, 1)
        if self._frame_idx != 1 and self._frame_idx % every != 0:
            ctx.active_ids.add(self.feature_id)
            result.ran = True
            return result

        fight = self._get().detect(ctx.frame, ctx.person_boxes, imgsz=ctx.cfg.imgsz)
        if fight:
            result.hazards.append(
                {
                    "type": fight.hazard_type,
                    "label": fight.label,
                    "confidence": fight.confidence,
                    "bbox": list(fight.bbox),
                    "message": fight.message,
                }
            )
            if ctx.safety:
                ctx.safety.incidents.report(
                    "fighting",
                    fight.message,
                    severity="high",
                    dedupe_key="fighting:live",
                    details={"confidence": fight.confidence},
                )
            result.alerts.append(
                {"type": "fighting", "message": fight.message, "severity": "high"}
            )

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
