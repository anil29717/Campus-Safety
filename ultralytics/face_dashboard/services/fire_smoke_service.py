"""Fire and smoke detection service."""

from __future__ import annotations

from face_dashboard.hazard_detector import FireSmokeDetector
from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class FireSmokeService(FeatureService):
    feature_id = "fire_smoke"

    def __init__(self) -> None:
        self._detector: FireSmokeDetector | None = None
        self._frame_idx = 0
        self._seen: set[str] = set()

    def _get(self) -> FireSmokeDetector:
        if self._detector is None:
            self._detector = FireSmokeDetector()
        return self._detector

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx):
            return result

        self._frame_idx += 1
        every = max(ctx.cfg.hazard_every, 1)
        if self._frame_idx != 1 and self._frame_idx % every != 0:
            ctx.active_ids.add(self.feature_id)
            result.ran = True
            return result

        hazards = self._get().detect(ctx.frame, imgsz=ctx.cfg.imgsz)
        for h in hazards:
            result.hazards.append(
                {
                    "type": h.hazard_type,
                    "label": h.label,
                    "confidence": h.confidence,
                    "bbox": list(h.bbox),
                    "message": h.message,
                }
            )
            key = f"{h.hazard_type}:{h.bbox[0]}:{h.bbox[1]}"
            if key not in self._seen and ctx.safety:
                self._seen.add(key)
                sev = "high" if h.hazard_type == "fire" else "warning"
                ctx.safety.incidents.report(
                    h.hazard_type,
                    h.message,
                    severity=sev,
                    dedupe_key=f"{h.hazard_type}:live",
                    details={"confidence": h.confidence, "bbox": list(h.bbox)},
                )
                result.alerts.append(
                    {"type": h.hazard_type, "message": h.message, "severity": sev}
                )

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
