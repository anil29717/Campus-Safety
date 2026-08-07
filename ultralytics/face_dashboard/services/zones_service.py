"""Restricted zone and overcrowding service."""

from __future__ import annotations

from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class ZonesService(FeatureService):
    feature_id = "safety_zones"
    dependencies = ("face_recognition", "person_detection")

    def enabled(self, ctx: ServiceContext) -> bool:
        if ctx.camera_role == "restricted":
            return True
        return super().enabled(ctx)

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx) or not self.deps_ok(ctx) or ctx.safety is None:
            return result

        fh, fw = ctx.frame.shape[:2]
        ctx.safety.enable_zones = True
        zone_alerts = ctx.safety.zones.evaluate(ctx.faces, ctx.person_boxes, fw, fh)
        for za in zone_alerts:
            sev = "high" if za.alert_type == "unauthorized" else "warning"
            ctx.safety.incidents.report(
                za.alert_type,
                za.message,
                severity=sev,
                zone_id=za.zone_id,
                dedupe_key=f"{za.alert_type}:{za.zone_id}",
                details={"count": za.count},
            )
            result.alerts.append(
                {"type": za.alert_type, "message": za.message, "severity": sev}
            )

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
