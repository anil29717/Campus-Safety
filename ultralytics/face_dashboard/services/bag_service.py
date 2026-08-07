"""Unattended bag tracking service."""

from __future__ import annotations

from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class BagService(FeatureService):
    feature_id = "bag_unattended"
    dependencies = ("object_detection",)

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx) or not self.deps_ok(ctx) or ctx.safety is None:
            return result

        ctx.safety.enable_unattended = True
        snap = ctx.safety.unattended.update(ctx.objects, ctx.person_boxes)
        for alert in snap.new_alerts:
            inc = ctx.safety.incidents.report(
                "unattended_bag",
                alert.message,
                severity="high",
                dedupe_key=f"unattended:{alert.track_id}",
                details={"label": alert.label, "dwell_sec": alert.dwell_sec},
            )
            if inc:
                result.alerts.append(
                    {"type": "unattended_bag", "message": alert.message, "severity": "high"}
                )

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
