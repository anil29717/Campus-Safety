"""Entry in/out tracking service."""

from __future__ import annotations

from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class EntryService(FeatureService):
    feature_id = "entry_tracking"
    dependencies = ("face_recognition",)

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx) or not self.deps_ok(ctx) or ctx.entry_tracker is None:
            return result

        if ctx.features.is_enabled("unknown_visitors") and ctx.unknown_store:
            ctx.entry_tracker._unknown_store = ctx.unknown_store
        else:
            ctx.entry_tracker._unknown_store = None

        ctx.entry_stats = ctx.entry_tracker.update(
            ctx.faces,
            mode=ctx.camera_role or "general",
            camera_id=ctx.camera_id,
            camera_name=ctx.camera_name,
        )
        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
