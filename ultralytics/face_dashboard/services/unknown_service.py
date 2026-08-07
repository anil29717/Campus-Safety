"""Unknown visitor photo capture when entry tracking is off."""

from __future__ import annotations

from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class UnknownService(FeatureService):
    feature_id = "unknown_visitors"
    dependencies = ("face_recognition",)

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx) or not self.deps_ok(ctx):
            return result
        if ctx.features.is_enabled("entry_tracking"):
            ctx.active_ids.add(self.feature_id)
            result.ran = True
            return result

        if ctx.unknown_store is None or ctx.entry_tracker is None:
            return result

        for face in ctx.faces:
            if face.name != "Unknown":
                continue
            feature = face.feature
            if feature is None:
                continue
            key, _, entry_type, person_id = ctx.entry_tracker._identity(
                face.name, feature, face.person_id
            )
            if entry_type == "unknown":
                ctx.unknown_store.record(
                    person_id,
                    face.aligned,
                    new_visit=True,
                )

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
