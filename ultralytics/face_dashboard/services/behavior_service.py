"""Pose-based behavior detection for registered users."""

from __future__ import annotations

from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class BehaviorService(FeatureService):
    feature_id = "behavior"
    dependencies = ("face_recognition",)

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx) or not self.deps_ok(ctx):
            return result

        pipeline = ctx.pipeline
        run = ctx.frame_idx == 1 or ctx.frame_idx % max(ctx.cfg.behavior_every, 1) == 0
        for face in ctx.faces:
            if face.name == "Unknown" or not face.person_id:
                continue
            key = face.person_id
            if run:
                try:
                    beh, conf = pipeline._get_behavior().predict(ctx.frame, face.bbox, key)
                    pipeline._last_behavior[key] = (beh, conf)
                except Exception:
                    pass
            if key in pipeline._last_behavior:
                face.behavior, face.behavior_conf = pipeline._last_behavior[key]

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
