"""Emotion detection for registered users."""

from __future__ import annotations

from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class EmotionService(FeatureService):
    feature_id = "emotion"
    dependencies = ("face_recognition",)

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx) or not self.deps_ok(ctx):
            return result

        pipeline = ctx.pipeline
        run = ctx.frame_idx == 1 or ctx.frame_idx % max(ctx.cfg.emotion_every, 1) == 0
        for face in ctx.faces:
            if face.name == "Unknown" or not face.person_id or face.aligned is None:
                continue
            key = face.person_id
            if run:
                try:
                    emo, conf = pipeline._get_emotion().predict(face.aligned, person_key=key)
                    pipeline._last_emotion[key] = (emo, conf)
                except Exception:
                    pass
            if key in pipeline._last_emotion:
                face.emotion, face.emotion_conf = pipeline._last_emotion[key]

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
