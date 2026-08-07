"""YuNet + SFace face recognition service."""

from __future__ import annotations

import cv2

from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class FaceService(FeatureService):
    feature_id = "face_recognition"
    dependencies = ()

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx):
            ctx.faces = []
            return result

        ctx.faces = ctx.pipeline.detect_faces(ctx.work_frame, ctx.inv_scale)
        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result

    @staticmethod
    def draw(ctx: ServiceContext) -> None:
        if ctx.display is None:
            return
        dscale = ctx.dscale
        for face in ctx.faces:
            x, y, fw, fh = face.bbox
            if dscale != 1.0:
                x, y, fw, fh = int(x * dscale), int(y * dscale), int(fw * dscale), int(fh * dscale)
            color = (0, 200, 0) if face.name != "Unknown" else (0, 0, 255)
            cv2.rectangle(ctx.display, (x, y), (x + fw, y + fh), color, 2)
            if face.name != "Unknown":
                label = face.name
                if face.person_id:
                    label += f" ({face.person_id})"
                if face.emotion:
                    label += f" | {face.emotion}"
                if face.behavior:
                    label += f" | {face.behavior}"
            else:
                label = "Unknown"
            cv2.putText(
                ctx.display, label, (x, max(y - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )
