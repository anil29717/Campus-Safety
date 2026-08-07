"""YOLO COCO object detection service."""

from __future__ import annotations

from face_dashboard.face_pipeline import DetectedObject
from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext
from face_dashboard.services.person_service import PersonService


class ObjectService(FeatureService):
    feature_id = "object_detection"

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx):
            ctx.objects = []
            return result

        pipeline = ctx.pipeline
        run_yolo = ctx.frame_idx == 1 or ctx.frame_idx % max(ctx.cfg.object_every, 1) == 0
        need_persons = ctx.features.is_enabled("person_detection")

        if run_yolo and (self.enabled(ctx) or need_persons):
            count, boxes, objects = pipeline._run_yolo(
                ctx.work_frame,
                ctx.cfg.conf,
                ctx.cfg.imgsz,
                detect_persons=need_persons,
                detect_objects=True,
            )
            if need_persons:
                pipeline._last_person_count = count
                pipeline._last_person_boxes = boxes
            pipeline._last_objects = objects

        objects = list(pipeline._last_objects)
        if ctx.inv_scale != 1.0:
            scaled: list[DetectedObject] = []
            for obj in objects:
                x1, y1, x2, y2 = obj.bbox
                scaled.append(
                    DetectedObject(
                        bbox=(
                            int(x1 * ctx.inv_scale),
                            int(y1 * ctx.inv_scale),
                            int(x2 * ctx.inv_scale),
                            int(y2 * ctx.inv_scale),
                        ),
                        label=obj.label,
                        class_id=obj.class_id,
                        confidence=obj.confidence,
                    )
                )
            objects = scaled
        ctx.objects = objects

        if need_persons and PersonService().enabled(ctx):
            PersonService().process(ctx)

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
