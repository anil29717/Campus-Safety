"""YOLO person detection service."""

from __future__ import annotations

import numpy as np

from face_dashboard.face_pipeline import _scale_boxes
from face_dashboard.services.base import FeatureResult, FeatureService, ServiceContext


class PersonService(FeatureService):
    feature_id = "person_detection"

    def process(self, ctx: ServiceContext) -> FeatureResult:
        result = FeatureResult(self.feature_id)
        if not self.enabled(ctx):
            return result

        pipeline = ctx.pipeline
        run_yolo = ctx.frame_idx == 1 or ctx.frame_idx % ctx.cfg.yolo_every == 0
        need_objects = ctx.features.is_enabled("object_detection")

        if run_yolo and (self.enabled(ctx) or need_objects):
            count, boxes, objects = pipeline._run_yolo(
                ctx.work_frame,
                ctx.cfg.conf,
                ctx.cfg.imgsz,
                detect_persons=True,
                detect_objects=need_objects,
            )
            if self.enabled(ctx):
                pipeline._last_person_count = count
                pipeline._last_person_boxes = boxes
            if need_objects:
                pipeline._last_objects = objects

        boxes = pipeline._last_person_boxes if self.enabled(ctx) else None
        if boxes is not None:
            boxes = _scale_boxes(boxes, ctx.inv_scale)
            ctx.person_count = len(boxes)
            ctx.person_boxes = [
                (int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes.astype(int)
            ]

        result.ran = True
        ctx.active_ids.add(self.feature_id)
        return result
