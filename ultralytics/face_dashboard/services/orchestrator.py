"""Orchestrates modular feature services per camera frame."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from face_dashboard.activity_log import ActivityLogger
from face_dashboard.config import ProcessConfig
from face_dashboard.entry_tracker import EntryTracker
from face_dashboard.face_pipeline import (
    FacePipeline,
    FrameResult,
    _resize_keep_aspect,
    _scale_boxes,
)
from face_dashboard.feature_config import FeatureConfig
from face_dashboard.safety_monitor import SafetyMonitor
from face_dashboard.services.bag_service import BagService
from face_dashboard.services.base import ServiceContext
from face_dashboard.services.behavior_service import BehaviorService
from face_dashboard.services.emotion_service import EmotionService
from face_dashboard.services.entry_service import EntryService
from face_dashboard.services.face_service import FaceService
from face_dashboard.services.fight_service import FightService
from face_dashboard.services.fire_smoke_service import FireSmokeService
from face_dashboard.services.object_service import ObjectService
from face_dashboard.services.person_service import PersonService
from face_dashboard.services.unknown_service import UnknownService
from face_dashboard.services.zones_service import ZonesService
from face_dashboard.unknown_face_store import UnknownFaceStore


@dataclass
class OrchestratorOutput:
    frame_result: FrameResult
    alerts: list[dict] = field(default_factory=list)
    hazards: list[dict] = field(default_factory=list)
    live_activity: list[dict] = field(default_factory=list)
    active_features: set[str] = field(default_factory=set)


class FeatureOrchestrator:
    """Run enabled feature services on each frame and merge results."""

    def __init__(
        self,
        pipeline: FacePipeline,
        features: FeatureConfig,
        entry_tracker: EntryTracker,
        unknown_store: UnknownFaceStore,
        activity_logger: ActivityLogger,
        safety: SafetyMonitor,
        process_config: ProcessConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.features = features
        self.entry_tracker = entry_tracker
        self.unknown_store = unknown_store
        self.activity_logger = activity_logger
        self.safety = safety
        self.cfg = process_config or ProcessConfig(
            max_width=480,
            display_width=720,
            imgsz=320,
            yolo_every=4,
            object_every=5,
            emotion_every=4,
            behavior_every=6,
            hazard_every=6,
        )
        self._frame_idx = 0
        self._bag_svc = BagService()
        self._fire_svc = FireSmokeService()
        self._fight_svc = FightService()
        self._zones_svc = ZonesService()
        self._face_svc = FaceService()
        self._emotion_svc = EmotionService()
        self._behavior_svc = BehaviorService()
        self._entry_svc = EntryService()
        self._unknown_svc = UnknownService()
        self._person_svc = PersonService()
        self._object_svc = ObjectService()
        self.camera_id: str | None = None
        self.camera_name: str | None = None
        self.camera_role: str = "none"

    def set_camera_context(
        self,
        camera_id: str | None = None,
        camera_name: str | None = None,
        camera_role: str = "none",
    ) -> None:
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.camera_role = (camera_role or "none").lower()

    def clear_camera_context(self) -> None:
        self.camera_id = None
        self.camera_name = None
        self.camera_role = "none"

    def reset_cache(self) -> None:
        self._frame_idx = 0
        self.pipeline.reset_cache()

    def _run_shared_yolo(self, ctx: ServiceContext) -> None:
        need_person = ctx.features.is_enabled("person_detection")
        need_object = ctx.features.is_enabled("object_detection")
        if not need_person and not need_object:
            self.pipeline._last_person_count = 0
            self.pipeline._last_person_boxes = None
            self.pipeline._last_objects = []
            return

        interval = ctx.cfg.yolo_every
        if need_object and not need_person:
            interval = ctx.cfg.object_every
        run = ctx.frame_idx == 1 or ctx.frame_idx % max(interval, 1) == 0
        if not run:
            if need_person and self.pipeline._last_person_boxes is not None:
                boxes = _scale_boxes(self.pipeline._last_person_boxes, ctx.inv_scale)
                ctx.person_count = len(boxes)
                ctx.person_boxes = [
                    (int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes.astype(int)
                ]
            if need_object:
                ctx.objects = self._scale_objects(list(self.pipeline._last_objects), ctx.inv_scale)
            return

        count, boxes, objects = self.pipeline._run_yolo(
            ctx.work_frame,
            ctx.cfg.conf,
            ctx.cfg.imgsz,
            detect_persons=need_person,
            detect_objects=need_object,
        )
        if need_person:
            self.pipeline._last_person_count = count
            self.pipeline._last_person_boxes = boxes
            if boxes is not None:
                boxes = _scale_boxes(boxes, ctx.inv_scale)
                ctx.person_count = len(boxes)
                ctx.person_boxes = [
                    (int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes.astype(int)
                ]
        if need_object:
            self.pipeline._last_objects = objects
            ctx.objects = self._scale_objects(objects, ctx.inv_scale)

        if need_person:
            ctx.active_ids.add("person_detection")
        if need_object:
            ctx.active_ids.add("object_detection")

    @staticmethod
    def _scale_objects(objects: list, inv_scale: float) -> list:
        if inv_scale == 1.0:
            return list(objects)
        from face_dashboard.face_pipeline import DetectedObject

        scaled = []
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            scaled.append(
                DetectedObject(
                    bbox=(
                        int(x1 * inv_scale),
                        int(y1 * inv_scale),
                        int(x2 * inv_scale),
                        int(y2 * inv_scale),
                    ),
                    label=obj.label,
                    class_id=obj.class_id,
                    confidence=obj.confidence,
                )
            )
        return scaled

    def _draw_overlays(self, ctx: ServiceContext) -> None:
        display = ctx.display
        if display is None:
            return
        dscale = ctx.dscale

        if ctx.features.is_enabled("person_detection"):
            for x1, y1, x2, y2 in ctx.person_boxes:
                if dscale != 1.0:
                    x1, y1, x2, y2 = int(x1 * dscale), int(y1 * dscale), int(x2 * dscale), int(y2 * dscale)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 128, 0), 2)
                cv2.putText(
                    display, "person", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2, cv2.LINE_AA,
                )

        if ctx.features.is_enabled("object_detection"):
            for obj in ctx.objects:
                x1, y1, x2, y2 = obj.bbox
                if dscale != 1.0:
                    x1, y1, x2, y2 = int(x1 * dscale), int(y1 * dscale), int(x2 * dscale), int(y2 * dscale)
                color = (255, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display, f"{obj.label} {obj.confidence:.0%}", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA,
                )

        if ctx.features.is_enabled("face_recognition"):
            FaceService.draw(ctx)

        hazard_colors = {"fire": (0, 0, 255), "smoke": (180, 180, 180), "fighting": (0, 0, 255)}
        for h in getattr(ctx, "_hazards_draw", []):
            x1, y1, x2, y2 = h["bbox"]
            if dscale != 1.0:
                x1, y1, x2, y2 = int(x1 * dscale), int(y1 * dscale), int(x2 * dscale), int(y2 * dscale)
            color = hazard_colors.get(h["type"], (0, 0, 255))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                display, f"{h['type'].upper()}", (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        if ctx.features.is_enabled("bag_unattended"):
            fh, fw = ctx.frame.shape[:2]
            dscale_s = display.shape[1] / fw if fw else 1.0
            for track in self.safety.unattended._tracks.values():
                x1, y1, x2, y2 = track.bbox
                if dscale_s != 1.0:
                    x1, y1, x2, y2 = int(x1 * dscale_s), int(y1 * dscale_s), int(x2 * dscale_s), int(y2 * dscale_s)
                if track.unattended:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        display, f"UNATTENDED {track.dwell_sec:.0f}s",
                        (x1, min(y2 + 16, display.shape[0] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA,
                    )
                elif track.label.lower() in {"backpack", "handbag", "suitcase"}:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 140, 255), 2)

        if ctx.features.is_enabled("safety_zones"):
            fh, fw = ctx.frame.shape[:2]
            dscale_s = display.shape[1] / fw if fw else 1.0
            for zone in self.safety.zones.zones:
                if not zone.enabled:
                    continue
                x1, y1, x2, y2 = self.safety.zones.pixel_rect(zone, fw, fh)
                if dscale_s != 1.0:
                    x1, y1, x2, y2 = int(x1 * dscale_s), int(y1 * dscale_s), int(x2 * dscale_s), int(y2 * dscale_s)
                color = (0, 165, 255) if zone.zone_type == "restricted" else (0, 255, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display, zone.name, (x1 + 4, max(y1 + 18, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )

        known = sum(1 for f in ctx.faces if f.name != "Unknown")
        unknown = len(ctx.faces) - known
        cv2.putText(
            display,
            f"Persons: {ctx.person_count}  Faces: {len(ctx.faces)}  Known: {known}  "
            f"Unknown: {unknown}  Objects: {len(ctx.objects)}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def process_frame(self, frame: np.ndarray) -> OrchestratorOutput:
        t0 = time.perf_counter()
        self._frame_idx += 1
        self.pipeline._frame_idx = self._frame_idx

        work_frame, scale = _resize_keep_aspect(frame, self.cfg.max_width)
        inv_scale = 1.0 / scale
        display, dscale = _resize_keep_aspect(frame, self.cfg.display_width)

        ctx = ServiceContext(
            frame=frame,
            work_frame=work_frame,
            inv_scale=inv_scale,
            dscale=dscale,
            cfg=self.cfg,
            features=self.features,
            pipeline=self.pipeline,
            frame_idx=self._frame_idx,
            entry_tracker=self.entry_tracker,
            unknown_store=self.unknown_store,
            activity_logger=self.activity_logger,
            safety=self.safety,
            display=display,
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            camera_role=self.camera_role,
        )

        self.safety.enable_unattended = self.features.is_enabled("bag_unattended")
        # Restricted cameras always evaluate safety zones when the feature exists
        self.safety.enable_zones = self.features.is_enabled("safety_zones") or (
            self.camera_role == "restricted"
        )
        self.safety.enable_fire_smoke = False
        self.safety.enable_fight = False

        self._run_shared_yolo(ctx)

        all_alerts: list[dict] = []
        all_hazards: list[dict] = []

        if self.features.is_enabled("face_recognition"):
            self._face_svc.process(ctx)

        if self.features.is_enabled("emotion"):
            self._emotion_svc.process(ctx)
        if self.features.is_enabled("behavior"):
            self._behavior_svc.process(ctx)

        if self.features.is_enabled("entry_tracking"):
            self._entry_svc.process(ctx)
        if self.features.is_enabled("unknown_visitors"):
            self._unknown_svc.process(ctx)

        activity_snap = None
        if self.features.is_enabled("emotion") or self.features.is_enabled("behavior"):
            activity_snap = self.activity_logger.update(ctx.faces)

        for svc in (self._bag_svc, self._fire_svc, self._fight_svc, self._zones_svc):
            if svc.enabled(ctx) and (not svc.dependencies or svc.deps_ok(ctx)):
                fr = svc.process(ctx)
                all_alerts.extend(fr.alerts)
                all_hazards.extend(fr.hazards)

        ctx._hazards_draw = all_hazards
        self._draw_overlays(ctx)

        elapsed = time.perf_counter() - t0
        fps = 1.0 / max(elapsed, 1e-6)

        known_count = sum(1 for f in ctx.faces if f.name != "Unknown")
        frame_result = FrameResult(
            frame=display,
            person_count=ctx.person_count,
            face_count=len(ctx.faces),
            known_count=known_count,
            unknown_count=len(ctx.faces) - known_count,
            faces=ctx.faces,
            objects=ctx.objects,
            object_count=len(ctx.objects),
            person_boxes=ctx.person_boxes,
            fps=fps,
        )

        self.features.set_active(ctx.active_ids)

        live_activity: list[dict] = []
        if activity_snap:
            live_activity = [
                {
                    "name": a.name,
                    "emotion": a.emotion,
                    "behavior": a.behavior,
                    "emotion_conf": round(a.emotion_conf, 2),
                    "behavior_conf": round(a.behavior_conf, 2),
                }
                for a in activity_snap.live
            ]

        return OrchestratorOutput(
            frame_result=frame_result,
            alerts=all_alerts,
            hazards=all_hazards,
            live_activity=live_activity,
            active_features=set(ctx.active_ids),
        )

    def get_feature_list(self, camera_running: bool) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.features.list_features(camera_running)]
