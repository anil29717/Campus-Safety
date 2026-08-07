"""Shared types for modular feature services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from face_dashboard.config import ProcessConfig
from face_dashboard.face_pipeline import DetectedFace, DetectedObject, FacePipeline

if TYPE_CHECKING:
    from face_dashboard.activity_log import ActivityLogger
    from face_dashboard.entry_tracker import EntryTracker
    from face_dashboard.feature_config import FeatureConfig
    from face_dashboard.safety_monitor import SafetyMonitor
    from face_dashboard.unknown_face_store import UnknownFaceStore


@dataclass
class FeatureResult:
    feature_id: str
    ran: bool = False
    alerts: list[dict] = field(default_factory=list)
    hazards: list[dict] = field(default_factory=list)


@dataclass
class ServiceContext:
    """Mutable per-frame context passed to each feature service."""

    frame: np.ndarray
    work_frame: np.ndarray
    inv_scale: float
    dscale: float
    cfg: ProcessConfig
    features: FeatureConfig
    pipeline: FacePipeline
    frame_idx: int

    person_count: int = 0
    person_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)
    faces: list[DetectedFace] = field(default_factory=list)

    entry_tracker: EntryTracker | None = None
    unknown_store: UnknownFaceStore | None = None
    activity_logger: ActivityLogger | None = None
    safety: SafetyMonitor | None = None

    display: np.ndarray | None = None
    safety_snap: Any = None
    activity_snap: Any = None
    entry_stats: Any = None

    camera_id: str | None = None
    camera_name: str | None = None
    camera_role: str = "none"  # none | entry | exit | restricted

    active_ids: set[str] = field(default_factory=set)


class FeatureService:
    """Base class for a single detection feature."""

    feature_id: str = ""
    dependencies: tuple[str, ...] = ()

    def enabled(self, ctx: ServiceContext) -> bool:
        return ctx.features.is_enabled(self.feature_id)

    def deps_ok(self, ctx: ServiceContext) -> bool:
        return all(ctx.features.is_enabled(d) for d in self.dependencies)

    def process(self, ctx: ServiceContext) -> FeatureResult:
        raise NotImplementedError
