"""Track bag-like objects and alert when left unattended."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

SUSPICIOUS_LABELS = frozenset({"backpack", "handbag", "suitcase"})
IOU_MATCH_THRESH = 0.3
STATIONARY_MOVE_PX = 35


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _person_near_object(
    obj_box: tuple[int, int, int, int],
    person_boxes: list[tuple[int, int, int, int]],
    margin_px: float,
) -> bool:
    if not person_boxes:
        return False
    oc = _center(obj_box)
    ox1, oy1, ox2, oy2 = obj_box
    expanded = (ox1 - margin_px, oy1 - margin_px, ox2 + margin_px, oy2 + margin_px)
    for pbox in person_boxes:
        if _iou(expanded, pbox) > 0.01:
            return True
        if _dist(oc, _center(pbox)) <= margin_px * 1.5:
            return True
    return False


@dataclass
class TrackedBag:
    track_id: int
    label: str
    bbox: tuple[int, int, int, int]
    first_seen: float
    stationary_since: float
    last_center: tuple[float, float]
    unattended: bool = False
    dwell_sec: float = 0.0


@dataclass
class UnattendedAlert:
    track_id: int
    label: str
    message: str
    dwell_sec: float
    bbox: tuple[int, int, int, int]


@dataclass
class UnattendedSnapshot:
    tracks: list[TrackedBag] = field(default_factory=list)
    new_alerts: list[UnattendedAlert] = field(default_factory=list)


class UnattendedObjectTracker:
    """IoU tracker with dwell-time alerts for bag-like COCO objects."""

    def __init__(
        self,
        dwell_sec: float = 30.0,
        person_margin_px: float = 100.0,
    ) -> None:
        self.dwell_sec = dwell_sec
        self.person_margin_px = person_margin_px
        self._tracks: dict[int, TrackedBag] = {}
        self._next_id = 1
        self._alerted: set[int] = set()

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._alerted.clear()

    def update(
        self,
        objects: list,
        person_boxes: list[tuple[int, int, int, int]],
    ) -> UnattendedSnapshot:
        now = time.time()
        bag_objects = [o for o in objects if getattr(o, "label", "").lower() in SUSPICIOUS_LABELS]
        matched: set[int] = set()
        new_alerts: list[UnattendedAlert] = []

        for obj in bag_objects:
            bbox = obj.bbox
            best_id: int | None = None
            best_iou = IOU_MATCH_THRESH
            for tid, track in self._tracks.items():
                if tid in matched:
                    continue
                iou = _iou(bbox, track.bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_id = tid

            center = _center(bbox)
            person_near = _person_near_object(bbox, person_boxes, self.person_margin_px)

            if best_id is not None:
                track = self._tracks[best_id]
                matched.add(best_id)
                if _dist(center, track.last_center) > STATIONARY_MOVE_PX:
                    track.stationary_since = now
                track.bbox = bbox
                track.last_center = center
                if person_near:
                    track.stationary_since = now
                    track.unattended = False
                    self._alerted.discard(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = TrackedBag(
                    track_id=tid,
                    label=obj.label,
                    bbox=bbox,
                    first_seen=now,
                    stationary_since=now,
                    last_center=center,
                )
                matched.add(tid)

        stale = [tid for tid in self._tracks if tid not in matched]
        for tid in stale:
            del self._tracks[tid]
            self._alerted.discard(tid)

        for track in self._tracks.values():
            person_near = _person_near_object(track.bbox, person_boxes, self.person_margin_px)
            dwell = now - track.stationary_since
            track.dwell_sec = dwell
            if person_near:
                track.unattended = False
                continue
            if dwell >= self.dwell_sec:
                track.unattended = True
                if track.track_id not in self._alerted:
                    self._alerted.add(track.track_id)
                    msg = f"Unattended {track.label} for {dwell:.0f}s"
                    new_alerts.append(
                        UnattendedAlert(
                            track_id=track.track_id,
                            label=track.label,
                            message=msg,
                            dwell_sec=dwell,
                            bbox=track.bbox,
                        )
                    )

        return UnattendedSnapshot(tracks=list(self._tracks.values()), new_alerts=new_alerts)
