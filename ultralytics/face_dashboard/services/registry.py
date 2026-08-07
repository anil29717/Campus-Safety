"""Static metadata for all detection features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureMeta:
    id: str
    name: str
    description: str
    load: str  # Low | Medium | High
    dependencies: tuple[str, ...] = ()


FEATURE_REGISTRY: list[FeatureMeta] = [
    FeatureMeta(
        "person_detection",
        "Person detection",
        "YOLO person bounding boxes",
        "Medium",
    ),
    FeatureMeta(
        "face_recognition",
        "Face recognition",
        "YuNet + SFace known/unknown faces",
        "Medium",
    ),
    FeatureMeta(
        "unknown_visitors",
        "Unknown visitors",
        "Save unknown ID and face photo",
        "Low",
        ("face_recognition",),
    ),
    FeatureMeta(
        "object_detection",
        "Object detection",
        "YOLO COCO objects",
        "Medium",
    ),
    FeatureMeta(
        "bag_unattended",
        "Bag / unattended",
        "Backpack, handbag, suitcase alerts",
        "Medium",
        ("object_detection",),
    ),
    FeatureMeta(
        "fire_smoke",
        "Fire & smoke",
        "Dedicated fire/smoke YOLO model",
        "High",
    ),
    FeatureMeta(
        "fighting",
        "Fighting",
        "Pose-based fight detection",
        "High",
        ("person_detection",),
    ),
    FeatureMeta(
        "emotion",
        "Emotion",
        "FER model for registered users",
        "Medium",
        ("face_recognition",),
    ),
    FeatureMeta(
        "behavior",
        "Behavior",
        "Pose sitting/standing/walking",
        "High",
        ("face_recognition",),
    ),
    FeatureMeta(
        "safety_zones",
        "Safety zones",
        "Restricted area and overcrowding",
        "Low",
        ("face_recognition", "person_detection"),
    ),
    FeatureMeta(
        "entry_tracking",
        "Entry tracking",
        "In/out times per person",
        "Low",
        ("face_recognition",),
    ),
]

FEATURE_IDS = {m.id for m in FEATURE_REGISTRY}
