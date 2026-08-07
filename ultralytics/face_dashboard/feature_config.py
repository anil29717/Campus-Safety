"""Persist and manage per-feature enable flags for the orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from face_dashboard.services.registry import FEATURE_REGISTRY, FeatureMeta

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "data" / "config" / "features.json"

# When disabling a feature, also disable dependents.
DISABLE_CASCADE: dict[str, list[str]] = {
    "face_recognition": ["unknown_visitors", "emotion", "behavior", "entry_tracking"],
    "object_detection": ["bag_unattended"],
    "person_detection": ["fighting"],
}

# When enabling a feature, auto-enable required dependencies.
ENABLE_REQUIRES: dict[str, list[str]] = {
    "unknown_visitors": ["face_recognition"],
    "emotion": ["face_recognition"],
    "behavior": ["face_recognition"],
    "entry_tracking": ["face_recognition"],
    "bag_unattended": ["object_detection"],
    "fighting": ["person_detection"],
    "safety_zones": ["face_recognition", "person_detection"],
}


@dataclass
class FeatureState:
    id: str
    name: str
    description: str
    load: str
    dependencies: list[str]
    enabled: bool = True
    active: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "load": self.load,
            "dependencies": self.dependencies,
            "enabled": self.enabled,
            "active": self.active,
        }


class FeatureConfig:
    """Load/save feature toggles and apply dependency rules."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_CONFIG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._enabled: dict[str, bool] = {
            meta.id: True for meta in FEATURE_REGISTRY
        }
        self._active: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._save()
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        saved = data.get("enabled", {})
        for meta in FEATURE_REGISTRY:
            if meta.id in saved:
                self._enabled[meta.id] = bool(saved[meta.id])

    def _save(self) -> None:
        payload = {"enabled": self._enabled}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_enabled(self, feature_id: str) -> bool:
        return self._enabled.get(feature_id, False)

    def set_active(self, feature_ids: set[str]) -> None:
        self._active = set(feature_ids)

    def set_enabled(self, feature_id: str, enabled: bool) -> list[str]:
        """Set one feature; apply cascade rules. Returns list of changed ids."""
        changed: list[str] = []
        if feature_id not in self._enabled:
            return changed

        if enabled:
            for dep in ENABLE_REQUIRES.get(feature_id, []):
                if not self._enabled.get(dep, False):
                    self._enabled[dep] = True
                    changed.append(dep)
            if not self._enabled.get(feature_id, False):
                self._enabled[feature_id] = True
                changed.append(feature_id)
        else:
            if self._enabled.get(feature_id, False):
                self._enabled[feature_id] = False
                changed.append(feature_id)
            for dep in DISABLE_CASCADE.get(feature_id, []):
                if self._enabled.get(dep, False):
                    self._enabled[dep] = False
                    changed.append(dep)

        if changed:
            self._save()
        return changed

    def update_bulk(self, updates: dict[str, bool]) -> list[str]:
        all_changed: list[str] = []
        for fid, val in updates.items():
            all_changed.extend(self.set_enabled(fid, val))
        return list(dict.fromkeys(all_changed))

    def list_features(self, camera_running: bool = False) -> list[FeatureState]:
        rows: list[FeatureState] = []
        for meta in FEATURE_REGISTRY:
            enabled = self.is_enabled(meta.id)
            active = enabled and camera_running and meta.id in self._active
            rows.append(
                FeatureState(
                    id=meta.id,
                    name=meta.name,
                    description=meta.description,
                    load=meta.load,
                    dependencies=list(meta.dependencies),
                    enabled=enabled,
                    active=active,
                )
            )
        return rows

    def to_process_flags(self) -> dict[str, bool]:
        """Map feature ids to legacy ProcessConfig enable_* keys."""
        return {
            "detect_persons": self.is_enabled("person_detection"),
            "detect_faces": self.is_enabled("face_recognition"),
            "detect_objects": self.is_enabled("object_detection"),
            "enable_emotion": self.is_enabled("emotion"),
            "enable_behavior": self.is_enabled("behavior"),
            "enable_bag": self.is_enabled("bag_unattended"),
            "enable_fire_smoke": self.is_enabled("fire_smoke"),
            "enable_fight": self.is_enabled("fighting"),
            "enable_zones": self.is_enabled("safety_zones"),
            "enable_entry": self.is_enabled("entry_tracking"),
            "enable_unknown": self.is_enabled("unknown_visitors"),
        }
