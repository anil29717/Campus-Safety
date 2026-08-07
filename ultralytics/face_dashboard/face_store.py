"""Persistent storage for registered face embeddings."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass
class KnownFace:
    """A registered person with face embedding."""

    id: str
    name: str
    feature: np.ndarray
    image_path: str | None = None
    created_at: str = ""

    @property
    def image_bgr(self) -> np.ndarray | None:
        if self.image_path and Path(self.image_path).exists():
            return cv2.imread(self.image_path)
        return None


class FaceStore:
    """Load/save known faces to disk."""

    def __init__(self, db_dir: Path) -> None:
        self.db_dir = db_dir
        self.features_dir = db_dir / "features"
        self.images_dir = db_dir / "images"
        self.meta_file = db_dir / "known_faces.json"
        for path in (self.db_dir, self.features_dir, self.images_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.faces: list[KnownFace] = []
        self.load()

    def load(self) -> None:
        self.faces.clear()
        if not self.meta_file.exists():
            return
        records = json.loads(self.meta_file.read_text(encoding="utf-8"))
        for rec in records:
            feat_path = self.features_dir / rec["feature_file"]
            if not feat_path.exists():
                continue
            feature = np.load(feat_path)
            self.faces.append(
                KnownFace(
                    id=rec["id"],
                    name=rec["name"],
                    feature=feature,
                    image_path=rec.get("image_path"),
                    created_at=rec.get("created_at", ""),
                )
            )

    def save(self) -> None:
        records = []
        for face in self.faces:
            records.append(
                {
                    "id": face.id,
                    "name": face.name,
                    "feature_file": f"{face.id}.npy",
                    "image_path": face.image_path,
                    "created_at": face.created_at,
                }
            )
        self.meta_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def add(self, name: str, feature: np.ndarray, face_bgr: np.ndarray | None = None) -> KnownFace:
        face_id = f"CSU-{uuid.uuid4().hex[:10].upper()}"
        np.save(self.features_dir / f"{face_id}.npy", feature)

        image_path = None
        if face_bgr is not None:
            image_path = str(self.images_dir / f"{name.replace(' ', '_')}_{face_id}.jpg")
            cv2.imwrite(image_path, face_bgr)

        known = KnownFace(
            id=face_id,
            name=name.strip(),
            feature=feature,
            image_path=image_path,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.faces.append(known)
        self.save()
        return known

    def remove(self, face_id: str) -> bool:
        for i, face in enumerate(self.faces):
            if face.id == face_id:
                feat_path = self.features_dir / f"{face.id}.npy"
                if feat_path.exists():
                    feat_path.unlink()
                if face.image_path and Path(face.image_path).exists():
                    Path(face.image_path).unlink()
                self.faces.pop(i)
                self.save()
                return True
        return False

    def names(self) -> list[str]:
        return [f.name for f in self.faces]

    def count(self) -> int:
        return len(self.faces)

    def get_by_name(self, name: str) -> KnownFace | None:
        for face in self.faces:
            if face.name == name:
                return face
        return None

    def get_by_id(self, person_id: str) -> KnownFace | None:
        for face in self.faces:
            if face.id == person_id:
                return face
        return None
