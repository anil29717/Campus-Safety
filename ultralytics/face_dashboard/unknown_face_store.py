"""Persistent storage for unknown visitors (ID + face photo)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass
class UnknownPerson:
    """A detected unknown visitor."""

    id: str
    first_seen: str
    last_seen: str
    image_path: str | None = None
    visit_count: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "image_path": self.image_path,
            "visit_count": self.visit_count,
        }


class UnknownFaceStore:
    """Save unknown person IDs and face photos when first detected."""

    def __init__(self, db_dir: Path) -> None:
        self.db_dir = db_dir
        self.images_dir = db_dir / "images"
        self.registry_file = db_dir / "unknown_registry.json"
        for path in (self.db_dir, self.images_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.persons: dict[str, UnknownPerson] = {}
        self.load()

    def load(self) -> None:
        self.persons.clear()
        if not self.registry_file.exists():
            return
        records = json.loads(self.registry_file.read_text(encoding="utf-8"))
        for rec in records:
            person = UnknownPerson(
                id=rec["id"],
                first_seen=rec.get("first_seen", ""),
                last_seen=rec.get("last_seen", ""),
                image_path=rec.get("image_path"),
                visit_count=int(rec.get("visit_count", 1)),
            )
            self.persons[person.id] = person

    def save(self) -> None:
        rows = sorted(self.persons.values(), key=lambda p: p.last_seen, reverse=True)
        payload = [p.to_dict() for p in rows]
        self.registry_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def has(self, person_id: str) -> bool:
        return person_id in self.persons

    def record(
        self,
        person_id: str,
        face_bgr: np.ndarray | None,
        *,
        new_visit: bool = False,
    ) -> UnknownPerson:
        """Register unknown person and save photo on first sighting."""
        now = _now_str()
        dirty = False

        if person_id not in self.persons:
            image_path = self._save_image(person_id, face_bgr)
            self.persons[person_id] = UnknownPerson(
                id=person_id,
                first_seen=now,
                last_seen=now,
                image_path=image_path,
                visit_count=1,
            )
            dirty = True
        else:
            person = self.persons[person_id]
            if new_visit:
                person.visit_count += 1
                person.last_seen = now
                dirty = True
            if not person.image_path and face_bgr is not None:
                person.image_path = self._save_image(person_id, face_bgr)
                dirty = True

        if dirty:
            self.save()
        return self.persons[person_id]

    def _save_image(self, person_id: str, face_bgr: np.ndarray | None) -> str | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        image_path = self.images_dir / f"{person_id}.jpg"
        cv2.imwrite(str(image_path), face_bgr)
        return str(image_path)

    def list_all(self) -> list[UnknownPerson]:
        return sorted(self.persons.values(), key=lambda p: p.last_seen, reverse=True)

    def get(self, person_id: str) -> UnknownPerson | None:
        return self.persons.get(person_id)

    def remove(self, person_id: str) -> bool:
        person = self.persons.pop(person_id, None)
        if person is None:
            return False
        if person.image_path and Path(person.image_path).exists():
            Path(person.image_path).unlink()
        self.save()
        return True

    def count(self) -> int:
        return len(self.persons)
