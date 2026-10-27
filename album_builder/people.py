"""Les personnes à reconnaître : un dossier par personne dans refs/, rempli de visages recadrés.

On peut ajouter une photo de groupe : chaque visage détecté est proposé à l'écran, la personne
lui donne un prénom, et le visage recadré est rangé dans refs/<prénom>/.
"""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from .errors import AlbumError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def safe_name(name: str) -> str:
    """Prénom utilisable comme nom de dossier (Windows compris)."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name or "").strip().strip(".")
    name = re.sub(r"\s+", " ", name)[:40]
    if not name:
        raise AlbumError("Donne un prénom à cette personne.", code="bad_name")
    return name


def _photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def list_people(refs_dir: Path) -> list[dict]:
    if not refs_dir.exists():
        return []
    people = []
    for d in sorted((p for p in refs_dir.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        photos = _photos(d)
        people.append({"name": d.name, "photos": [p.name for p in photos]})
    return people


def find_person_dir(refs_dir: Path, name: str) -> Path:
    """Dossier existant de la personne (sans tenir compte des majuscules), ou nouveau dossier."""
    name = safe_name(name)
    if refs_dir.exists():
        for d in refs_dir.iterdir():
            if d.is_dir() and d.name.lower() == name.lower():
                return d
    return refs_dir / name


def save_face(refs_dir: Path, name: str, jpeg_bytes: bytes) -> Path:
    folder = find_person_dir(refs_dir, name)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex[:10]}.jpg"
    path.write_bytes(jpeg_bytes)
    return path


def remove_photo(refs_dir: Path, name: str, photo: str) -> None:
    folder = find_person_dir(refs_dir, name)
    target = (folder / Path(photo).name)
    if target.exists():
        target.unlink()
    if folder.exists() and not _photos(folder):
        shutil.rmtree(folder)


def remove_person(refs_dir: Path, name: str) -> None:
    folder = find_person_dir(refs_dir, name)
    if folder.exists():
        shutil.rmtree(folder)
