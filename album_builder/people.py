"""Les personnes à reconnaître : un dossier par personne dans refs/, rempli de visages recadrés.

On peut ajouter une photo de groupe : chaque visage détecté est proposé à l'écran, la personne
lui donne un prénom, et le visage recadré est rangé dans refs/<prénom>/.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import date
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
        birth = _info(d).get("birth")
        people.append({"name": d.name, "photos": [p.name for p in photos], "birth": birth})
    return people


def _info(folder: Path) -> dict:
    try:
        return json.loads((folder / "person.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def set_birth(refs_dir: Path, name: str, birth: str | None) -> None:
    """Date de naissance (AAAA-MM-JJ) ou None pour l'effacer."""
    folder = find_person_dir(refs_dir, name)
    if not folder.exists():
        raise AlbumError(f"Je ne connais pas encore {name}.", code="unknown_person")
    if birth:
        try:
            date.fromisoformat(birth)
        except ValueError:
            raise AlbumError("Date de naissance non comprise (format attendu : AAAA-MM-JJ).", code="bad_date")
    info = {**_info(folder), "birth": birth or None}
    (folder / "person.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


def births(refs_dir: Path) -> dict[str, date]:
    out = {}
    for p in list_people(refs_dir):
        if p["birth"]:
            out[p["name"]] = date.fromisoformat(p["birth"])
    return out


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
    if folder.exists() and not _photos(folder) and not _info(folder).get("birth"):
        shutil.rmtree(folder)


def remove_person(refs_dir: Path, name: str) -> None:
    folder = find_person_dir(refs_dir, name)
    if folder.exists():
        shutil.rmtree(folder)
