"""Photos choisies dans Google Photos avec le sélecteur officiel de Google.

Depuis mars 2025, une application ne peut plus parcourir une bibliothèque Google Photos : la
personne choisit elle-même ses photos (2 000 au plus à chaque fois) dans une fenêtre de Google.
Albumia récupère alors seulement une miniature de chaque photo (les adresses de Google ne
restent valables qu'une heure), puis les traite comme des photos de Drive.

Les photos choisies s'accumulent d'une sélection à l'autre dans .cache/gphotos.json.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from .drive import DriveFile, _parse_rfc3339
from .errors import AlbumError

SCOPES = ["https://www.googleapis.com/auth/photospicker.mediaitems.readonly"]
API = "https://photospicker.googleapis.com/v1"
SOURCE = "gphotos"   # remplace le lien de dossier Drive quand les photos viennent de Google Photos
PREFIX = "gp_"       # les photos de Google Photos ont un identifiant à part
MAX_PICK = 2000


def store_path(workdir: Path) -> Path:
    return workdir / ".cache" / "gphotos.json"


def load_store(workdir: Path) -> dict[str, dict]:
    try:
        return json.loads(store_path(workdir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_store(workdir: Path, store: dict) -> None:
    path = store_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(store), encoding="utf-8")
    tmp.replace(path)


def summary(workdir: Path) -> dict:
    store = load_store(workdir)
    whens = sorted(v["when"] for v in store.values() if v.get("when"))
    return {"count": len(store), "from": whens[0][:10] if whens else None, "to": whens[-1][:10] if whens else None}


class PickerClient:
    """Sessions du sélecteur Google Photos (API Picker)."""

    def __init__(self, creds):
        from google.auth.transport.requests import AuthorizedSession

        self.http = AuthorizedSession(creds)

    def _call(self, method: str, url: str, **kw) -> dict:
        for attempt in range(4):
            try:
                resp = self.http.request(method, url, timeout=30, **kw)
            except OSError:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 403 and "SERVICE_DISABLED" in resp.text:
                raise AlbumError(
                    "Le sélecteur Google Photos n'est pas encore activé dans ton projet Google. "
                    "Active « Google Photos Picker API » dans la console Google Cloud, puis réessaie.",
                    code="picker_disabled",
                )
            if resp.status_code >= 400:
                raise AlbumError(f"Google Photos a refusé la demande ({resp.status_code}).", code="gphotos")
            return resp.json() if resp.content else {}
        raise AlbumError("Google Photos ne répond pas. Vérifie la connexion Internet.", code="network")

    def create_session(self) -> dict:
        return self._call("POST", f"{API}/sessions", json={"pickingConfig": {"maxItemCount": str(MAX_PICK)}})

    def get_session(self, session_id: str) -> dict:
        return self._call("GET", f"{API}/sessions/{session_id}")

    def delete_session(self, session_id: str) -> None:
        try:
            self._call("DELETE", f"{API}/sessions/{session_id}")
        except AlbumError:
            pass  # elle expire toute seule

    def picked_items(self, session_id: str) -> Iterator[dict]:
        token = None
        while True:
            params = {"sessionId": session_id, "pageSize": 100}
            if token:
                params["pageToken"] = token
            page = self._call("GET", f"{API}/mediaItems", params=params)
            yield from page.get("mediaItems", [])
            token = page.get("nextPageToken")
            if not token:
                return

    def download(self, base_url: str, size: int = 1024) -> bytes | None:
        for attempt in range(3):
            try:
                resp = self.http.get(f"{base_url}=w{size}-h{size}", timeout=30)
            except OSError:
                time.sleep(1 + attempt)
                continue
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                return resp.content
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return None
        return None


def describe(item: dict) -> dict:
    """Ce qu'on garde d'une photo choisie (jamais l'original)."""
    media = item.get("mediaFile") or {}
    meta = media.get("mediaFileMetadata") or {}
    camera = " ".join(x for x in (meta.get("cameraMake"), meta.get("cameraModel")) if x)
    return {
        "name": media.get("filename") or item["id"],
        "mime": media.get("mimeType", ""),
        "when": item.get("createTime"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "camera": camera,
    }


def import_session(picker: PickerClient, session_id: str, workdir: Path, reporter=None, workers: int = 6) -> dict:
    """Range les photos choisies et télécharge leur miniature tout de suite (adresses valables 1 h)."""
    items = [i for i in picker.picked_items(session_id)
             if (i.get("type") in (None, "PHOTO")) and (i.get("mediaFile") or {}).get("baseUrl")]
    store = load_store(workdir)
    raw_dir = workdir / ".cache" / "gphotos"
    raw_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in items if not (raw_dir / f"{PREFIX}{i['id']}.jpg").exists()]
    done = 0

    def fetch(item):
        data = picker.download(item["mediaFile"]["baseUrl"])
        if data:
            (raw_dir / f"{PREFIX}{item['id']}.jpg").write_bytes(data)
        return item, bool(data)

    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item, ok in pool.map(fetch, todo):
            done += 1
            failed += not ok
            if reporter:
                reporter.advance(done, len(todo))
    added = 0
    for item in items:
        key = f"{PREFIX}{item['id']}"
        if (raw_dir / f"{key}.jpg").exists():
            added += key not in store
            store[key] = describe(item)
    save_store(workdir, store)
    picker.delete_session(session_id)
    return {"picked": len(items), "added": added, "failed": failed, **summary(workdir)}


class GooglePhotosSource:
    """Même interface que DriveClient pour l'analyse : liste des photos et miniatures."""

    def __init__(self, workdir: Path):
        self.workdir = workdir

    def iter_images(self, folder_ids: Iterable[str], recursive: bool = True) -> Iterator[DriveFile]:
        for key, v in load_store(self.workdir).items():
            when = _parse_rfc3339(v.get("when")) or datetime(1970, 1, 1)
            yield DriveFile(
                id=key, name=v.get("name", key), mime=v.get("mime", ""), thumbnail_link=None,
                when=when, date_source="exif" if v.get("when") else "modified",
                modified=v.get("when") or "", width=v.get("width"), height=v.get("height"),
                camera=v.get("camera", ""),
            )

    def fetch_thumbnail(self, f: DriveFile, size: int = 1024) -> bytes | None:
        try:
            return (self.workdir / ".cache" / "gphotos" / f"{f.id}.jpg").read_bytes()
        except OSError:
            return None
