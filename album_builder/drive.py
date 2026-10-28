"""Accès à Google Drive : authentification, listing, miniatures, création d'album.

Principe de confidentialité : on ne télécharge JAMAIS les originaux (RAW de 30 Mo).
On récupère uniquement la miniature générée par Drive (~1024 px), qui suffit pour
juger netteté, visages et sourires.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .errors import AlbumError

# Moindre privilège : lecture de tout le Drive + écriture limitée aux fichiers
# créés par cette application (le dossier d'album et ses raccourcis).
SCOPES_LIGHT = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
# Accès complet, seulement si vous voulez écrire dans un dossier existant (--parent).
SCOPES_FULL = ["https://www.googleapis.com/auth/drive"]

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"

IMAGE_EXT = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff",
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf",
}

LIST_FIELDS = (
    "nextPageToken, files(id,name,mimeType,size,thumbnailLink,createdTime,modifiedTime,starred,"
    "imageMediaMetadata(time,width,height,cameraModel))"
)


@dataclass
class DriveFile:
    id: str
    name: str
    mime: str
    thumbnail_link: str | None
    when: datetime            # date de prise de vue (EXIF) sinon date de modification
    date_source: str          # "exif" ou "modified"
    modified: str             # chaîne RFC3339, sert à invalider le cache
    width: int | None = None   # dimensions de l'original (pour juger l'impression)
    height: int | None = None
    camera: str = ""           # vide : capture d'écran, image reçue par messagerie…
    starred: bool = False      # étoilée dans Drive : un choix de la personne

    @classmethod
    def from_api(cls, f: dict) -> "DriveFile":
        meta = f.get("imageMediaMetadata") or {}
        taken = _parse_exif_time(meta.get("time"))
        modified = _parse_rfc3339(f.get("modifiedTime")) or datetime(1970, 1, 1)
        return cls(
            id=f["id"],
            name=f.get("name", f["id"]),
            mime=f.get("mimeType", ""),
            thumbnail_link=f.get("thumbnailLink"),
            when=taken or modified,
            date_source="exif" if taken else "modified",
            modified=f.get("modifiedTime", ""),
            width=meta.get("width"),
            height=meta.get("height"),
            camera=meta.get("cameraModel") or "",
            starred=bool(f.get("starred")),
        )


def _parse_exif_time(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None  # y compris les "0000:00:00 00:00:00" invalides


def _parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _is_image(f: dict) -> bool:
    mime = f.get("mimeType", "")
    if mime.startswith("image/"):
        return True
    return Path(f.get("name", "")).suffix.lower() in IMAGE_EXT


LOGIN_DONE_MESSAGE = (
    "C'est bon, Albumia est connecté à ton Google Drive. "
    "Tu peux fermer cet onglet et revenir à Albumia."
)


def load_saved_credentials(token_path: Path, scopes: list[str]) -> Credentials | None:
    """Jeton déjà enregistré et utilisable, sans jamais ouvrir de navigateur."""
    if not token_path.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    except (ValueError, OSError):
        return None
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds, token_path)
            return creds
        except (RefreshError, OSError):
            return None
    return None


def get_credentials(credentials_path: Path, token_path: Path, scopes: list[str]) -> Credentials:
    """OAuth « application de bureau » : une fenêtre de navigateur s'ouvre la 1re fois."""
    creds = load_saved_credentials(token_path, scopes)
    if creds:
        return creds
    if not credentials_path.exists():
        raise AlbumError(
            f"Fichier {credentials_path} introuvable. Suivez l'étape « Google Cloud » du README.",
            code="no_credentials",
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes)
    try:
        creds = flow.run_local_server(port=0, success_message=LOGIN_DONE_MESSAGE, timeout_seconds=300)
    except Exception as e:  # onglet fermé, délai dépassé, accès refusé…
        raise AlbumError(
            f"La connexion à Google n'a pas abouti ({e.__class__.__name__}). Réessaie.", code="login_failed"
        )
    _save_token(creds, token_path)
    return creds


def _save_token(creds: Credentials, token_path: Path) -> None:
    token_path.write_text(creds.to_json())
    try:
        os.chmod(token_path, 0o600)
    except OSError:
        pass


class DriveClient:
    def __init__(self, creds: Credentials):
        self.creds = creds
        self.service = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._local = threading.local()

    # ---- utilitaires ------------------------------------------------------
    def _session(self) -> AuthorizedSession:
        s = getattr(self._local, "session", None)
        if s is None:
            s = AuthorizedSession(self.creds)
            self._local.session = s
        return s

    @staticmethod
    def _execute(request, retries: int = 5):
        for attempt in range(retries):
            try:
                return request.execute()
            except HttpError as e:
                status = getattr(e.resp, "status", 0)
                if status in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except OSError as e:  # coupure réseau, délai dépassé, erreur SSL passagère
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise AlbumError(
                    "La connexion avec Google Drive a été coupée. Vérifie internet, puis relance : "
                    "les photos déjà analysées ne seront pas refaites.",
                    code="network",
                ) from e

    # ---- lecture ----------------------------------------------------------
    def whoami(self) -> str:
        about = self._execute(self.service.about().get(fields="user(emailAddress,displayName)"))
        user = about.get("user", {})
        return user.get("emailAddress") or user.get("displayName") or ""

    def get_folder(self, folder_id: str) -> dict | None:
        """Nom d'un dossier, ou None s'il n'existe pas / n'est pas accessible / n'est pas un dossier."""
        try:
            f = self._execute(
                self.service.files().get(
                    fileId=folder_id, fields="id,name,mimeType,trashed,webViewLink", supportsAllDrives=True
                )
            )
        except HttpError:
            return None
        if f.get("mimeType") != FOLDER_MIME or f.get("trashed"):
            return None
        return f

    def iter_images(self, folder_ids: Iterable[str], recursive: bool = True) -> Iterator[DriveFile]:
        """Liste les images des dossiers donnés (sous-dossiers inclus).

        Les raccourcis sont ignorés : cela évite de recompter les albums déjà créés.
        """
        seen: set[str] = set()
        stack = list(folder_ids)
        while stack:
            folder_id = stack.pop()
            page = None
            while True:
                resp = self._execute(
                    self.service.files().list(
                        q=f"'{folder_id}' in parents and trashed = false",
                        fields=LIST_FIELDS,
                        pageSize=1000,
                        pageToken=page,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                )
                for f in resp.get("files", []):
                    mime = f.get("mimeType", "")
                    if mime == FOLDER_MIME:
                        if recursive:
                            stack.append(f["id"])
                        continue
                    if mime == SHORTCUT_MIME or not _is_image(f) or f["id"] in seen:
                        continue
                    seen.add(f["id"])
                    yield DriveFile.from_api(f)
                page = resp.get("nextPageToken")
                if not page:
                    break

    def fetch_thumbnail(self, f: DriveFile, size: int = 1024) -> bytes | None:
        """Miniature générée par Drive (fonctionne aussi pour les RAW)."""
        if not f.thumbnail_link:
            return None
        url = re.sub(r"=s\d+(-c)?$", f"=s{size}", f.thumbnail_link)
        if url == f.thumbnail_link and not re.search(r"=s\d+", url):
            url = f"{url}=s{size}"
        for attempt in range(3):
            try:
                resp = self._session().get(url, timeout=30)
            except Exception:
                time.sleep(1 + attempt)
                continue
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                return resp.content
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return None
        return None

    # ---- écriture ---------------------------------------------------------
    def create_folder(self, name: str, parent_id: str | None = None) -> dict:
        body = {"name": name, "mimeType": FOLDER_MIME}
        if parent_id:
            body["parents"] = [parent_id]
        return self._execute(
            self.service.files().create(body=body, fields="id,name,webViewLink", supportsAllDrives=True)
        )

    def add_shortcut(self, target_id: str, name: str, parent_id: str) -> dict:
        body = {
            "name": name,
            "mimeType": SHORTCUT_MIME,
            "parents": [parent_id],
            "shortcutDetails": {"targetId": target_id},
        }
        return self._execute(self.service.files().create(body=body, fields="id", supportsAllDrives=True))

    def copy_file(self, file_id: str, name: str, parent_id: str) -> dict:
        body = {"name": name, "parents": [parent_id]}
        return self._execute(
            self.service.files().copy(fileId=file_id, body=body, fields="id", supportsAllDrives=True)
        )
