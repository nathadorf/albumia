"""Analyse d'image 100 % locale : visages, reconnaissance, netteté, sourire, exposition.

Modèles utilisés (OpenCV Zoo, téléchargés une seule fois dans ./models) :
  - YuNet  : détection de visages
  - SFace  : empreinte de visage (pour reconnaître « ma fille », « mes fils »…)
Aucune photo ne quitte l'ordinateur dans ce module.
"""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .errors import AlbumError

# Plusieurs sources : si GitHub est bloqué, on essaie le miroir officiel OpenCV sur Hugging Face.
YUNET_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
)
SFACE_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
)
MIN_FACE_PX = 20
FACE_CROP = 112


def ensure_model(path: Path, urls: tuple[str, ...]) -> Path:
    if path.exists() and path.stat().st_size > 100_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement du modèle {path.name} (une seule fois)…")
    errors = []
    for url in urls:
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # réseau, proxy…
            path.unlink(missing_ok=True)
            errors.append(f"{url} ({e})")
            continue
        if path.stat().st_size > 100_000:
            return path
        path.unlink(missing_ok=True)
        errors.append(f"{url} (fichier invalide)")
    raise AlbumError(
        "Impossible de télécharger le modèle de reconnaissance des visages "
        f"{path.name}. Vérifie la connexion Internet, ou télécharge-le à la main et "
        f"place-le ici : {path}\n" + "\n".join(errors),
        code="models",
    )


class FaceEngine:
    """Un moteur par thread (les objets OpenCV DNN ne sont pas thread-safe)."""

    def __init__(self, model_dir: Path):
        yunet = ensure_model(model_dir / "face_detection_yunet_2023mar.onnx", YUNET_URLS)
        sface = ensure_model(model_dir / "face_recognition_sface_2021dec.onnx", SFACE_URLS)
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.8, 0.3, 200)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def detect_faces(self, img: np.ndarray, with_embedding: bool = True) -> list[dict]:
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(img)
        if rows is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = []
        for row in rows:
            x, y, fw, fh = (int(round(v)) for v in row[:4])
            x, y = max(x, 0), max(y, 0)
            fw, fh = min(fw, w - x), min(fh, h - y)
            if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
                continue
            crop = gray[y : y + fh, x : x + fw]
            crop_sq = cv2.resize(crop, (FACE_CROP, FACE_CROP), interpolation=cv2.INTER_AREA)
            face = {
                "box": [x, y, fw, fh],
                "score": float(row[14]),
                "lap": float(cv2.Laplacian(crop_sq, cv2.CV_64F).var()),
                "smile": self._smile(crop),
                "eyes": self._eyes(crop),
            }
            if with_embedding:
                feat = self.recognizer.feature(self.recognizer.alignCrop(img, row)).flatten()
                norm = float(np.linalg.norm(feat)) or 1.0
                face["emb"] = (feat / norm).astype(float).tolist()
            faces.append(face)
        return faces

    # Heuristiques Haar : rapides mais imparfaites. Pour un tri fin, utilisez --use-claude.
    def _smile(self, face_gray: np.ndarray) -> float:
        if self.smile_cascade.empty():  # cascades absentes (ex. OpenCV 5) : note neutre
            return 0.5
        h, w = face_gray.shape[:2]
        roi = face_gray[int(h * 0.55) :, :]
        if roi.size == 0 or w < 40:
            return 0.0
        scale = 96.0 / w
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        found = self.smile_cascade.detectMultiScale(roi, scaleFactor=1.6, minNeighbors=20, minSize=(24, 12))
        return 1.0 if len(found) else 0.0

    def _eyes(self, face_gray: np.ndarray) -> float:
        if self.eye_cascade.empty():
            return 0.5
        h, w = face_gray.shape[:2]
        roi = face_gray[: int(h * 0.6), :]
        if roi.size == 0 or w < 40:
            return 0.0
        scale = 120.0 / w
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        found = self.eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=6, minSize=(14, 14))
        return min(1.0, len(found) / 2.0)


_thread_local = threading.local()


def get_engine(model_dir: Path) -> FaceEngine:
    eng = getattr(_thread_local, "engine", None)
    if eng is None:
        eng = FaceEngine(model_dir)
        _thread_local.engine = eng
    return eng


def imread(path: Path) -> np.ndarray | None:
    """cv2.imread ne sait pas lire un chemin accentué sous Windows (« refs/Arié/… ») : on passe par numpy."""
    try:
        return decode(np.fromfile(str(path), dtype=np.uint8).tobytes())
    except OSError:
        return None


def imwrite(path: Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img, params or [])
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def decode(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def dhash(gray: np.ndarray) -> str:
    """Empreinte perceptuelle 64 bits (hex) pour repérer les quasi-doublons."""
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"


def analyze(img: np.ndarray, engine: FaceEngine | None, with_embedding: bool = True) -> dict:
    """Extrait les caractéristiques brutes d'une image (mises en cache, sans score)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cy0, cy1 = int(h * 0.1), int(h * 0.9)
    cx0, cx1 = int(w * 0.1), int(w * 0.9)
    center = gray[cy0:cy1, cx0:cx1]
    return {
        "w": w,
        "h": h,
        "lap_center": float(cv2.Laplacian(center, cv2.CV_64F).var()),
        "mean": float(gray.mean()) / 255.0,
        "clip_low": float((gray <= 5).mean()),
        "clip_high": float((gray >= 250).mean()),
        "dhash": dhash(gray),
        "faces": engine.detect_faces(img, with_embedding) if engine else [],
    }


def build_references(engine: FaceEngine, refs_dir: Path) -> dict[str, list[float]]:
    """refs/<étiquette>/*.jpg -> une empreinte moyenne par personne ou groupe."""
    refs: dict[str, list[float]] = {}
    if not refs_dir.exists():
        return refs
    for label_dir in sorted(p for p in refs_dir.iterdir() if p.is_dir()):
        embs = []
        for path in sorted(label_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            img = imread(path)
            if img is None:
                print(f"  ! {path} illisible, ignoré")
                continue
            img = resize_max(img, 1600)
            faces = engine.detect_faces(img)
            if not faces and max(img.shape[:2]) < 800:  # petit visage recadré : on réessaie en plus grand
                img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                faces = engine.detect_faces(img)
            if not faces:
                print(f"  ! aucun visage détecté dans {path.name}, ignoré")
                continue
            biggest = max(faces, key=lambda f: f["box"][2] * f["box"][3])
            if len(faces) > 1:
                print(f"  ! {path.name} contient {len(faces)} visages : le plus grand est utilisé")
            embs.append(np.array(biggest["emb"]))
        if embs:
            mean = np.mean(embs, axis=0)
            refs[label_dir.name] = (mean / (np.linalg.norm(mean) or 1.0)).tolist()
            print(f"  référence « {label_dir.name} » : {len(embs)} photo(s)")
        else:
            print(f"  ! dossier de référence « {label_dir.name} » sans visage exploitable")
    return refs


def face_crop(img: np.ndarray, box: list[int], margin: float = 0.6, min_side: int = 320) -> np.ndarray:
    """Recadre un visage avec de la marge (pour qu'il reste détectable seul), agrandi si trop petit."""
    h, w = img.shape[:2]
    x, y, fw, fh = box
    cx, cy = x + fw / 2, y + fh / 2
    half = max(fw, fh) * (0.5 + margin)
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    crop = img[y0:y1, x0:x1]
    side = max(crop.shape[:2])
    if 0 < side < min_side:
        scale = min_side / side
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop
