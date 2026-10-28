"""Règles de composition des photographes, mesurées avec OpenCV sur la miniature.

Ce que regardent les éditeurs d'albums (et les travaux de Ke et al., 2006, sur la qualité
photo) : un sujet bien placé et pas collé au bord, de l'air au-dessus de la tête, un fond
simple qui ne distrait pas, l'œil attiré par le sujet, un horizon droit, des couleurs vivantes
et un bon contraste. Chaque critère donne une note de 0 à 1.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

VERSION = 1


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def placement(box, w: int, h: int) -> float:
    """Visage près d'un point fort (tiers ou centre, à 1/3 ou 0,45 de la hauteur) ; 0 au bord."""
    x, y, fw, fh = box
    cx, cy = (x + fw / 2) / w, (y + fh / 2) / h
    d = min(math.hypot(cx - a, cy - b) for a in (1 / 3, 0.5, 2 / 3) for b in (1 / 3, 0.45))
    return _clip(1.0 - d / 0.30)


def headroom(box, w: int, h: int) -> float:
    """De l'air au-dessus de la tête, et le visage dans le haut de l'image plutôt qu'en bas."""
    x, y, fw, fh = box
    top, cy = y / h, (y + fh / 2) / h
    return _clip(top / 0.06) * _clip((0.85 - cy) / 0.25)


def _expand(box, k: float, w: int, h: int):
    x, y, fw, fh = box
    cx, cy = x + fw / 2, y + fh / 2
    # le corps est sous le visage : on étend surtout vers le bas
    x0, x1 = cx - fw * k, cx + fw * k
    y0, y1 = cy - fh * k * 0.8, cy + fh * k * 2.0
    return max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))


def _subject_mask(boxes, w: int, h: int, k: float = 1.6) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = _expand(b, k, w, h)
        mask[y0:y1, x0:x1] = True
    return mask


def clutter(gray: np.ndarray, boxes) -> float:
    """Fond simple = peu de contours hors du sujet (1 = fond épuré, 0 = fond très chargé)."""
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160) > 0
    outside = ~_subject_mask(boxes, w, h) if boxes else np.ones((h, w), dtype=bool)
    density = edges[outside].mean() if outside.any() else 0.0
    return _clip(1.0 - (density - 0.03) / 0.12)


def saliency_map(gray: np.ndarray) -> np.ndarray:
    """Saillance par résidu spectral (Hou & Zhang, 2007) : ce qui attire l'œil."""
    small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-6)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 2.5)
    return cv2.resize(sal, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)


def attention(gray: np.ndarray, boxes) -> float:
    """Part de l'attention sur le sujet, comparée à sa surface (1 = l'œil va droit au sujet)."""
    if not boxes:
        return 0.5
    h, w = gray.shape
    sal = saliency_map(gray)
    mask = _subject_mask(boxes, w, h, k=1.2)
    area = mask.mean()
    if area <= 0 or area >= 0.95:
        return 0.5
    share = sal[mask].sum() / (sal.sum() or 1.0)
    return _clip((share - area) / max(0.15, (1 - area) * 0.6) + 0.5)


def level(gray: np.ndarray) -> float:
    """Horizon et verticales droits (1) ou penchés de quelques degrés (0)."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=int(0.3 * max(w, h)), maxLineGap=8)
    if lines is None:
        return 1.0
    devs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 90
        dev = min(ang, 90 - ang)
        if dev <= 15:  # presque horizontale ou verticale : elle devrait être droite
            devs.append(dev)
    if len(devs) < 2:
        return 1.0
    return _clip(1.0 - (float(np.median(devs)) - 1.5) / 6.0)


def colorfulness(img: np.ndarray) -> float:
    """Indice de Hasler & Süsstrunk (2003), ramené à 0-1 (≈ 60 et plus : couleurs vivantes)."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(img)]
    rg, yb = r - g, 0.5 * (r + g) - b
    m = math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    return _clip(m / 60.0)


def contrast(gray: np.ndarray) -> float:
    """Largeur de l'histogramme (98 % central) : image terne (0) ou bien contrastée (1)."""
    lo, hi = np.percentile(gray, [1, 99])
    return _clip(((hi - lo) / 255.0 - 0.35) / 0.5)


def measure(img: np.ndarray, boxes) -> dict[str, float]:
    """Critères qui demandent l'image (les autres se calculent depuis les visages en cache)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "clean": round(clutter(gray, boxes), 4),
        "attention": round(attention(gray, boxes), 4),
        "level": round(level(gray), 4),
        "color": round(colorfulness(img), 4),
        "contrast": round(contrast(gray), 4),
    }
