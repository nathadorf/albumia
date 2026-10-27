"""Scoring, dédoublonnage des rafales et sélection répartie dans le temps.

Tout est ici en Python pur (+ numpy) : testable sans OpenCV ni Drive.
Les constantes de netteté sont des points de départ : regardez plan.html et ajustez
si besoin (voir README, section « Ajuster »).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# Variance du Laplacien (échelle log10) correspondant à 0 et 1 de score de netteté.
SHARP_RANGE_GLOBAL = (1.0, 2.7)   # image entière (miniature ~1024 px)
SHARP_RANGE_FACE = (0.8, 2.4)     # visage redimensionné à 112x112

DEFAULT_MATCH_THRESHOLD = 0.363   # seuil cosinus recommandé pour SFace
REJECT_SHARPNESS = 0.12           # en dessous : photo floue, écartée


@dataclass
class Weights:
    sharpness: float = 0.30
    faces: float = 0.22
    smile: float = 0.14
    eyes: float = 0.10
    exposure: float = 0.08
    bonus: float = 0.16


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def sharp_score(lap_var: float, rng: tuple[float, float]) -> float:
    lo, hi = rng
    return _clip((math.log10(max(lap_var, 0.0) + 1.0) - lo) / (hi - lo))


def exposure_score(mean: float, clip_low: float, clip_high: float) -> float:
    # Pénalise les images très sombres / cramées, et les gros aplats noirs ou blancs.
    tone = 1.0 - _clip(abs(mean - 0.45) / 0.45)
    clipping = _clip((clip_low + clip_high) * 3.0)
    return _clip(0.6 * tone + 0.4 * (1.0 - clipping))


def faces_score(faces: list[dict], w: int, h: int) -> float:
    if not faces:
        return 0.0
    largest = max(f["box"][2] * f["box"][3] for f in faces) / float(w * h)
    size = _clip(math.sqrt(largest) / 0.25)
    count = min(len(faces), 3) / 3.0
    return _clip(0.6 * size + 0.4 * count)


def subject_sharpness(feat: dict) -> float:
    faces = feat.get("faces") or []
    if faces:
        biggest = max(faces, key=lambda f: f["box"][2] * f["box"][3])
        return sharp_score(biggest["lap"], SHARP_RANGE_FACE)
    return sharp_score(feat["lap_center"], SHARP_RANGE_GLOBAL)


def match_labels(faces: list[dict], refs: dict[str, list[float]], threshold: float) -> set[str]:
    """Étiquettes de personnes reconnues dans la photo (similarité cosinus)."""
    found: set[str] = set()
    if not faces or not refs:
        return found
    embs = np.array([f["emb"] for f in faces if "emb" in f])
    if embs.size == 0:
        return found
    for label, ref in refs.items():
        if float((embs @ np.array(ref)).max()) >= threshold:
            found.add(label)
    return found


@dataclass
class Candidate:
    file_id: str
    name: str
    when: datetime
    feat: dict
    labels: set[str] = field(default_factory=set)
    score: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)
    local_score: float = 0.0
    note: str = ""


def local_score(
    feat: dict,
    labels: set[str],
    bonus_labels: list[str],
    weights: Weights = Weights(),
) -> tuple[float, dict[str, float]]:
    faces = feat.get("faces") or []
    parts = {
        "sharpness": subject_sharpness(feat),
        "faces": faces_score(faces, feat["w"], feat["h"]),
        "smile": float(np.mean([f["smile"] for f in faces])) if faces else 0.0,
        "eyes": float(np.mean([f["eyes"] for f in faces])) if faces else 0.0,
        "exposure": exposure_score(feat["mean"], feat["clip_low"], feat["clip_high"]),
    }
    w = {
        "sharpness": weights.sharpness,
        "faces": weights.faces,
        "smile": weights.smile,
        "eyes": weights.eyes,
        "exposure": weights.exposure,
    }
    if bonus_labels:
        parts["bonus"] = sum(1 for b in bonus_labels if b in labels) / len(bonus_labels)
        w["bonus"] = weights.bonus
    total_w = sum(w.values())
    score = sum(parts[k] * w[k] for k in w) / total_w
    return score, parts


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def dedupe_bursts(
    items: list[Candidate], hash_threshold: int = 6, window_seconds: int = 120
) -> list[Candidate]:
    """Garde la meilleure photo de chaque rafale (images proches dans le temps ET visuellement)."""
    ordered = sorted(items, key=lambda c: c.when)
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    start = 0
    for i, ci in enumerate(ordered):
        while (ci.when - ordered[start].when).total_seconds() > window_seconds:
            start += 1
        for j in range(start, i):
            if hamming_hex(ci.feat["dhash"], ordered[j].feat["dhash"]) <= hash_threshold:
                parent[find(i)] = find(j)

    best: dict[int, Candidate] = {}
    for i, c in enumerate(ordered):
        root = find(i)
        if root not in best or c.score > best[root].score:
            best[root] = c
    return sorted(best.values(), key=lambda c: c.when)


def allocate(counts: dict, total: int) -> dict:
    """Répartit `total` places entre les périodes, au prorata de la racine carrée de leur taille.

    La racine carrée évite qu'un mois à 800 photos écrase un mois à 40 photos.
    """
    total = min(total, sum(counts.values()))
    alloc = {k: 0 for k in counts}
    remaining = total
    while remaining > 0:
        active = [k for k in counts if alloc[k] < counts[k]]
        if not active:
            break
        weights = {k: math.sqrt(counts[k]) for k in active}
        wsum = sum(weights.values())
        quotas = {k: remaining * weights[k] / wsum for k in active}
        given = 0
        for k in active:
            n = min(int(quotas[k]), counts[k] - alloc[k])
            alloc[k] += n
            given += n
        left = remaining - given
        if left > 0:
            by_fraction = sorted(active, key=lambda k: quotas[k] - int(quotas[k]), reverse=True)
            for k in by_fraction:
                if left == 0:
                    break
                if alloc[k] < counts[k]:
                    alloc[k] += 1
                    left -= 1
        remaining = left
    return alloc


def select_spread(items: list[Candidate], n: int) -> list[Candidate]:
    """Choisit n photos en couvrant toute la période (par mois), meilleure note d'abord."""
    months: dict[tuple[int, int], list[Candidate]] = {}
    for c in items:
        months.setdefault((c.when.year, c.when.month), []).append(c)
    alloc = allocate({k: len(v) for k, v in months.items()}, n)
    chosen: list[Candidate] = []
    for key, quota in alloc.items():
        top = sorted(months[key], key=lambda c: c.score, reverse=True)[:quota]
        chosen.extend(top)
    return sorted(chosen, key=lambda c: c.when)
