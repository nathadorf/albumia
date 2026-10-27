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
# Calibré sur 501 vraies photos (miniatures ~1024 px) : médiane visage ≈ 2,56.
SHARP_RANGE_GLOBAL = (2.0, 3.6)   # image entière
SHARP_RANGE_FACE = (1.8, 3.0)     # visage redimensionné à 112x112
REJECT_SHARPNESS = 0.05           # en dessous : photo floue, écartée

# Reconnaissance : chaque visage reçoit AU PLUS une personne, la plus ressemblante, et seulement
# si elle se détache nettement de la deuxième (des frères et sœurs se ressemblent beaucoup).
DEFAULT_MATCH_THRESHOLD = 0.45
LENIENT_MATCH_THRESHOLD = 0.38
MATCH_MARGIN = 0.05
MIN_FACE_AREA = 0.005             # visage < 0,5 % de la photo : fausse détection ou figurant


@dataclass
class Weights:
    sharpness: float = 0.30
    faces: float = 0.22
    smile: float = 0.14
    eyes: float = 0.04                # détecteur d'yeux ouverts peu fiable : poids faible
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


def faces_score(faces: list[dict], w: int, h: int, total: int | None = None) -> float:
    """Taille du plus grand visage de `faces` ; `total` = nombre de visages dans la photo."""
    if not faces:
        return 0.0
    largest = max(f["box"][2] * f["box"][3] for f in faces) / float(w * h)
    size = _clip(math.sqrt(largest) / 0.25)
    count = min(total if total is not None else len(faces), 3) / 3.0
    return _clip(0.6 * size + 0.4 * count)


def real_faces(feat: dict) -> list[dict]:
    """Visages assez grands pour compter (écarte les fausses détections minuscules)."""
    area = float(feat["w"] * feat["h"]) or 1.0
    return [f for f in feat.get("faces") or [] if f["box"][2] * f["box"][3] / area >= MIN_FACE_AREA]


def subject_sharpness(feat: dict, faces: list[dict] | None = None) -> float:
    faces = real_faces(feat) if faces is None else faces
    if faces:
        biggest = max(faces, key=lambda f: f["box"][2] * f["box"][3])
        return sharp_score(biggest["lap"], SHARP_RANGE_FACE)
    return sharp_score(feat["lap_center"], SHARP_RANGE_GLOBAL)


def face_labels(
    faces: list[dict], refs: dict[str, list[float]], threshold: float, margin: float = MATCH_MARGIN
) -> list[str | None]:
    """Pour chaque visage : la personne reconnue, ou None (inconnu ou trop ambigu)."""
    if not faces or not refs:
        return [None] * len(faces)
    names = list(refs)
    ref_mat = np.array([refs[n] for n in names])
    out: list[str | None] = []
    for f in faces:
        if "emb" not in f:
            out.append(None)
            continue
        sims = ref_mat @ np.array(f["emb"])
        order = np.argsort(sims)[::-1]
        best = float(sims[order[0]])
        second = float(sims[order[1]]) if len(order) > 1 else -1.0
        out.append(names[order[0]] if best >= threshold and best - second >= margin else None)
    return out


def match_labels(
    faces: list[dict], refs: dict[str, list[float]], threshold: float, margin: float = MATCH_MARGIN
) -> set[str]:
    """Personnes reconnues dans la photo."""
    return {label for label in face_labels(faces, refs, threshold, margin) if label}


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
    labels_per_face: list[str | None] | None = None,
    targets: list[str] = (),
) -> tuple[float, dict[str, float]]:
    """Note de 0 à 1. Avec `targets`, cadrage, netteté et sourire se jugent sur les visages de ces
    personnes (et non sur l'adulte en gros plan d'un selfie)."""
    faces = real_faces(feat)
    focus = faces
    if targets and labels_per_face is not None:
        focus = [f for f, label in zip(faces, labels_per_face) if label in targets] or faces
    parts = {
        "sharpness": subject_sharpness(feat, focus),
        "faces": faces_score(focus, feat["w"], feat["h"], total=len(faces)),
        "smile": float(np.mean([f["smile"] for f in focus])) if focus else 0.0,
        "eyes": float(np.mean([f["eyes"] for f in focus])) if focus else 0.0,
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
    items: list[Candidate],
    hash_threshold: int = 10,
    window_seconds: int = 900,
    twin_threshold: int = 4,
) -> list[Candidate]:
    """Garde la meilleure photo de chaque groupe de doublons.

    Doublons = images proches visuellement ET prises à moins de `window_seconds` (rafale, même
    scène), ou quasi identiques quelle que soit la date (copie, photo réenregistrée).
    """
    ordered = sorted(items, key=lambda c: c.when)
    n = len(ordered)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    if n:
        hashes = np.array([int(c.feat["dhash"], 16) for c in ordered], dtype=np.uint64)
        times = np.array([c.when.timestamp() for c in ordered])
        for i in range(n - 1):
            xor = np.bitwise_xor(hashes[i + 1 :], hashes[i])
            dist = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
            close = (times[i + 1 :] - times[i]) <= window_seconds
            for k in np.nonzero((dist <= twin_threshold) | (close & (dist <= hash_threshold)))[0]:
                union(i + 1 + int(k), i)

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


def select_spread(items: list[Candidate], n: int, max_per_day: int = 2) -> list[Candidate]:
    """Choisit n photos en couvrant toute la période (par mois), meilleure note d'abord.

    Au plus `max_per_day` photos d'une même journée (même scène sous d'autres angles), sauf s'il
    n'y a pas assez d'autres photos dans le mois pour remplir sa part.
    """
    months: dict[tuple[int, int], list[Candidate]] = {}
    for c in items:
        months.setdefault((c.when.year, c.when.month), []).append(c)
    alloc = allocate({k: len(v) for k, v in months.items()}, n)
    chosen: list[Candidate] = []
    for key, quota in alloc.items():
        ranked = sorted(months[key], key=lambda c: c.score, reverse=True)
        per_day: dict = {}
        picked: list[Candidate] = []
        for c in ranked:
            if len(picked) == quota:
                break
            if per_day.get(c.when.date(), 0) < max_per_day:
                per_day[c.when.date()] = per_day.get(c.when.date(), 0) + 1
                picked.append(c)
        if len(picked) < quota:  # pas assez de journées différentes : on complète
            picked += [c for c in ranked if c not in picked][: quota - len(picked)]
        chosen.extend(picked)
    return sorted(chosen, key=lambda c: c.when)
