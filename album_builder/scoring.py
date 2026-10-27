"""Scoring, dédoublonnage des rafales et sélection répartie dans le temps.

Tout est ici en Python pur (+ numpy) : testable sans OpenCV ni Drive.
Les constantes de netteté sont des points de départ : regardez plan.html et ajustez
si besoin (voir README, section « Ajuster »).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime

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
MIN_SUBJECT_AREA = 0.01           # personne choisie plus petite que ça : elle n'est pas le sujet

PRINT_MIN_SIDE = (800, 1500)      # côté court de l'original (px) : 0 → 1 de note « impression »
STAR_BONUS = 0.12                 # photo étoilée dans Drive
MOMENT_GAP_HOURS = 2              # plus de 2 h sans photo : un nouveau moment commence
SHOT_TARGETS = {"gros plan": 0.4, "plan moyen": 0.4, "scène": 0.2}

SCREENSHOT_RE = re.compile(r"screenshot|screen[ _-]?shot|capture d.?[eé]cran|^scr_", re.I)
MESSAGING_RE = re.compile(r"^IMG-\d{8}-WA\d+|whatsapp|^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-", re.I)


@dataclass
class Weights:
    sharpness: float = 0.28
    faces: float = 0.20
    smile: float = 0.08               # détecteur de sourire approximatif (≈ 1 sur 2 juste)
    eyes: float = 0.03                # détecteur d'yeux ouverts peu fiable : poids faible
    frontal: float = 0.06             # regarde vers l'objectif
    exposure: float = 0.07
    resolution: float = 0.06          # assez grande pour un tirage
    bonus: float = 0.16


def photo_kind(name: str, camera: str = "", width: int | None = None, height: int | None = None) -> str:
    """« capture » (à écarter), « messagerie » (souvent trop petite) ou « photo »."""
    if SCREENSHOT_RE.search(name or ""):
        return "capture"
    if not camera and MESSAGING_RE.search(name or ""):
        return "messagerie"
    if not camera and width and height and max(width, height) / max(1, min(width, height)) >= 1.95:
        return "capture"  # format d'écran de téléphone, sans appareil photo
    return "photo"


def print_score(width: int | None, height: int | None) -> float:
    if not width or not height:
        return 1.0  # inconnu (fréquent pour les RAW) : on ne pénalise pas
    lo, hi = PRINT_MIN_SIDE
    return _clip((min(width, height) - lo) / (hi - lo))


def shot_type(feat: dict, focus: list[dict]) -> str:
    if not focus:
        return "scène"
    area = max(f["box"][2] * f["box"][3] for f in focus) / float(feat["w"] * feat["h"])
    return "gros plan" if area >= 0.06 else "plan moyen" if area >= 0.015 else "scène"


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
    faces: list[dict],
    refs: dict[str, list[float]],
    threshold: float,
    margin: float = MATCH_MARGIN,
    allowed: set[str] | None = None,
) -> list[str | None]:
    """Pour chaque visage : la personne reconnue, ou None (inconnu ou trop ambigu).

    `allowed` limite les candidats (ex. : pas Arié sur une photo prise avant sa naissance).
    """
    if allowed is not None:
        refs = {k: v for k, v in refs.items() if k in allowed}
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
    meta: dict = field(default_factory=dict)   # starred, kind, shot, width, height


def local_score(
    feat: dict,
    labels: set[str],
    bonus_labels: list[str],
    weights: Weights = Weights(),
    labels_per_face: list[str | None] | None = None,
    targets: list[str] = (),
    meta: dict | None = None,
) -> tuple[float, dict[str, float]]:
    """Note de 0 à 1. Avec `targets`, cadrage, netteté et sourire se jugent sur les visages de ces
    personnes (et non sur l'adulte en gros plan d'un selfie)."""
    meta = meta or {}
    faces = real_faces(feat)
    focus = faces
    if targets and labels_per_face is not None:
        focus = [f for f, label in zip(faces, labels_per_face) if label in targets] or faces
    framing = faces_score(focus, feat["w"], feat["h"], total=len(faces))
    if focus and max(focus, key=lambda f: f["box"][2] * f["box"][3]).get("edge"):
        framing *= 0.6  # visage principal coupé par le bord
    resolution = print_score(meta.get("width"), meta.get("height"))
    if meta.get("kind") == "messagerie":
        resolution *= 0.5
    parts = {
        "sharpness": subject_sharpness(feat, focus),
        "faces": framing,
        "smile": float(np.mean([f["smile"] for f in focus])) if focus else 0.0,
        "eyes": float(np.mean([f["eyes"] for f in focus])) if focus else 0.0,
        "frontal": float(np.mean([f.get("frontal", 1.0) for f in focus])) if focus else 0.5,
        "exposure": exposure_score(feat["mean"], feat["clip_low"], feat["clip_high"]),
        "resolution": resolution,
    }
    w = {k: getattr(weights, k) for k in parts}
    if bonus_labels:
        parts["bonus"] = sum(1 for b in bonus_labels if b in labels) / len(bonus_labels)
        w["bonus"] = weights.bonus
    total_w = sum(w.values())
    score = sum(parts[k] * w[k] for k in w) / total_w
    if meta.get("starred"):
        score = min(1.0, score + STAR_BONUS)
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


def allocate(counts: dict, total: int, caps: dict | None = None) -> dict:
    """Répartit `total` places entre les périodes, au prorata de la racine carrée de leur taille.

    La racine carrée évite qu'un mois à 800 photos écrase un mois à 40 photos.
    `caps` : nombre maximal de places par période (par défaut sa taille).
    """
    caps = caps or counts
    total = min(total, sum(caps.values()))
    alloc = {k: 0 for k in counts}
    remaining = total
    while remaining > 0:
        active = [k for k in counts if alloc[k] < caps[k]]
        if not active:
            break
        weights = {k: math.sqrt(counts[k]) for k in active}
        wsum = sum(weights.values())
        quotas = {k: remaining * weights[k] / wsum for k in active}
        given = 0
        for k in active:
            n = min(int(quotas[k]), caps[k] - alloc[k])
            alloc[k] += n
            given += n
        left = remaining - given
        if left > 0:
            by_fraction = sorted(active, key=lambda k: quotas[k] - int(quotas[k]), reverse=True)
            for k in by_fraction:
                if left == 0:
                    break
                if alloc[k] < caps[k]:
                    alloc[k] += 1
                    left -= 1
        remaining = left
    return alloc


def moments(items: list[Candidate], gap_hours: float = MOMENT_GAP_HOURS) -> list[list[Candidate]]:
    """Regroupe les photos en moments : une coupure de plus de `gap_hours` sépare deux moments."""
    out: list[list[Candidate]] = []
    for c in sorted(items, key=lambda c: c.when):
        if out and (c.when - out[-1][-1].when).total_seconds() <= gap_hours * 3600:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def moment_cap(size: int) -> int:
    """Un grand moment (beaucoup de photos prises) mérite plus de place qu'un instant isolé."""
    return 1 if size < 5 else 2 if size < 15 else 3


def similarity(a: Candidate, b: Candidate) -> float:
    """0 à 1 : même décor, mêmes couleurs, même cadrage."""
    visual = max(0.0, 1.0 - hamming_hex(a.feat["dhash"], b.feat["dhash"]) / 32.0)
    ha, hb = a.feat.get("hues"), b.feat.get("hues")
    if not ha or not hb:
        return visual
    return 0.5 * visual + 0.5 * float(sum(min(x, y) for x, y in zip(ha, hb)))


def _share(chosen: list[Candidate], key, value) -> float:
    return sum(1 for o in chosen if key(o) == value) / len(chosen) if chosen else 0.0


def diversity_penalty(c: Candidate, chosen: list[Candidate], strength: float = 0.25) -> float:
    """Pénalité pour une photo trop semblable à celles déjà retenues (scène, plan, personnes)."""
    if not chosen:
        return 0.0
    sim = max(similarity(c, o) for o in chosen)
    penalty = strength * max(0.0, sim - 0.55) / 0.45
    if len(chosen) >= 4:
        shot = c.meta.get("shot", "scène")
        penalty += 0.06 * max(0.0, _share(chosen, lambda o: o.meta.get("shot", "scène"), shot) - SHOT_TARGETS.get(shot, 0.4))
        portrait = c.feat["h"] > c.feat["w"]
        penalty += 0.03 * max(0.0, _share(chosen, lambda o: o.feat["h"] > o.feat["w"], portrait) - 0.65)
        who = frozenset(c.labels)
        penalty += 0.04 * max(0.0, _share(chosen, lambda o: frozenset(o.labels), who) - 0.5)
    return penalty


def age_bucket(birth: date):
    """Tranches d'âge : par trimestre la 1re année, par semestre jusqu'à 3 ans, puis par an."""
    def key(c: Candidate):
        months = (c.when.year - birth.year) * 12 + c.when.month - birth.month - (c.when.day < birth.day)
        if months < 0:
            return (-1, 0)
        if months < 12:
            return (0, months // 3)
        if months < 36:
            return (1, months // 6)
        return (2, months // 12)
    return key


def select_spread(items: list[Candidate], n: int, bucket=None, diversity: float = 0.25) -> list[Candidate]:
    """Choisit n photos qui racontent la période.

    1. Les places sont réparties entre les périodes (mois, ou tranches d'âge avec `bucket`).
    2. Dans chaque période, chaque moment a un plafond : 1 photo pour un instant isolé, jusqu'à 3
       pour un grand moment.
    3. On prend d'abord la meilleure note, en pénalisant ce qui ressemble à ce qui est déjà choisi.
    """
    bucket = bucket or (lambda c: (c.when.year, c.when.month))
    groups: dict = {}
    for c in items:
        groups.setdefault(bucket(c), []).append(c)
    alloc = allocate({k: len(v) for k, v in groups.items()}, n)
    chosen: list[Candidate] = []
    for key in sorted(groups):
        quota = alloc[key]
        if not quota:
            continue
        ms = moments(groups[key])
        left = {i: min(len(m), moment_cap(len(m))) for i, m in enumerate(ms)}
        if sum(left.values()) < quota:  # pas assez de moments : on relâche le plafond
            left = {i: len(m) for i, m in enumerate(ms)}
        moment_of = {id(c): i for i, m in enumerate(ms) for c in m}
        pool = [c for c in groups[key]]
        for _ in range(quota):
            open_ = [c for c in pool if left[moment_of[id(c)]] > 0]
            if not open_:
                break
            best = max(open_, key=lambda c: c.score - diversity_penalty(c, chosen, diversity))
            chosen.append(best)
            pool.remove(best)
            left[moment_of[id(best)]] -= 1
    return sorted(chosen, key=lambda c: c.when)
