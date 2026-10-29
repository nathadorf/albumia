"""Le tri d'un photographe : un moment, une photo ; des chapitres ; une couverture.

Règles relevées sur la sélection « VDEF » faite à la main pour la famille (octobre 2025), puis
confirmées par ses relectures (19 % de retraits contre 29 à 43 % pour les albums automatiques) :

1. Un moment = des photos prises à moins de 10 minutes d'écart, de scène voisine. On en garde une
   (deux si le moment est fort : au moins 5 prises, et une autre personne ou un autre plan).
2. Dans un moment, on garde la prise la plus vivante : l'émotion, les gestes entre les personnes,
   plusieurs de la famille, de face. Sur ses relectures (35 prises gardées contre leur « doublon »), ce sont
   elles qui ont départagé ; les yeux ouverts, le sourire et la netteté, non (40 à 46 %).
3. Jamais deux fois la même pose dans l'album, même à des heures d'écart.
4. Un séjour se raconte en chapitres (3 à 8 photos chacun), coupés aux grands écarts de temps ou
   de scène.
5. Pas de plan large avec des enfants minuscules, pas d'adulte seul ; un quart de plans larges
   au plus, pour situer le lieu.
6. Chacun de la famille est présent au moins 2 fois (3 pour un bébé), avec une photo de groupe
   qui sert de couverture.
7. Au plus quelques photos d'une même activité (piscine, jeux gonflables, repas).

Tout est local, sans réseau. Les seuils viennent des mesures faites sur les photos de la famille :
`python -m album_builder.curate --compare auto/plan.json main/plan.json` compare un album
automatique à une sélection faite à la main.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import scenes, scoring
from .scoring import Candidate

# ---- un moment = une photo ------------------------------------------------------------------
MOMENT_GAP = 600                      # secondes : 10 minutes entre deux prises du même moment
MOMENT_SPAN = 1800                    # … et un moment dure 30 minutes au plus (une sortie en compte plusieurs)
MOMENT_COS = {"L": 0.80, "B": 0.75}   # scène voisine (œil plus précis / œil habituel)
POSE_COS = {"L": 0.88, "B": 0.93}     # même pose : ses paires gardées à la main sont à 0,86 au plus
SAME_FRAME_BITS = 4                   # dhash : même cadrage (copie, rafale serrée)
STRONG_TAKES = 5                      # un moment de 5 prises ou plus peut donner une 2e photo…
SECOND_TAKE_SLACK = 0.12              # … si elle apporte quelqu'un d'autre ou un autre plan
# part des paires (prise gardée, prise retirée en « doublon ») où la prise gardée est la meilleure, sur 35 paires :
# émotion (œil précis) 76 %, note 70 %, personnes 69 %, gestes entre eux 66 %, moment et hauteur d'œil 62 %,
# de face 59 % ; visage 50 %, beauté 50 %, sourire 46 %, netteté 46 %, yeux ouverts 40 % (rien à en tirer)
TAKE_WEIGHTS = {"emotion": 0.25, "score": 0.20, "people": 0.20, "interaction": 0.15, "frontal": 0.10,
                "moment": 0.05, "eye_level": 0.05}
TAKE_PRESETS = {                      # pour comparer des réglages sur les paires relues (`--takes`)
    "avant": {"frontal": 0.18, "eyes": 0.12, "smile": 0.08, "emotion": 0.10, "face": 0.10,
              "people": 0.08, "score": 0.26, "sharp": 0.08},
    "actuel": TAKE_WEIGHTS,
    "émotion et monde": {"emotion": 0.5, "people": 0.5},
    "note seule": {"score": 1.0},
    "sans la note": {"emotion": 0.3, "people": 0.3, "interaction": 0.2, "frontal": 0.1, "moment": 0.1},
}
FACE_FULL = 0.02                      # visage de la famille à 2 % de l'image : crédit plein (plus gros : pas mieux)
SHARP_FULL = 0.6                      # au-delà de cette netteté, plus de crédit : un filtre, pas un but

# ---- ce qui n'a pas sa place dans un album de famille ---------------------------------------
TINY_SUBJECT = 0.015                  # visage de la famille sous 1,5 % de l'image : sujet minuscule
TINY_PENALTY = 0.03                   # un simple départage : il a remis lui-même une vue large aux enfants minuscules
ADULT_ALONE_PENALTY = 0.12
# plans larges (visage sous 1,5 % de l'image : un groupe au loin, un lieu). Ce qu'il garde en compte 44 % (4 albums
# relus) et 53 % (sa sélection finale), mais 54 % de ceux qu'on lui propose sont retirés (18 % des autres) : un plafond
# à 40 %, et ce sont la note, le moment et l'ouverture des chapitres qui choisissent lesquels
WIDE_SHARE = 0.40
SCENE_SHARE = 0.08                    # une même scène (piscine…) : 8 % de l'album, 2 photos au moins
SCENE_CAP_MIN = 2
TITLE_SCENE_SHARE = 0.5               # un chapitre ne prend le nom d'une scène que si elle en fait la moitié

# ---- la famille est présente ----------------------------------------------------------------
PERSON_MIN, BABY_MIN = 2, 3
FLAGSHIP_SIZES = (4, 3)               # photo phare : au moins 4 de la famille (sinon 3)
SEED_PERCENTILE = 40                  # une photo imposée doit être au-dessus de ce rang de qualité
BABY_MONTHS = 24

# ---- chapitres ------------------------------------------------------------------------------
OPENER_SLACK = 0.10                   # un plan large qui ouvre un chapitre : note ≥ médiane de l'album − 0,10
OPENER_REACH = 3 * 3600               # … pris au plus 3 h avant le chapitre (on arrive, on situe le lieu)
PHOTOS_PER_CHAPTER = 5.5
MIN_CHAPTER = 3
MAX_CHAPTERS = 12
MIN_FOR_CHAPTERS = 8                  # en dessous, un seul chapitre
DAY_CHANGE = 0.8                      # force d'une coupure : changement de jour…
SCENE_SHIFT = 2.0                     # … changement de scène…
SIZE_WEIGHT = 1.0                     # … contre des chapitres de taille régulière

WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
          "octobre", "novembre", "décembre"]


# ---- moments --------------------------------------------------------------------------------
def _same_moment(a: Candidate, b: Candidate, cos: tuple[float, str] | None) -> bool:
    gap = abs((a.when - b.when).total_seconds())
    if gap > MOMENT_GAP:
        return False
    bits = scoring.hamming_hex(a.feat["dhash"], b.feat["dhash"])
    if gap < 1.0 or bits <= SAME_FRAME_BITS:
        return True
    if cos is None:  # sans œil d'éditeur : seulement des photos visuellement proches, prises coup sur coup
        return gap <= 120 and bits <= 12
    return cos[0] >= MOMENT_COS[cos[1]]


def find_moments(items: list[Candidate]) -> list[list[Candidate]]:
    """Regroupe les prises d'un même moment : moins de 10 minutes d'écart et une scène voisine.

    Une photo rejoint le moment dont l'une des trois dernières prises lui ressemble, tant que ce
    moment dure moins de 30 minutes (une longue sortie reste plusieurs moments). Numérote chaque
    photo (meta["moment"])."""
    vecs = scoring.Vectors(items)
    index = {id(c): i for i, c in enumerate(items)}
    clusters: list[list[Candidate]] = []
    for c in sorted(items, key=lambda c: c.when):
        home = None
        for cluster in reversed(clusters[-6:]):
            if (c.when - cluster[0].when).total_seconds() > MOMENT_SPAN:
                continue
            if any(_same_moment(c, o, vecs.cos(index[id(c)], index[id(o)])) for o in cluster[-3:]):
                home = cluster
                break
        if home is None:
            clusters.append([c])
        else:
            home.append(c)
    for k, cluster in enumerate(clusters):
        for c in cluster:
            c.meta["moment"] = k
    return clusters


def take_features(c: Candidate) -> dict[str, float]:
    """Ce qui peut départager deux prises d'un même moment, de 0 à 1 (0,5 quand ce n'est pas mesuré)."""
    p = c.parts
    return {
        "frontal": p.get("frontal", 0.5), "eyes": p.get("eyes", 0.5), "smile": p.get("smile", 0.5),
        "emotion": p.get("emotion_l", p.get("emotion", 0.5)),
        "face": min(1.0, c.meta.get("family_area", 0.0) / FACE_FULL),
        "people": min(1.0, len(c.labels) / 3.0),
        "score": c.score,
        "sharp": min(1.0, p.get("sharpness", 0.5) / SHARP_FULL),
        "interaction": p.get("interaction", 0.5), "moment": p.get("moment", 0.5),
        "eye_level": p.get("eye_level", 0.5), "face_seen": p.get("face_seen", 0.5),
        "light": p.get("light", 0.5), "tidy": p.get("tidy", 0.5),
        "beauty": p.get("beauty_l", p.get("beauty", 0.5)),
    }


def take_value(c: Candidate, weights: dict[str, float] | None = None) -> float:
    """Quelle prise garder dans un moment : la plus vivante (émotion, gestes entre les personnes, plusieurs
    de la famille, de face), selon ce qui a départagé les prises qu'il a gardées de leurs doublons."""
    w, f = weights or TAKE_WEIGHTS, take_features(c)
    return sum(w[k] * f[k] for k in w) / sum(w.values())


def _second_take(ranked: list[Candidate]) -> Candidate | None:
    first = ranked[0]
    floor = take_value(first) - SECOND_TAKE_SLACK
    for c in ranked[1:]:
        if take_value(c) < floor:
            break
        if (c.labels - first.labels) or c.meta.get("shot") != first.meta.get("shot"):
            return c
    return None


def heroes(groups: list[list[Candidate]], pinned=frozenset()) -> tuple[list[Candidate], dict[int, list[Candidate]]]:
    """(la ou les photos retenues de chaque moment, les autres prises de chaque moment, meilleures d'abord).

    Une photo gardée à la relecture est toujours la photo de son moment. Les prises retirées à une relecture
    (`meta["banned"]`) comptent pour la force du moment (le nombre de prises) mais ne sont jamais proposées."""
    picks: list[Candidate] = []
    others: dict[int, list[Candidate]] = {}
    for k, group in enumerate(groups):
        ranked = sorted((c for c in group if not c.meta.get("banned")), key=take_value, reverse=True)
        if not ranked:
            others[k] = []
            continue
        keep = [c for c in ranked if c.file_id in pinned]
        if not keep:
            keep = [ranked[0]]
            if len(group) >= STRONG_TAKES and len(ranked) > 1:
                second = _second_take(ranked)
                if second is not None:
                    keep.append(second)
        picks.extend(keep)
        kept = {id(c) for c in keep}
        others[k] = [c for c in ranked if id(c) not in kept]
    return picks, others


def event_cap(size: int) -> int:
    """Photos au plus pour une sortie de `size` prises (un après-midi de jeux, une fête…)."""
    return 1 if size < 5 else 2 if size < 15 else 3 if size < 40 else 4


def penalize(c: Candidate) -> None:
    """Sujet minuscule dans un plan large, adulte seul : moins bien classés (sans être exclus)."""
    penalty = (TINY_PENALTY if c.meta.get("tiny") else 0.0) + (ADULT_ALONE_PENALTY if c.meta.get("adult_only") else 0.0)
    if penalty:
        c.score = max(0.0, c.score - penalty)


def scene_cap(count: int) -> int:
    return max(SCENE_CAP_MIN, round(SCENE_SHARE * count))


def wide_cap(count: int) -> int:
    return max(1, math.ceil(WIDE_SHARE * count))


# ---- la famille est présente ----------------------------------------------------------------
def coverage_seeds(pool: list[Candidate], family: list[str], babies=(), pinned=frozenset(),
                   limit: int = 8) -> set[str]:
    """Photos à prendre d'abord pour que chacun de la famille soit présent (2 fois, 3 pour un bébé) et
    qu'une photo de groupe serve de photo phare. Jamais une photo médiocre, un selfie ou un repas."""
    if not family or not pool:
        return set()
    scores = sorted(c.score for c in pool)
    floor = scores[int(len(scores) * SEED_PERCENTILE / 100)] if len(scores) >= 10 else -1.0
    vecs = scoring.Vectors(pool)
    index = {id(c): i for i, c in enumerate(pool)}
    chosen = [c for c in pool if c.file_id in pinned]
    used = {c.meta.get("moment") for c in chosen}
    near = np.zeros(len(pool), dtype=bool)
    for c in chosen:
        near |= vecs.near(index[id(c)], POSE_COS)
    seeds: list[Candidate] = []

    def usable(c: Candidate) -> bool:
        return (c.score >= floor and not c.meta.get("selfie") and not c.meta.get("meal")
                and not c.meta.get("tiny") and c.meta.get("moment") not in used
                and not near[index[id(c)]])

    def add(c: Candidate) -> None:
        seeds.append(c)
        chosen.append(c)
        used.add(c.meta.get("moment"))
        near[:] |= vecs.near(index[id(c)], POSE_COS)

    members = set(family)
    for size in FLAGSHIP_SIZES:
        if any(len(c.labels & members) >= size for c in chosen):
            break
        group = [c for c in pool if len(c.labels & members) >= size and usable(c)]
        if group:
            add(max(group, key=lambda c: c.score))
            break
    for person in family:
        need = BABY_MIN if person in babies else PERSON_MIN
        have = sum(1 for c in chosen if person in c.labels)
        for c in sorted((c for c in pool if person in c.labels), key=lambda c: c.score, reverse=True):
            if have >= need or len(seeds) >= limit:
                break
            if usable(c):
                add(c)
                have += 1
    return {c.file_id for c in seeds}


# ---- album plus court que demandé -----------------------------------------------------------
EXTRA_POSE_COS = {"L": 0.93, "B": 0.97}   # une prise en plus doit être une autre image (0,96 et plus : la même)
EXTRA_STRENGTH = 0.04                     # à valeur égale, les moments qui ont donné le plus de prises d'abord
EXTRA_SAME_FRAME_BITS = 10                # sans vecteur : deux cadrages à 10 bits (dhash) ou moins sont la même image


def add_extra_takes(selected: list[Candidate], others: dict[int, list[Candidate]], sizes: dict[int, int],
                    count: int, decor_max: int | None = None) -> list[Candidate]:
    """Un dossier qui n'a que 25 moments pour 31 places : tous les moments d'abord, puis les moments forts
    donnent une prise de plus (le Kotel en quatre vues, le loulav en deux) jusqu'à `event_cap` photos chacun.

    D'abord une prise qui apporte quelqu'un d'autre ou un autre plan ; s'il reste des places, une autre vue
    du même lieu suffit (le Kotel de trois côtés : il en avait gardé quatre). Jamais la même image qu'une photo
    déjà choisie ; ni selfie, ni adulte seul, ni repas, ni visages minuscules ; au plus `decor_max` photos sans visage."""
    out = list(selected)
    chosen = {o.file_id for o in out}
    spare = [c for takes in others.values() for c in takes if c.file_id not in chosen]
    if len(out) >= count or not spare:
        return out
    pool = out + spare
    vecs = scoring.Vectors(pool)
    index = {id(c): i for i, c in enumerate(pool)}
    near = np.zeros(len(pool), dtype=bool)
    mates: dict = {}
    for c in out:
        near |= vecs.near(index[id(c)], EXTRA_POSE_COS)
        mates.setdefault(c.meta.get("moment"), []).append(c)
    decor = [sum(1 for c in out if not scoring.real_faces(c.feat))]

    def usable(c: Candidate, fresh: bool) -> bool:
        k = c.meta.get("moment")
        if len(mates.get(k, ())) >= event_cap(sizes.get(k, 1)) or near[index[id(c)]]:
            return False
        if c.meta.get("selfie") or c.meta.get("adult_only") or c.meta.get("meal") or c.meta.get("tiny"):
            return False
        if decor_max is not None and decor[0] >= decor_max and not scoring.real_faces(c.feat):
            return False
        i = index[id(c)]
        if not (vecs.hasL[i] or vecs.hasB[i]) and any(      # sans œil d'éditeur : seul le cadrage dit si c'est une autre image
                scoring.hamming_hex(c.feat["dhash"], o.feat["dhash"]) <= EXTRA_SAME_FRAME_BITS for o in mates.get(k, ())):
            return False
        if not fresh or not mates.get(k):
            return True
        known = set().union(*(o.labels for o in mates.get(k, ())))
        shots = {o.meta.get("shot") for o in mates.get(k, ())}
        return bool(c.labels - known) or c.meta.get("shot") not in shots

    for fresh in (True, False):
        while len(out) < count:
            options = [c for c in spare if usable(c, fresh)]
            if not options:
                break
            best = max(options, key=lambda c: take_value(c) + EXTRA_STRENGTH * math.log2(max(1, sizes.get(c.meta.get("moment"), 1))))
            out.append(best)
            spare.remove(best)
            mates.setdefault(best.meta.get("moment"), []).append(best)
            near |= vecs.near(index[id(best)], EXTRA_POSE_COS)
            decor[0] += int(not scoring.real_faces(best.feat))
    return sorted(out, key=lambda c: c.when)


# ---- chapitres ------------------------------------------------------------------------------
def _strength(ordered: list[Candidate]) -> np.ndarray:
    """Force d'une coupure avant chaque photo : écart de temps, changement de jour, changement de scène."""
    n = len(ordered)
    vecs = scoring.Vectors(ordered)
    out = np.zeros(n)
    for i in range(1, n):
        gap_h = max(0.0, (ordered[i].when - ordered[i - 1].when).total_seconds() / 3600.0)
        value = math.log1p(gap_h)
        if ordered[i].when.date() != ordered[i - 1].when.date():
            value += DAY_CHANGE
        left, right = list(range(max(0, i - 2), i)), list(range(i, min(n, i + 2)))
        for mat, has in ((vecs.L, vecs.hasL), (vecs.B, vecs.hasB)):
            if all(has[j] for j in left + right):
                a, b = mat[left].mean(axis=0), mat[right].mean(axis=0)
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                if norm:
                    value += SCENE_SHIFT * (1.0 - float(a @ b) / norm)
                break
        out[i] = value
    return out


def _best_cuts(strength: np.ndarray, n: int, k: int, min_size: int) -> list[int]:
    """Les k−1 coupures qui maximisent la force des coupures, avec des chapitres de taille régulière."""
    ideal = n / k
    neg = float("-inf")
    best = [[neg] * (n + 1) for _ in range(k + 1)]
    back = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for seg in range(1, k + 1):
        for end in range(seg * min_size, n - (k - seg) * min_size + 1):
            for start in range((seg - 1) * min_size, end - min_size + 1):
                if best[seg - 1][start] == neg:
                    continue
                value = (best[seg - 1][start] + (float(strength[start]) if start else 0.0)
                         - SIZE_WEIGHT * ((end - start - ideal) / ideal) ** 2)
                if value > best[seg][end]:
                    best[seg][end], back[seg][end] = value, start
    cuts, end = [], n
    for seg in range(k, 0, -1):
        end = back[seg][end]
        cuts.append(end)
    return sorted(c for c in cuts if c > 0)


def cut_chapters(selected: list[Candidate], target: int | None = None) -> list[list[Candidate]]:
    """Coupe l'album (dans l'ordre du temps) en chapitres : aux grands écarts de temps ou de scène."""
    ordered = sorted(selected, key=lambda c: c.when)
    n = len(ordered)
    if n < MIN_FOR_CHAPTERS:
        return [ordered] if ordered else []
    k = target or max(2, min(MAX_CHAPTERS, round(n / PHOTOS_PER_CHAPTER)))
    k = max(1, min(k, n // MIN_CHAPTER))
    if k == 1:
        return [ordered]
    cuts = [0] + _best_cuts(_strength(ordered), n, k, MIN_CHAPTER) + [n]
    return [ordered[a:b] for a, b in zip(cuts, cuts[1:])]


def add_openers(selected: list[Candidate], pool: list[Candidate], count: int,
                protected=frozenset(), wide_limit: int | None = None) -> list[Candidate]:
    """Chaque chapitre a un plan large qui situe le lieu : on en ajoute un quand il manque.

    C'est la photo d'ouverture d'un album de photographe (le Kotel en grand, la salle de jeux) ; il en
    a remis trois lui-même à la relecture. Le plan large vient de la même époque, sans refaire un moment
    ni une pose déjà là. L'album ne dépasse jamais `count` : la photo la plus faible d'un chapitre
    bien fourni lui cède la place (jamais une photo `protected`), ni `wide_limit` plans larges en tout."""
    out = sorted(selected, key=lambda c: c.when)
    groups = cut_chapters(out)
    if len(out) < MIN_FOR_CHAPTERS or len(groups) < 2:
        return out
    floor = float(np.median([c.score for c in out])) - OPENER_SLACK
    vecs = scoring.Vectors(pool)
    index = {id(c): i for i, c in enumerate(pool)}
    near = np.zeros(len(pool), dtype=bool)
    for c in out:
        if id(c) in index:
            near |= vecs.near(index[id(c)], POSE_COS)
    moments_used = {c.meta.get("moment") for c in out}
    for group in groups:
        if any(c.meta.get("shot") == "scène" for c in group):
            continue
        if wide_limit is not None and sum(1 for c in out if c.meta.get("shot") == "scène") >= wide_limit:
            break
        start, end = group[0].when, group[-1].when
        options = [c for c in pool if c.meta.get("shot") == "scène" and c.score >= floor
                   and c.meta.get("moment") not in moments_used and id(c) in index and not near[index[id(c)]]
                   and (start - c.when).total_seconds() <= OPENER_REACH and (c.when - end).total_seconds() <= 0
                   and not c.meta.get("meal")]
        if not options:
            continue
        place = lambda c: (scenes.KIND.get(c.note) == scenes.PLACE) * 0.05 + (0.03 if c.labels else 0.0)
        best = max(options, key=lambda c: c.score + place(c))
        if len(out) >= count:
            roomy = [g for g in cut_chapters(out) if len(g) > MIN_CHAPTER]
            removable = [c for g in roomy for c in g if c.file_id not in protected]
            if not removable:
                continue
            out.remove(min(removable, key=lambda c: c.score))
        out.append(best)
        out.sort(key=lambda c: c.when)
        moments_used.add(best.meta.get("moment"))
        near |= vecs.near(index[id(best)], POSE_COS)
    return out


def date_label(start: datetime, end: datetime) -> str:
    a, b = start.date(), end.date()
    day = lambda d: "1er" if d.day == 1 else str(d.day)
    if a == b:
        return f"{WEEKDAYS[a.weekday()].capitalize()} {day(a)} {MONTHS[a.month - 1]}"
    if (b - a).days <= 31:
        if (a.year, a.month) == (b.year, b.month):
            return f"Du {day(a)} au {day(b)} {MONTHS[b.month - 1]}"
        return f"Du {day(a)} {MONTHS[a.month - 1]} au {day(b)} {MONTHS[b.month - 1]}"
    if a.year == b.year:
        return f"{MONTHS[a.month - 1].capitalize()} à {MONTHS[b.month - 1]} {b.year}"
    return f"{MONTHS[a.month - 1].capitalize()} {a.year} – {MONTHS[b.month - 1]} {b.year}"


def chapter_title(group: list[Candidate]) -> str:
    """« Mardi 7 octobre · Piscine » : le jour, et la scène quand elle fait au moins la moitié du chapitre
    (sinon le titre mentirait : un chapitre de jeux, de repas et de rue reste « Mardi 7 octobre »)."""
    label = date_label(min(c.when for c in group), max(c.when for c in group))
    counts = Counter(c.note for c in group if c.note and c.note not in scenes.GENERIC)
    if counts:
        scene, n = counts.most_common(1)[0]
        if n >= 2 and n >= TITLE_SCENE_SHARE * len(group):
            return f"{label} · {scene}"
    return label


def pick_cover(selected: list[Candidate], family: list[str] | None = None, kids=()) -> str | None:
    """La photo de couverture : les enfants et du monde, de face, un sourire, belle et nette.

    Jamais un selfie, un adulte seul, un repas ni des visages minuscules, sauf s'il n'y a rien d'autre."""
    if not selected:
        return None
    members, children = set(family or ()), set(kids)

    def value(c: Candidate) -> float:
        p = c.parts
        crowd = len(c.labels & members) if members else len(c.labels)
        value = (0.20 * min(crowd, 4) / 4 + 0.12 * p.get("frontal", 0.5) + 0.12 * p.get("smile", 0.5)
                 + 0.12 * p.get("beauty_l", p.get("beauty", 0.5)) + 0.06 * min(1.0, p.get("sharpness", 0.5) / SHARP_FULL)
                 + 0.10 * min(1.0, c.meta.get("family_area", 0.0) / FACE_FULL) + 0.18 * c.score)
        if children:
            value += 0.20 * min(len(c.labels & children), 3) / 3
        return value

    flawed = lambda c: c.meta.get("selfie") or c.meta.get("adult_only") or c.meta.get("meal") or c.meta.get("tiny")
    return max([c for c in selected if not flawed(c)] or selected, key=value).file_id


def layout(selected: list[Candidate], reserve: list[Candidate], family: list[str] | None = None, kids=()) -> dict:
    """Chapitres, couverture et, pour chaque photo (choisie ou en réserve), son chapitre et son moment."""
    groups = cut_chapters(selected)
    chapters = [{"title": chapter_title(g), "start": g[0].when.isoformat(timespec="seconds"),
                 "end": g[-1].when.isoformat(timespec="seconds")} for g in groups]
    of: dict[str, dict] = {}
    for k, g in enumerate(groups):
        for c in g:
            of[c.file_id] = {"chapter": k, "moment": c.meta.get("moment")}
    spans = [(g[0].when, g[-1].when) for g in groups]
    for c in reserve:  # une photo de la réserve rejoint le chapitre le plus proche dans le temps
        if spans:
            k = min(range(len(spans)), key=lambda i: 0.0 if spans[i][0] <= c.when <= spans[i][1]
                    else min(abs((c.when - spans[i][0]).total_seconds()), abs((c.when - spans[i][1]).total_seconds())))
            of[c.file_id] = {"chapter": k, "moment": c.meta.get("moment")}
    return {"chapters": chapters, "cover": pick_cover(selected, family, kids), "of": of}


# ---- réserve --------------------------------------------------------------------------------
ALTERNATES = 2   # autres prises proposées pour chaque photo choisie (bouton « autres photos de ce moment »)


def build_reserve(selected: list[Candidate], others: dict[int, list[Candidate]], pool: list[Candidate],
                  limit: int) -> list[Candidate]:
    """Pour chaque photo choisie, ses meilleures autres prises ; puis les meilleures photos d'autres moments."""
    chosen = {c.file_id for c in selected}
    out: list[Candidate] = []
    seen: set[str] = set()
    for c in selected:
        for o in others.get(c.meta.get("moment"), [])[:ALTERNATES]:
            if o.file_id not in chosen and o.file_id not in seen:
                out.append(o)
                seen.add(o.file_id)
    rest = sorted((c for c in pool if c.file_id not in chosen and c.file_id not in seen),
                  key=lambda c: c.score, reverse=True)
    return out + rest[:limit]


# ---- comparaison avec une sélection faite à la main -----------------------------------------
def _load_candidates(plan: dict, workdir: Path) -> list[Candidate]:
    cache = _read(workdir / ".cache" / "features.json")
    editor = _read(workdir / ".cache" / "editor.json")
    current = _read(workdir / ".cache" / "parts.json")   # critères du dernier passage : les mêmes pour toutes
    out = []
    for item in plan.get("photos", []) + plan.get("reserve", []):
        fid = item["id"]
        feat = (cache.get(fid) or {}).get("feat") or {"dhash": "0" * 16, "w": 1, "h": 1, "faces": []}
        c = Candidate(fid, item["name"], datetime.fromisoformat(item["date"]), feat, set(item.get("people", [])))
        c.score, c.parts, c.note = item.get("score", 0.0), current.get(fid) or item.get("parts", {}), item.get("note", "")
        c.meta = {"vec": (editor.get(fid) or {}).get("vec"), "vec_l": (editor.get(fid) or {}).get("vec_l"),
                  "shot": item.get("shot", ""), "family_area": item.get("face", 0.0)}
        out.append(c)
    return out


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def compare(auto_path: Path, ref_path: Path, workdir: Path, out=print) -> dict:
    """Combien des photos de la sélection à la main (`ref`) l'album automatique retrouve-t-il ?

    Sont comparés les photos elles-mêmes et leurs moments (une autre prise du même moment compte)."""
    auto, ref = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (auto_path, ref_path))
    ref_ids = (ref.get("review") or {}).get("ids") or [p["id"] for p in ref["photos"]]
    auto_ids = (auto.get("review") or {}).get("ids") or [p["id"] for p in auto["photos"]]
    merged = {"photos": auto["photos"] + ref["photos"], "reserve": auto.get("reserve", []) + ref.get("reserve", [])}
    seen: dict = {}
    for item in merged["photos"] + merged["reserve"]:
        seen.setdefault(item["id"], item)
    cands = _load_candidates({"photos": list(seen.values())}, workdir)
    find_moments(cands)
    by_id = {c.file_id: c for c in cands}
    moment = {fid: by_id[fid].meta["moment"] for fid in by_id}
    auto_moments = {moment[i] for i in auto_ids if i in moment}
    exact = [i for i in ref_ids if i in set(auto_ids)]
    same_moment = [i for i in ref_ids if i not in exact and moment.get(i) in auto_moments]
    missing = [i for i in ref_ids if i not in exact and i not in same_moment]
    ref_moments = {moment[i] for i in ref_ids if i in moment}
    extra = [i for i in auto_ids if moment.get(i) not in ref_moments]
    why = _read(workdir / ".cache" / "rejected.json")
    name = lambda i: (f"{by_id[i].name} ({by_id[i].when:%d/%m %H:%M})" if i in by_id else i) + _why(why.get(i))
    out(f"Sélection à la main : {len(ref_ids)} photos ; album automatique : {len(auto_ids)} photos.")
    out(f"  mêmes photos : {len(exact)} ; autre prise du même moment : {len(same_moment)} ; "
        f"moment absent de l'album automatique : {len(missing)}.")
    out(f"  moments de l'album automatique que la sélection à la main n'a pas : {len(extra)}.")
    for title, ids in (("Autre prise du même moment", same_moment), ("Moment absent", missing),
                       ("Moment en plus", extra)):
        if ids:
            out(f"{title} :")
            for i in ids:
                out(f"  - {name(i)}")
    table = moments_table(_read(workdir / ".cache" / "moments.json"), ref_ids, auto_ids)
    if table:
        out("Moments où la sélection à la main garde plusieurs photos ou qui comptent beaucoup de prises "
            "(prises = toutes les prises du moment, dont celles retirées avant) :")
        for row in table:
            out(f"  - {row['start']} → {row['end'][11:]} : {row['takes']} prises ({row['removed_before']} retirées avant), "
                f"à la main {row['ref']}, album automatique {row['auto']}")
    merged = merged_moments(cands, ref_ids)
    if merged:
        out(f"Moments où la sélection à la main garde plusieurs photos ({len(merged)}) — "
            "trop regroupés pour elle, à comparer à MOMENT_COS / MOMENT_GAP :")
        for m in merged:
            out(f"  - {', '.join(by_id[i].name for i in m['ids'])} : {m['span_min']:.0f} min d'écart, "
                f"ressemblance la plus faible {m['min_cos']}")
    return {"exact": exact, "same_moment": same_moment, "missing": missing, "extra": extra, "merged": merged}


def moments_table(moments: list, ref_ids: list[str], auto_ids: list[str], strong: int = 5) -> list[dict]:
    """Les moments (voir `.cache/moments.json`) où la sélection à la main garde 2 photos ou plus, ou qui ont donné au
    moins `strong` prises : combien de prises, combien gardées à la main, combien dans l'album automatique."""
    out = []
    for m in moments if isinstance(moments, list) else []:
        ref = [i for i in m.get("ids", []) if i in set(ref_ids)]
        auto = [i for i in m.get("ids", []) if i in set(auto_ids)]
        if len(ref) >= 2 or (m.get("takes", 0) >= strong and (ref or auto)):
            out.append({"start": m["start"], "end": m["end"], "takes": m["takes"],
                        "removed_before": m.get("removed_before", 0), "ref": len(ref), "auto": len(auto)})
    return out


def _why(record: dict | None) -> str:
    """« — écartée : sans intérêt (scène Rue, écart +0,031, visage 1,2 %) » : le dossier d'une photo, en chiffres."""
    if not record:
        return ""
    bits = [record.get("scene") and f"scène {record['scene']}"]
    if record.get("why") == "sans intérêt" and record.get("gap") is not None:
        bits.append(f"écart {record['gap']:+.3f}")
    if record.get("face") is not None:
        bits.append(f"visage {record['face']:.1%}")
    if record.get("score") is not None:
        bits.append(f"note {record['score']:.2f}")
    if record.get("twin"):
        bits.append("ressemble à {} ({})".format(*record["twin"][:2]))
    if record.get("blocked"):
        bits.append("plafonds : " + ", ".join(record["blocked"]))
    bits = [b for b in bits if b]
    return f" — {record.get('why', '?')}" + (f" ({', '.join(bits)})" if bits else "")


def merged_moments(cands: list[Candidate], ref_ids: list[str]) -> list[dict]:
    """Moments (voir `find_moments`, déjà calculés) où la sélection à la main garde au moins deux photos :
    leur écart de temps et leur ressemblance la plus faible disent de combien il faudrait resserrer les seuils."""
    vecs = scoring.Vectors(cands)
    index = {c.file_id: k for k, c in enumerate(cands)}
    by_moment: dict = {}
    for i in ref_ids:
        if i in index:
            by_moment.setdefault(cands[index[i]].meta["moment"], []).append(i)
    out = []
    for ids in by_moment.values():
        if len(ids) < 2:
            continue
        times = [cands[index[i]].when for i in ids]
        sims = [s for a, b in itertools.combinations(ids, 2) if (s := vecs.cos(index[a], index[b]))]
        out.append({"ids": ids, "span_min": (max(times) - min(times)).total_seconds() / 60,
                    "min_cos": round(min(s[0] for s in sims), 3) if sims else None})
    return sorted(out, key=lambda m: m["span_min"])


MIN_PAIRS = 5   # en dessous, on ne dit rien d'un critère
TAKE_FEATURES = ("frontal", "eyes", "smile", "emotion", "emotion_l", "sharpness", "beauty", "beauty_l", "tidy",
                 "light", "eye_level", "face_seen", "interaction", "moment")


def _take_row(c: Candidate) -> dict[str, float]:
    """Les critères de la prise ; un critère non mesuré est absent (jamais un 0,5 qui ferait gagner l'autre)."""
    row = {k: float(c.parts[k]) for k in TAKE_FEATURES if k in c.parts}
    row["face"] = min(1.0, c.meta.get("family_area", 0.0) / FACE_FULL)
    row["people"] = min(1.0, len(c.labels) / 3.0)
    row["score"] = c.score
    row["take_value"] = take_value(c)
    return row


def takes_report(plan_paths: list[Path], workdir: Path, out=print, min_pairs: int = MIN_PAIRS) -> dict:
    """Parmi les prises d'un même moment, laquelle la famille a-t-elle gardée ? Pour chaque critère :
    part des paires (prise gardée, prise retirée en « doublon » ou échangée) où la prise gardée a la meilleure valeur.
    Puis des poids ajustés sur ces paires (peu de paires : à prendre comme une indication)."""
    pairs, split = [], []
    for path in plan_paths:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
        review = plan.get("review") or {}
        kept = set(review.get("ids", []))
        # « doublon » : une autre prise du moment est gardée ; « échange » : remplacée par une autre prise (⇄)
        dups = {r["id"] for r in review.get("removed", []) if r.get("reason") in ("doublon", "echange")}
        cands = _load_candidates(plan, workdir)
        vecs = scoring.Vectors(cands)
        index = {c.file_id: k for k, c in enumerate(cands)}
        kept_at = [index[i] for i in kept if i in index]
        for group in find_moments(cands):
            pairs += [(w, l) for w in group if w.file_id in kept for l in group if l.file_id in dups]
            if not any(c.file_id in kept for c in group):   # doublon sans photo gardée dans son moment
                for l in (c for c in group if c.file_id in dups):
                    near = min(kept_at, key=lambda k: abs((cands[k].when - l.when).total_seconds()), default=None)
                    if near is not None:
                        sim = vecs.cos(index[l.file_id], near)
                        split.append((abs((cands[near].when - l.when).total_seconds()), sim and round(sim[0], 3)))
    out(f"{len(pairs)} paires (prise gardée, prise retirée en doublon) dans un même moment.")
    if split:
        gaps = sorted(g for g, _ in split)
        sims = sorted(s for _, s in split if s is not None)
        out(f"{len(split)} doublons retirés que je n'ai pas regroupés avec la photo gardée : écart de temps "
            f"médian {gaps[len(gaps) // 2] / 60:.0f} min" + (f", ressemblance médiane {sims[len(sims) // 2]:.2f}" if sims else "") + ".")
    if not pairs:
        return {}
    rows = [(_take_row(w), _take_row(l)) for w, l in pairs]
    paired = lambda k: sum(k in a and k in b for a, b in rows)
    names = [k for k in dict.fromkeys(k for a, _ in rows for k in a) if paired(k) >= min_pairs]
    acc = {}
    for k in names:
        both = [(a, b) for a, b in rows if k in a and k in b]
        wins = sum(1.0 if a[k] > b[k] else 0.5 if a[k] == b[k] else 0.0 for a, b in both)
        acc[k] = wins / len(both)
        out(f"  {k:<12} la prise gardée est la meilleure dans {acc[k]:.0%} des {len(both)} paires")
    presets = {}
    out("Réglages du choix de la prise (part des paires où la prise gardée a la meilleure valeur ; ils ont été "
        "réglés en regardant ces mêmes paires : un chiffre optimiste) :")
    for label, weights in TAKE_PRESETS.items():
        wins = sum(1.0 if take_value(w, weights) > take_value(l, weights) else
                   0.5 if take_value(w, weights) == take_value(l, weights) else 0.0 for w, l in pairs)
        presets[label] = wins / len(pairs)
        out(f"  {label:<18} {presets[label]:.0%}")
    from . import taste

    # les critères mesurés pour presque toutes les paires (une prise sans œil précis n'en a pas), sur les paires qui les ont
    fit = [k for k in names if k not in ("score", "take_value") and paired(k) >= 0.8 * len(rows)]
    usable = [(a, b) for a, b in rows if all(k in a and k in b for k in fit)]
    if len(usable) < min_pairs or not fit:
        return {"pairs": len(pairs), "accuracy": acc, "presets": presets, "weights": {}}
    diffs = np.array([[a[k] - b[k] for k in fit] for a, b in usable])

    def fitted(rows: np.ndarray):
        scale = np.sqrt((rows ** 2).mean(axis=0)) + 1e-9   # même échelle pour tous : on compare des poids, pas des unités
        X = np.vstack([rows / scale, -rows / scale])
        y = np.array([1.0] * len(rows) + [0.0] * len(rows))
        return taste.train(X, y, l2=3.0, lr=None), scale

    model, scale = fitted(diffs)
    weights = dict(zip(fit, (round(float(w), 2) for w in model.w)))
    out(f"Poids ajustés sur {len(usable)} paires, critères à la même échelle (plus c'est grand, plus le critère fait choisir la prise gardée) :")
    out("  " + ", ".join(f"{k} {w:+.2f}" for k, w in sorted(weights.items(), key=lambda kv: -abs(kv[1]))))
    # le vrai chiffre : chaque paire jugée par des poids ajustés sans elle
    hits = 0.0
    for i in range(len(diffs)):
        m, sc = fitted(np.delete(diffs, i, axis=0))
        z = float((diffs[i] / sc) @ m.w + m.b)
        hits += 1.0 if z > 0 else 0.5 if z == 0 else 0.0
    loo = hits / len(diffs)
    out(f"Ces poids, essayés sur chaque paire mise de côté (ajustés sans elle) : la prise gardée est trouvée dans {loo:.0%} des cas.")
    return {"pairs": len(pairs), "accuracy": acc, "presets": presets, "weights": weights, "held_out": loo}


# ---- les paires presque pareilles et ce que la famille en a fait --------------------------------
def pairs_report(plan_paths: list[Path], workdir: Path, out=print, max_gap: int = 6 * 3600, min_cos: float = 0.70) -> list[dict]:
    """Chaque paire de photos qui se ressemblent (ressemblance ≥ `min_cos`, moins de 6 h d'écart) dont l'une au moins
    est gardée : la famille garde les deux, ou retire l'autre en « doublon » ? Pour régler ce qui compte comme
    « la même pose » : ressemblance de chaque œil, écart de dhash et de temps, même plan, mêmes personnes."""
    rows = []
    for path in plan_paths:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
        review = plan.get("review") or {}
        kept = set(review.get("ids", []))
        removed = {r["id"]: r.get("reason", "") for r in review.get("removed", [])}
        role = lambda i: ("gardée" if i in kept else {"doublon": "doublon", "echange": "échange"}.get(removed.get(i), "retirée")
                          if i in removed else "réserve")
        cands = _load_candidates(plan, workdir)
        vecs = scoring.Vectors(cands)
        find_moments(cands)
        for i, a in enumerate(cands):
            for j in range(i + 1, len(cands)):
                b = cands[j]
                roles = sorted((role(a.file_id), role(b.file_id)))
                if "gardée" not in roles or roles[1] not in ("gardée", "doublon", "échange"):
                    continue
                gap = abs((a.when - b.when).total_seconds())
                sim = vecs.cos(i, j)
                if gap > max_gap or not sim or sim[0] < min_cos:
                    continue
                both = lambda has, mat: round(float(mat[i] @ mat[j]), 3) if has[i] and has[j] else None
                rows.append({
                    "album": plan.get("album_name", ""), "a": a.name, "b": b.name, "roles": "+".join(roles),
                    "cos_l": both(vecs.hasL, vecs.L), "cos_b": both(vecs.hasB, vecs.B),
                    "bits": scoring.hamming_hex(a.feat["dhash"], b.feat["dhash"]), "gap_s": int(gap),
                    "same_moment": a.meta["moment"] == b.meta["moment"],
                    "same_shot": a.meta.get("shot") == b.meta.get("shot"),
                    "same_people": a.labels == b.labels, "same_note": a.note == b.note,
                })
    out(f"{len(rows)} paires de photos qui se ressemblent, dont au moins une gardée.")
    for kind in dict.fromkeys(r["roles"] for r in rows):
        group = [r for r in rows if r["roles"] == kind]
        cos = sorted(r["cos_l"] if r["cos_l"] is not None else r["cos_b"] for r in group)
        out(f"{kind} : {len(group)} paires ; ressemblance médiane {cos[len(cos) // 2]:.2f} "
            f"(de {cos[0]:.2f} à {cos[-1]:.2f}) ; même moment {sum(r['same_moment'] for r in group)}, "
            f"même plan {sum(r['same_shot'] for r in group)}, mêmes personnes {sum(r['same_people'] for r in group)}.")
    for r in sorted(rows, key=lambda r: (r["roles"], -(r["cos_l"] or r["cos_b"] or 0))):
        flags = "".join(f for f, on in (("M", r["same_moment"]), ("P", r["same_shot"]), ("G", r["same_people"]),
                                         ("S", r["same_note"])) if on)
        out(f"  {r['roles']:<16} {r['a']} {r['b']}  L={r['cos_l']} B={r['cos_b']}  dhash={r['bits']}  "
            f"{r['gap_s']} s  {flags}")
    out("(M = même moment, P = même plan, G = mêmes personnes, S = même scène)")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Outils du tri de photographe.")
    parser.add_argument("--compare", nargs=2, metavar=("AUTO", "MAIN"),
                        help="plan.json d'un album automatique et plan.json d'une sélection à la main")
    parser.add_argument("--takes", nargs="+", metavar="PLAN",
                        help="plan.json relus : quelle prise du moment la famille garde-t-elle ?")
    parser.add_argument("--pairs", nargs="+", metavar="PLAN",
                        help="plan.json relus : parmi les photos qui se ressemblent, lesquelles garde-t-elle ?")
    parser.add_argument("--workdir", default=".", help="dossier d'Albumia (celui qui contient .cache)")
    args = parser.parse_args(argv)
    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]), Path(args.workdir))
    elif args.takes:
        takes_report([Path(p) for p in args.takes], Path(args.workdir))
    elif args.pairs:
        pairs_report([Path(p) for p in args.pairs], Path(args.workdir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
