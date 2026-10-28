"""Le tri d'un photographe : un moment, une photo ; des chapitres ; une couverture.

Règles relevées sur la sélection « VDEF » faite à la main pour la famille (octobre 2025), puis
confirmées par ses relectures (19 % de retraits contre 29 à 43 % pour les albums automatiques) :

1. Un moment = des photos prises à moins de 10 minutes d'écart, de scène voisine. On en garde une
   (deux si le moment est fort : au moins 5 prises, et une autre personne ou un autre plan).
2. Dans un moment, on garde la prise la plus naturelle : l'enfant de face, les yeux ouverts, le
   visage bien visible, plusieurs de la famille ; la netteté ne départage qu'à la fin.
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
TAKE_WEIGHTS = {"frontal": 0.18, "eyes": 0.12, "smile": 0.08, "emotion": 0.10, "face": 0.10,
                "people": 0.08, "score": 0.26, "sharp": 0.08}
FACE_FULL = 0.02                      # visage de la famille à 2 % de l'image : crédit plein (plus gros : pas mieux)
SHARP_FULL = 0.6                      # au-delà de cette netteté, plus de crédit : un filtre, pas un but

# ---- ce qui n'a pas sa place dans un album de famille ---------------------------------------
TINY_SUBJECT = 0.015                  # visage de la famille sous 1,5 % de l'image : sujet minuscule
TINY_PENALTY = 0.03                   # un simple départage : il a remis lui-même une vue large aux enfants minuscules
ADULT_ALONE_PENALTY = 0.12
WIDE_SHARE = 0.25                     # plans larges : un quart de l'album au plus
SCENE_SHARE = 0.08                    # une même scène (piscine…) : 8 % de l'album, 2 photos au moins
SCENE_CAP_MIN = 2

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


def take_value(c: Candidate) -> float:
    """Quelle prise garder dans un moment : l'enfant de face, les yeux ouverts, un visage de la
    famille bien visible, plusieurs de la famille, ce qui plaît ; la netteté ne départage qu'à la fin."""
    w, p = TAKE_WEIGHTS, c.parts
    emotion = p.get("emotion_l", p.get("emotion", 0.5))
    face = min(1.0, c.meta.get("family_area", 0.0) / FACE_FULL)
    people = min(1.0, len(c.labels) / 3.0)
    sharp = min(1.0, p.get("sharpness", 0.5) / SHARP_FULL)
    return (w["frontal"] * p.get("frontal", 0.5) + w["eyes"] * p.get("eyes", 0.5)
            + w["smile"] * p.get("smile", 0.5) + w["emotion"] * emotion + w["face"] * face
            + w["people"] * people + w["score"] * c.score + w["sharp"] * sharp)


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

    Une photo gardée à la relecture est toujours la photo de son moment."""
    picks: list[Candidate] = []
    others: dict[int, list[Candidate]] = {}
    for k, group in enumerate(groups):
        ranked = sorted(group, key=take_value, reverse=True)
        keep = [c for c in ranked if c.file_id in pinned]
        if not keep:
            keep = [ranked[0]]
            if len(ranked) >= STRONG_TAKES:
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
                protected=frozenset()) -> list[Candidate]:
    """Chaque chapitre a un plan large qui situe le lieu : on en ajoute un quand il manque.

    C'est la photo d'ouverture d'un album de photographe (le Kotel en grand, la salle de jeux) ; il en
    a remis trois lui-même à la relecture. Le plan large vient de la même époque, sans refaire un moment
    ni une pose déjà là. L'album ne dépasse jamais `count` : la photo la plus faible d'un chapitre
    bien fourni lui cède la place (jamais une photo `protected`)."""
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
    """« Mardi 7 octobre · Piscine » : le jour, et la scène qui revient le plus souvent."""
    label = date_label(min(c.when for c in group), max(c.when for c in group))
    counts = Counter(c.note for c in group if c.note and c.note not in scenes.GENERIC)
    if counts:
        scene, n = counts.most_common(1)[0]
        if n >= 2:
            return f"{label} · {scene}"
    return label


def pick_cover(selected: list[Candidate], family: list[str] | None = None) -> str | None:
    """La photo de couverture : du monde, de face, un sourire, belle et nette."""
    if not selected:
        return None
    members = set(family or ())

    def value(c: Candidate) -> float:
        p = c.parts
        crowd = len(c.labels & members) if members else len(c.labels)
        return (0.30 * min(crowd, 4) / 4 + 0.15 * p.get("frontal", 0.5) + 0.15 * p.get("smile", 0.5)
                + 0.15 * p.get("beauty_l", p.get("beauty", 0.5)) + 0.10 * min(1.0, p.get("sharpness", 0.5) / SHARP_FULL)
                + 0.15 * c.score)

    return max(selected, key=value).file_id


def layout(selected: list[Candidate], reserve: list[Candidate], family: list[str] | None = None) -> dict:
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
    return {"chapters": chapters, "cover": pick_cover(selected, family), "of": of}


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
    out = []
    for item in plan.get("photos", []) + plan.get("reserve", []):
        fid = item["id"]
        feat = (cache.get(fid) or {}).get("feat") or {"dhash": "0" * 16, "w": 1, "h": 1, "faces": []}
        c = Candidate(fid, item["name"], datetime.fromisoformat(item["date"]), feat, set(item.get("people", [])))
        c.score, c.parts, c.note = item.get("score", 0.0), item.get("parts", {}), item.get("note", "")
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
    name = lambda i: f"{by_id[i].name} ({by_id[i].when:%d/%m %H:%M})" if i in by_id else i
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
    return {"exact": exact, "same_moment": same_moment, "missing": missing, "extra": extra}


TAKE_FEATURES = ("frontal", "eyes", "smile", "emotion", "emotion_l", "sharpness", "beauty", "beauty_l", "tidy",
                 "light", "eye_level", "face_seen", "interaction", "moment")


def _take_row(c: Candidate) -> dict[str, float]:
    row = {k: float(c.parts.get(k, 0.5)) for k in TAKE_FEATURES}
    row["face"] = min(1.0, c.meta.get("family_area", 0.0) / FACE_FULL)
    row["people"] = min(1.0, len(c.labels) / 3.0)
    row["score"] = c.score
    row["take_value"] = take_value(c)
    return row


def takes_report(plan_paths: list[Path], workdir: Path, out=print) -> dict:
    """Parmi les prises d'un même moment, laquelle la famille a-t-elle gardée ? Pour chaque critère :
    part des paires (prise gardée, prise retirée « doublon ») où la prise gardée a la meilleure valeur.
    Puis des poids ajustés sur ces paires (peu de paires : à prendre comme une indication)."""
    pairs = []
    for path in plan_paths:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
        review = plan.get("review") or {}
        kept = set(review.get("ids", []))
        dups = {r["id"] for r in review.get("removed", []) if r.get("reason") == "doublon"}
        for group in find_moments(_load_candidates(plan, workdir)):
            pairs += [(w, l) for w in group if w.file_id in kept for l in group if l.file_id in dups]
    out(f"{len(pairs)} paires (prise gardée, prise retirée en doublon) dans un même moment.")
    if not pairs:
        return {}
    rows = [(_take_row(w), _take_row(l)) for w, l in pairs]
    names = list(rows[0][0])
    acc = {}
    for k in names:
        wins = sum(1.0 if a[k] > b[k] else 0.5 if a[k] == b[k] else 0.0 for a, b in rows)
        acc[k] = wins / len(rows)
        out(f"  {k:<12} la prise gardée est la meilleure dans {acc[k]:.0%} des paires")
    from . import taste

    fit = [k for k in names if k not in ("score", "take_value")]
    X = np.array([[a[k] - b[k] for k in fit] for a, b in rows])
    X = np.vstack([X, -X])
    y = np.array([1.0] * len(rows) + [0.0] * len(rows))
    model = taste.train(X, y, l2=3.0, lr=None)
    weights = dict(zip(fit, (round(float(w), 2) for w in model.w)))
    out("Poids ajustés (plus c'est grand, plus le critère fait choisir la prise gardée) :")
    out("  " + ", ".join(f"{k} {w:+.2f}" for k, w in sorted(weights.items(), key=lambda kv: -abs(kv[1]))))
    return {"pairs": len(pairs), "accuracy": acc, "weights": weights}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Outils du tri de photographe.")
    parser.add_argument("--compare", nargs=2, metavar=("AUTO", "MAIN"),
                        help="plan.json d'un album automatique et plan.json d'une sélection à la main")
    parser.add_argument("--takes", nargs="+", metavar="PLAN",
                        help="plan.json relus : quelle prise du moment la famille garde-t-elle ?")
    parser.add_argument("--workdir", default=".", help="dossier d'Albumia (celui qui contient .cache)")
    args = parser.parse_args(argv)
    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]), Path(args.workdir))
    elif args.takes:
        takes_report([Path(p) for p in args.takes], Path(args.workdir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
