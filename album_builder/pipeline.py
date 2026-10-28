"""Préparation d'un album, indépendante de l'interface (ligne de commande ou page web).

`build_plan` lit Drive, analyse les miniatures en local et écrit un plan à relire.
`apply_plan` crée ensuite le dossier d'album (raccourcis) dans Drive.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from . import aesthetic, compose, curate, intent, people, scenes, scoring, taste, vision
from .errors import AlbumError, Cancelled
from .report import slugify, write_plan

STEPS = 5


def date_references(ref_photos: dict, known: dict, cache: dict, min_sim: float = 0.85) -> dict:
    """Retrouve la date des photos de référence découpées dans des photos déjà analysées.

    Le visage d'une référence nommée depuis « Qui est sur tes photos ? » est presque identique à
    un visage du cache : on lui donne la date de cette photo.
    """
    import numpy as np

    todo = [(n, photo, e) for n, v in ref_photos.items() for photo, e in v if photo not in known.get(n, {})]
    if not todo:
        return {}
    embs, whens = [], []
    for entry in cache.values():
        if not entry.get("when"):
            continue
        for face in entry["feat"].get("faces", []):
            if "emb" in face:
                embs.append(face["emb"])
                whens.append(entry["when"])
    if not embs:
        return {}
    mat = np.asarray(embs, dtype=float)
    out: dict = {}
    for name, photo, e in todo:
        sims = mat @ np.asarray(e, dtype=float)
        i = int(np.argmax(sims))
        if sims[i] >= min_sim:
            out.setdefault(name, {})[photo] = date.fromisoformat(whens[i])
    return out


def judge_photos(files, cache, thumbs_dir: Path, editor_path: Path, model_dir: Path, rep, cancel) -> dict:
    """Composition (OpenCV) et œil d'éditeur (CLIP) sur chaque miniature, gardés en cache."""
    editor = _load_cache(editor_path)
    ready = [f for f in files if f.id in cache and (thumbs_dir / f"{f.id}.jpg").exists()]

    def stale(f, key, version):
        e = editor.get(f.id, {})
        return (e.get("mod"), e.get(key)) != (f.modified, version)

    need_comp = [f for f in ready if stale(f, "cv", compose.VERSION)]
    need_clip = [f for f in ready if stale(f, "v", aesthetic.VERSION)]
    if not need_comp and not need_clip:
        return editor
    judge = aesthetic.load_judge(model_dir, rep.note) if need_clip else None
    todo = {f.id: f for f in need_comp + (need_clip if judge else [])}
    if not todo:
        return editor
    rep.note(f"Regard d'éditeur sur {len(todo)} photos…")
    for i, f in enumerate(todo.values(), 1):
        if cancel.is_set():
            break
        img = vision.imread(thumbs_dir / f"{f.id}.jpg")
        if img is None:
            continue
        e = editor.get(f.id, {})
        if e.get("mod") != f.modified:
            e = {}
        e["mod"] = f.modified
        if stale(f, "cv", compose.VERSION) or "comp" not in e:
            faces = scoring.real_faces(cache[f.id]["feat"])
            boxes = [x["box"] for x in faces]
            scale = img.shape[1] / float(cache[f.id]["feat"]["w"] or img.shape[1])
            boxes = [[v * scale for v in b] for b in boxes]
            e["comp"], e["cv"] = compose.measure(img, boxes), compose.VERSION
        if judge and (e.get("v") != aesthetic.VERSION or "vec" not in e):
            e["scores"], e["vec"] = judge.judge_with_vector(img)
            e["scene"], e["v"] = judge.scene(e["vec"]), aesthetic.VERSION
        editor[f.id] = e
        if i % 200 == 0:
            _save_cache(editor_path, editor)
    _save_cache(editor_path, editor)
    if cancel.is_set():
        raise Cancelled()
    return editor


def look_closer(ids, thumbs_dir: Path, editor_path: Path, editor: dict, model_dir: Path, rep, cancel) -> None:
    """Œil plus précis (ViT-L/14) sur ces photos : une seule fois, gardé en cache."""
    todo = [i for i in dict.fromkeys(ids) if i in editor and editor[i].get("vl") != aesthetic.LARGE_VERSION
            and (thumbs_dir / f"{i}.jpg").exists()]
    if not todo:
        return
    eye = aesthetic.load_large(model_dir, rep.note)
    if eye is None:
        return
    minutes = max(1, round(len(todo) * 2 / 60))
    rep.note(f"Regard plus précis sur {len(todo)} photos (une seule fois, environ {minutes} min)…")
    for n, fid in enumerate(todo, 1):
        if cancel.is_set():
            break
        img = vision.imread(thumbs_dir / f"{fid}.jpg")
        if img is not None:
            editor[fid]["vec_l"], editor[fid]["vl"] = eye.vector(img), aesthetic.LARGE_VERSION
        rep.advance(n, len(todo))
        if n % 50 == 0:
            _save_cache(editor_path, editor)
    _save_cache(editor_path, editor)
    if cancel.is_set():
        raise Cancelled()


def is_selfie(faces, per_face, births, when, feat) -> bool:
    """Un adulte de la famille en très gros plan : selfie pris à bout de bras."""
    area = float(feat["w"] * feat["h"]) or 1.0
    for face, label in zip(faces, per_face):
        adult = label and (label not in births or scoring.months_between(births[label], when) >= 18 * 12)
        if adult and face["box"][2] * face["box"][3] / area >= scoring.SELFIE_FACE_AREA:
            return True
    return False


ADULT_ALONE_MIN_AREA = 0.01   # visage d'au moins 1 % de l'image : « le » sujet de la photo


def is_tiny(c) -> bool:
    """Visages minuscules dans une scène large (enfants au loin sur une pelouse), sauf un lieu qui compte."""
    area = c.meta.get("subject_area", 0.0)
    return 0.0 < area < curate.TINY_SUBJECT and scenes.KIND.get(c.note) != scenes.PLACE


def is_adult_alone(faces, per_face, births, when, feat) -> bool:
    """Un seul visage bien visible, celui d'un adulte de la famille (papa devant son téléphone).
    Sans date de naissance connue pour personne, on ne sait pas qui est un adulte : jamais."""
    if not births:
        return False
    area = float(feat["w"] * feat["h"]) or 1.0
    big = [label for x, label in zip(faces, per_face) if x["box"][2] * x["box"][3] / area >= ADULT_ALONE_MIN_AREA]
    if len(big) != 1 or not big[0]:
        return False
    return big[0] not in births or scoring.months_between(births[big[0]], when) >= 18 * 12


# ce qui n'est pas un souvenir de famille, même avec un visage dessus (affiche, écran, publicité)
OBJECT_PROMPTS = [
    "an advertisement poster on a wall", "a photo of a phone screen or a computer screen",
    "a banner or a sign with text", "a supermarket shelf with products", "a shop or a restaurant sign",
    "an empty room or a corridor", "a city street with buildings and a tram", "a fridge door with magnets",
    "a restaurant menu", "a projection screen in a hall", "a building exterior", "a hotel lobby or a library",
]
PEOPLE_PROMPTS = ["a photo of a child", "a family photo", "a baby", "children playing", "a portrait of a person"]
OBJECT_MARGIN = 0.02


def object_photos(files, editor, model_dir: Path, rep) -> set[str]:
    """Photos qui ressemblent nettement plus à un objet ou un décor qu'à des gens."""
    vecs = {f.id: (editor.get(f.id) or {}).get("vec") for f in files}
    vecs = {k: v for k, v in vecs.items() if v}
    if not vecs:
        return set()
    both = aesthetic.theme_vectors(model_dir, OBJECT_PROMPTS + PEOPLE_PROMPTS, rep.note)
    obj, ppl = both[:len(OBJECT_PROMPTS)], both[len(OBJECT_PROMPTS):]
    if any(v is None for v in obj + ppl):
        return set()
    ids = list(vecs)
    mat = np.asarray([vecs[i] for i in ids], dtype=float)
    gap = (mat @ np.vstack(obj).T).max(axis=1) - (mat @ np.vstack(ppl).T).max(axis=1)
    return {fid for fid, g in zip(ids, gap) if g > OBJECT_MARGIN}


def previous_review(out_dir: Path) -> tuple[set[str], set[str]]:
    """(photos gardées, photos retirées pour « pas les bonnes personnes ») à la dernière relecture."""
    try:
        review = json.loads((out_dir / "plan.json").read_text(encoding="utf-8")).get("review") or {}
    except (OSError, ValueError):
        return set(), set()
    removed = {r["id"] for r in review.get("removed", []) if r.get("reason") == "personnes"}
    return set(review.get("ids", [])) - removed, removed


SHORTLIST_FACTOR, SHORTLIST_FLOOR, SHORTLIST_CEILING = 6, 120, 500
SHORTLIST_PER_MOMENT, SHORTLIST_PER_DAY = 3, 4


def pick_shortlist(cands, count: int, protected=frozenset()):
    """Les candidates qui méritent le regard précis (plus lent) : les mieux notées, au plus 3 prises par
    moment, 4 par jour au moins pour que chaque journée soit représentée, et les photos protégées."""
    k = min(SHORTLIST_CEILING, max(SHORTLIST_FLOOR, SHORTLIST_FACTOR * count))
    if len(cands) <= k:
        return list(cands)
    ranked = sorted(cands, key=lambda c: c.score, reverse=True)
    rank_in_moment: dict[str, int] = {}
    for group in curate.find_moments(cands):
        for r, c in enumerate(sorted(group, key=lambda c: c.score, reverse=True)):
            rank_in_moment[c.file_id] = r
    keep = {c.file_id for c in [c for c in ranked if rank_in_moment[c.file_id] < SHORTLIST_PER_MOMENT][:k]}
    by_day: dict = {}
    for c in ranked:
        by_day.setdefault(c.when.date(), []).append(c)
    for items in by_day.values():
        keep.update(c.file_id for c in items[:SHORTLIST_PER_DAY])
    keep |= set(protected)
    return [c for c in cands if c.file_id in keep]


def must_missing(must, labels, known, per_face) -> bool:
    """Une personne obligatoire manque. Un enfant qu'Albumia ne sait pas encore reconnaître à cet
    âge (pas de référence proche) peut être n'importe quel visage inconnu de la photo."""
    if any(m in known and m not in labels for m in must):
        return True
    unknown = [m for m in must if m not in known]
    return bool(unknown) and sum(1 for label in per_face if not label) < len(unknown)


MIN_THEMED = 8  # en dessous, un thème strict ne ferait qu'un album minuscule


def album_theme(params, files, editor, model_dir: Path, rep) -> dict[str, tuple[float, bool]]:
    """Note du thème pour chaque photo ; un thème strict trop rare devient une simple préférence."""
    if not params.theme:
        return {}
    label, prompt, strict_kind = intent.THEMES[params.theme][1:4]
    params.theme_strict = params.theme_strict and strict_kind  # une ambiance n'écarte jamais
    prompts = prompt if isinstance(prompt, list) else [prompt]
    vecs = {f.id: (editor.get(f.id) or {}).get("vec") for f in files}
    tvecs = aesthetic.theme_vectors(model_dir, prompts, rep.note) if any(vecs.values()) else [None]
    tvec = None if any(v is None for v in tvecs) else (tvecs[0] if len(tvecs) == 1 else np.vstack(tvecs))
    if tvec is None:
        rep.note(f"Thème « {label} » non pris en compte : l'œil d'éditeur n'est pas disponible.")
        return {}
    margin = intent.THEME_MARGIN.get(params.theme, scoring.THEME_STRICT_MARGIN)
    found = intent.THEME_FOUND.get(params.theme, margin + scoring.THEME_FOUND_EXTRA)
    # une ambiance ne fait que mettre en avant : pas besoin de prouver qu'elle existe
    # (un dossier entièrement « vacances » n'a pas de photos qui ressortent du lot)
    # une fête juive : ses dates suffisent, le thème ne fait que mettre en avant
    if strict_kind and params.theme not in intent.HOLIDAYS and not scoring.theme_found(vecs, tvec, found):
        rep.note(f"Je ne trouve pas de photos « {label} » dans ce dossier : thème ignoré.")
        params.theme_strict = False
        return {}
    not_prompt = intent.THEME_NOT.get(params.theme)
    not_vec = aesthetic.theme_vectors(model_dir, [not_prompt], rep.note)[0] if not_prompt else None
    themed = scoring.theme_scores(vecs, {f.id: f.when.date() for f in files}, tvec,
                                  margin=margin, not_vec=not_vec)
    inside = sum(1 for _, ok in themed.values() if ok)
    if params.theme_strict and inside < min(params.count, MIN_THEMED):
        rep.note(f"Seulement {inside} photo(s) sur le thème « {label} » : je les mets en avant "
                 "sans écarter les autres.")
        params.theme_strict = False
    else:
        rep.note(f"Thème « {label} » : {inside} photos y correspondent.")
    return themed


def parse_folder_id(value: str) -> str:
    value = (value or "").strip()
    if value == "gphotos":  # photos choisies dans Google Photos (voir gphotos.py)
        return value
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", value) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", value)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):
        return value
    raise AlbumError(
        "Ce lien ne ressemble pas à un dossier Google Drive. Ouvre le dossier dans Drive "
        "et copie l'adresse affichée en haut du navigateur.",
        code="bad_folder",
    )


@dataclass
class BuildParams:
    name: str
    folders: list[str]
    count: int = 50
    date_from: datetime | None = None
    date_to: datetime | None = None          # exclue
    must: list[str] = field(default_factory=list)
    bonus: list[str] = field(default_factory=list)
    match_threshold: float = scoring.DEFAULT_MATCH_THRESHOLD
    faces_only: bool = False
    recursive: bool = True
    use_claude: bool = False
    describe: str = ""
    weights: scoring.Weights = field(default_factory=scoring.Weights)
    workers: int = 6
    reserve: int = 60                         # photos de remplacement proposées à la relecture
    theme: str = ""                           # clé de intent.THEMES (plage, anniversaire…)
    theme_strict: bool = False                # seulement les photos du thème
    family: bool = False                      # « en famille » : au moins un des `bonus` par photo


class Reporter:
    """Reçoit l'avancement. La version de base écrit dans la console (ligne de commande)."""

    def step(self, n: int, label: str) -> None:
        print(f"{n}/{STEPS} {label}")

    def advance(self, done: int, total: int) -> None:
        if total and (done % 25 == 0 or done == total):
            print(f"\r    {done}/{total}", end="" if done < total else "\n", flush=True)

    def note(self, message: str) -> None:
        print(f"    {message}")


def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(path)


def refine_with_large_eye(short, examples, editor, model_dir: Path, rep, rate, all_parts, rejected, pinned,
                          liked_scene) -> list:
    """Redécrit les scènes, juge les critères de photographe et note à nouveau les candidates que
    l'œil plus précis a vues ; écarte celles qu'il trouve sans intérêt."""
    ids = {c.file_id for c in short} | set(examples["pos"]) | set(examples["neg"])
    vecs_l = {fid: v["vec_l"] for fid, v in editor.items() if v.get("vec_l") and fid in ids}
    described = scenes.describe_all(vecs_l, model_dir, rep.note, large=True)
    crit = aesthetic.criteria_large(model_dir, rep.note)
    liked = scenes.interest(examples, {k: v[0] for k, v in described.items()}) if described else liked_scene
    out = []
    for c in short:
        vec_l = (editor.get(c.file_id) or {}).get("vec_l")
        if vec_l:
            c.meta["vec_l"] = vec_l
            if crit:
                c.meta["editor_l"] = aesthetic.judge_vector(np.asarray(vec_l, dtype=float), crit)
            if c.file_id in described:
                scene, dull, gap = described[c.file_id]
                c.note = scene
                c.meta["meal"] = scenes.KIND.get(scene) == scenes.MEAL
                c.meta["dull"] = dull or (not scoring.real_faces(c.feat) and gap >= 0)
                if c.meta["dull"] and not c.meta["shield"] and c.file_id not in pinned:
                    rejected["sans intérêt"] += 1
                    continue
            c.meta["tiny"] = is_tiny(c)
            rate(c, liked)
            all_parts[c.file_id] = c.parts
        out.append(c)
    return out


def build_plan(
    client,
    params: BuildParams,
    workdir: Path,
    out_root: Path,
    refs_dir: Path,
    reporter: Reporter | None = None,
    cancel: threading.Event | None = None,
) -> Path:
    """Construit le plan et renvoie le chemin de plan.json. Ne modifie rien dans Drive."""
    import cv2  # import tardif : le reste du module reste importable sans OpenCV

    rep = reporter or Reporter()
    cancel = cancel or threading.Event()
    cache_dir = workdir / ".cache"
    thumbs_dir = cache_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    features_path = cache_dir / "features.json"
    model_dir = workdir / "models"
    folder_ids = [parse_folder_id(x) for x in params.folders]

    rep.step(1, "Recherche des photos dans Drive…")
    files = list(client.iter_images(folder_ids, recursive=params.recursive))
    listed = len(files)
    span = (min(f.when for f in files), max(f.when for f in files)) if files else None
    lo, hi = params.date_from, params.date_to
    files = [f for f in files if (lo is None or f.when >= lo) and (hi is None or f.when < hi)]
    from_modified = sum(1 for f in files if f.date_source == "modified")
    rep.note(f"{listed} images trouvées, {len(files)} dans la période.")
    if from_modified:
        rep.note(f"{from_modified} photo(s) sans date de prise de vue : date du fichier utilisée.")
    if not listed:
        raise AlbumError("Je n'ai trouvé aucune photo dans ce dossier Drive.", code="empty_folder")
    if not files:
        raise AlbumError(
            f"Le dossier contient {listed} photos, mais aucune dans la période demandée. "
            f"Elles vont du {span[0]:%d/%m/%Y} au {span[1]:%d/%m/%Y} : ajoute l'année "
            "dans ta description (par exemple « Souccot 2025 ») ou corrige les dates.",
            code="empty_period",
        )

    wanted = list(dict.fromkeys(params.must + params.bonus))
    refs: dict[str, list[float]] = {}
    ref_photos: dict[str, list[tuple[str, list[float]]]] = {}
    if wanted:
        rep.step(2, "Apprentissage des visages…")
        ref_photos = vision.reference_embeddings(vision.get_engine(model_dir), refs_dir)
        refs = {n: vision.mean_embedding([e for _, e in v]) for n, v in ref_photos.items()}
        missing = [w for w in wanted if w not in refs]
        if params.family and [w for w in params.bonus if w in missing]:
            skipped = [w for w in params.bonus if w in missing]
            rep.note(f"Pas encore de photo de référence pour {', '.join(skipped)} : "
                     "je ne peux pas le reconnaître, l'album se fait avec les autres.")
            params.bonus = [w for w in params.bonus if w not in missing]
            wanted = list(dict.fromkeys(params.must + params.bonus))
            missing = [w for w in missing if w in params.must]
        if missing:
            raise AlbumError(
                f"Je n'ai pas de photo de référence exploitable pour : {', '.join(missing)}. "
                "Ajoute une photo où l'on voit bien son visage.",
                code="missing_refs",
            )
    else:
        rep.step(2, "Aucune personne ciblée : choix sur la qualité des photos.")

    rep.step(3, "Analyse des photos sur ton ordinateur…")
    cache = _load_cache(features_path)
    todo = [
        f for f in files
        if cache.get(f.id, {}).get("mod") != f.modified
        or cache[f.id].get("v") != vision.FEATURES_VERSION
        or not (thumbs_dir / f"{f.id}.jpg").exists()
    ]
    rep.note(f"{len(files) - len(todo)} déjà analysées, {len(todo)} à analyser.")

    def process(f):
        if cancel.is_set():
            return f.id, None
        data = client.fetch_thumbnail(f, 1024)
        if data is None:
            return f.id, None
        img = vision.decode(data)
        if img is None:
            return f.id, None
        feat = vision.analyze(img, vision.get_engine(model_dir))
        small = vision.resize_max(img, 640)
        vision.imwrite(thumbs_dir / f"{f.id}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return f.id, feat

    by_id = {f.id: f for f in files}
    failed = done = 0
    rep.advance(0, len(todo))
    with ThreadPoolExecutor(max_workers=max(1, params.workers)) as pool:
        futures = [pool.submit(process, f) for f in todo]
        try:
            for fut in as_completed(futures):
                fid, feat = fut.result()
                done += 1
                if feat is None:
                    if not cancel.is_set():
                        failed += 1
                else:
                    cache[fid] = {"mod": by_id[fid].modified, "v": vision.FEATURES_VERSION, "feat": feat}
                rep.advance(done, len(todo))
                if done % 200 == 0:
                    _save_cache(features_path, cache)
                if cancel.is_set():
                    break
        finally:
            if cancel.is_set():
                for fut in futures:
                    fut.cancel()
            _save_cache(features_path, cache)  # le travail déjà fait sert au prochain essai
    if cancel.is_set():
        raise Cancelled()
    if failed:
        rep.note(f"{failed} photo(s) sans aperçu Drive, ignorées.")
    editor_path = workdir / ".cache" / "editor.json"
    editor = judge_photos(files, cache, thumbs_dir, editor_path, model_dir, rep, cancel)

    rep.step(4, "Choix des meilleures photos…")
    dated = False
    for f in files:
        if f.id in cache and cache[f.id].get("when") != f.when.date().isoformat():
            cache[f.id]["when"] = f.when.date().isoformat()
            dated = True
    if dated:  # « Qui est sur tes photos ? » regroupe les visages par période
        _save_cache(features_path, cache)
    dates = people.photo_dates(refs_dir)
    found = date_references(ref_photos, dates, cache)
    if found:
        people.set_photo_dates(refs_dir, found)
        for name, photos in found.items():
            dates.setdefault(name, {}).update(photos)
        _save_cache(features_path, cache)
    births = people.births(refs_dir)
    ref_sets = {n: [(e, dates.get(n, {}).get(photo)) for photo, e in v] for n, v in ref_photos.items()}
    refs_by_month: dict = {}
    themed = album_theme(params, files, editor, model_dir, rep)
    # 1re passe, rapide, sur toutes les photos : l'œil habituel (ViT-B/32)
    described = scenes.describe_all({f.id: (editor.get(f.id) or {}).get("vec") for f in files},
                                    model_dir, rep.note)
    objects = object_photos(files, editor, model_dir, rep)
    examples = taste.load_examples(workdir)
    liked_scene = scenes.interest(examples, {k: v[0] for k, v in described.items()})
    all_parts: dict[str, dict] = {}
    cands: list[scoring.Candidate] = []
    reasons = taste.excluded_reasons(workdir)
    banned = set(reasons)
    kept = set(examples["pos"])  # gardées à une relecture : jamais écartées par ressemblance
    # relecture précédente de ce même album : les photos gardées reviennent, celles retirées
    # pour « pas les bonnes personnes » non (elles restent possibles dans d'autres albums)
    pinned, banned_here = previous_review(out_root / slugify(params.name))
    banned |= banned_here
    # les photos retirées « pas belle » / « goût », par jour : leur quasi-jumelle (même série) ne revient
    # pas à leur place. Pas pour un « doublon » : sa jumelle est justement la photo qu'on a gardée.
    by_id = {f.id: f for f in files}
    banned_vecs: dict = {}
    for fid, why in reasons.items():
        if why == "doublon":
            continue
        vec = (editor.get(fid) or {}).get("vec")
        if vec and fid in by_id:
            banned_vecs.setdefault(by_id[fid].when.date(), []).append((by_id[fid].name, vec))

    def rate(c, liked):
        """Note de base d'une candidate : critères, bonus des personnes, intérêt de la scène."""
        c.local_score, c.parts = scoring.local_score(
            c.feat, c.labels, params.bonus, params.weights, c.meta["per_face"], wanted, c.meta)
        if c.note:
            c.parts["scene"] = liked.get(c.note, 0.5)
            c.local_score = (1 - scenes.WEIGHT) * c.local_score + scenes.WEIGHT * c.parts["scene"]
        c.score = c.local_score

    rejected = {"retirée avant": 0, "comme une photo retirée": 0, "sans analyse": 0, "pas une photo": 0, "pas pour un album": 0, "objet ou décor": 0, "sans intérêt": 0, "trop petite pour l'impression": 0, "floue": 0, "personne absente": 0,
                "personne trop petite": 0, "sans visage": 0, "personne de la famille": 0, "hors thème": 0}
    for f in files:
        entry = cache.get(f.id)
        if f.id in banned:
            rejected["retirée avant"] += 1
            continue
        if not entry:
            rejected["sans analyse"] += 1
            continue
        feat = entry["feat"]
        faces = scoring.real_faces(feat)
        # personne ne peut apparaître sur une photo prise avant sa naissance
        allowed = {n for n in refs if n not in births or births[n] <= f.when.date()}
        month = (f.when.year, f.when.month)
        if month not in refs_by_month:
            refs_by_month[month] = scoring.refs_at(ref_sets, births, f.when.date().replace(day=15))
        per_face = scoring.face_labels(faces, refs_by_month[month], params.match_threshold, allowed=allowed)
        labels = {label for label in per_face if label}
        kind = scoring.photo_kind(f.name, f.camera, f.width, f.height)
        c = scoring.Candidate(f.id, f.name, f.when, feat, labels)
        focus = [x for x, label in zip(faces, per_face) if label in wanted] if wanted else faces
        c.meta = {"starred": f.starred, "kind": kind, "width": f.width, "height": f.height,
                  "editor": (editor.get(f.id) or {}).get("scores"),
                  "comp": (editor.get(f.id) or {}).get("comp"),
                  "vec": (editor.get(f.id) or {}).get("vec"),
                  "scene": (editor.get(f.id) or {}).get("scene"),
                  "shot": scoring.shot_type(feat, focus or faces),
                  "selfie": is_selfie(faces, per_face, births, f.when.date(), feat)}
        c.meta["adult_only"] = is_adult_alone(faces, per_face, births, f.when.date(), feat)
        c.meta["selfie"] = c.meta["selfie"] or c.meta["adult_only"]
        scene, dull, gap = described.get(f.id, ("", False, -1.0))
        dull = dull or (not faces and gap >= 0)  # sans visage, le moindre doute suffit
        c.meta["meal"] = scenes.KIND.get(scene) == scenes.MEAL
        c.meta["per_face"] = per_face
        c.meta["dull"] = dull
        c.note = scene
        rate(c, liked_scene)
        all_parts[f.id] = c.parts
        must_faces = [x for x, label in zip(faces, per_face) if label in params.must]
        area = float(feat["w"] * feat["h"]) or 1.0
        family_close = any(label in (wanted or refs) and x["box"][2] * x["box"][3] / area >= scenes.FAMILY_FACE_AREA
                           for x, label in zip(faces, per_face))
        areas = [x["box"][2] * x["box"][3] / area for x in faces]
        named = [a for a, label in zip(areas, per_face) if label and label in (wanted or refs)]
        c.meta["family_area"] = max(named, default=0.0) if (wanted or refs) else max(areas, default=0.0)
        c.meta["subject_area"] = max(areas, default=0.0)
        c.meta["tiny"] = is_tiny(c)
        # ce qui protège une photo d'être jugée « sans intérêt » : un visage de la famille bien visible,
        # une photo gardée à une relecture, le thème de l'album
        c.meta["shield"] = bool(family_close or f.id in kept or themed.get(f.id, (0, False))[1])
        twins = banned_vecs.get(f.when.date())
        if f.id in pinned:
            cands.append(c)
        elif twins and c.meta["vec"] and f.id not in kept and any(
                scoring.same_series(name, f.name, float(np.dot(v, c.meta["vec"]))) for name, v in twins):
            rejected["comme une photo retirée"] += 1
        elif kind == "capture":
            rejected["pas une photo"] += 1
        elif (c.meta["editor"] or {}).get("not_album", 0) > aesthetic.NOT_ALBUM_REJECT:
            rejected["pas pour un album"] += 1
        # un décor sans personne de la famille n'est pas sauvé par le thème ni par un « lieu »
        # (banderole, enseigne, hall d'hôtel prenaient le libellé « Kotel » ou « fête »)
        elif f.id in objects and not labels and f.id not in kept:
            rejected["objet ou décor"] += 1
        elif dull and not c.meta["shield"]:
            rejected["sans intérêt"] += 1
        elif scoring.too_small_to_print(f.width, f.height, kind):
            rejected["trop petite pour l'impression"] += 1
        elif c.parts["sharpness"] < scoring.REJECT_SHARPNESS:
            rejected["floue"] += 1
        elif params.must and must_missing(params.must, labels, refs_by_month[month], per_face):
            rejected["personne absente"] += 1
        elif must_faces and max(x["box"][2] * x["box"][3] for x in must_faces) / area < scoring.MIN_SUBJECT_AREA:
            rejected["personne trop petite"] += 1
        elif params.family and faces and not labels & set(wanted) and set(wanted) <= set(refs_by_month[month]):
            rejected["personne de la famille"] += 1
        elif params.faces_only and not faces:
            rejected["sans visage"] += 1
        elif params.theme_strict and themed and not themed.get(f.id, (0, False))[1]:
            rejected["hors thème"] += 1
        else:
            cands.append(c)
    unknown_age = [w for w in wanted if refs_by_month and not any(w in r for r in refs_by_month.values())]
    if unknown_age:
        rep.note(f"Je ne connais pas encore {', '.join(unknown_age)} à cet âge : donne-leur un prénom "
                 "dans « Qui est sur tes photos ? », puis relance l'album, il sera bien meilleur.")
    _save_cache(workdir / ".cache" / "parts.json", all_parts)  # pour mesurer le goût (taste --eval)

    # 2e passe, plus précise (CLIP ViT-L/14), sur les seules meilleures candidates : 5 000 photos ne
    # coûtent pas des heures, et chaque photo n'est regardée qu'une fois (gardé en cache). Elle redécrit
    # les scènes, juge les critères de photographe et prépare la ressemblance « même pose ».
    # Sans cet œil (pas de modèle, pas de réseau), la 1re passe suffit et rien ne change.
    if cands and not cancel.is_set():
        short = pick_shortlist(cands, params.count, protected=pinned | kept)
        look_closer([c.file_id for c in short] + list(examples["pos"]) + list(examples["neg"]),
                    thumbs_dir, editor_path, editor, model_dir, rep, cancel)
        if any((editor.get(c.file_id) or {}).get("vec_l") for c in short):
            cands = refine_with_large_eye(short, examples, editor, model_dir, rep, rate, all_parts,
                                          rejected, pinned, liked_scene)
            _save_cache(workdir / ".cache" / "parts.json", all_parts)
            rep.note(f"Regard précis : {len(cands)} photos gardées pour le tri final.")

    sharp_vecs = {k: v["vec_l"] for k, v in editor.items() if v.get("vec_l")}
    day_of = {k: v["when"] for k, v in cache.items() if v.get("when")}
    learned, kind = taste.best_model(workdir, sharp_vecs, all_parts, day_of)
    if learned:
        rep.note(f"Ton goût, appris sur {learned.size} photos gardées ou retirées, compte dans le choix.")
        weight = taste.IMAGE_WEIGHT if kind == "image" else taste.WEIGHT
        for c in cands:
            vec = sharp_vecs.get(c.file_id)
            if kind == "image" and not vec:
                continue  # pas encore vue par l'œil plus précis : sa note reste celle des critères
            c.parts["taste"] = learned.score(vec, c.parts)
            c.local_score = c.score = (1 - weight) * c.score + weight * c.parts["taste"]
    if themed:
        for c in cands:
            c.parts["theme"] = themed.get(c.file_id, (0.5, False))[0]
            c.local_score = c.score = (1 - scoring.THEME_WEIGHT) * c.score + scoring.THEME_WEIGHT * c.parts["theme"]
    rep.note(f"{len(cands)} photos retenues ; écartées : "
             + ", ".join(f"{k} {v}" for k, v in rejected.items() if v))
    if not cands:
        if params.must and rejected["personne absente"]:
            raise AlbumError(
                f"Je n'ai reconnu {', '.join(params.must)} sur aucune photo. Essaie d'être moins "
                "strict sur la reconnaissance, ou ajoute des photos de référence plus récentes.",
                code="no_match",
            )
        raise AlbumError("Aucune photo ne passe les filtres (trop floues ou sans visage).", code="no_match")

    # Le tri d'un photographe (voir curate.py) : un moment = une photo, la prise la plus naturelle ;
    # jamais deux fois la même pose ; la famille présente ; des plafonds par activité
    for c in cands:
        curate.penalize(c)
    pool = scoring.dedupe_bursts(cands, window_seconds=0, burst_seconds=0)   # copies exactes seulement
    kept_ids = {c.file_id for c in pool}
    pool += [c for c in cands if c.file_id in pinned and c.file_id not in kept_ids]  # jamais « doublon »
    groups = curate.find_moments(pool)
    pool, others = curate.heroes(groups, pinned)
    size_of = {c.file_id: len(groups[c.meta["moment"]]) for c in pool}
    rep.note(f"{len(groups)} moments distincts parmi {len(cands)} photos.")
    if pinned:
        rep.note(f"{len(pinned & {c.file_id for c in pool})} photos gardées à ta relecture reprises.")

    if params.use_claude:
        from .claude_rerank import rerank

        top = sorted(pool, key=lambda c: c.score, reverse=True)[: max(params.count * 3, params.count)]
        rep.note(f"Avis de Claude sur {len(top)} miniatures…")
        results = rerank(top, thumbs_dir, params.describe or params.name)
        for c in top:
            if c.file_id in results:
                claude_score, note = results[c.file_id]
                c.score = 0.55 * c.local_score + 0.45 * claude_score
                c.note = note
        pool = top

    # album centré sur un enfant dont on connaît la date de naissance : réparti par âge
    main = next((m for m in params.must if m in births), None)
    bucket = scoring.age_bucket(births[main]) if main else None
    if main:
        rep.note(f"Répartition selon l'âge de {main}.")
    elif (max(c.when for c in pool) - min(c.when for c in pool)).days <= 31:
        bucket = lambda c: c.when.date()  # un séjour : réparti jour par jour
    # peu de photos pour beaucoup de places : pas de seuil, sinon l'album sort raccourci
    floor = scoring.quality_floor([c.score for c in pool]) if len(pool) >= 2 * params.count else None
    family = list(dict.fromkeys(params.must + params.bonus)) if params.family else None
    decor = scoring.DECOR_MAX_SHARE if wanted else None  # album de personnes : peu de décor
    end_day = max(c.when for c in pool).date()
    babies = [m for m in (family or []) if m in births
              and scoring.months_between(births[m], end_day) < curate.BABY_MONTHS]
    seeds = curate.coverage_seeds(pool, family, babies, pinned, limit=max(2, params.count // 3)) if family else set()
    rules = dict(bucket=bucket, floor=floor, family=family, decor_max=decor,
                 selfie_max=scoring.SELFIE_MAX if wanted else None, meal_max=scenes.MEAL_MAX,
                 size_of=size_of, event_cap=curate.event_cap)
    cap = None if params.theme else curate.scene_cap(params.count)
    relaxed = {k: v + 0.05 for k, v in curate.POSE_COS.items()}
    expected = min(params.count, len(pool))
    selected: list = []
    # les règles du photographe d'abord ; si elles laissent trop de places vides (un dossier où tout se
    # ressemble), on les assouplit puis on les lève pour compléter l'album avec ce qui reste
    for pose, scene_limit, wide in ((curate.POSE_COS, cap, curate.wide_cap(params.count) if wanted else None),
                                    (relaxed, cap and cap + 2, None), (None, None, None)):
        selected = scoring.select_spread(
            pool, params.count, pinned=pinned | seeds | {c.file_id for c in selected}, pose_cos=pose,
            scene_cap=scene_limit, scene_exempt=scenes.GENERIC, wide_max=wide, **rules)
        if len(selected) >= 0.8 * expected:
            break
    if len(selected) < params.count and params.theme_strict and themed:
        rep.note(f"{len(selected)} photos seulement : il n'y en a pas plus de belles sur ce thème.")
    elif len(selected) < params.count:
        rep.note(f"{len(selected)} photos seulement : les autres sont le même moment, des selfies ou des "
                 "photos sans personne. Tu peux en reprendre dans la réserve.")
    selected = curate.add_openers(selected, pool, params.count, protected=pinned | seeds)
    reserve = curate.build_reserve(selected, others, pool, params.reserve)
    plan_layout = curate.layout(selected, reserve, family or list(refs))

    rep.step(5, "Préparation de l'aperçu…")
    out_dir = out_root / slugify(params.name)
    summary = {
        "count": params.count, "folders": folder_ids,
        "from": params.date_from.date().isoformat() if params.date_from else None,
        "to": params.date_to.date().isoformat() if params.date_to else None,
        "must": params.must, "bonus": params.bonus, "use_claude": params.use_claude,
        "describe": params.describe, "theme": params.theme, "theme_strict": params.theme_strict,
        "family": params.family,
    }
    stats = {
        "listed": listed, "analyzed": len(files), "after_filters": len(cands),
        "after_dedupe": len(pool), "moments": len(groups), "no_thumbnail": failed,
    }
    return write_plan(out_dir, params.name, summary, selected, thumbs_dir, stats, reserve, plan_layout)


def apply_plan(
    client,
    album_name: str,
    photos: list[dict],
    parent_id: str | None = None,
    copy: bool = False,
    reporter: Reporter | None = None,
) -> dict:
    """Crée le dossier d'album et un raccourci (ou une copie) par photo. Renvoie le dossier créé."""
    rep = reporter or Reporter()
    if not photos:
        raise AlbumError("L'album est vide.", code="empty_plan")
    folder = client.create_folder(album_name, parent_id)
    for i, p in enumerate(photos, 1):
        name = f"{p['date'][:10]}_{p['date'][11:16].replace(':', 'h')}_{p['name']}"
        if copy:
            client.copy_file(p["id"], name, folder["id"])
        else:
            client.add_shortcut(p["id"], name, folder["id"])
        rep.advance(i, len(photos))
    return folder
