"""Préparation d'un album, indépendante de l'interface (ligne de commande ou page web).

`build_plan` lit Drive, analyse les miniatures en local et écrit un plan à relire.
`apply_plan` crée ensuite le dossier d'album (raccourcis) dans Drive.
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np

from . import aesthetic, compose, intent, people, scoring, taste, vision
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


MIN_THEMED = 8  # en dessous, un thème strict ne ferait qu'un album minuscule


def album_theme(params, files, editor, model_dir: Path, rep) -> dict[str, tuple[float, bool]]:
    """Note du thème pour chaque photo ; un thème strict trop rare devient une simple préférence."""
    if not params.theme:
        return {}
    label, prompt = intent.THEMES[params.theme][1:3]
    vecs = {f.id: (editor.get(f.id) or {}).get("vec") for f in files}
    tvec = aesthetic.theme_vector(model_dir, prompt, rep.note) if any(vecs.values()) else None
    if tvec is None:
        rep.note(f"Thème « {label} » non pris en compte : l'œil d'éditeur n'est pas disponible.")
        return {}
    margin = intent.THEME_MARGIN.get(params.theme, scoring.THEME_STRICT_MARGIN)
    if not scoring.theme_found(vecs, tvec, margin):
        rep.note(f"Je ne trouve pas de photos « {label} » dans ce dossier : thème ignoré.")
        params.theme_strict = False
        return {}
    not_prompt = intent.THEME_NOT.get(params.theme)
    not_vec = aesthetic.theme_vector(model_dir, not_prompt, rep.note) if not_prompt else None
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
            f"Le dossier contient {listed} photos, mais aucune dans la période demandée.",
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
    editor = judge_photos(files, cache, thumbs_dir, workdir / ".cache" / "editor.json", model_dir, rep, cancel)

    rep.step(4, "Choix des meilleures photos…")
    for f in files:
        if f.id in cache:
            cache[f.id]["when"] = f.when.date().isoformat()
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
    all_parts: dict[str, dict] = {}
    cands: list[scoring.Candidate] = []
    reasons = taste.excluded_reasons(workdir)
    banned = set(reasons)
    kept = set(taste.load_examples(workdir)["pos"])  # gardées à une relecture : jamais écartées par ressemblance
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
    rejected = {"retirée avant": 0, "comme une photo retirée": 0, "sans analyse": 0, "pas une photo": 0, "pas pour un album": 0, "trop petite pour l'impression": 0, "floue": 0, "personne absente": 0,
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
                  "shot": scoring.shot_type(feat, focus or faces)}
        c.local_score, c.parts = scoring.local_score(
            feat, labels, params.bonus, params.weights, per_face, wanted, c.meta
        )
        c.score = c.local_score
        all_parts[f.id] = c.parts
        must_faces = [x for x, label in zip(faces, per_face) if label in params.must]
        area = float(feat["w"] * feat["h"]) or 1.0
        twins = banned_vecs.get(f.when.date())
        if twins and c.meta["vec"] and f.id not in kept and any(
                scoring.same_series(name, f.name, float(np.dot(v, c.meta["vec"]))) for name, v in twins):
            rejected["comme une photo retirée"] += 1
        elif kind == "capture":
            rejected["pas une photo"] += 1
        elif (c.meta["editor"] or {}).get("not_album", 0) > aesthetic.NOT_ALBUM_REJECT:
            rejected["pas pour un album"] += 1
        elif scoring.too_small_to_print(f.width, f.height, kind):
            rejected["trop petite pour l'impression"] += 1
        elif c.parts["sharpness"] < scoring.REJECT_SHARPNESS:
            rejected["floue"] += 1
        elif params.must and not all(m in labels for m in params.must):
            rejected["personne absente"] += 1
        elif must_faces and max(x["box"][2] * x["box"][3] for x in must_faces) / area < scoring.MIN_SUBJECT_AREA:
            rejected["personne trop petite"] += 1
        elif params.family and faces and not labels & set(wanted):
            rejected["personne de la famille"] += 1
        elif params.faces_only and not faces:
            rejected["sans visage"] += 1
        elif params.theme_strict and themed and not themed.get(f.id, (0, False))[1]:
            rejected["hors thème"] += 1
        else:
            cands.append(c)
    _save_cache(workdir / ".cache" / "parts.json", all_parts)  # pour mesurer le goût (taste --eval)
    liked = taste.model_for(workdir, all_parts)
    if liked:
        rep.note(f"Ton goût, appris sur {liked.size} photos gardées ou retirées, compte dans le choix.")
        for c in cands:
            c.parts["taste"] = liked.predict(c.parts)
            c.local_score = c.score = (1 - taste.WEIGHT) * c.score + taste.WEIGHT * c.parts["taste"]
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

    pool = scoring.dedupe_bursts(cands)
    rep.note(f"{len(pool)} après suppression des rafales.")

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
    floor = scoring.quality_floor([c.score for c in pool])
    family = list(dict.fromkeys(params.must + params.bonus)) if params.family else None
    selected = scoring.select_spread(pool, params.count, bucket=bucket, floor=floor, family=family)
    if len(selected) < params.count and params.theme_strict and themed:
        rep.note(f"{len(selected)} photos seulement : il n'y en a pas plus de belles sur ce thème.")
    elif len(selected) < params.count:
        rep.note(f"{len(selected)} photos seulement : les autres étaient trop faibles pour un album.")
    chosen = {c.file_id for c in selected}
    reserve = sorted((c for c in pool if c.file_id not in chosen), key=lambda c: c.score, reverse=True)
    reserve = reserve[: params.reserve]

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
        "after_dedupe": len(pool), "no_thumbnail": failed,
    }
    return write_plan(out_dir, params.name, summary, selected, thumbs_dir, stats, reserve)


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
        name = f"{p['date'][:10]}_{p['name']}"
        if copy:
            client.copy_file(p["id"], name, folder["id"])
        else:
            client.add_shortcut(p["id"], name, folder["id"])
        rep.advance(i, len(photos))
    return folder
