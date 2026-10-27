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
from datetime import datetime
from pathlib import Path

from . import people, scoring, vision
from .errors import AlbumError, Cancelled
from .report import slugify, write_plan

STEPS = 5


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
    if wanted:
        rep.step(2, "Apprentissage des visages…")
        refs = vision.build_references(vision.get_engine(model_dir), refs_dir)
        missing = [w for w in wanted if w not in refs]
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

    rep.step(4, "Choix des meilleures photos…")
    births = people.births(refs_dir)
    cands: list[scoring.Candidate] = []
    rejected = {"sans analyse": 0, "pas une photo": 0, "floue": 0, "personne absente": 0,
                "personne trop petite": 0, "sans visage": 0}
    for f in files:
        entry = cache.get(f.id)
        if not entry:
            rejected["sans analyse"] += 1
            continue
        feat = entry["feat"]
        faces = scoring.real_faces(feat)
        # personne ne peut apparaître sur une photo prise avant sa naissance
        allowed = {n for n in refs if n not in births or births[n] <= f.when.date()}
        per_face = scoring.face_labels(faces, refs, params.match_threshold, allowed=allowed)
        labels = {label for label in per_face if label}
        kind = scoring.photo_kind(f.name, f.camera, f.width, f.height)
        c = scoring.Candidate(f.id, f.name, f.when, feat, labels)
        focus = [x for x, label in zip(faces, per_face) if label in wanted] if wanted else faces
        c.meta = {"starred": f.starred, "kind": kind, "width": f.width, "height": f.height,
                  "shot": scoring.shot_type(feat, focus or faces)}
        c.local_score, c.parts = scoring.local_score(
            feat, labels, params.bonus, params.weights, per_face, wanted, c.meta
        )
        c.score = c.local_score
        must_faces = [x for x, label in zip(faces, per_face) if label in params.must]
        area = float(feat["w"] * feat["h"]) or 1.0
        if kind == "capture":
            rejected["pas une photo"] += 1
        elif c.parts["sharpness"] < scoring.REJECT_SHARPNESS:
            rejected["floue"] += 1
        elif params.must and not all(m in labels for m in params.must):
            rejected["personne absente"] += 1
        elif must_faces and max(x["box"][2] * x["box"][3] for x in must_faces) / area < scoring.MIN_SUBJECT_AREA:
            rejected["personne trop petite"] += 1
        elif params.faces_only and not faces:
            rejected["sans visage"] += 1
        else:
            cands.append(c)
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
    selected = scoring.select_spread(pool, params.count, bucket=bucket)
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
        "describe": params.describe,
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
