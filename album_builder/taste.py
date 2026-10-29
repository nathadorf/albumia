"""Le goût de la famille, appris à partir de ses propres choix.

Chaque photo est décrite par une quinzaine de critères lisibles (netteté, sourire, émotion,
sujet bien placé, fond simple, horizon droit, couleurs…). Les photos gardées dans un album et
celles retirées pour « pas belle » ou « je ne l'aime pas » servent d'exemples : une petite
régression logistique apprend quels critères comptent pour cette famille. Avec si peu de
critères, quelques dizaines de photos suffisent (contre des centaines pour des vecteurs bruts),
et on peut dire pourquoi une photo plaît. Tout reste sur l'ordinateur.

Démarrer avec une sélection faite à la main (photos retenues = aimées, réserve = moins aimées) :
    python -m album_builder.taste --seed out/<album>/plan.json
Mesurer la qualité (validation croisée) et voir ce qui compte :
    python -m album_builder.taste --eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES = ("sharpness", "faces", "smile", "eyes", "frontal", "exposure", "resolution",
            "beauty", "emotion", "placement", "headroom", "clean", "attention", "level",
            "color", "contrast", "other_first",
            # œil plus précis (ViT-L/14) : mêmes jugements, plus justes, et critères de photographes
            "beauty_l", "emotion_l", "interaction", "eye_level", "light", "tidy", "face_seen", "moment")
# L'intérêt de la scène (parts["scene"]) n'en fait pas partie : il est calculé sur ces mêmes exemples, donc
# chaque photo contribue à sa propre note (AUC 0,83 pour ce seul critère, trompeur). Il compte déjà à part
# dans la note de base (scenes.WEIGHT).
MIN_EXAMPLES = 8   # par classe, avant de faire confiance au goût appris
L2 = 1.0           # régularisation : peu d'exemples, on reste prudent
MIN_AUC = 0.70     # le goût ne compte que s'il prédit mieux que le critère « émotion » seul
WEIGHT = 0.30      # part du goût appris dans la note finale, quand il est fiable


def _path(workdir: Path) -> Path:
    return workdir / ".cache" / "taste.json"


def load_examples(workdir: Path) -> dict[str, dict[str, dict]]:
    """{"pos": {photo: critères}, "neg": {photo: critères}} (critères vides si inconnus)."""
    try:
        data = json.loads(_path(workdir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {k: {fid: (v if isinstance(v, dict) else {}) for fid, v in data.get(k, {}).items()}
            for k in ("pos", "neg")}


def add_examples(workdir: Path, liked: list[str], disliked: list[str],
                 parts: dict[str, dict] | None = None) -> None:
    """Une photo aimée puis retirée (ou l'inverse) prend le dernier avis."""
    parts = parts or {}
    ex = load_examples(workdir)
    for fid in liked:
        old = ex["neg"].pop(fid, None) or ex["pos"].get(fid) or {}
        ex["pos"][fid] = parts.get(fid) or old
    for fid in disliked:
        old = ex["pos"].pop(fid, None) or ex["neg"].get(fid) or {}
        ex["neg"][fid] = parts.get(fid) or old
    path = _path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ex), encoding="utf-8")


def seed_from_plan(workdir: Path, plan_path: Path) -> tuple[int, int]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    liked = plan.get("photos", [])
    disliked = plan.get("reserve", [])
    add_examples(workdir, [p["id"] for p in liked], [p["id"] for p in disliked],
                 parts={p["id"]: p.get("parts", {}) for p in liked + disliked})
    return len(liked), len(disliked)


def features(parts: dict) -> np.ndarray:
    """Critères en vecteur, centrés sur 0,5 (un critère inconnu vaut 0,5 : neutre)."""
    return np.array([float(parts.get(k, 0.5)) - 0.5 for k in FEATURES])


class Taste:
    def __init__(self, w: np.ndarray, b: float, size: int):
        self.w, self.b, self.size = w, b, size
        self.auc = float("nan")   # justesse sur des journées jamais vues (validation croisée par groupes)

    def predict(self, parts: dict) -> float:
        z = float(features(parts) @ self.w + self.b)
        return round(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), 4)

    def score(self, vec, parts: dict) -> float:
        return self.predict(parts)

    def explain(self, top: int = 5) -> list[tuple[str, float]]:
        """Les critères qui comptent le plus pour cette famille (+ : aimé, − : pas aimé)."""
        order = np.argsort(-np.abs(self.w))[:top]
        return [(FEATURES[i], round(float(self.w[i]), 2)) for i in order]


def train(X: np.ndarray, y: np.ndarray, l2: float = L2, steps: int = 600, lr: float | None = 1.0) -> Taste:
    """Régression logistique (descente de gradient), classes équilibrées.

    `lr=None` : pas calculé d'après la matrice (stable même avec des centaines de dimensions)."""
    n, d = X.shape
    w, bias = np.zeros(d), 0.0
    pos = max(1, int(y.sum()))
    sample_w = np.where(y == 1, n / (2 * pos), n / (2 * max(1, n - pos)))
    if lr is None:
        # la courbure de la perte logistique est au plus 1/4 de celle des moindres carrés
        lr = 1.0 / (0.25 * float(sample_w.max()) * np.linalg.norm(X, 2) ** 2 / n + l2 / n + 1e-9)
        steps = max(steps, 1500)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + bias, -30, 30)))
        g = (p - y) * sample_w
        w -= lr * (X.T @ g / n + l2 * w / n)
        bias -= lr * float(g.mean())
    return Taste(w, bias, n)


def _dataset(examples: dict, current: dict[str, dict] | None = None):
    """Les critères recalculés à ce passage (`current`) priment sur ceux enregistrés.
    Renvoie (critères, 1 = aimée / 0 = retirée, identifiants)."""
    current = current or {}
    rows, ys, ids = [], [], []
    for label, group in ((1, examples["pos"]), (0, examples["neg"])):
        for fid, saved in group.items():
            parts = current.get(fid) or saved
            if parts:
                rows.append(features(parts))
                ys.append(label)
                ids.append(fid)
    if not rows:
        return np.zeros((0, len(FEATURES))), np.zeros(0), []
    return np.asarray(rows), np.asarray(ys, dtype=float), ids


def group_ids(ids: list[str], day_of: dict[str, str] | None = None,
              vectors: dict[str, list[float]] | None = None, twin_cos: float = 0.9) -> dict[str, str]:
    """Un groupe par photo : celles du même jour, et les quasi-jumelles d'un autre jour, vont ensemble.

    Sans cela, la validation croisée triche : une photo et sa rafale de sept quasi identiques se
    retrouvent des deux côtés, et le goût paraît bien meilleur qu'il ne l'est sur de nouvelles photos
    (sur 3 albums relus : justesse 0,96 en théorie, pas mieux que l'ancien goût en pratique)."""
    day_of = day_of or {}
    parent = {fid: fid for fid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    first_of_day: dict[str, str] = {}
    for fid in ids:
        day = day_of.get(fid)
        if day:
            union(fid, first_of_day.setdefault(day, fid))
    known = [fid for fid in ids if vectors and vectors.get(fid)]
    if len(known) > 1:
        V = np.asarray([vectors[f] for f in known], dtype=float)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        close = np.argwhere(np.triu(V @ V.T, 1) >= twin_cos)
        for i, j in close:
            union(known[i], known[j])
    return {fid: find(fid) for fid in ids}


def model_for(workdir: Path, current: dict[str, dict] | None = None, min_auc: float = MIN_AUC,
              groups: dict[str, str] | None = None) -> Taste | None:
    X, y, ids = _dataset(load_examples(workdir), current)
    if len(y) == 0 or y.sum() < MIN_EXAMPLES or (len(y) - y.sum()) < MIN_EXAMPLES:
        return None
    score = cross_val_auc(X, y, groups=[(groups or {}).get(f, f) for f in ids])
    if min_auc and not score >= min_auc:
        return None
    m = train(X, y)
    m.auc = round(score, 3)
    return m


# ---- goût appris sur l'image entière ---------------------------------------------------------
# Les ~17 critères lisibles résument mal une photo. Appris directement sur le vecteur de l'œil
# plus précis (CLIP ViT-L/14) et ces critères, le goût retrouve bien mieux les choix de la famille
# (relectures d'octobre 2025, 106 photos : AUC 0,97 contre 0,90 ; un album laissé de côté : moitié
# moins de photos retirées parmi les proposées).
IMAGE_MIN = 25          # exemples par classe avant de s'y fier
IMAGE_L2 = (10.0, 30.0, 100.0, 300.0)
IMAGE_MIN_AUC = 0.70    # sur des journées jamais vues
IMAGE_WEIGHT = 0.30     # part du goût dans la note quand il est fiable (comme le goût sur les critères)
IMAGE_BETTER_BY = 0.02  # le goût sur l'image ne remplace celui sur les critères que s'il généralise mieux


class ImageTaste:
    def __init__(self, w, b, mean, std, size: int, auc_: float):
        self.w, self.b, self.mean, self.std, self.size, self.auc = w, b, mean, std, size, auc_

    def predict(self, vec, parts: dict) -> float:
        x = np.clip((np.concatenate([np.asarray(vec, dtype=float), features(parts)]) - self.mean) / self.std, -6, 6)
        return round(float(1.0 / (1.0 + np.exp(-np.clip(x @ self.w + self.b, -30, 30)))), 4)

    def score(self, vec, parts: dict) -> float:
        return self.predict(vec, parts)


def _image_dataset(workdir: Path, vectors: dict[str, list[float]], current: dict[str, dict] | None = None):
    current = current or {}
    ex = load_examples(workdir)
    rows, ys, ids = [], [], []
    for label, group in ((1, ex["pos"]), (0, ex["neg"])):
        for fid, saved in group.items():
            vec = vectors.get(fid)
            if vec:
                rows.append(np.concatenate([np.asarray(vec, dtype=float), features(current.get(fid) or saved)]))
                ys.append(label)
                ids.append(fid)
    if not rows:
        return np.zeros((0, 1)), np.zeros(0), []
    return np.asarray(rows), np.asarray(ys, dtype=float), ids


def image_model_for(workdir: Path, vectors: dict[str, list[float]], current: dict[str, dict] | None = None,
                    groups: dict[str, str] | None = None) -> ImageTaste | None:
    """Goût appris sur [vecteur de l'image, critères] ; None tant qu'il n'est pas fiable.
    `vectors` : {photo: vecteur ViT-L/14} pour toutes les photos connues (editor.json)."""
    X, y, ids = _image_dataset(workdir, vectors, current)
    if y.sum() < IMAGE_MIN or len(y) - y.sum() < IMAGE_MIN:
        return None
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    by = [(groups or {}).get(f, f) for f in ids]
    scores = {l2: cross_val_auc(Z, y, l2=l2, lr=None, groups=by) for l2 in IMAGE_L2}
    best = max(scores, key=lambda l2: scores[l2] if scores[l2] == scores[l2] else -1)
    if not scores[best] >= IMAGE_MIN_AUC:
        return None
    m = train(Z, y, l2=best, lr=None)
    return ImageTaste(m.w, m.b, mean, std, len(y), round(scores[best], 3))


def best_model(workdir: Path, vectors: dict[str, list[float]], current: dict[str, dict] | None = None,
               day_of: dict[str, str] | None = None):
    """Le goût qui généralise le mieux à des journées jamais vues : (modèle, genre) ou (None, None).

    genre : « image » (vecteur de l'image + critères) ou « critères » (la quinzaine de notes lisibles).
    Les deux sont jugés par validation croisée sur des groupes (journées, rafales), jamais sur des photos
    isolées : une rafale entière est toujours d'un seul côté."""
    ex = load_examples(workdir)
    groups = group_ids(list(ex["pos"]) + list(ex["neg"]), day_of, vectors)
    image = image_model_for(workdir, vectors, current, groups)
    simple = model_for(workdir, current, groups=groups)
    if image and (not simple or image.auc >= simple.auc + IMAGE_BETTER_BY):
        return image, "image"
    if simple:
        return simple, "critères"
    return None, None


def auc(scores, labels) -> float:
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def _group_folds(groups, folds: int, seed: int) -> list[np.ndarray]:
    """Parties de taille proche, sans jamais couper un groupe."""
    labels = list(groups)
    members: dict = {}
    for i, g in enumerate(labels):
        members.setdefault(g, []).append(i)
    k = min(folds, len(members))
    order = list(members)
    np.random.default_rng(seed).shuffle(order)
    order.sort(key=lambda g: -len(members[g]))     # les plus gros groupes d'abord
    parts: list[list[int]] = [[] for _ in range(k)]
    for g in order:
        min(parts, key=len).extend(members[g])
    return [np.asarray(sorted(p)) for p in parts]


def cross_val_auc(X: np.ndarray, y: np.ndarray, folds: int = 5, l2: float = L2, seed: int = 0,
                  lr: float | None = 1.0, groups=None) -> float:
    """Justesse (AUC) en validation croisée. Avec `groups` (une étiquette par ligne), jamais deux photos
    d'un même groupe de part et d'autre : c'est la justesse sur de nouvelles journées."""
    n = len(y)
    if n < 2 * folds:
        return float("nan")
    if groups is None:
        order = np.random.default_rng(seed).permutation(n)
        tests = [order[k::folds] for k in range(folds)]
    else:
        tests = _group_folds(groups, folds, seed)
        if len(tests) < 2:
            return float("nan")
    preds = np.full(n, 0.5)
    for test in tests:
        train_idx = np.setdiff1d(np.arange(n), test)
        if y[train_idx].min() == y[train_idx].max():
            continue
        m = train(X[train_idx], y[train_idx], l2=l2, lr=lr)
        preds[test] = 1.0 / (1.0 + np.exp(-np.clip(X[test] @ m.w + m.b, -30, 30)))
    return auc(preds, y)


def shot_report(X: np.ndarray, y: np.ndarray, ids: list[str], cache: dict, out=print, min_each: int = 4) -> dict:
    """Les photos gardées et retirées, selon la taille des visages (d'après `features.json`) : combien de
    chaque sorte, et quels critères séparent les gardées des retirées dans chaque type de plan. Les plans larges
    sont ce qu'il retire le plus ET ce qu'il garde le plus : qu'est-ce qui fait un bon plan large ?"""
    from . import scoring

    def kind(fid: str) -> str:
        feat = (cache.get(fid) or {}).get("feat")
        if not feat:
            return "inconnu"
        faces = scoring.real_faces(feat)
        return scoring.shot_type(feat, faces) if faces else "sans visage"

    kinds = np.array([kind(f) for f in ids])
    report: dict = {}
    out("Par taille des visages (gardées / retirées) :")
    for name in ("scène", "plan moyen", "gros plan", "sans visage", "inconnu"):
        rows = kinds == name
        kept, removed = int(y[rows].sum()), int(rows.sum() - y[rows].sum())
        if not rows.any():
            continue
        entry = {"kept": kept, "removed": removed, "auc": {}}
        out(f"  {name:<12} {kept} gardées, {removed} retirées ({removed / (kept + removed):.0%} retirées)")
        if kept >= min_each and removed >= min_each:
            scores = {FEATURES[i]: auc(X[rows, i], y[rows]) for i in range(len(FEATURES))}
            entry["auc"] = {k: round(v, 3) for k, v in scores.items() if v == v}
            best = sorted(entry["auc"].items(), key=lambda kv: -abs(kv[1] - 0.5))[:6]
            out("    ce qui sépare les gardées des retirées (AUC seul) : " + ", ".join(f"{k} {v:.2f}" for k, v in best))
        report[name] = entry
    return report


# ---- photos retirées pour de bon -------------------------------------------------------------
def _excluded_path(workdir: Path) -> Path:
    return workdir / ".cache" / "excluded.json"


def excluded_reasons(workdir: Path) -> dict[str, str]:
    """{photo retirée: raison} (« pas-belle », « gout » ou « doublon »)."""
    try:
        data = json.loads(_excluded_path(workdir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, list):  # ancien format : une simple liste
        return {fid: "pas-belle" for fid in data}
    return dict(data)


def excluded(workdir: Path) -> set[str]:
    return set(excluded_reasons(workdir))


def _write_excluded(workdir: Path, reasons: dict[str, str]) -> None:
    path = _excluded_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reasons, sort_keys=True), encoding="utf-8")


def exclude(workdir: Path, ids: list[str], reason: str = "pas-belle") -> None:
    """Une photo retirée (pas belle, je ne l'aime pas, doublon) ne revient plus dans les albums."""
    if ids:
        _write_excluded(workdir, {**excluded_reasons(workdir), **{i: reason for i in ids}})


def unexclude(workdir: Path, ids: list[str]) -> None:
    """« Annuler » après un retrait : la photo redevient possible (et n'est plus un contre-exemple)."""
    if not ids:
        return
    _write_excluded(workdir, {k: v for k, v in excluded_reasons(workdir).items() if k not in set(ids)})
    ex = load_examples(workdir)
    for fid in ids:
        ex["neg"].pop(fid, None)
    _path(workdir).write_text(json.dumps(ex), encoding="utf-8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Goût appris d'Albumia")
    p.add_argument("--workdir", default=".")
    p.add_argument("--seed", metavar="PLAN_JSON", help="Ajoute une sélection faite à la main comme exemples")
    p.add_argument("--eval", action="store_true", help="Validation croisée (AUC) et critères qui comptent")
    p.add_argument("--exclude", nargs="+", metavar="FILE_ID", help="Retire ces photos des albums pour de bon")
    p.add_argument("--reason", default="pas-belle", choices=("pas-belle", "gout", "doublon"))
    args = p.parse_args(argv)
    workdir = Path(args.workdir)
    if args.seed:
        n_pos, n_neg = seed_from_plan(workdir, Path(args.seed))
        print(f"{n_pos} photos aimées, {n_neg} moins aimées ajoutées.")
    if args.exclude:
        exclude(workdir, args.exclude, args.reason)
        print(f"{len(args.exclude)} photo(s) retirée(s) pour de bon.")
    if args.eval:
        def load(name):
            try:
                return json.loads((workdir / ".cache" / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}

        current, editor, cache = load("parts.json"), load("editor.json"), load("features.json")
        day_of = {k: v["when"] for k, v in cache.items() if v.get("when")}
        vectors = {k: v["vec_l"] for k, v in editor.items() if v.get("vec_l")}
        examples = load_examples(workdir)
        groups = group_ids(list(examples["pos"]) + list(examples["neg"]), day_of, vectors)
        X, y, ids = _dataset(examples, current)
        by = [groups.get(f, f) for f in ids]
        print(f"{int(y.sum())} aimées, {int(len(y) - y.sum())} moins aimées, "
              f"{len(set(by))} groupes (journées, rafales).")
        for l2 in (0.1, 1.0, 10.0):
            alone = np.nanmean([cross_val_auc(X, y, l2=l2, seed=s) for s in range(3)])
            grouped = np.nanmean([cross_val_auc(X, y, l2=l2, seed=s, groups=by) for s in range(3)])
            print(f"Critères, L2={l2}: AUC photo par photo={alone:.3f} ; sur des journées jamais vues={grouped:.3f}")
        for k in FEATURES:  # pouvoir de chaque critère seul
            i = FEATURES.index(k)
            print(f"  {k:<11} AUC seul = {auc(X[:, i], y):.3f}")
        if len(y):
            print("Ce qui compte :", train(X, y).explain(8))
            shot_report(X, y, ids, cache)
        XI, yi, ids_i = _image_dataset(workdir, vectors, current)
        print(f"Sur l'image (ViT-L/14) : {int(yi.sum())} aimées, {int(len(yi) - yi.sum())} moins aimées.")
        if len(yi) >= 10:
            Z = (XI - XI.mean(axis=0)) / (XI.std(axis=0) + 1e-6)
            by_i = [groups.get(f, f) for f in ids_i]
            for l2 in IMAGE_L2:
                alone = np.nanmean([cross_val_auc(Z, yi, l2=l2, lr=None, seed=s) for s in range(3)])
                grouped = np.nanmean([cross_val_auc(Z, yi, l2=l2, lr=None, seed=s, groups=by_i) for s in range(3)])
                print(f"Image, L2={l2:g}: AUC photo par photo={alone:.3f} ; sur des journées jamais vues={grouped:.3f}")
        model, kind = best_model(workdir, vectors, current, day_of)
        print("Goût retenu :", f"{kind} (justesse {model.auc} sur des journées jamais vues)" if model else "aucun (pas assez fiable)")


if __name__ == "__main__":
    main()
