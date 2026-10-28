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
            "color", "contrast")
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

    def predict(self, parts: dict) -> float:
        z = float(features(parts) @ self.w + self.b)
        return round(1.0 / (1.0 + np.exp(-z)), 4)

    def explain(self, top: int = 5) -> list[tuple[str, float]]:
        """Les critères qui comptent le plus pour cette famille (+ : aimé, − : pas aimé)."""
        order = np.argsort(-np.abs(self.w))[:top]
        return [(FEATURES[i], round(float(self.w[i]), 2)) for i in order]


def train(X: np.ndarray, y: np.ndarray, l2: float = L2, steps: int = 600) -> Taste:
    """Régression logistique (descente de gradient), classes équilibrées."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    pos = max(1, int(y.sum()))
    sample_w = np.where(y == 1, n / (2 * pos), n / (2 * max(1, n - pos)))
    lr = 1.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = (p - y) * sample_w
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * float(g.mean())
    return Taste(w, b, n)


def _dataset(examples: dict, current: dict[str, dict] | None = None):
    """Les critères recalculés à ce passage (`current`) priment sur ceux enregistrés."""
    current = current or {}
    rows, ys = [], []
    for label, group in ((1, examples["pos"]), (0, examples["neg"])):
        for fid, saved in group.items():
            parts = current.get(fid) or saved
            if parts:
                rows.append(features(parts))
                ys.append(label)
    if not rows:
        return np.zeros((0, len(FEATURES))), np.zeros(0)
    return np.asarray(rows), np.asarray(ys, dtype=float)


def model_for(workdir: Path, current: dict[str, dict] | None = None, min_auc: float = MIN_AUC) -> Taste | None:
    examples = load_examples(workdir)
    X, y = _dataset(examples, current)
    if len(y) == 0 or y.sum() < MIN_EXAMPLES or (len(y) - y.sum()) < MIN_EXAMPLES:
        return None
    if min_auc and not cross_val_auc(X, y) >= min_auc:
        return None
    return train(X, y)


def auc(scores, labels) -> float:
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def cross_val_auc(X: np.ndarray, y: np.ndarray, folds: int = 5, l2: float = L2, seed: int = 0) -> float:
    if len(y) < 2 * folds:
        return float("nan")
    order = np.random.default_rng(seed).permutation(len(y))
    preds = np.zeros(len(y))
    for k in range(folds):
        test = order[k::folds]
        train_idx = np.setdiff1d(order, test)
        m = train(X[train_idx], y[train_idx], l2=l2)
        preds[test] = 1.0 / (1.0 + np.exp(-(X[test] @ m.w + m.b)))
    return auc(preds, y)


# ---- photos retirées pour de bon -------------------------------------------------------------
def _excluded_path(workdir: Path) -> Path:
    return workdir / ".cache" / "excluded.json"


def excluded(workdir: Path) -> set[str]:
    try:
        return set(json.loads(_excluded_path(workdir).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def _write_excluded(workdir: Path, ids: set[str]) -> None:
    path = _excluded_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids)), encoding="utf-8")


def exclude(workdir: Path, ids: list[str]) -> None:
    """Une photo retirée pour « pas belle » ou « je ne l'aime pas » ne revient plus dans les albums."""
    if ids:
        _write_excluded(workdir, excluded(workdir) | set(ids))


def unexclude(workdir: Path, ids: list[str]) -> None:
    """« Annuler » après un retrait : la photo redevient possible (et n'est plus un contre-exemple)."""
    if not ids:
        return
    _write_excluded(workdir, excluded(workdir) - set(ids))
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
    args = p.parse_args(argv)
    workdir = Path(args.workdir)
    if args.seed:
        n_pos, n_neg = seed_from_plan(workdir, Path(args.seed))
        print(f"{n_pos} photos aimées, {n_neg} moins aimées ajoutées.")
    if args.exclude:
        exclude(workdir, args.exclude)
        print(f"{len(args.exclude)} photo(s) retirée(s) pour de bon.")
    if args.eval:
        try:
            current = json.loads((workdir / ".cache" / "parts.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
        X, y = _dataset(load_examples(workdir), current)
        print(f"{int(y.sum())} aimées, {int(len(y) - y.sum())} moins aimées.")
        for l2 in (0.1, 1.0, 10.0):
            print(f"L2={l2}: AUC={cross_val_auc(X, y, l2=l2):.3f}")
        for k in FEATURES:  # pouvoir de chaque critère seul
            i = FEATURES.index(k)
            print(f"  {k:<11} AUC seul = {auc(X[:, i], y):.3f}")
        if len(y):
            print("Ce qui compte :", train(X, y).explain(8))


if __name__ == "__main__":
    main()
