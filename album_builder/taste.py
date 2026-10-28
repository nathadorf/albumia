"""Le goût de la famille, appris à partir de ses propres choix.

Chaque photo a un vecteur CLIP (ce qu'elle montre, son ambiance, son cadrage). Les photos gardées
dans un album et celles retirées pour « pas belle » ou « je ne l'aime pas » servent d'exemples :
une petite régression logistique apprend ce qui plaît, et donne à chaque nouvelle photo une note
de 0 à 1. Plus on relit d'albums, plus la note est juste. Tout reste sur l'ordinateur.

Démarrer avec une sélection faite à la main (photos retenues = aimées, réserve = moins aimées) :
    python -m album_builder.taste --seed out/<album>/plan.json
Mesurer la qualité (validation croisée) :
    python -m album_builder.taste --eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_EXAMPLES = 8   # par classe, avant de faire confiance au goût appris
L2 = 1.0           # régularisation : peu d'exemples, on reste prudent


def _path(workdir: Path) -> Path:
    return workdir / ".cache" / "taste.json"


def load_examples(workdir: Path) -> dict[str, dict[str, float]]:
    try:
        data = json.loads(_path(workdir).read_text(encoding="utf-8"))
        return {"pos": dict(data.get("pos", {})), "neg": dict(data.get("neg", {}))}
    except (OSError, ValueError):
        return {"pos": {}, "neg": {}}


def add_examples(workdir: Path, liked: list[str], disliked: list[str]) -> None:
    """Une photo aimée puis retirée (ou l'inverse) prend le dernier avis."""
    ex = load_examples(workdir)
    for fid in liked:
        ex["neg"].pop(fid, None)
        ex["pos"][fid] = 1.0
    for fid in disliked:
        ex["pos"].pop(fid, None)
        ex["neg"][fid] = 1.0
    path = _path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ex), encoding="utf-8")


def seed_from_plan(workdir: Path, plan_path: Path) -> tuple[int, int]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    liked = [p["id"] for p in plan.get("photos", [])]
    disliked = [p["id"] for p in plan.get("reserve", [])]
    add_examples(workdir, liked, disliked)
    return len(liked), len(disliked)


class Taste:
    def __init__(self, w: np.ndarray, b: float, size: int):
        self.w, self.b, self.size = w, b, size

    def predict(self, vec) -> float:
        z = float(np.asarray(vec, dtype=float) @ self.w + self.b)
        return round(1.0 / (1.0 + np.exp(-z)), 4)


def train(X: np.ndarray, y: np.ndarray, l2: float = L2, steps: int = 400) -> Taste:
    """Régression logistique (descente de gradient), classes équilibrées."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    pos = max(1, int(y.sum()))
    sample_w = np.where(y == 1, n / (2 * pos), n / (2 * max(1, n - pos)))
    scale = 10.0  # vecteurs normalisés : on étire les écarts (cos ~ 0,2 à 0,3)
    Xs = X * scale
    lr = 0.5
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        g = (p - y) * sample_w
        w -= lr * (Xs.T @ g / n + l2 * w / n)
        b -= lr * float(g.mean())
    return Taste(w * scale, b, n)


def _dataset(editor: dict, examples: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows, ys, ids = [], [], []
    for label, group in ((1, examples["pos"]), (0, examples["neg"])):
        for fid in group:
            vec = (editor.get(fid) or {}).get("vec")
            if vec:
                rows.append(vec)
                ys.append(label)
                ids.append(fid)
    if not rows:
        return np.zeros((0, 0)), np.zeros(0), []
    return np.asarray(rows, dtype=float), np.asarray(ys, dtype=float), ids


def model_for(workdir: Path, editor: dict) -> Taste | None:
    X, y, _ = _dataset(editor, load_examples(workdir))
    if len(y) == 0 or y.sum() < MIN_EXAMPLES or (len(y) - y.sum()) < MIN_EXAMPLES:
        return None
    return train(X, y)


def auc(scores, labels) -> float:
    scores, labels = np.asarray(scores), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def cross_val_auc(editor: dict, examples: dict, folds: int = 5, l2: float = L2, seed: int = 0) -> float:
    X, y, _ = _dataset(editor, examples)
    if len(y) < 2 * folds:
        return float("nan")
    order = np.random.default_rng(seed).permutation(len(y))
    preds = np.zeros(len(y))
    for k in range(folds):
        test = order[k::folds]
        train_idx = np.setdiff1d(order, test)
        m = train(X[train_idx], y[train_idx], l2=l2)
        preds[test] = [m.predict(X[i]) for i in test]
    return auc(preds, y)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Goût appris d'Albumia")
    p.add_argument("--workdir", default=".")
    p.add_argument("--seed", metavar="PLAN_JSON", help="Ajoute une sélection faite à la main comme exemples")
    p.add_argument("--eval", action="store_true", help="Validation croisée (AUC) sur les exemples")
    args = p.parse_args(argv)
    workdir = Path(args.workdir)
    if args.seed:
        n_pos, n_neg = seed_from_plan(workdir, Path(args.seed))
        print(f"{n_pos} photos aimées, {n_neg} moins aimées ajoutées.")
    if args.eval:
        editor = json.loads((workdir / ".cache" / "editor.json").read_text(encoding="utf-8"))
        ex = load_examples(workdir)
        for l2 in (0.1, 1.0, 10.0):
            print(f"L2={l2}: AUC={cross_val_auc(editor, ex, l2=l2):.3f}")


if __name__ == "__main__":
    main()
