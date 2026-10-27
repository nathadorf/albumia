"""Albumia apprend des corrections faites à la relecture.

Chaque photo retirée (et sa raison) est notée dans .cache/feedback.json, avec les notes détaillées
des photos gardées. Les poids de la note sont ensuite ajustés doucement : si les photos retirées
sont en moyenne plus sombres que celles gardées, l'exposition compte un peu plus, etc.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

from .scoring import Weights

REASONS = {
    "pas-belle": "Pas belle (floue, sombre, ratée)",
    "doublon": "Déjà une photo de ce moment",
    "personnes": "Pas les bonnes personnes",
    "gout": "Je ne l'aime pas",
}
MAX_CHANGE = 0.5   # un poids varie au plus d'un facteur e^0,5 ≈ 1,65
MIN_REMOVED = 5    # pas d'ajustement avant 5 photos retirées


def _path(workdir: Path) -> Path:
    return workdir / ".cache" / "feedback.json"


def load(workdir: Path) -> list[dict]:
    try:
        return json.loads(_path(workdir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def record(workdir: Path, album: str, kept: list[dict], removed: list[dict]) -> None:
    """`kept` : photos de l'album ; `removed` : photos retirées, chacune avec sa `reason`."""
    if not removed:
        return
    entries = load(workdir)
    entries.append({
        "album": album,
        "at": datetime.now().isoformat(timespec="seconds"),
        "kept": [p.get("parts", {}) for p in kept],
        "removed": [{"parts": p.get("parts", {}), "reason": p.get("reason", "gout")} for p in removed],
    })
    path = _path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def learned_weights(workdir: Path, base: Weights | None = None) -> Weights:
    """Poids ajustés d'après les corrections passées (les poids de base s'il y en a trop peu)."""
    base = base or Weights()
    entries = load(workdir)
    # un doublon ou une erreur de personne ne dit rien sur la qualité : seuls « pas belle » et « goût » comptent
    removed = [r["parts"] for e in entries for r in e["removed"] if r["reason"] in ("pas-belle", "gout")]
    kept = [p for e in entries for p in e["kept"]]
    if len(removed) < MIN_REMOVED or not kept:
        return base
    values = asdict(base)
    for f in fields(Weights):
        k = f.name
        r = [p[k] for p in removed if k in p]
        g = [p[k] for p in kept if k in p]
        if len(r) < MIN_REMOVED or not g:
            continue
        gap = sum(g) / len(g) - sum(r) / len(r)   # > 0 : les photos gardées sont meilleures sur ce critère
        values[k] = values[k] * math.exp(max(-MAX_CHANGE, min(MAX_CHANGE, 2 * gap)))
    return Weights(**values)
