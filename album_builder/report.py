"""Écriture du plan (plan.json) et de la planche-contact (plan.html) pour relecture."""
from __future__ import annotations

import html
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "album"


def _item(c) -> dict:
    return {
                "id": c.file_id,
                "name": c.name,
                "date": c.when.isoformat(timespec="seconds"),
                "score": round(c.score, 3),
                "local_score": round(c.local_score, 3),
                "parts": {k: round(v, 2) for k, v in c.parts.items()},
                "people": sorted(c.labels),
                "note": c.note,
                "starred": bool(c.meta.get("starred")),
                "shot": c.meta.get("shot", ""),
                "moment": c.meta.get("moment"),
                "face": round(c.meta.get("family_area", 0.0), 4),   # plus grand visage de la famille (part de l'image)
            }


def write_plan(
    out_dir: Path, name: str, args_summary: dict, selected, thumbs_dir: Path, stats: dict, reserve=(),
    layout: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thumbs").mkdir(exist_ok=True)
    for c in [*selected, *reserve]:
        src = thumbs_dir / f"{c.file_id}.jpg"
        if src.exists():
            shutil.copyfile(src, out_dir / "thumbs" / f"{c.file_id}.jpg")
    layout = layout or {}
    where = layout.get("of", {})
    items = [{**_item(c), **where.get(c.file_id, {})} for c in selected]
    plan = {
        "album_name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings": args_summary,
        "stats": stats,
        "photos": items,
        "reserve": [{**_item(c), **where.get(c.file_id, {})} for c in reserve],  # remplaçantes proposées à la relecture
    }
    if layout.get("chapters"):
        plan["chapters"], plan["cover"] = layout["chapters"], layout.get("cover")
    (out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(out_dir, plan)
    return out_dir / "plan.json"


def _write_html(out_dir: Path, plan: dict) -> None:
    cards = []
    for p in plan["photos"]:
        people = ", ".join(p["people"]) or "—"
        parts = " · ".join(f"{k} {v:.2f}" for k, v in p["parts"].items())
        note = f"<div class='note'>{html.escape(p['note'])}</div>" if p["note"] else ""
        cards.append(
            f"<figure><img loading='lazy' src='thumbs/{html.escape(p['id'])}.jpg'>"
            f"<figcaption><b>{html.escape(p['date'][:10])}</b> · {html.escape(p['name'])}<br>"
            f"score {p['score']:.2f} · personnes : {html.escape(people)}<br>"
            f"<small>{html.escape(parts)}</small>{note}</figcaption></figure>"
        )
    s = plan["stats"]
    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<title>{html.escape(plan['album_name'])}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}}
 figure{{margin:0;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px #0002}}
 img{{width:100%;display:block}} figcaption{{padding:8px 10px;font-size:13px;line-height:1.4}}
 .note{{margin-top:4px;color:#555;font-style:italic}} small{{color:#777}}
</style>
<h1>{html.escape(plan['album_name'])}</h1>
<p>{len(plan['photos'])} photos retenues sur {s.get('listed', '?')} listées
 ({s.get('analyzed', '?')} analysées, {s.get('after_filters', '?')} après filtres,
 {s.get('after_dedupe', '?')} après dédoublonnage des rafales).</p>
<div class="grid">{''.join(cards)}</div></html>"""
    (out_dir / "plan.html").write_text(doc, encoding="utf-8")
