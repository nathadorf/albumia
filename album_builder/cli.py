"""Ligne de commande : `python -m album_builder ...` (voir README)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from . import pipeline, scoring
from .drive import SCOPES_FULL, SCOPES_LIGHT, DriveClient, get_credentials
from .errors import AlbumError


# --------------------------------------------------------------------------- utilitaires
def date_range(args) -> tuple[datetime | None, datetime | None]:
    lo = hi = None
    if args.year:
        lo, hi = datetime(args.year, 1, 1), datetime(args.year + 1, 1, 1)
    if args.date_from:
        lo = datetime.strptime(args.date_from, "%Y-%m-%d")
    if args.date_to:
        hi = datetime.strptime(args.date_to, "%Y-%m-%d") + timedelta(days=1)
    return lo, hi


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="album_builder",
        description="Crée un album (dossier de raccourcis) dans Google Drive avec les meilleures photos.",
    )
    p.add_argument("--name", help="Nom de l'album à créer (ex. « Eydel 2026 »)")
    p.add_argument("--count", type=int, default=50, help="Nombre de photos voulues (défaut : 50)")
    p.add_argument("--folder", action="append", default=[], help="Dossier Drive source (URL ou ID), répétable")
    p.add_argument("--no-recursive", action="store_true", help="Ne pas explorer les sous-dossiers")
    p.add_argument("--year", type=int, help="Année (ex. 2026)")
    p.add_argument("--from", dest="date_from", help="Date de début AAAA-MM-JJ")
    p.add_argument("--to", dest="date_to", help="Date de fin AAAA-MM-JJ (incluse)")
    p.add_argument("--refs-dir", default="refs", help="Dossier des photos de référence (refs/<personne>/*.jpg)")
    p.add_argument("--must", action="append", default=[], help="Personne obligatoire sur la photo (répétable : toutes requises)")
    p.add_argument("--bonus", action="append", default=[], help="Personne qui rapporte des points si présente (répétable)")
    p.add_argument("--match-threshold", type=float, default=scoring.DEFAULT_MATCH_THRESHOLD,
                   help="Seuil de reconnaissance de visage (défaut %(default)s ; baissez-le si trop de ratés)")
    p.add_argument("--faces-only", action="store_true", help="Exclure les photos sans visage")
    p.add_argument("--describe", default="", help="Description de l'album (utilisée seulement avec --use-claude)")
    p.add_argument("--use-claude", action="store_true",
                   help="Reclasse les meilleures candidates avec Claude (envoie des miniatures à l'API)")
    p.add_argument("--workers", type=int, default=6, help="Analyses en parallèle (défaut : 6)")
    p.add_argument("--workdir", default=".", help="Dossier contenant credentials.json, models/, .cache/")
    p.add_argument("--out", default="out", help="Dossier de sortie du plan (défaut : out)")
    p.add_argument("--full-scope", action="store_true",
                   help="Autorisation Drive complète (nécessaire pour --parent vers un dossier existant)")
    p.add_argument("--apply-plan", metavar="PLAN_JSON", help="Crée l'album dans Drive à partir d'un plan relu")
    p.add_argument("--parent", help="Dossier Drive où créer l'album (URL ou ID) ; défaut : racine de Mon Drive")
    p.add_argument("--copy", action="store_true", help="Copier les fichiers au lieu de créer des raccourcis (occupe du quota !)")
    args = p.parse_args(argv)

    if args.apply_plan:
        return args
    if not args.name:
        p.error("--name est requis")
    if not args.folder:
        p.error("au moins un --folder est requis")
    if args.count < 1:
        p.error("--count doit être ≥ 1")
    return args


def _client(args) -> DriveClient:
    root = Path(args.workdir)
    scopes = SCOPES_FULL if args.full_scope else SCOPES_LIGHT
    token = root / ("token_full.json" if args.full_scope else "token.json")
    return DriveClient(get_credentials(root / "credentials.json", token, scopes))


# --------------------------------------------------------------------------- construction du plan
def cmd_build(args) -> None:
    lo, hi = date_range(args)
    params = pipeline.BuildParams(
        name=args.name, folders=args.folder, count=args.count, date_from=lo, date_to=hi,
        must=args.must, bonus=args.bonus, match_threshold=args.match_threshold,
        faces_only=args.faces_only, recursive=not args.no_recursive, use_claude=args.use_claude,
        describe=args.describe, workers=args.workers,
    )
    plan_path = pipeline.build_plan(
        _client(args), params, Path(args.workdir), Path(args.out), Path(args.refs_dir)
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    print(f"\n{len(plan['photos'])} photos sélectionnées.")
    print(f"→ Relisez : {plan_path.parent / 'plan.html'}")
    print(f"→ Puis créez l'album dans Drive :\n   python -m album_builder --apply-plan {plan_path}")


# --------------------------------------------------------------------------- création dans Drive
def cmd_apply(args) -> None:
    plan = json.loads(Path(args.apply_plan).read_text(encoding="utf-8"))
    client = _client(args)
    parent = pipeline.parse_folder_id(args.parent) if args.parent else None
    try:
        folder = pipeline.apply_plan(client, plan["album_name"], plan["photos"], parent, args.copy)
    except AlbumError:
        raise
    except Exception as e:
        raise SystemExit(
            f"Création de l'album impossible ({e}).\n"
            "Si --parent désigne un dossier existant, relancez avec --full-scope."
        )
    print(f"Album « {plan['album_name']} » créé ({'copies' if args.copy else 'raccourcis'}) : "
          f"{folder.get('webViewLink', folder['id'])}")


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):  # console Windows (cp1252) : pas d'erreur sur « → »
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        if args.apply_plan:
            cmd_apply(args)
        else:
            cmd_build(args)
    except AlbumError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main(sys.argv[1:])
