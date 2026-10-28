"""Albumia : la page web locale (http://127.0.0.1:8765) qui pilote tout le reste.

Le serveur n'écoute que sur cette machine. Les photos ne quittent pas l'ordinateur,
sauf si l'option Claude est activée (désactivée par défaut, voir README).
"""
from __future__ import annotations

import base64
import json
import os
import threading
import uuid
import webbrowser
from datetime import date, datetime, timedelta
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from googleapiclient.errors import HttpError

from . import intent, learn, people, pipeline, scoring, taste
from .errors import AlbumError, Cancelled
from .report import slugify

HOST, PORT = "127.0.0.1", 8765
ROOT = Path(os.environ.get("ALBUMIA_HOME") or Path.cwd()).resolve()
STATIC = Path(__file__).resolve().parent / "static"
REFS = ROOT / "refs"
OUT = ROOT / "out"
SETTINGS = ROOT / ".cache" / "settings.json"
CREDENTIALS = ROOT / "credentials.json"
TOKEN_LIGHT, TOKEN_FULL = ROOT / "token.json", ROOT / "token_full.json"
APP_FOLDER_NAME = "Albums (Albumia)"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


# --------------------------------------------------------------------------- réglages
def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(values: dict) -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps({**load_settings(), **values}, indent=2), encoding="utf-8")


def claude_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- Google Drive
_clients: dict[bool, object] = {}
_clients_lock = threading.Lock()


def drive(full: bool | None = None):
    """Client Drive à partir du jeton enregistré ; lève `needs_login` s'il faut se connecter."""
    from .drive import SCOPES_FULL, SCOPES_LIGHT, DriveClient, load_saved_credentials

    candidates = [True] if full else [True, False] if full is None else [False]
    with _clients_lock:
        for is_full in candidates:
            if is_full in _clients:
                return _clients[is_full]
            token, scopes = (TOKEN_FULL, SCOPES_FULL) if is_full else (TOKEN_LIGHT, SCOPES_LIGHT)
            creds = load_saved_credentials(token, scopes)
            if creds:
                _clients[is_full] = DriveClient(creds)
                return _clients[is_full]
    raise AlbumError("Connecte d'abord Albumia à ton Google Drive.", code="needs_login")


def login(full: bool) -> str:
    from .drive import SCOPES_FULL, SCOPES_LIGHT, DriveClient, get_credentials

    token, scopes = (TOKEN_FULL, SCOPES_FULL) if full else (TOKEN_LIGHT, SCOPES_LIGHT)
    creds = get_credentials(CREDENTIALS, token, scopes)
    client = DriveClient(creds)
    with _clients_lock:
        _clients[full] = client
    email = client.whoami()
    save_settings({"email": email})
    return email


def albums_folder(client) -> str:
    """Dossier « Albums (Albumia) » à la racine de Mon Drive, créé une seule fois."""
    folder_id = load_settings().get("albums_folder_id")
    if folder_id and client.get_folder(folder_id):
        return folder_id
    folder = client.create_folder(APP_FOLDER_NAME)
    save_settings({"albums_folder_id": folder["id"]})
    return folder["id"]


# --------------------------------------------------------------------------- tâches de fond
class Job(pipeline.Reporter):
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.state = "running"
        self.step_n, self.label = 0, "Démarrage…"
        self.done = self.total = 0
        self.notes: list[str] = []
        self.result: dict | None = None
        self.error: dict | None = None
        self.cancel = threading.Event()

    def step(self, n, label):
        self.step_n, self.label, self.done, self.total = n, label, 0, 0

    def advance(self, done, total):
        self.done, self.total = done, total

    def note(self, message):
        self.notes.append(message)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "state": self.state, "step": self.step_n,
            "steps": pipeline.STEPS, "label": self.label, "done": self.done, "total": self.total,
            "notes": self.notes[-6:], "result": self.result, "error": self.error,
        }


JOBS: dict[str, Job] = {}


def start_job(kind: str, work) -> Job:
    job = Job(kind)
    JOBS[job.id] = job

    def run():
        try:
            job.result = work(job)
            job.state = "done"
        except Cancelled:
            job.state = "cancelled"
        except AlbumError as e:
            job.state, job.error = "error", {"message": str(e), "code": e.code}
        except Exception as e:  # erreur inattendue : on l'affiche telle quelle
            app.logger.exception("tâche %s", kind)
            job.state, job.error = "error", {"message": f"Erreur inattendue : {e}", "code": "unexpected"}

    threading.Thread(target=run, daemon=True).start()
    return job


def error_response(e: AlbumError, status: int = 400):
    return jsonify({"error": str(e), "code": e.code}), status


@app.errorhandler(AlbumError)
def _album_error(e):
    return error_response(e)


# --------------------------------------------------------------------------- pages
@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_file(name):
    return send_from_directory(STATIC, name)


@app.get("/refs/<person>/<photo>")
def ref_photo(person, photo):
    return send_from_directory(people.find_person_dir(REFS, person), photo)


@app.get("/plans/<slug>/thumbs/<photo>")
def plan_thumb(slug, photo):
    return send_from_directory(OUT / slugify(slug) / "thumbs", photo)


# --------------------------------------------------------------------------- API : connexion
@app.get("/api/status")
def status():
    connected, email = False, ""
    try:
        drive()
        connected, email = True, load_settings().get("email", "")
    except AlbumError:
        pass
    return jsonify({
        "credentials": CREDENTIALS.exists(), "connected": connected, "email": email,
        "claude": claude_available(),
    })


@app.post("/api/credentials")
def upload_credentials():
    f = request.files.get("file")
    try:
        data = json.loads(f.read().decode("utf-8")) if f else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = None
    if not data or "installed" not in data:
        raise AlbumError(
            "Ce fichier n'est pas un identifiant « Application de bureau » Google. "
            "Revois l'étape 4 du guide d'installation.", code="bad_credentials",
        )
    CREDENTIALS.write_text(json.dumps(data), encoding="utf-8")
    return jsonify({"ok": True})


@app.post("/api/login")
def start_login():
    full = bool((request.get_json(silent=True) or {}).get("full"))
    if not CREDENTIALS.exists():
        raise AlbumError("Il manque le fichier credentials.json.", code="no_credentials")
    job = start_job("login", lambda job: {"email": login(full)})
    return jsonify(job.to_dict())


# --------------------------------------------------------------------------- API : personnes
def _people_payload():
    return [
        {**p, "photos": [f"/refs/{p['name']}/{x}" for x in p["photos"]]}
        for p in people.list_people(REFS)
    ]


@app.get("/api/people")
def list_people():
    return jsonify(_people_payload())


_pending_faces: dict[str, bytes] = {}


@app.post("/api/faces")
def detect_faces():
    """Détecte les visages des photos envoyées et propose un prénom pour chacun."""
    import cv2

    from . import vision

    engine = vision.get_engine(ROOT / "models")
    known = vision.build_references(engine, REFS)
    results = []
    for f in request.files.getlist("files"):
        img = vision.decode(f.read())
        if img is None:
            results.append({"file": f.filename, "error": "Format non lu : envoie plutôt un JPEG ou un PNG."})
            continue
        img = vision.resize_max(img, 1600)
        faces = engine.detect_faces(img)
        faces.sort(key=lambda x: x["box"][0])  # de gauche à droite, comme sur la photo
        items = []
        for face in faces:
            crop = vision.face_crop(img, face["box"])
            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                continue
            token = uuid.uuid4().hex
            _pending_faces[token] = buf.tobytes()
            guess = scoring.face_labels([face], known, scoring.DEFAULT_MATCH_THRESHOLD)[0] or ""
            items.append({
                "token": token, "guess": guess,
                "preview": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
            })
        results.append({"file": f.filename, "faces": items})
    return jsonify(results)


@app.post("/api/faces/save")
def save_faces():
    saved = 0
    for item in (request.get_json(silent=True) or {}).get("items", []):
        data = _pending_faces.pop(item.get("token", ""), None)
        if data and (item.get("name") or "").strip():
            people.save_face(REFS, item["name"], data)
            saved += 1
    return jsonify({"saved": saved, "people": _people_payload()})


@app.put("/api/people/<person>")
def update_person(person):
    body = request.get_json(silent=True) or {}
    if "birth" in body:
        people.set_birth(REFS, person, body.get("birth") or None)
    return jsonify(_people_payload())


_face_groups: dict[str, list[dict]] = {}


def _thumb_face(item: dict):
    """Visage recadré depuis la miniature en cache (640 px), à partir de sa boîte (analyse en 1024 px)."""
    from . import vision

    img = vision.imread(ROOT / ".cache" / "thumbs" / f"{item['file']}.jpg")
    if img is None:
        return None
    scale = img.shape[1] / float(item["w"])
    box = [int(round(v * scale)) for v in item["box"]]
    return vision.face_crop(img, box)


@app.get("/api/face-groups")
def face_groups():
    """« Qui est sur tes photos ? » : regroupe les visages déjà analysés qu'Albumia ne connaît pas."""
    import cv2
    import numpy as np

    from . import vision

    cache = pipeline._load_cache(ROOT / ".cache" / "features.json")
    if not cache:
        raise AlbumError("Prépare d'abord un album : je regroupe ensuite les visages de tes photos.",
                         code="no_analysis")
    try:
        known = vision.build_references(vision.get_engine(ROOT / "models"), REFS) if REFS.exists() else {}
    except AlbumError:
        known = {}
    faces = []
    for fid, entry in cache.items():
        feat = entry["feat"]
        real = scoring.real_faces(feat)
        labels = scoring.face_labels(real, known, scoring.DEFAULT_MATCH_THRESHOLD)
        for face, label in zip(real, labels):
            if "emb" in face and not label:
                faces.append({"file": fid, "box": face["box"], "w": feat["w"], "when": entry.get("when"),
                              "size": face["box"][2] * face["box"][3], "emb": np.array(face["emb"])})
    faces.sort(key=lambda f: f["size"], reverse=True)
    clusters: list[dict] = []
    for f in faces[:3000]:
        best = max(clusters, key=lambda c: float(c["center"] @ f["emb"]), default=None)
        if best is not None and float(best["center"] @ f["emb"]) >= 0.6:
            best["items"].append(f)
            center = best["center"] * (len(best["items"]) - 1) + f["emb"]
            best["center"] = center / (np.linalg.norm(center) or 1.0)
        else:
            clusters.append({"center": f["emb"], "items": [f]})
    clusters = sorted((c for c in clusters if len(c["items"]) >= 3), key=lambda c: len(c["items"]), reverse=True)[:8]
    _face_groups.clear()
    out = []
    for c in clusters:
        gid = uuid.uuid4().hex[:10]
        _face_groups[gid] = c["items"]
        previews = []
        for item in c["items"][:4]:
            crop = _thumb_face(item)
            if crop is not None:
                ok, buf = cv2.imencode(".jpg", vision.resize_max(crop, 160), [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    previews.append("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode())
        out.append({"id": gid, "count": len(c["items"]), "previews": previews})
    return jsonify(out)


@app.post("/api/face-groups/<gid>")
def name_face_group(gid):
    """Donne un prénom à un groupe : ses visages les plus grands deviennent des références."""
    import cv2

    name = (request.get_json(silent=True) or {}).get("name", "").strip()
    items = _face_groups.pop(gid, None)
    if not items or not name:
        raise AlbumError("Ce groupe n'existe plus : recharge la liste.", code="bad_group")
    saved = 0
    for item in items[:6]:
        crop = _thumb_face(item)
        if crop is None:
            continue
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            taken = date.fromisoformat(item["when"]) if item.get("when") else None
            people.save_face(REFS, name, buf.tobytes(), taken)
            saved += 1
    return jsonify({"saved": saved, "people": _people_payload()})


@app.delete("/api/people/<person>")
def delete_person(person):
    people.remove_person(REFS, person)
    return jsonify(_people_payload())


@app.delete("/api/people/<person>/<photo>")
def delete_person_photo(person, photo):
    people.remove_photo(REFS, person, photo)
    return jsonify(_people_payload())


# --------------------------------------------------------------------------- API : album
@app.post("/api/interpret")
def interpret():
    body = request.get_json(silent=True) or {}
    names = [p["name"] for p in people.list_people(REFS)]
    selected = [s for s in body.get("selected", []) if s in names]
    births = people.births(REFS)
    return jsonify(intent.interpret(body.get("prompt", ""), names, selected, births=births).to_dict())


@app.post("/api/folder")
def check_folder():
    link = (request.get_json(silent=True) or {}).get("link", "")
    folder_id = pipeline.parse_folder_id(link)
    folder = drive().get_folder(folder_id)
    if not folder:
        raise AlbumError(
            "Je ne trouve pas ce dossier. Vérifie que c'est bien un dossier de ton Drive "
            "(ou partagé avec toi), pas une photo.", code="bad_folder",
        )
    save_settings({"last_folder": link})
    return jsonify({"id": folder["id"], "name": folder["name"]})


def _date(value: str | None, end: bool = False) -> datetime | None:
    if not value:
        return None
    d = datetime.strptime(value, "%Y-%m-%d")
    return d + timedelta(days=1) if end else d


@app.post("/api/build")
def build():
    b = request.get_json(silent=True) or {}
    folder = (b.get("folder") or "").strip()
    pipeline.parse_folder_id(folder)
    name = (b.get("name") or "").strip() or "Mon album"
    weights = learn.learned_weights(ROOT)  # ajustés d'après les corrections passées
    if b.get("favor_smiles"):
        weights.smile *= 2.5
    use_claude = bool(b.get("use_claude")) and claude_available()
    params = pipeline.BuildParams(
        name=name, folders=[folder], count=max(1, min(intent.MAX_COUNT, int(b.get("count") or 50))),
        date_from=_date(b.get("date_from")), date_to=_date(b.get("date_to"), end=True),
        must=list(b.get("must") or []), bonus=list(b.get("bonus") or []),
        match_threshold=scoring.LENIENT_MATCH_THRESHOLD if b.get("lenient") else scoring.DEFAULT_MATCH_THRESHOLD,
        faces_only=bool(b.get("faces_only")), use_claude=use_claude,
        describe=b.get("prompt", ""), weights=weights,
    )
    client = drive()

    def work(job: Job):
        plan_path = pipeline.build_plan(client, params, ROOT, OUT, REFS, job, job.cancel)
        return {"slug": plan_path.parent.name}

    return jsonify(start_job("build", work).to_dict())


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id) or abort(404)
    return jsonify(job.to_dict())


@app.post("/api/jobs/<job_id>/cancel")
def job_cancel(job_id):
    job = JOBS.get(job_id) or abort(404)
    job.cancel.set()
    return jsonify(job.to_dict())


def _read_plan(slug: str) -> tuple[Path, dict]:
    path = OUT / slugify(slug) / "plan.json"
    if not path.exists():
        abort(404)
    return path, json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/plans/<slug>")
def get_plan(slug):
    return jsonify(_read_plan(slug)[1])


@app.get("/api/albums")
def list_albums():
    albums = []
    for path in sorted(OUT.glob("*/plan.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        first = plan["photos"][0]["id"] if plan.get("photos") else None
        albums.append({
            "slug": path.parent.name, "name": plan["album_name"], "count": len(plan.get("photos", [])),
            "created": plan.get("created"), "drive_url": plan.get("drive_url"),
            "cover": f"/plans/{path.parent.name}/thumbs/{first}.jpg" if first else None,
        })
    return jsonify(albums)


@app.post("/api/apply")
def apply():
    b = request.get_json(silent=True) or {}
    plan_path, plan = _read_plan(b.get("slug", ""))
    known = {p["id"]: p for p in plan["photos"] + plan.get("reserve", [])}
    photos = [known[i] for i in b.get("ids", []) if i in known]
    removed = [{**known[r["id"]], "reason": r.get("reason", "gout")}
               for r in b.get("removed", []) if r.get("id") in known]
    name = (b.get("name") or "").strip() or plan["album_name"]
    full = bool(b.get("full"))
    client = drive(full=True if full else None)

    def work(job: Job):
        job.step(1, "Création de l'album dans ton Drive…")
        try:
            parent = albums_folder(client)
            folder = pipeline.apply_plan(client, name, photos, parent, reporter=job)
        except HttpError as e:
            if getattr(e.resp, "status", 0) in (403, 404) and not full:
                raise AlbumError(
                    "Google demande une autorisation de plus pour créer les raccourcis.",
                    code="needs_full_scope",
                )
            raise AlbumError(f"Google Drive a refusé la création ({e.reason}).", code="drive")
        url = folder.get("webViewLink") or f"https://drive.google.com/drive/folders/{folder['id']}"
        learn.record(ROOT, name, photos, removed)
        disliked = [r["id"] for r in removed if r["reason"] in ("pas-belle", "gout")]
        taste.add_examples(ROOT, [p["id"] for p in photos], disliked)
        taste.exclude(ROOT, disliked)
        plan.update(album_name=name, photos=photos, drive_url=url)
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"url": url, "count": len(photos), "name": name}

    return jsonify(start_job("apply", work).to_dict())


@app.get("/api/ping")
def ping():
    return jsonify({"app": "albumia"})


def main() -> None:
    url = f"http://{HOST}:{PORT}/"
    print(f"\n  Albumia est ouvert dans ton navigateur : {url}")
    print("  (Pour l'arrêter, ferme cette fenêtre ou appuie sur Ctrl+C.)\n")
    if not os.environ.get("ALBUMIA_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    # pas de bannière technique ni de journal de chaque requête dans la fenêtre
    import logging

    import flask.cli

    flask.cli.show_server_banner = lambda *a, **k: None
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host=HOST, port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    main()
