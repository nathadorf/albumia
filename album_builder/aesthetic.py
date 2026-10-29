"""L'œil d'éditeur : beauté de l'image et émotion du moment, jugées sur l'ordinateur.

OpenCV sait dire si une photo est nette et qui est dessus, pas si elle est belle ou émouvante.
Ce module utilise CLIP (un modèle image-texte libre, ~90 Mo, téléchargé une fois) pour comparer
chaque miniature à des descriptions d'éditeur : « une belle photo bien composée » contre
« un instantané banal et encombré », « un moment de joie » contre « un enfant qui s'ennuie »…
Gratuit, local, aucune photo n'est envoyée. Si le modèle n'est pas disponible (pas d'Internet,
Python trop récent pour onnxruntime), Albumia fonctionne comme avant, sans ces critères.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import vision
from .errors import AlbumError

HF = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
MIRROR = "https://hf-mirror.com/Xenova/clip-vit-base-patch32/resolve/main/"
FILES = {
    "clip_vision.onnx": "onnx/vision_model_quantized.onnx",
    "clip_text.onnx": "onnx/text_model_quantized.onnx",
    "clip_tokenizer.json": "tokenizer.json",
}
# œil plus précis (CLIP ViT-L/14, ~430 Mo, ~2 s par photo la première fois) : sert à décrire les
# scènes (« sans intérêt » repéré deux fois mieux sur les relectures de la famille, oct. 2025)
LARGE_HF = "https://huggingface.co/Xenova/clip-vit-large-patch14/resolve/main/"
LARGE_MIRROR = "https://hf-mirror.com/Xenova/clip-vit-large-patch14/resolve/main/"
LARGE_FILES = {
    "clipL_vision.onnx": "onnx/vision_model_quantized.onnx",
    "clipL_text.onnx": "onnx/text_model_quantized.onnx",
    "clip_tokenizer.json": "tokenizer.json",  # le même que celui de ViT-B/32
}
LARGE_VERSION = 1
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
VERSION = 3  # à changer si les descriptions ci-dessous changent (3 : types de scène)

# critère -> (descriptions positives, descriptions négatives)
CRITERIA = {
    "beauty": (
        ["a beautiful, well composed photograph with a clear subject",
         "an artistic family photo with lovely light",
         "a professional portrait of a child"],
        ["a poorly composed snapshot with the subject at the edge of the frame",
         "a cluttered, boring snapshot of a room",
         "a dull photo taken from above"],
    ),
    "emotion": (
        ["a child laughing, a joyful moment",
         "a tender moment between family members",
         "a happy child looking at the camera with a big smile"],
        ["a bored child with a blank expression",
         "a child looking away, an ordinary moment",
         "a tired child sitting in a stroller"],
    ),
    "not_album": (
        ["a photo of a computer screen",
         "a photo of a printed photograph",
         "a child sitting on a potty",
         "a naked baby",
         "a blurry accidental photo"],
        ["a family photo", "a photo of a child"],
    ),
}
# critères de photographes (culling, photo de famille), jugés par l'œil plus précis seulement :
# gestes entre les personnes, hauteur de prise de vue, lumière, fond, visage visible, vrai moment
PRO_CRITERIA = {
    "interaction": (
        ["children laughing and playing together",
         "a parent and a child sharing a tender moment",
         "family members hugging and looking at each other"],
        ["people lined up stiffly facing the camera",
         "a person alone with a blank expression",
         "people looking in different directions, not interacting"],
    ),
    "eye_level": (
        ["a photo taken at a child's eye level, seen from the front"],
        ["a photo taken from above looking down at a child",
         "the top of a child's head seen from above"],
    ),
    "light": (
        ["soft natural light, golden hour, warm and gentle",
         "gentle light in the shade, even and flattering"],
        ["harsh midday sun with deep shadows and squinting faces",
         "a dark underexposed photo",
         "a flash photo with a harsh glare"],
    ),
    "tidy": (
        ["a clean, simple background behind the subject"],
        ["a cluttered, busy background full of objects",
         "a crowd of strangers around a small child"],
    ),
    "face_seen": (
        ["a child facing the camera with the face clearly visible"],
        ["a child seen from behind", "the back of a person's head"],
    ),
    "moment": (
        ["a memorable, authentic moment of a family trip",
         "a child doing something exciting"],
        ["a boring routine moment", "an accidental snapshot of nothing"],
    ),
}
# type de scène : la routine (chaise haute, siège auto…) ne doit pas envahir l'album
SCENES = {
    "chaise haute": "a baby sitting in a high chair",
    "siège auto": "a child sitting in a car seat",
    "poussette": "a child sitting in a stroller",
    "repas": "a child eating a meal at a table",
    "bain": "a child in the bath",
    "dodo": "a sleeping baby in bed",
    "jeu à la maison": "a child playing with toys on the floor at home",
    "dehors": "a child playing outdoors in a park or garden",
    "fête": "a birthday party with a cake and decorations",
    "sortie": "a family day out at the beach, the zoo or a fair",
    "câlin": "a parent hugging a child",
    "portrait": "a close-up portrait of a child's face",
}
ROUTINE = {"chaise haute", "siège auto", "poussette", "repas"}
NOT_ALBUM_REJECT = 0.97   # au-delà : photo d'écran, pot… écartée (plus bas, trop de belles photos partaient)


class Judge:
    def __init__(self, model_dir: Path):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:  # Python trop récent ou installation incomplète
            raise AlbumError("Œil d'éditeur indisponible (onnxruntime absent).", code="no_aesthetic") from e
        paths = {name: vision.ensure_model(model_dir / name, (HF + src, MIRROR + src))
                 for name, src in FILES.items()}
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.vision = ort.InferenceSession(str(paths["clip_vision.onnx"]), opts, providers=["CPUExecutionProvider"])
        self.text_vectors = self._text_vectors(ort, Tokenizer, paths, model_dir / "clip_criteria.json")

    # ---- descriptions -> vecteurs (calculés une fois, gardés dans models/) -------------------
    def _text_vectors(self, ort, Tokenizer, paths, cache_path: Path) -> dict:
        try:
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            if saved.get("v") == VERSION:
                return {k: (np.array(v[0]), np.array(v[1])) for k, v in saved["criteria"].items()}
        except (OSError, ValueError, KeyError):
            pass
        embed = _text_embedder(ort, Tokenizer, paths)
        vectors = {k: (embed(pos), embed(neg)) for k, (pos, neg) in CRITERIA.items()}
        vectors["_scenes"] = (embed(list(SCENES.values())), np.zeros((0, 1)))
        cache_path.write_text(json.dumps({"v": VERSION, "criteria": {
            k: [p.round(5).tolist(), n.round(5).tolist()] for k, (p, n) in vectors.items()}}), encoding="utf-8")
        return vectors

    # ---- image -> notes ----------------------------------------------------------------------
    def embed_image(self, img_bgr: np.ndarray) -> np.ndarray:
        x = letterbox(img_bgr)
        out = _pick_embeds(self.vision, self.vision.run(None, {self.vision.get_inputs()[0].name: x}))[0]
        return out / (np.linalg.norm(out) or 1.0)

    def judge(self, img_bgr: np.ndarray) -> dict[str, float]:
        return judge_vector(self.embed_image(img_bgr), self.text_vectors)

    def judge_with_vector(self, img_bgr: np.ndarray) -> tuple[dict[str, float], list[float]]:
        vec = self.embed_image(img_bgr)
        return judge_vector(vec, self.text_vectors), [round(float(x), 4) for x in vec]

    def scene(self, vec) -> str:
        return scene_of(np.asarray(vec, dtype=float), self.text_vectors["_scenes"][0])


class LargeEye:
    """Vecteur CLIP ViT-L/14 d'une miniature (même découpage que l'œil d'éditeur)."""

    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        path = vision.ensure_model(model_dir / "clipL_vision.onnx",
                                   tuple(u + LARGE_FILES["clipL_vision.onnx"] for u in (LARGE_HF, LARGE_MIRROR)))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.vision = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    def vector(self, img_bgr: np.ndarray) -> list[float]:
        out = _pick_embeds(self.vision, self.vision.run(None, {self.vision.get_inputs()[0].name: letterbox(img_bgr)}))[0]
        return [round(float(x), 4) for x in out / (np.linalg.norm(out) or 1.0)]


def load_large(model_dir: Path, note=print) -> LargeEye | None:
    if os.environ.get("ALBUMIA_NO_EDITOR"):
        return None
    try:
        return LargeEye(model_dir)
    except Exception as e:  # pas de réseau, modèle corrompu : on garde l'œil habituel
        note(f"Œil plus précis indisponible ({e}).")
    return None


def _text_embedder(ort, Tokenizer, paths):
    tok = Tokenizer.from_file(str(paths["clip_tokenizer.json"]))
    tok.enable_padding(length=77, pad_id=49407)
    tok.enable_truncation(max_length=77)
    sess = ort.InferenceSession(str(paths["clip_text.onnx"]), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    def embed(texts):
        enc = tok.encode_batch(texts)
        feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64)}
        if "attention_mask" in names:
            feeds["attention_mask"] = np.array([e.attention_mask for e in enc], dtype=np.int64)
        out = _pick_embeds(sess, sess.run(None, feeds))
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    return embed


def theme_vectors(model_dir: Path, prompts: list[str], note=print, large: bool = False) -> list[np.ndarray | None]:
    """Descriptions -> vecteurs CLIP. Celles qui manquent sont calculées en une seule fois puis gardées
    dans models/clip_themes.json (ou clipL_themes.json pour l'œil plus précis)."""
    cache_path = model_dir / ("clipL_themes.json" if large else "clip_themes.json")
    try:
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    missing = [x for x in dict.fromkeys(prompts) if x not in saved]
    if missing and not os.environ.get("ALBUMIA_NO_EDITOR"):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            hf, mirror, files = (LARGE_HF, LARGE_MIRROR, LARGE_FILES) if large else (HF, MIRROR, FILES)
            paths = {name: vision.ensure_model(model_dir / name, (hf + src, mirror + src))
                     for name, src in files.items() if "vision" not in name}
            text = paths.get("clipL_text.onnx") or paths["clip_text.onnx"]
            vecs = _text_embedder(ort, Tokenizer, {**paths, "clip_text.onnx": text})(missing)
        except Exception as e:  # pas de modèle : l'album se fait sans
            note(f"Descriptions non prises en compte ({e}).")
        else:
            for x, vec in zip(missing, vecs):
                saved[x] = [round(float(v), 5) for v in vec]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(saved), encoding="utf-8")
    return [np.asarray(saved[x], dtype=float) if x in saved else None for x in prompts]


def theme_vector(model_dir: Path, prompt: str, note=print, large: bool = False) -> np.ndarray | None:
    return theme_vectors(model_dir, [prompt], note, large)[0]


def criteria_large(model_dir: Path, note=print) -> dict | None:
    """Les critères (beauté, émotion…) avec les descriptions de l'œil plus précis."""
    every = {**CRITERIA, **PRO_CRITERIA}
    prompts = [x for pos, neg in every.values() for x in (*pos, *neg)]
    vecs = dict(zip(prompts, theme_vectors(model_dir, prompts, note, large=True)))
    if any(v is None for v in vecs.values()):
        return None
    return {k: (np.vstack([vecs[x] for x in pos]), np.vstack([vecs[x] for x in neg]))
            for k, (pos, neg) in every.items()}


def scene_of(img_vec: np.ndarray, scene_vectors: np.ndarray) -> str:
    return list(SCENES)[int(np.argmax(scene_vectors @ img_vec))]


def judge_vector(img_vec: np.ndarray, text_vectors: dict, odds: bool = False) -> dict[str, float]:
    """Pour chaque critère : probabilité (0 à 1) que l'image ressemble aux descriptions positives.

    Avec `odds`, le logarithme du rapport de chances à la place : il ne sature pas (une probabilité
    arrondie à 1,0 ou à 0,02 ne départage plus personne), `calibrate` le met ensuite à l'échelle."""
    out = {}
    for k, (pos, neg) in text_vectors.items():
        if k.startswith("_"):
            continue
        logits = 100.0 * np.concatenate([pos @ img_vec, neg @ img_vec])
        top = logits.max()
        weights = np.exp(logits - top)
        if odds:
            out[k] = round(float(np.log(weights[: len(pos)].sum()) - np.log(weights[len(pos):].sum())), 3)
        else:
            out[k] = round(float(weights[: len(pos)].sum() / weights.sum()), 4)
    return out


CALIBRATE_MIN = 20   # en dessous, trop peu de photos pour se situer parmi elles


def calibrate(odds: dict[str, dict[str, float]], keys) -> dict[str, dict[str, float]]:
    """Rang (0 à 1) de chaque photo parmi les autres, critère par critère, à partir des rapports de chances.

    Les critères de photographe (fond rangé, geste, vrai moment…) sortent presque tous près de 0 ou de 1
    avec CLIP : leur rang, lui, sépare les photos et pèse pareil d'un critère à l'autre."""
    if len(odds) < CALIBRATE_MIN:
        return {}
    out: dict[str, dict[str, float]] = {fid: {} for fid in odds}
    for k in keys:
        ids = [fid for fid, o in odds.items() if k in o]
        if len(ids) < CALIBRATE_MIN:
            continue
        values = np.array([odds[fid][k] for fid in ids])
        ordered = np.sort(values)
        rank = (np.searchsorted(ordered, values, "left") + np.searchsorted(ordered, values, "right")) / 2.0
        for fid, r in zip(ids, rank):
            out[fid][k] = round(float((r + 0.5) / (len(ids) + 1)), 4)
    return out


def letterbox(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Image entière (sans recadrer : le cadrage fait partie du jugement), bordée de gris, normalisée."""
    import cv2

    h, w = img_bgr.shape[:2]
    s = size / max(h, w)
    small = cv2.resize(img_bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def _pick_embeds(sess, outputs) -> np.ndarray:
    names = [o.name for o in sess.get_outputs()]
    for i, n in enumerate(names):
        if n.endswith("embeds"):
            return np.asarray(outputs[i], dtype=np.float32)
    return np.asarray(outputs[0], dtype=np.float32)


def load_judge(model_dir: Path, note=print) -> Judge | None:
    if os.environ.get("ALBUMIA_NO_EDITOR"):  # tests : pas de téléchargement
        return None
    try:
        return Judge(model_dir)
    except AlbumError as e:
        note(str(e))
    except Exception as e:  # modèle corrompu, format inattendu : on continue sans
        note(f"Œil d'éditeur indisponible ({e}).")
    return None
