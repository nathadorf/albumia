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
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
VERSION = 1  # à changer si les descriptions ci-dessous changent

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
NOT_ALBUM_REJECT = 0.75   # au-delà : photo d'écran, photo de photo, pot… écartée


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

        vectors = {k: (embed(pos), embed(neg)) for k, (pos, neg) in CRITERIA.items()}
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


def judge_vector(img_vec: np.ndarray, text_vectors: dict) -> dict[str, float]:
    """Pour chaque critère : probabilité (0 à 1) que l'image ressemble aux descriptions positives."""
    out = {}
    for k, (pos, neg) in text_vectors.items():
        logits = 100.0 * np.concatenate([pos @ img_vec, neg @ img_vec])
        p = np.exp(logits - logits.max())
        p /= p.sum()
        out[k] = round(float(p[: len(pos)].sum()), 4)
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
