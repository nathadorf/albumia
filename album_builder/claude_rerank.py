"""Reclassement OPTIONNEL par Claude (--use-claude).

⚠ Confidentialité : ce module envoie à l'API Anthropic des MINIATURES (≤ 768 px) des
seules photos présélectionnées (environ 3× le nombre voulu), jamais les originaux ni tout
le Drive. Il n'est utilisé que si vous passez --use-claude ET définissez ANTHROPIC_API_KEY.
Vérifiez la politique de conservation/entraînement d'Anthropic pour l'API avant usage.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import cv2

from .errors import AlbumError

PROMPT = """Tu aides à composer un album photo de famille.
Brief de l'album : {brief}

Voici {n} photos numérotées. Pour CHAQUE photo, donne une note de 0 à 10 selon :
netteté du sujet, expressions (sourires naturels, yeux ouverts), émotion / moment fort,
composition, intérêt narratif, adéquation avec le brief. Pénalise les clignements,
les visages coupés, les poses ratées et les photos banales ou redondantes.

Réponds UNIQUEMENT par un tableau JSON, sans texte autour :
[{{"i": 1, "score": 7.5, "note": "phrase courte"}}, ...]"""


def _encode(path: Path, max_side: int = 768) -> str | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.standard_b64encode(buf.tobytes()).decode() if ok else None


def _parse_json_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("pas de tableau JSON dans la réponse")
    return json.loads(text[start : end + 1])


def rerank(candidates, thumbs_dir: Path, brief: str, model: str | None = None, batch_size: int = 10):
    """Renvoie {file_id: (score_0_a_1, note)}. Les photos non notées sont simplement absentes."""
    try:
        import anthropic
    except ImportError:
        raise AlbumError("Installez le SDK : pip install -r requirements-claude.txt", code="claude")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise AlbumError("Définissez la variable d'environnement ANTHROPIC_API_KEY pour utiliser Claude.", code="claude")

    client = anthropic.Anthropic()
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    results: dict[str, tuple[float, str]] = {}

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        content, indexed = [], []
        for c in batch:
            data = _encode(thumbs_dir / f"{c.file_id}.jpg")
            if data is None:
                continue
            indexed.append(c)
            content.append({"type": "text", "text": f"Photo {len(indexed)} :"})
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}
            )
        if not indexed:
            continue
        content.append({"type": "text", "text": PROMPT.format(brief=brief or "meilleures photos", n=len(indexed))})

        for attempt in range(2):
            try:
                msg = client.messages.create(
                    model=model, max_tokens=1500, messages=[{"role": "user", "content": content}]
                )
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                for row in _parse_json_array(text):
                    idx = int(row["i"]) - 1
                    if 0 <= idx < len(indexed):
                        results[indexed[idx].file_id] = (
                            max(0.0, min(10.0, float(row["score"]))) / 10.0,
                            str(row.get("note", ""))[:200],
                        )
                break
            except Exception as e:  # JSON invalide, limite de débit…
                if attempt == 1:
                    print(f"  ! lot {start // batch_size + 1} non noté par Claude ({e})")
        print(f"  Claude : {min(start + batch_size, len(candidates))}/{len(candidates)} photos notées")
    return results
