"""Ce que montre une photo, en quelques mots, pour ne garder que les souvenirs.

Chaque photo reçoit la description la plus proche dans une liste de scènes (CLIP, sur
l'ordinateur, gratuit) : des souvenirs de famille (« jeux gonflables », « enfant avec le loulav »),
des lieux qui ont du sens (« le Kotel ») et des scènes sans intérêt pour un album (« intérieur de
bus », « rayon de magasin », « homme sur son téléphone »). Une photo « sans intérêt » n'est pas
proposée, sauf si quelqu'un de la famille est bien visible ; la description s'affiche sous la photo.

Chaque description a une note d'intérêt : un a priori selon son genre, corrigé par les photos
gardées ou retirées aux relectures (« goût par description »).

Listes réglées sur les relectures de la famille (octobre 2025) : transports, documents, objets,
écrans, salles vides et selfies retirés ; piscine, balançoire, aires de jeux, Kotel, gonflables gardés.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import aesthetic

SOUVENIR, PLACE, MEAL, BANAL = "souvenir", "lieu", "repas", "sans intérêt"

# (description affichée, description pour CLIP, genre)
SCENES = [
    ("Jeux dehors", "children playing outdoors in a park", SOUVENIR),
    ("Aire de jeux", "children playing at a playground", SOUVENIR),
    ("Toboggan", "a child on a slide", SOUVENIR),
    ("Balançoire", "a child on a swing", SOUVENIR),
    ("Jeux gonflables", "kids in a bouncy castle or on an inflatable slide", SOUVENIR),
    ("Course dans l'herbe", "children running on grass", SOUVENIR),
    ("Piscine", "kids at a swimming pool", SOUVENIR),
    ("Plage", "a family at the beach", SOUVENIR),
    ("Bébé au bain", "a baby in the bath", SOUVENIR),
    ("Bébé", "a baby lying on a blanket", SOUVENIR),
    ("Bébé avec une kippa", "a baby wearing a kippah", SOUVENIR),
    ("Câlin", "a parent hugging a child", SOUVENIR),
    ("Portrait", "a close-up portrait of a smiling child", SOUVENIR),
    ("Les enfants ensemble", "siblings together holding hands", SOUVENIR),
    ("En famille", "a family posing together for a photo", SOUVENIR),
    ("Souccah", "a family inside a sukkah decorated with branches", SOUVENIR),
    ("Loulav", "a child holding a lulav and an etrog", SOUVENIR),
    ("Bougies de fête", "lighting candles for a holiday", SOUVENIR),
    ("Anniversaire", "a birthday cake with candles", SOUVENIR),
    ("Déguisement", "a child wearing a costume", SOUVENIR),
    ("Vélo", "a child riding a bike", SOUVENIR),
    ("Animaux", "a child looking at animals at a zoo or a farm", SOUVENIR),
    ("Dans l'avion", "children sitting in an airplane seat", SOUVENIR),
    ("Rue le soir", "a city street at night with people", PLACE),
    ("Le Kotel", "a family at the Western Wall in Jerusalem", PLACE),
    ("Monument", "a famous landmark or monument", PLACE),
    ("Paysage", "a beautiful landscape view", PLACE),
    ("Objet de fête", "an ornate decorated chair or a religious object", PLACE),
    ("Repas de fête", "a festive family meal at a decorated table", MEAL),
    ("Repas", "children eating a meal", MEAL),
    ("Intérieur de bus", "the inside of a bus or a tram", BANAL),
    ("Valises", "suitcases at an airport or in a hall", BANAL),
    ("Enfants de dos", "children seen from behind", BANAL),
    ("Station de tram", "a tram station", BANAL),
    ("Rue", "a city street with traffic and buildings", BANAL),
    ("Magasin", "a supermarket aisle", BANAL),
    ("Menu", "a restaurant menu or a price list", BANAL),
    ("Document", "a printed document", BANAL),
    ("Livre", "a book on a table", BANAL),
    ("Écran", "a phone screen or a computer screen", BANAL),
    ("Affiche", "an advertisement poster or a banner", BANAL),
    ("Objet", "a toy box or an object on a table", BANAL),
    ("Plante", "a houseplant in a pot", BANAL),
    ("Frigo", "a fridge door with magnets", BANAL),
    ("Salle vide", "an empty conference hall", BANAL),
    ("Couloir", "a dark corridor", BANAL),
    ("Chambre en désordre", "a messy bedroom", BANAL),
    ("Au téléphone", "a man looking at his phone", BANAL),
    ("Selfie", "a selfie of a man", BANAL),
    ("Foule", "a crowded market full of strangers", BANAL),
    ("Pieds", "people's feet on the floor", BANAL),
]
KIND = {label: kind for label, _, kind in SCENES}
BANAL_MARGIN = 0.025    # « sans intérêt » seulement s'il l'emporte nettement sur toute autre scène
FAMILY_FACE_AREA = 0.03  # … et si personne de la famille n'est bien visible (visage ≥ 3 % de l'image)
MEAL_MAX = 2            # la famille garde peu de repas : 2 par album au plus
PRIOR = {SOUVENIR: 0.65, PLACE: 0.6, MEAL: 0.4, BANAL: 0.2}
PRIOR_WEIGHT = 3        # nombre de photos « imaginaires » derrière l'a priori
WEIGHT = 0.15           # part de l'intérêt de la description dans la note


def describe_all(vecs: dict[str, list[float]], model_dir: Path, note=print) -> dict[str, tuple[str, bool]]:
    """{photo: (description, sans intérêt net)} ; vide si l'œil d'éditeur n'est pas disponible.

    La description affichée est la meilleure scène qui n'est pas « sans intérêt » quand celle-ci
    ne l'emporte pas nettement (hésitation : on donne sa chance à la photo)."""
    vecs = {k: v for k, v in vecs.items() if v}
    if not vecs:
        return {}
    texts = [aesthetic.theme_vector(model_dir, prompt, note) for _, prompt, _ in SCENES]
    if any(t is None for t in texts):
        return {}
    ids = list(vecs)
    sims = np.asarray([vecs[i] for i in ids], dtype=float) @ np.vstack(texts).T
    banal = np.array([k == BANAL for _, _, k in SCENES])
    others = np.flatnonzero(~banal)
    out = {}
    for row, fid in zip(sims, ids):
        best = int(np.argmax(row))
        clear = bool(banal[best]) and row[banal].max() - row[~banal].max() >= BANAL_MARGIN
        if banal[best] and not clear:
            best = int(others[np.argmax(row[others])])
        out[fid] = (SCENES[best][0], clear)
    return out


def interest(examples: dict[str, dict], labels: dict[str, str]) -> dict[str, float]:
    """Intérêt de chaque description : a priori du genre, corrigé par les photos gardées (pos) et
    retirées (neg) qui portent cette description."""
    counts: dict[str, list[int]] = {}
    for group, liked in (("pos", 1), ("neg", 0)):
        for fid in examples.get(group, {}):
            if fid in labels:
                counts.setdefault(labels[fid], [0, 0])[liked] += 1
    out = {}
    for label, _, kind in SCENES:
        neg, pos = counts.get(label, [0, 0])
        out[label] = round((PRIOR[kind] * PRIOR_WEIGHT + pos) / (PRIOR_WEIGHT + pos + neg), 3)
    return out
