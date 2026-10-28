import json
import random
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from album_builder import scenes, scoring, taste
from album_builder.pipeline import previous_review
from tests.test_scoring import cand, face


def _unit(i, d=len(scenes.SCENES)):
    v = np.zeros(d)
    v[i] = 1.0
    return v


class SceneTests(unittest.TestCase):
    def setUp(self):
        index = {prompt: i for i, (_, prompt, _) in enumerate(scenes.SCENES)}
        patcher = unittest.mock.patch.object(scenes.aesthetic, "theme_vector",
                                             side_effect=lambda _d, prompt, _n=None, large=False: _unit(index[prompt]))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.at = {label: i for i, (label, _, _) in enumerate(scenes.SCENES)}

    def vec(self, **weights):
        v = np.zeros(len(scenes.SCENES))
        for label, w in weights.items():
            v[self.at[label.replace("_", " ")]] = w
        return list(v)

    def test_clear_banal_scene_is_flagged(self):
        out = scenes.describe_all({"bus": self.vec(Magasin=0.30, Piscine=0.20)}, Path("."), note=lambda *_: None)
        self.assertEqual(out["bus"][:2], ("Magasin", True))

    def test_hesitation_keeps_the_photo_with_the_best_souvenir(self):
        out = scenes.describe_all({"p": self.vec(Magasin=0.30, Piscine=0.29)}, Path("."), note=lambda *_: None)
        self.assertEqual(out["p"][:2], ("Piscine", False))

    def test_interest_learns_from_reviews(self):
        labels = {"a": "Piscine", "b": "Piscine", "c": "Au téléphone", "d": "Au téléphone"}
        liked = scenes.interest({"pos": {"a": {}, "b": {}}, "neg": {"c": {}, "d": {}}}, labels)
        self.assertGreater(liked["Piscine"], scenes.PRIOR[scenes.SOUVENIR])
        self.assertLess(liked["Au téléphone"], scenes.PRIOR[scenes.BANAL])
        self.assertEqual(liked["Balançoire"], scenes.PRIOR[scenes.SOUVENIR])


class PinAndMealTests(unittest.TestCase):
    def items(self):
        t0 = datetime(2025, 10, 5, 9)
        out = []
        for i in range(20):
            c = cand(f"p{i}", t0 + timedelta(hours=5 * i), 0.9 - i * 0.02,
                     dhash=format(random.Random(300 + i).getrandbits(64), "016x"))
            c.feat["faces"] = [face(box=[100, 100, 200, 200])]
            c.meta["meal"] = i < 6
            out.append(c)
        return out

    def test_meals_capped(self):
        sel = scoring.select_spread(self.items(), 10, meal_max=scenes.MEAL_MAX)
        self.assertEqual(sum(1 for c in sel if c.meta["meal"]), 2)
        self.assertEqual(len(sel), 10)

    def test_pinned_photos_always_come_back(self):
        items = self.items()
        pinned = {"idp17", "idp18", "idp19"}  # les moins bien notées
        sel = scoring.select_spread(items, 5, pinned=pinned, floor=0.8)
        self.assertTrue(pinned <= {c.file_id for c in sel})
        self.assertEqual(len(sel), 5)

    def test_previous_review(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "plan.json").write_text(json.dumps({"review": {
                "ids": ["a", "b"], "removed": [{"id": "c", "reason": "personnes"}, {"id": "e", "reason": "gout"}]}}))
            self.assertEqual(previous_review(Path(d)), ({"a", "b"}, {"c"}))
            self.assertEqual(previous_review(Path(d) / "absent"), (set(), set()))


class ImageTasteTests(unittest.TestCase):
    @staticmethod
    def embeddings(n=120, seed=0):
        """Comme de vrais vecteurs CLIP : peu de directions comptent (ici 8), le reste est du bruit."""
        rng = np.random.default_rng(seed)
        basis = rng.normal(size=(8, 768))
        vecs, liked = {}, []
        for i in range(n):
            z = rng.normal(size=8)
            z[0] += 1.5 if i % 2 else -1.5  # la direction qui sépare « gardée » de « retirée »
            v = z @ basis + 0.3 * rng.normal(size=768)
            vecs[f"p{i}"] = list(v / np.linalg.norm(v))
            liked.append(i % 2 == 1)
        return vecs, liked

    def test_learns_from_image_vectors(self):
        vecs, liked = self.embeddings()
        ids = list(vecs)
        pos = [f for f, k in zip(ids, liked) if k]
        neg = [f for f, k in zip(ids, liked) if not k]
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), pos[:30], neg[:30])
            m = taste.image_model_for(Path(d), vecs)
            self.assertIsNotNone(m)
            held = [(m.predict(vecs[f], {}), 1) for f in pos[30:]] + [(m.predict(vecs[f], {}), 0) for f in neg[30:]]
            self.assertGreater(taste.auc([x for x, _ in held], [y for _, y in held]), 0.9)

    def test_needs_enough_examples_and_the_large_eye(self):
        vecs, liked = self.embeddings()
        ids = list(vecs)
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), ids[1:20:2], ids[0:20:2])  # 10 par classe : trop peu
            self.assertIsNone(taste.image_model_for(Path(d), vecs))
            taste.add_examples(Path(d), ids[1::2], ids[0::2])
            self.assertIsNone(taste.image_model_for(Path(d), {}))  # pas d'œil plus précis : rien

    def test_unseen_values_do_not_blow_up(self):
        vecs, liked = self.embeddings()
        ids = list(vecs)
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), ids[1::2], ids[0::2])
            m = taste.image_model_for(Path(d), vecs)
            score = m.predict(vecs[ids[0]], {"scene": 0.9, "smile": 1.0})  # critères constants à l'entraînement
            self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
