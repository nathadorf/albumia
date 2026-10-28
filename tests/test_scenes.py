import json
import random
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from album_builder import scenes, scoring
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
                                             side_effect=lambda _d, prompt, _n=None: _unit(index[prompt]))
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
        self.assertEqual(out["bus"], ("Magasin", True))

    def test_hesitation_keeps_the_photo_with_the_best_souvenir(self):
        out = scenes.describe_all({"p": self.vec(Magasin=0.30, Piscine=0.29)}, Path("."), note=lambda *_: None)
        self.assertEqual(out["p"], ("Piscine", False))

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


if __name__ == "__main__":
    unittest.main()
