"""Critères v3 : moments, variété, naissance, impression, apprentissage."""
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

from album_builder import learn, people, scoring
from album_builder.intent import interpret
from tests.test_scoring import cand, face, feat


def h(i):
    return f"{i * 0x9E3779B97F4A7C15 % 2**64:016x}"


class PhotoKindTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(scoring.photo_kind("Screenshot_20230101.png"), "capture")
        self.assertEqual(scoring.photo_kind("Capture d’écran 2023.png"), "capture")
        self.assertEqual(scoring.photo_kind("IMG-20230312-WA0004.jpg"), "messagerie")
        self.assertEqual(scoring.photo_kind("4C841891-1A2B-4C3D-9E8F-001122334455.jpg"), "messagerie")
        self.assertEqual(scoring.photo_kind("4C841891-1A2B-4C3D-9E8F-001122334455.jpg", camera="iPhone 13"), "photo")
        self.assertEqual(scoring.photo_kind("x.png", width=1170, height=2532), "capture")
        self.assertEqual(scoring.photo_kind("IMG_2382.JPG", camera="Canon EOS", width=6000, height=4000), "photo")

    def test_print_score(self):
        self.assertEqual(scoring.print_score(None, None), 1.0)
        self.assertEqual(scoring.print_score(6000, 4000), 1.0)
        self.assertEqual(scoring.print_score(640, 480), 0.0)


class SelectionTests(unittest.TestCase):
    def test_big_moment_gets_more_than_isolated_shot(self):
        t = datetime(2023, 5, 7, 10)
        party = [cand(i, t + timedelta(minutes=5 * i), 0.9 - i * 0.001, dhash=h(i)) for i in range(20)]
        lone = [cand(100 + i, datetime(2023, 5, 10 + i, 12), 0.95, dhash=h(100 + i)) for i in range(3)]
        chosen = scoring.select_spread(party + lone, 6)
        from_party = [c for c in chosen if c.file_id.startswith("id") and int(c.file_id[2:]) < 100]
        self.assertEqual(len(from_party), 3)   # plafond d'un grand moment
        self.assertEqual(len(chosen), 6)

    def test_similar_scene_is_penalised(self):
        t = datetime(2023, 5, 1, 10)
        a = cand(1, t, 0.9, dhash="0" * 16)
        twin_scene = cand(2, t + timedelta(hours=5), 0.85, dhash="00000000000000ff")  # 8 bits : même décor
        other = cand(3, t + timedelta(hours=10), 0.8, dhash="f0f0f0f0f0f0f0f0")
        chosen = scoring.select_spread([a, twin_scene, other], 2)
        self.assertEqual({c.file_id for c in chosen}, {"id1", "id3"})

    def test_age_buckets(self):
        key = scoring.age_bucket(date(2021, 7, 14))
        mk = lambda d: cand(0, d, 0.5)
        self.assertEqual(key(mk(datetime(2021, 8, 1))), (0, 0))
        self.assertEqual(key(mk(datetime(2021, 11, 1))), (0, 1))
        self.assertEqual(key(mk(datetime(2023, 1, 20))), (1, 3))
        self.assertEqual(key(mk(datetime(2025, 1, 1))), (2, 3))
        self.assertEqual(key(mk(datetime(2021, 1, 1))), (-1, 0))


class ScoreTests(unittest.TestCase):
    def test_cut_face_and_profile_score_lower(self):
        good = feat(faces=[face(box=(300, 200, 250, 250)) | {"frontal": 1.0, "edge": False}])
        cut = feat(faces=[face(box=(0, 200, 250, 250)) | {"frontal": 1.0, "edge": True}])
        profile = feat(faces=[face(box=(300, 200, 250, 250)) | {"frontal": 0.0, "edge": False}])
        s_good, _ = scoring.local_score(good, set(), [])
        self.assertGreater(s_good, scoring.local_score(cut, set(), [])[0])
        self.assertGreater(s_good, scoring.local_score(profile, set(), [])[0])

    def test_starred_and_low_resolution(self):
        f = feat(faces=[face()])
        base, _ = scoring.local_score(f, set(), [], meta={"width": 4000, "height": 3000})
        star, _ = scoring.local_score(f, set(), [], meta={"width": 4000, "height": 3000, "starred": True})
        small, _ = scoring.local_score(f, set(), [], meta={"width": 800, "height": 600})
        self.assertGreater(star, base)
        self.assertLess(small, base)

    def test_birth_date_excludes_unborn_sibling(self):
        refs = {"eydel": [1.0, 0.0], "arie": [0.8, 0.6]}
        baby = face(emb=[0.949, 0.316])  # ressemble autant aux deux
        self.assertEqual(scoring.face_labels([baby], refs, 0.45), [None])
        self.assertEqual(scoring.face_labels([baby], refs, 0.45, allowed={"eydel"}), ["eydel"])


class BirthAndIntentTests(unittest.TestCase):
    def test_birth_roundtrip_and_first_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            refs = Path(tmp)
            people.save_face(refs, "Eydel", b"jpeg")
            people.set_birth(refs, "eydel", "2021-07-14")
            self.assertEqual(people.births(refs), {"Eydel": date(2021, 7, 14)})
            self.assertEqual(people.list_people(refs)[0]["birth"], "2021-07-14")
        it = interpret("la première année d'Eydel", ["Eydel", "Arié"], births={"Eydel": date(2021, 7, 14)},
                       today=date(2026, 10, 27))
        self.assertEqual((it.date_from, it.date_to), ("2021-07-14", "2022-07-13"))
        self.assertEqual(it.name, "La première année d'Eydel")
        it = interpret("Arié à 2 ans", ["Eydel", "Arié"], births={"Arié": date(2022, 9, 9)}, today=date(2026, 10, 27))
        self.assertEqual((it.date_from, it.date_to), ("2024-09-09", "2025-09-08"))


class LearnTests(unittest.TestCase):
    def test_dark_removals_raise_exposure_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            kept = [{"parts": {"exposure": 0.9, "smile": 0.5}} for _ in range(20)]
            removed = [{"parts": {"exposure": 0.2, "smile": 0.5}, "reason": "pas-belle"} for _ in range(6)]
            learn.record(tmp, "Test", kept, removed[:3])
            self.assertEqual(learn.learned_weights(tmp), scoring.Weights())  # trop peu d'exemples
            learn.record(tmp, "Test 2", kept, removed[3:])
            w = learn.learned_weights(tmp)
            self.assertGreater(w.exposure, scoring.Weights().exposure)
            self.assertAlmostEqual(w.smile, scoring.Weights().smile)

    def test_duplicates_do_not_teach_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            removed = [{"parts": {"exposure": 0.1}, "reason": "doublon"} for _ in range(10)]
            learn.record(tmp, "T", [{"parts": {"exposure": 0.9}}], removed)
            self.assertEqual(learn.learned_weights(tmp), scoring.Weights())


if __name__ == "__main__":
    unittest.main()


class EditorTests(unittest.TestCase):
    """Retours de la comparaison v2 / v3 sur les vraies photos."""

    def test_same_second_copy_and_selfie_series_are_duplicates(self):
        t = datetime(2021, 7, 16, 11, 21, 12)
        copy_a = cand(1, t, 0.9, dhash="0" * 16)
        copy_b = cand(2, t, 0.8, dhash="00000000ffffffff")  # recadrée : empreinte très différente
        selfies = [cand(10 + i, datetime(2021, 7, 26, 15) + timedelta(seconds=50 * i), 0.7 + i / 100,
                        dhash=f"{int(h(5), 16) ^ ((1 << (6 * i)) - 1):016x}") for i in range(3)]  # 0, 6 et 12 bits d'écart
        kept = {c.file_id for c in scoring.dedupe_bursts([copy_a, copy_b] + selfies)}
        self.assertEqual(kept, {"id1", "id12"})

    def test_same_instant_prefers_another_moment(self):
        t = datetime(2023, 5, 7, 10)
        a = cand(1, t, 0.9, dhash=h(1))
        sofa = cand(2, t + timedelta(minutes=3), 0.85, dhash=h(2))
        later = cand(3, t + timedelta(minutes=50), 0.8, dhash=h(3))
        chosen = scoring.select_spread([a, sofa, later], 2)
        self.assertEqual({c.file_id for c in chosen}, {"id1", "id3"})

    def test_too_small_to_print(self):
        self.assertTrue(scoring.too_small_to_print(720, 1280, "messagerie"))
        self.assertFalse(scoring.too_small_to_print(720, 1280, "photo"))
        self.assertTrue(scoring.too_small_to_print(480, 640, "photo"))
        self.assertFalse(scoring.too_small_to_print(None, None, "messagerie"))

    def test_selfie_of_someone_else_scores_lower(self):
        child = face(box=(400, 300, 150, 150)) | {"frontal": 1.0, "edge": False}
        dad = face(box=(50, 50, 400, 400)) | {"frontal": 1.0, "edge": False}
        alone, _ = scoring.local_score(feat(faces=[child]), {"Eydel"}, [], labels_per_face=["Eydel"], targets=["Eydel"])
        selfie, _ = scoring.local_score(feat(faces=[child, dad]), {"Eydel"}, [],
                                        labels_per_face=["Eydel", None], targets=["Eydel"])
        self.assertLess(selfie, alone)


class NetworkTests(unittest.TestCase):
    def test_drive_retries_timeouts(self):
        from unittest import mock
        from album_builder import drive
        from album_builder.errors import AlbumError

        calls = []

        class Req:
            def execute(self):
                calls.append(1)
                if len(calls) < 3:
                    raise TimeoutError("The read operation timed out")
                return {"ok": True}

        with mock.patch.object(drive.time, "sleep"):
            self.assertEqual(drive.DriveClient._execute(Req()), {"ok": True})

            class Down:
                def execute(self):
                    raise TimeoutError("The read operation timed out")

            with self.assertRaises(AlbumError):
                drive.DriveClient._execute(Down())


class AgeMatchTests(unittest.TestCase):
    """Arié bébé ne doit pas être pris pour Eydel bébé."""

    def setUp(self):
        import numpy as np
        unit = lambda v: (np.asarray(v, dtype=float) / np.linalg.norm(v)).tolist()
        self.baby = unit([1, 0, 0, 0])          # « visage de bébé » commun aux deux
        self.eydel_big = unit([0, 1, 0, 0])
        self.arie_big = unit([0, 0, 1, 0])
        self.births = {"Eydel": date(2021, 7, 14), "Arié": date(2022, 9, 9)}
        self.sets = {
            "Eydel": [(self.baby, date(2021, 7, 20)), (self.eydel_big, date(2023, 6, 1))],
            "Arié": [(self.baby, date(2022, 9, 15)), (self.arie_big, None)],
        }

    def test_baby_in_2023_is_not_matched_to_newborn_eydel(self):
        import numpy as np
        refs = scoring.refs_at(self.sets, self.births, date(2023, 8, 24))
        self.assertGreater(refs["Eydel"][1], 0.99)   # Eydel à 2 ans : seulement sa photo de 2023
        face_emb = (np.asarray(self.baby) * 0.8 + np.asarray(self.arie_big) * 0.6).tolist()
        labels = scoring.face_labels([{"box": [0, 0, 10, 10], "emb": face_emb}], refs, 0.45, margin=0.05)
        self.assertEqual(labels, ["Arié"])

    def test_newborn_eydel_still_uses_her_baby_photo(self):
        refs = scoring.refs_at(self.sets, self.births, date(2021, 8, 1))
        self.assertGreater(refs["Eydel"][0], 0.99)

    def test_reference_dates_found_from_cache(self):
        from album_builder import pipeline
        cache = {"a": {"when": "2021-07-20", "feat": {"faces": [{"emb": self.baby}]}},
                 "b": {"feat": {"faces": [{"emb": self.eydel_big}]}}}
        found = pipeline.date_references({"Eydel": [("x.jpg", self.baby), ("y.jpg", self.eydel_big)]}, {}, cache)
        self.assertEqual(found, {"Eydel": {"x.jpg": date(2021, 7, 20)}})

    def test_photo_dates_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = people.save_face(Path(d), "Eydel", b"jpg", date(2021, 7, 20))
            self.assertEqual(people.photo_dates(Path(d)), {"Eydel": {p.name: date(2021, 7, 20)}})


class EditorEyeTests(unittest.TestCase):
    def test_judge_vector_prefers_matching_description(self):
        import numpy as np
        from album_builder import aesthetic
        pos, neg = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
        scores = aesthetic.judge_vector(np.array([0.9, 0.1]), {"beauty": (pos, neg)})
        self.assertGreater(scores["beauty"], 0.99)

    def test_letterbox_keeps_whole_image(self):
        import numpy as np
        from album_builder import aesthetic
        x = aesthetic.letterbox(np.zeros((640, 360, 3), dtype=np.uint8))
        self.assertEqual(x.shape, (1, 3, 224, 224))

    def test_editor_scores_change_ranking(self):
        f = feat(faces=[face()])
        dull, _ = scoring.local_score(f, set(), [], meta={"editor": {"beauty": 0.1, "emotion": 0.1}})
        nice, parts = scoring.local_score(f, set(), [], meta={"editor": {"beauty": 0.9, "emotion": 0.9}})
        self.assertGreater(nice, dull + 0.1)
        self.assertIn("beauty", parts)
        plain, parts = scoring.local_score(f, set(), [])
        self.assertNotIn("beauty", parts)


class TasteTests(unittest.TestCase):
    def data(self, n=40, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        editor, pos, neg = {}, [], []
        for i in range(n):
            liked = i % 2 == 0
            v = rng.normal(size=16) + (2.0 if liked else -2.0) * np.eye(16)[0]
            editor[f"p{i}"] = {"vec": (v / np.linalg.norm(v)).tolist()}
            (pos if liked else neg).append(f"p{i}")
        return editor, pos, neg

    def test_learns_and_generalises(self):
        from album_builder import taste
        editor, pos, neg = self.data()
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), pos, neg)
            self.assertGreater(taste.cross_val_auc(editor, taste.load_examples(Path(d))), 0.85)
            m = taste.model_for(Path(d), editor)
            self.assertGreater(m.predict(editor[pos[0]]["vec"]), m.predict(editor[neg[0]]["vec"]))

    def test_needs_enough_examples_and_last_opinion_wins(self):
        from album_builder import taste
        editor, pos, neg = self.data()
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), pos[:3], neg[:3])
            self.assertIsNone(taste.model_for(Path(d), editor))
            taste.add_examples(Path(d), [], [pos[0]])
            ex = taste.load_examples(Path(d))
            self.assertIn(pos[0], ex["neg"])
            self.assertNotIn(pos[0], ex["pos"])

    def test_seed_from_plan(self):
        import json
        from album_builder import taste
        with tempfile.TemporaryDirectory() as d:
            plan = Path(d) / "plan.json"
            plan.write_text(json.dumps({"photos": [{"id": "a"}], "reserve": [{"id": "b"}, {"id": "c"}]}))
            self.assertEqual(taste.seed_from_plan(Path(d), plan), (1, 2))
