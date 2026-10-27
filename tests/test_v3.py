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
