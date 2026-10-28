"""Critères v3 : moments, variété, naissance, impression, apprentissage."""
import random

import numpy as np
import tempfile
import unittest
import unittest.mock
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
    """Goût appris sur des critères lisibles : ici, la famille n'aime que les fonds simples."""

    def data(self, n=40, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        parts, pos, neg = {}, [], []
        for i in range(n):
            liked = i % 2 == 0
            p = {k: float(rng.uniform(0.2, 0.8)) for k in ("sharpness", "smile", "emotion", "color")}
            p["clean"] = float(rng.uniform(0.6, 1.0) if liked else rng.uniform(0.0, 0.5))
            parts[f"p{i}"] = p
            (pos if liked else neg).append(f"p{i}")
        return parts, pos, neg

    def test_learns_what_matters_and_generalises(self):
        from album_builder import taste
        parts, pos, neg = self.data()
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), pos, neg, parts=parts)
            X, y = taste._dataset(taste.load_examples(Path(d)))
            self.assertGreater(taste.cross_val_auc(X, y), 0.9)
            m = taste.model_for(Path(d))
            self.assertEqual(m.explain(1)[0][0], "clean")
            self.assertGreater(m.predict(parts[pos[0]]), m.predict(parts[neg[0]]))

    def test_needs_enough_examples_and_last_opinion_wins(self):
        from album_builder import taste
        parts, pos, neg = self.data()
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), pos[:3], neg[:3], parts=parts)
            self.assertIsNone(taste.model_for(Path(d)))
            taste.add_examples(Path(d), [], [pos[0]])
            ex = taste.load_examples(Path(d))
            self.assertIn(pos[0], ex["neg"])
            self.assertEqual(ex["neg"][pos[0]], parts[pos[0]])   # critères gardés
            self.assertNotIn(pos[0], ex["pos"])

    def test_current_parts_override_saved(self):
        from album_builder import taste
        parts, pos, neg = self.data()
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), pos, neg)   # sans critères enregistrés
            self.assertIsNone(taste.model_for(Path(d)))
            self.assertIsNotNone(taste.model_for(Path(d), parts))

    def test_seed_from_plan(self):
        import json
        from album_builder import taste
        with tempfile.TemporaryDirectory() as d:
            plan = Path(d) / "plan.json"
            plan.write_text(json.dumps({"photos": [{"id": "a", "parts": {"smile": 1}}],
                                        "reserve": [{"id": "b"}, {"id": "c"}]}))
            self.assertEqual(taste.seed_from_plan(Path(d), plan), (1, 2))
            self.assertEqual(taste.load_examples(Path(d))["pos"]["a"], {"smile": 1})

    def test_unexclude_after_undo(self):
        from album_builder import taste
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), [], ["x"], parts={"x": {"smile": 0}})
            taste.exclude(Path(d), ["x"])
            taste.unexclude(Path(d), ["x"])
            self.assertEqual(taste.excluded(Path(d)), set())
            self.assertNotIn("x", taste.load_examples(Path(d))["neg"])


class FinalTouchTests(unittest.TestCase):
    def test_floor_leaves_weak_photos_out(self):
        t = datetime(2023, 1, 1, 10)
        good = [cand(i, t + timedelta(days=i), 0.8, dhash=h(i)) for i in range(5)]
        weak = cand(99, datetime(2023, 6, 1, 10), 0.2, dhash=h(99))  # seul mois de juin
        chosen = scoring.select_spread(good + [weak], 6, floor=0.3)
        self.assertNotIn("id99", {c.file_id for c in chosen})
        self.assertEqual(len(chosen), 5)

    def test_floor_slot_goes_to_another_period(self):
        t = datetime(2023, 1, 1, 10)
        good = [cand(i, t + timedelta(days=3 * i), 0.8, dhash=h(i)) for i in range(8)]
        weak = cand(99, datetime(2023, 6, 1, 10), 0.2, dhash=h(99))
        chosen = scoring.select_spread(good + [weak], 6, floor=0.3)
        self.assertEqual(len(chosen), 6)

    def test_exclusion_list(self):
        from album_builder import taste
        with tempfile.TemporaryDirectory() as d:
            taste.exclude(Path(d), ["a", "b"])
            taste.exclude(Path(d), ["b", "c"])
            self.assertEqual(taste.excluded(Path(d)), {"a", "b", "c"})

    def test_taste_needs_good_cross_validation(self):
        from album_builder import taste
        import numpy as np
        rng = np.random.default_rng(3)
        parts = {f"p{i}": {"smile": float(rng.uniform()), "clean": float(rng.uniform())} for i in range(40)}
        with tempfile.TemporaryDirectory() as d:
            taste.add_examples(Path(d), [f"p{i}" for i in range(20)], [f"p{i}" for i in range(20, 40)], parts=parts)
            self.assertIsNone(taste.model_for(Path(d)))   # bruit : pas de goût fiable
            self.assertIsNotNone(taste.model_for(Path(d), min_auc=0))


class AlbumRulesTests(unittest.TestCase):
    """Règles des éditeurs d'albums : pas de redite (même sous un autre angle), peu de routine."""

    def vcand(self, i, when, score, vec, scene=None):
        import numpy as np
        c = cand(i, when, score, dhash=h(i))
        c.meta = {"vec": (np.asarray(vec, float) / np.linalg.norm(vec)).tolist(), "scene": scene}
        return c

    def test_same_content_other_angle_is_duplicate(self):
        t = datetime(2023, 5, 7, 10)
        a = self.vcand(1, t, 0.9, [1, 0.1, 0])
        b = self.vcand(2, t + timedelta(minutes=6), 0.8, [1, 0.12, 0.02])   # même canapé, autre angle
        c = self.vcand(3, t + timedelta(minutes=8), 0.7, [0, 1, 0])
        self.assertEqual({x.file_id for x in scoring.dedupe_bursts([a, b, c])}, {"id1", "id3"})

    def test_routine_scenes_are_capped(self):
        t = datetime(2023, 1, 1, 9)
        routine = [self.vcand(i, t + timedelta(days=i), 0.9, [1, i, 0], "chaise haute" if i % 2 else "siège auto")
                   for i in range(8)]
        outings = [self.vcand(100 + i, t + timedelta(days=i, hours=6), 0.75, [0, 1, i], "dehors") for i in range(8)]
        with unittest.mock.patch.object(scoring, "ROUTINE_MAX", 3):  # désactivé par défaut (scènes peu fiables)
            chosen = scoring.select_spread(routine + outings, 8)
        self.assertLessEqual(sum(1 for c in chosen if c.meta["scene"] != "dehors"), 3)

    def test_composition_metrics(self):
        import cv2
        import numpy as np
        from album_builder import compose
        simple = np.full((480, 360, 3), 200, np.uint8)
        cv2.circle(simple, (180, 200), 60, (30, 60, 200), -1)
        busy = np.random.default_rng(0).integers(0, 255, (480, 360, 3)).astype(np.uint8)
        box = [130, 150, 100, 100]
        self.assertGreater(compose.measure(simple, [box])["clean"], compose.measure(busy, [box])["clean"])
        self.assertGreater(compose.placement(box, 360, 480), compose.placement([0, 150, 60, 60], 360, 480))
        tilt = np.full((360, 480), 255, np.uint8)
        cv2.line(tilt, (0, 150), (479, 200), 0, 3)
        cv2.line(tilt, (0, 250), (479, 300), 0, 3)
        self.assertLess(compose.level(tilt), 0.5)


class SeriesTests(unittest.TestCase):
    def test_file_numbers_and_series(self):
        self.assertEqual(scoring.file_number("IMG_0238.JPG"), 238)
        self.assertEqual(scoring.file_number("20210726_123722.jpg"), 123722)
        self.assertIsNone(scoring.file_number("photo.jpg"))
        self.assertTrue(scoring.same_series("IMG_0235.JPG", "IMG_0238.JPG", 0.82))
        self.assertFalse(scoring.same_series("IMG_0235.JPG", "IMG_0298.JPG", 0.82))
        self.assertTrue(scoring.same_series("A.jpg", "B.jpg", 0.91))
        self.assertFalse(scoring.same_series("A.jpg", "B.jpg", 0.86))


class ExclusionReasonTests(unittest.TestCase):
    def test_reasons_kept_and_old_format_read(self):
        import json
        from album_builder import taste
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".cache").mkdir()
            (Path(d) / ".cache" / "excluded.json").write_text(json.dumps(["old"]))
            taste.exclude(Path(d), ["twin"], "doublon")
            self.assertEqual(taste.excluded_reasons(Path(d)), {"old": "pas-belle", "twin": "doublon"})
            taste.unexclude(Path(d), ["twin"])
            self.assertEqual(taste.excluded(Path(d)), {"old"})

    def test_dedupe_works_when_some_vectors_missing(self):
        import numpy as np
        t = datetime(2023, 10, 12, 10)
        a, b = cand(1, t, 0.9, dhash=h(1)), cand(2, t + timedelta(minutes=2), 0.8, dhash=h(2))
        a.meta = {"vec": [1.0, 0.0]}
        b.meta = {"vec": (np.array([1.0, 0.4]) / np.linalg.norm([1.0, 0.4])).tolist()}   # cos 0,93
        lone = cand(3, t + timedelta(minutes=5), 0.7, dhash=h(3))   # sans vecteur
        self.assertEqual({c.file_id for c in scoring.dedupe_bursts([a, b, lone])}, {"id1", "id3"})


class ThemeTests(unittest.TestCase):
    def test_theme_words_found_in_description(self):
        from album_builder.intent import interpret, theme_of

        self.assertEqual(theme_of("nos vacances à la plage"), ("plage", True))
        self.assertEqual(theme_of("des câlins"), ("calins", False))
        self.assertEqual(theme_of("des câlins, seulement"), ("calins", False))  # une ambiance n'écarte pas
        self.assertEqual(theme_of("vacances, seulement"), ("vacances", False))
        self.assertEqual(theme_of("parce que je l'aime"), ("", False))
        it = interpret("30 photos de l'anniversaire d'Arié", ["Arié"])
        self.assertEqual(it.theme, "anniversaire")
        self.assertIn("thème : anniversaire (seulement)", it.understood)

    def test_theme_scores_relative_to_folder_and_event_days(self):
        from datetime import date

        from album_builder import scoring

        t = np.array([1.0, 0.0])
        def v(sim):
            return [sim, float(np.sqrt(1 - sim * sim))]
        d1, d2 = date(2026, 5, 1), date(2026, 6, 1)
        vecs, days = {}, {}
        for i in range(10):  # jour ordinaire : loin du thème
            vecs[f"o{i}"], days[f"o{i}"] = v(0.20), d2
        for i in range(4):   # anniversaire : gâteau bien visible…
            vecs[f"g{i}"], days[f"g{i}"] = v(0.30), d1
        vecs["invite"], days["invite"] = v(0.23), d1  # …et un invité, le même jour
        vecs["ailleurs"], days["ailleurs"] = v(0.23), d2
        out = scoring.theme_scores(vecs, days, t)
        self.assertTrue(out["g0"][1])
        self.assertGreater(out["g0"][0], 0.9)
        self.assertTrue(out["invite"][1])        # même journée que l'événement
        self.assertFalse(out["ailleurs"][1])     # même ressemblance, autre jour
        self.assertFalse(out["o0"][1])

    def test_lookalike_and_missing_theme(self):
        from datetime import date

        from album_builder import scoring

        t, empty_tub = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        def v(a, b):
            return [a, b, float(np.sqrt(1 - a * a - b * b))]
        vecs = {f"o{i}": v(0.20, 0.10) for i in range(10)}
        vecs["bain"] = v(0.32, 0.10)
        vecs["baignoire vide"] = v(0.31, 0.40)   # papa habillé dans la baignoire vide
        days = dict.fromkeys(vecs, date(2026, 1, 1))
        out = scoring.theme_scores(vecs, days, t, not_vec=empty_tub)
        self.assertTrue(out["bain"][1])
        self.assertFalse(out["baignoire vide"][1])
        self.assertFalse(scoring.theme_found(vecs, t))  # 2 photos seulement : pas un vrai thème
        vecs.update({f"b{i}": v(0.32, 0.10) for i in range(5)})
        self.assertTrue(scoring.theme_found(vecs, t))


class FamilyHolidayTests(unittest.TestCase):
    FAMILY = ["Eydel", "Arié", "Hillel", "Perla", "Nathanaël"]
    TODAY = date(2026, 10, 28)

    def test_jewish_holidays_dates(self):
        from album_builder.hebcal import holiday

        self.assertEqual(holiday("souccot", 2025), (date(2025, 10, 2), date(2025, 10, 15)))
        self.assertEqual(holiday("hanouka", 2025)[0], date(2025, 12, 14))
        self.assertEqual(holiday("pessah", 2025)[0], date(2025, 4, 12))
        self.assertEqual(holiday("pourim", 2024)[0], date(2024, 3, 23))
        self.assertEqual(holiday("roch-hachana", 2025)[0], date(2025, 9, 22))
        self.assertEqual(holiday("chavouot", 2025)[0], date(2025, 6, 1))

    def test_family_album_wants_anyone_of_the_family(self):
        it = interpret("Souccot 2025 en famille", self.FAMILY, today=self.TODAY)
        self.assertEqual((it.date_from, it.date_to), ("2025-10-02", "2025-10-15"))
        self.assertEqual(it.name, "Souccot 2025 en famille")
        self.assertTrue(it.family)
        self.assertEqual(it.must, [])
        self.assertEqual(it.bonus, self.FAMILY)
        self.assertEqual(it.theme, "souccot")
        self.assertFalse(it.theme_strict)  # les dates suffisent

        it = interpret("Eté 2025 en famille", self.FAMILY, selected=self.FAMILY, today=self.TODAY)
        self.assertEqual(it.must, [])           # pas les 5 sur chaque photo
        self.assertEqual(it.name, "Été 2025 en famille")

    def test_holiday_without_year_is_the_last_one(self):
        it = interpret("Hanouka", self.FAMILY, today=self.TODAY)
        self.assertEqual(it.date_from, "2025-12-14")

    def test_reference_taken_before_birth_is_ignored(self):
        refs = scoring.refs_at({"Arié": [([1.0, 0.0], date(2021, 7, 14)), ([0.0, 1.0], date(2022, 11, 1))]},
                               {"Arié": date(2022, 9, 9)}, date(2023, 6, 1))
        self.assertAlmostEqual(refs["Arié"][1], 1.0)

    def test_family_album_shares_places(self):
        items = []
        t0 = datetime(2025, 7, 1, 10)
        for i in range(40):  # Eydel seule : les plus belles photos
            items.append(cand(f"e{i}", t0 + timedelta(hours=3 * i), 0.9, dhash=format(random.Random(i).getrandbits(64), "016x")))
            items[-1].labels = {"Eydel"}
        for i in range(10):
            items.append(cand(f"p{i}", t0 + timedelta(days=10, hours=3 * i), 0.8, dhash=format(random.Random(100 + i).getrandbits(64), "016x")))
            items[-1].labels = {"Perla", "Arié"}
        plain = scoring.select_spread(items, 20)
        fam = scoring.select_spread(items, 20, family=["Eydel", "Arié", "Perla"])
        others = lambda sel: sum(1 for c in sel if "Perla" in c.labels)
        self.assertGreater(others(fam), others(plain))
        self.assertGreaterEqual(others(fam), 3)

    def test_period_containing_a_holiday_is_flagged(self):
        it = interpret("Vacances en Israël octobre 2025", self.FAMILY, today=self.TODAY)
        self.assertEqual(it.theme, "vacances")
        self.assertFalse(it.theme_strict)
        self.assertEqual(it.holiday_hint, "souccot")
        self.assertEqual(interpret("Été 2025", self.FAMILY, today=self.TODAY).holiday_hint, "")

    def test_flat_but_sharp_scene_is_not_blurry(self):
        f = feat()
        f["lap_center"], f["faces"] = 60.0, []
        low, _ = scoring.local_score(f, set(), [], scoring.Weights(), [], [], {})
        _, parts = scoring.local_score(f, set(), [], scoring.Weights(), [], [], {"comp": {"lap_tiles": 900.0}})
        self.assertLess(scoring.subject_sharpness(f, []), scoring.REJECT_SHARPNESS)
        self.assertGreater(parts["sharpness"], 0.5)
        self.assertNotIn("lap_tiles", parts)

    def test_people_album_keeps_little_decor(self):
        t0 = datetime(2025, 10, 5, 9)
        items = []
        for i in range(30):
            c = cand(f"d{i}", t0 + timedelta(hours=2 * i), 0.9, dhash=format(random.Random(500 + i).getrandbits(64), "016x"))
            items.append(c)  # pas de visage : rue, plats, affiches
        for i in range(30):
            c = cand(f"f{i}", t0 + timedelta(hours=2 * i + 1), 0.7, dhash=format(random.Random(900 + i).getrandbits(64), "016x"))
            c.feat["faces"] = [face(box=[100, 100, 200, 200])]
            items.append(c)
        sel = scoring.select_spread(items, 20, decor_max=scoring.DECOR_MAX_SHARE)
        self.assertLessEqual(sum(1 for c in sel if not c.feat["faces"]), 2)
        self.assertEqual(len(sel), 20)
