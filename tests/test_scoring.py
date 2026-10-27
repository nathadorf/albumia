import unittest
from datetime import datetime, timedelta

from album_builder import scoring
from album_builder.scoring import Candidate


def feat(lap=500.0, faces=None, mean=0.45, dhash="0" * 16):
    return {
        "w": 1000, "h": 750, "lap_center": lap, "mean": mean,
        "clip_low": 0.0, "clip_high": 0.0, "dhash": dhash, "faces": faces or [],
    }


def face(box=(100, 100, 250, 250), lap=300.0, smile=1.0, eyes=1.0, emb=None):
    f = {"box": list(box), "score": 0.95, "lap": lap, "smile": smile, "eyes": eyes}
    if emb is not None:
        f["emb"] = emb
    return f


def cand(i, when, score, dhash="0" * 16):
    c = Candidate(f"id{i}", f"IMG_{i}.CR2", when, feat(dhash=dhash))
    c.score = c.local_score = score
    return c


class SharpnessTests(unittest.TestCase):
    def test_monotonic_and_bounded(self):
        vals = [scoring.sharp_score(v, scoring.SHARP_RANGE_GLOBAL) for v in (0, 5, 50, 500, 5000)]
        self.assertEqual(vals, sorted(vals))
        self.assertEqual(vals[0], 0.0)
        self.assertEqual(vals[-1], 1.0)

    def test_blurry_photo_is_below_reject_threshold(self):
        self.assertLess(scoring.sharp_score(3, scoring.SHARP_RANGE_GLOBAL), scoring.REJECT_SHARPNESS)


class LocalScoreTests(unittest.TestCase):
    def test_smiling_sharp_face_beats_blurry_no_face(self):
        good, _ = scoring.local_score(feat(lap=600, faces=[face()]), set(), [])
        bad, _ = scoring.local_score(feat(lap=20), set(), [])
        self.assertGreater(good, bad)

    def test_bonus_label_raises_score(self):
        f = feat(faces=[face()])
        with_bonus, _ = scoring.local_score(f, {"freres"}, ["freres"])
        without, _ = scoring.local_score(f, set(), ["freres"])
        self.assertGreater(with_bonus, without)

    def test_score_stays_in_unit_interval(self):
        s, parts = scoring.local_score(feat(faces=[face(), face()]), {"a"}, ["a"])
        self.assertTrue(0.0 <= s <= 1.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in parts.values()))

    def test_exposure_penalises_dark_and_blown(self):
        ok = scoring.exposure_score(0.45, 0.0, 0.0)
        self.assertGreater(ok, scoring.exposure_score(0.05, 0.6, 0.0))
        self.assertGreater(ok, scoring.exposure_score(0.95, 0.0, 0.6))


class MatchingTests(unittest.TestCase):
    def test_match_labels(self):
        faces = [face(emb=[1.0, 0.0]), face(emb=[0.0, 1.0])]
        refs = {"fille": [1.0, 0.0], "autre": [0.7071, -0.7071]}
        found = scoring.match_labels(faces, refs, 0.8)
        self.assertEqual(found, {"fille"})

    def test_lookalike_siblings_are_not_both_matched(self):
        refs = {"eydel": [1.0, 0.0], "arie": [0.8, 0.6]}       # similarité 0,8 entre les deux
        ambiguous = face(emb=[0.949, 0.316])                    # 0,95 et 0,95 : on ne sait pas
        clear = face(emb=[1.0, 0.0])                            # 1,0 contre 0,8
        self.assertEqual(scoring.face_labels([ambiguous, clear], refs, 0.45), [None, "eydel"])
        self.assertEqual(scoring.match_labels([ambiguous, clear], refs, 0.45), {"eydel"})

    def test_tiny_false_face_is_ignored(self):
        f = feat(faces=[face(box=(10, 10, 15, 15)), face()])
        self.assertEqual(len(scoring.real_faces(f)), 1)

    def test_framing_judged_on_target_child_not_on_adult(self):
        adult = face(box=(0, 0, 500, 500))
        child = face(box=(800, 600, 100, 100))
        f = feat(faces=[adult, child])
        _, generic = scoring.local_score(f, {"eydel"}, [])
        _, focused = scoring.local_score(f, {"eydel"}, [], labels_per_face=[None, "eydel"], targets=["eydel"])
        self.assertLess(focused["faces"], generic["faces"])

    def test_no_refs_or_faces(self):
        self.assertEqual(scoring.match_labels([], {"a": [1.0]}, 0.3), set())
        self.assertEqual(scoring.match_labels([face(emb=[1.0])], {}, 0.3), set())


class DedupeTests(unittest.TestCase):
    def test_burst_keeps_best_only(self):
        t = datetime(2026, 5, 1, 12, 0, 0)
        burst = [cand(i, t + timedelta(seconds=i * 2), s) for i, s in enumerate([0.4, 0.9, 0.5])]
        other = cand(9, t + timedelta(hours=3), 0.3, dhash="f" * 16)
        kept = scoring.dedupe_bursts(burst + [other])
        self.assertEqual({c.file_id for c in kept}, {"id1", "id9"})

    def test_similar_hash_far_in_time_is_kept(self):
        t = datetime(2026, 5, 1)
        a = cand(1, t, 0.5)
        b = cand(2, t + timedelta(days=2), 0.6, dhash="000000000000003f")  # 6 bits d'écart
        self.assertEqual(len(scoring.dedupe_bursts([a, b])), 2)

    def test_identical_copy_far_in_time_is_removed(self):
        t = datetime(2023, 5, 1)
        a, b = cand(1, t, 0.5), cand(2, t + timedelta(days=400), 0.6)  # même image, recopiée en 2024
        self.assertEqual([c.file_id for c in scoring.dedupe_bursts([a, b])], ["id2"])

    def test_same_scene_within_fifteen_minutes(self):
        t = datetime(2026, 5, 1, 12)
        a = cand(1, t, 0.5)
        b = cand(2, t + timedelta(minutes=10), 0.6, dhash="00000000000003ff")  # 10 bits d'écart
        self.assertEqual(len(scoring.dedupe_bursts([a, b])), 1)

    def test_close_in_time_but_different_scene_is_kept(self):
        t = datetime(2026, 5, 1)
        a = cand(1, t, 0.5, dhash="0" * 16)
        b = cand(2, t + timedelta(seconds=10), 0.6, dhash="f" * 16)
        self.assertEqual(len(scoring.dedupe_bursts([a, b])), 2)


class AllocationTests(unittest.TestCase):
    def test_sum_and_caps(self):
        counts = {"jan": 800, "feb": 40, "mar": 3}
        alloc = scoring.allocate(counts, 50)
        self.assertEqual(sum(alloc.values()), 50)
        for k in counts:
            self.assertLessEqual(alloc[k], counts[k])

    def test_sqrt_damping(self):
        alloc = scoring.allocate({"big": 900, "small": 100}, 40)
        self.assertEqual(sum(alloc.values()), 40)
        self.assertGreater(alloc["small"], 40 * 100 / 1000)  # plus que le prorata brut

    def test_total_larger_than_available(self):
        self.assertEqual(sum(scoring.allocate({"a": 2, "b": 1}, 10).values()), 3)

    def test_select_spread_covers_months_and_is_chronological(self):
        items = []
        for m in range(1, 13):
            for k in range(10):
                items.append(cand(m * 100 + k, datetime(2026, m, 1 + k), k / 10))
        chosen = scoring.select_spread(items, 24)
        self.assertEqual(len(chosen), 24)
        self.assertEqual({c.when.month for c in chosen}, set(range(1, 13)))
        self.assertEqual(chosen, sorted(chosen, key=lambda c: c.when))
        # dans chaque mois on prend les mieux notées
        for c in chosen:
            self.assertGreaterEqual(c.score, 0.7)


if __name__ == "__main__":
    unittest.main()
