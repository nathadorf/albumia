"""Le tri d'un photographe : moments, prises, chapitres, couverture, famille présente."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from album_builder import curate, scoring
from tests.test_scoring import cand


def unit(*xs):
    v = np.array(xs, dtype=float)
    return list(v / np.linalg.norm(v))


def photo(i, when, score=0.5, vec=None, vec_l=None, dhash=None, labels=(), shot="plan moyen", note="", **parts):
    c = cand(i, when, score, dhash=dhash or format((i * 0x9E3779B97F4A7C15) % 2**64, "016x"))
    c.labels, c.note, c.parts = set(labels), note, dict(parts)
    c.meta = {"vec": vec, "vec_l": vec_l, "shot": shot}
    return c


T0 = datetime(2025, 10, 7, 10, 0)
A, B = unit(1, 0, 0), unit(0, 1, 0)


class MomentTests(unittest.TestCase):
    def test_close_similar_shots_make_one_moment(self):
        items = [photo(1, T0, vec_l=A), photo(2, T0 + timedelta(minutes=3), vec_l=A),
                 photo(3, T0 + timedelta(minutes=6), vec_l=A),
                 photo(4, T0 + timedelta(minutes=4), vec_l=B),          # autre scène au même moment
                 photo(5, T0 + timedelta(hours=1), vec_l=A)]            # même scène, une heure plus tard
        groups = curate.find_moments(items)
        self.assertEqual(sorted(sorted(c.file_id for c in g) for g in groups),
                         [["id1", "id2", "id3"], ["id4"], ["id5"]])
        self.assertEqual(items[0].meta["moment"], items[1].meta["moment"])
        self.assertNotEqual(items[0].meta["moment"], items[3].meta["moment"])

    def test_the_eye_used_is_the_precise_one_when_both_photos_have_it(self):
        # l'œil habituel les trouve voisines (0,9), l'œil plus précis non (0,6) : le précis décide
        a = photo(1, T0, vec=unit(1, 0.1, 0), vec_l=unit(1, 0, 0))
        b = photo(2, T0 + timedelta(minutes=2), vec=unit(1, 0.1, 0), vec_l=unit(0.6, 0.8, 0))
        self.assertEqual(len(curate.find_moments([a, b])), 2)

    def test_without_vectors_only_look_alikes_taken_one_after_the_other_group(self):
        base = "0" * 16
        near = format(0xFF, "016x")                                     # 8 bits d'écart
        quick = [photo(1, T0, dhash=base), photo(2, T0 + timedelta(seconds=30), dhash=near)]
        slow = [photo(3, T0, dhash=base), photo(4, T0 + timedelta(minutes=5), dhash=near)]
        self.assertEqual(len(curate.find_moments(quick)), 1)
        self.assertEqual(len(curate.find_moments(slow)), 2)

    def test_a_moment_lasts_half_an_hour_at_most(self):
        items = [photo(i, T0 + timedelta(minutes=5 * i), vec_l=A) for i in range(10)]   # 45 minutes de suite
        self.assertGreaterEqual(len(curate.find_moments(items)), 2)


class TakeTests(unittest.TestCase):
    def test_natural_take_beats_a_sharper_stiff_one(self):
        natural = photo(1, T0, 0.5, vec_l=A, frontal=1.0, eyes=1.0, smile=0.9, sharpness=0.5, labels={"Eydel"})
        stiff = photo(2, T0 + timedelta(minutes=1), 0.5, vec_l=A, frontal=0.2, eyes=0.5, smile=0.2,
                      sharpness=1.0, labels={"Eydel"})
        self.assertGreater(curate.take_value(natural), curate.take_value(stiff))
        picks, others = curate.heroes(curate.find_moments([stiff, natural]))
        self.assertEqual([c.file_id for c in picks], ["id1"])
        self.assertEqual([c.file_id for c in others[0]], ["id2"])

    def test_what_the_family_kept_counts_more_than_open_eyes_and_sharpness(self):
        # 35 paires relues : émotion, gestes et personnes ont départagé ; yeux, sourire et netteté non
        alive = photo(1, T0, 0.55, vec_l=A, labels={"Eydel", "Arié"}, emotion_l=0.9, interaction=0.9, frontal=0.6,
                      eyes=0.2, smile=0.2, sharpness=0.4)
        stiff = photo(2, T0 + timedelta(minutes=1), 0.5, vec_l=A, labels={"Eydel"}, emotion_l=0.3, interaction=0.2,
                      frontal=0.6, eyes=1.0, smile=1.0, sharpness=1.0)
        self.assertGreater(curate.take_value(alive), curate.take_value(stiff))
        self.assertGreater(curate.take_value(stiff, curate.TAKE_PRESETS["avant"]) + 0.2,
                           curate.take_value(alive, curate.TAKE_PRESETS["avant"]))   # l'ancien réglage le payait

    def test_more_of_the_family_wins_a_tie(self):
        alone = photo(1, T0, 0.5, vec_l=A, labels={"Eydel"})
        together = photo(2, T0 + timedelta(minutes=1), 0.5, vec_l=A, labels={"Eydel", "Perla"})
        self.assertEqual(curate.heroes(curate.find_moments([alone, together]))[0][0].file_id, "id2")

    def test_a_strong_moment_gives_a_second_photo_with_someone_else(self):
        takes = [photo(i, T0 + timedelta(minutes=i), 0.6 - 0.005 * i, vec_l=A, labels={"Eydel"}, frontal=1.0)
                 for i in range(5)]
        takes.append(photo(9, T0 + timedelta(minutes=6), 0.58, vec_l=A, labels={"Arié"}, frontal=0.9))
        picks, _ = curate.heroes(curate.find_moments(takes))
        self.assertEqual(len(picks), 2)
        self.assertEqual({tuple(sorted(c.labels)) for c in picks}, {("Eydel",), ("Arié",)})
        weak = curate.find_moments([photo(i, T0 + timedelta(minutes=i), vec_l=A, labels={"Eydel"}) for i in range(4)])
        self.assertEqual(len(curate.heroes(weak)[0]), 1)

    def test_five_identical_takes_stay_one_photo(self):
        takes = [photo(i, T0 + timedelta(minutes=i), 0.6, vec_l=A, labels={"Eydel"}) for i in range(6)]
        self.assertEqual(len(curate.heroes(curate.find_moments(takes))[0]), 1)

    def test_removed_takes_count_for_the_strength_of_a_moment_but_are_never_offered(self):
        shots = ["gros plan", "plan moyen"]
        takes = [photo(i, T0 + timedelta(minutes=i), 0.7 - 0.05 * i, vec_l=A, labels={"Eydel"}, shot=shots[i % 2])
                 for i in range(2)]
        gone = [photo(10 + i, T0 + timedelta(minutes=3 + i), 0.99, vec_l=A, labels={"Eydel"}) for i in range(3)]
        for c in gone:
            c.meta["banned"] = True
        groups = curate.find_moments(takes + gone)
        self.assertEqual(len(groups), 1)
        picks, others = curate.heroes(groups)
        self.assertEqual(len(picks), 2)                                  # 5 prises en tout : un moment fort
        self.assertFalse({c.file_id for c in picks + others[0]} & {"id10", "id11", "id12"})
        alone, _ = curate.heroes(curate.find_moments(takes))             # sans elles : 2 prises, une seule photo
        self.assertEqual(len(alone), 1)
        only_gone = curate.find_moments([photo(20, T0 + timedelta(days=2), 0.9, vec_l=B)])
        only_gone[0][0].meta["banned"] = True
        self.assertEqual(curate.heroes(only_gone), ([], {0: []}))

    def test_a_kept_photo_is_always_the_photo_of_its_moment(self):
        best = photo(1, T0, 0.9, vec_l=A, frontal=1.0, eyes=1.0)
        kept = photo(2, T0 + timedelta(minutes=1), 0.2, vec_l=A)
        picks, _ = curate.heroes(curate.find_moments([best, kept]), pinned={"id2"})
        self.assertEqual([c.file_id for c in picks], ["id2"])


class RuleTests(unittest.TestCase):
    def test_the_same_pose_is_never_taken_twice(self):
        a = photo(1, T0, 0.9, vec_l=A)
        again = photo(2, T0 + timedelta(hours=3), 0.85, vec_l=unit(1, 0.2, 0))   # cos 0,98 : même pose
        other = photo(3, T0 + timedelta(hours=6), 0.5, vec_l=B)
        both = {c.file_id for c in scoring.select_spread([a, again, other], 2)}
        self.assertEqual(both, {"id1", "id2"})
        kept = {c.file_id for c in scoring.select_spread([a, again, other], 2, pose_cos=curate.POSE_COS)}
        self.assertEqual(kept, {"id1", "id3"})

    def test_an_activity_is_capped(self):
        pool = [photo(i, T0 + timedelta(hours=3 * i), 0.9 - 0.01 * i, note="Piscine") for i in range(6)]
        pool += [photo(10 + i, T0 + timedelta(hours=40 + 3 * i), 0.5, note="Portrait") for i in range(4)]
        pool += [photo(20 + i, T0 + timedelta(hours=80 + 3 * i), 0.4, note="Plage") for i in range(4)]
        chosen = scoring.select_spread(pool, 8, scene_cap=2, scene_exempt={"Portrait"})
        self.assertEqual(sum(1 for c in chosen if c.note == "Piscine"), 2)
        self.assertEqual(sum(1 for c in chosen if c.note == "Portrait"), 4)    # un genre, pas une activité
        self.assertEqual(len(chosen), 8)

    def test_wide_shots_are_capped(self):
        pool = [photo(i, T0 + timedelta(hours=3 * i), 0.9 - 0.01 * i, shot="scène") for i in range(6)]
        pool += [photo(10 + i, T0 + timedelta(hours=40 + 3 * i), 0.5, shot="plan moyen") for i in range(6)]
        chosen = scoring.select_spread(pool, 8, wide_max=curate.wide_cap(8))
        self.assertEqual(sum(1 for c in chosen if c.meta["shot"] == "scène"), 2)
        self.assertEqual(len(chosen), 8)

    def test_a_big_outing_made_of_many_takes_deserves_more_than_one_photo(self):
        basis = lambda i: unit(*[1.0 if j == i else 0.0 for j in range(8)])
        outing = [photo(i, T0 + timedelta(minutes=40 * i), 0.9 - 0.01 * i, vec_l=basis(i)) for i in range(4)]
        later = [photo(10 + i, T0 + timedelta(days=3 + 2 * i), 0.5, vec_l=basis(4 + i)) for i in range(2)]
        small = scoring.select_spread(outing + later, 3, event_cap=curate.event_cap)
        big = scoring.select_spread(outing + later, 3, event_cap=curate.event_cap,
                                    size_of={c.file_id: 12 for c in outing})
        from_outing = lambda sel: sum(1 for c in sel if c.file_id in {o.file_id for o in outing})
        self.assertEqual(from_outing(small), 1)    # 4 photos prises en 2 h : un seul souvenir
        self.assertEqual(from_outing(big), 3)      # 48 prises : jusqu'à 3 (4 au plus)

    def test_penalties(self):
        c = photo(1, T0, 0.8)
        c.meta.update(tiny=True, adult_only=True)
        curate.penalize(c)
        self.assertAlmostEqual(c.score, 0.8 - curate.TINY_PENALTY - curate.ADULT_ALONE_PENALTY)
        ok = photo(2, T0, 0.8)
        curate.penalize(ok)
        self.assertEqual(ok.score, 0.8)

    def test_caps_scale_with_the_album(self):
        self.assertEqual(curate.scene_cap(31), 2)
        self.assertEqual(curate.scene_cap(100), 8)
        self.assertEqual(curate.wide_cap(31), 8)


class ExtraTakesTests(unittest.TestCase):
    """Un dossier avec peu de moments : les moments forts donnent des prises en plus."""

    def setUp(self):
        self.base = lambda i: unit(*[1.0 if j == i else 0.0 for j in range(16)])
        self.heroes = [photo(1, T0, 0.8, vec_l=self.base(0), labels={"Eydel"}, shot="plan moyen"),
                       photo(2, T0 + timedelta(hours=5), 0.7, vec_l=self.base(1), labels={"Arié"}, shot="plan moyen")]
        for k, c in enumerate(self.heroes):
            c.meta["moment"] = k

    def take(self, i, moment, vec, **kw):
        c = photo(i, T0 + timedelta(minutes=i), kw.pop("score", 0.6), vec_l=vec, **kw)
        c.meta["moment"] = moment
        return c

    def test_a_strong_moment_gives_a_take_with_a_new_shot_or_person(self):
        wide = self.take(10, 0, self.base(2), labels={"Eydel"}, shot="scène")
        again = self.take(11, 0, self.base(3), labels={"Eydel"}, shot="plan moyen")       # rien de neuf
        close = self.take(12, 0, self.base(4), labels={"Eydel", "Perla"}, shot="plan moyen", score=0.5)  # quelqu'un d'autre
        out = curate.add_extra_takes(self.heroes, {0: [wide, again, close], 1: []}, {0: 30, 1: 1}, 5)
        ids = {c.file_id for c in out}
        self.assertEqual(ids, {"id1", "id2", "id10", "id12"})
        self.assertEqual(len(out), 4)                            # moment de 30 prises : 3 photos au plus (event_cap)

    def test_a_small_moment_stays_one_photo_and_the_album_never_exceeds_the_count(self):
        take = self.take(10, 1, self.base(2), labels={"Arié", "Perla"}, shot="scène")
        self.assertEqual(len(curate.add_extra_takes(self.heroes, {0: [], 1: [take]}, {0: 1, 1: 3}, 5)), 2)
        strong = [self.take(20 + i, 0, self.base(5 + i), labels={"Eydel", f"P{i}"}) for i in range(5)]
        out = curate.add_extra_takes(self.heroes, {0: strong, 1: []}, {0: 60, 1: 1}, 3)
        self.assertEqual(len(out), 3)

    def test_never_the_same_image_a_selfie_a_meal_or_tiny_faces(self):
        same = self.take(10, 0, unit(1, 0.1, *[0] * 14), labels={"Eydel", "Perla"})      # 0,99 de ressemblance
        flawed = []
        for k, flaw in enumerate(("selfie", "adult_only", "meal", "tiny")):
            c = self.take(30 + k, 0, self.base(6 + k), labels={"Eydel", "Perla"}, shot="scène")
            c.meta[flaw] = True
            flawed.append(c)
        out = curate.add_extra_takes(self.heroes, {0: [same] + flawed, 1: []}, {0: 60, 1: 1}, 6)
        self.assertEqual(len(out), 2)

    def test_the_strongest_moment_comes_first(self):
        weak = self.take(10, 0, self.base(2), labels={"Eydel", "Perla"}, shot="scène", score=0.7)
        strong = self.take(11, 1, self.base(3), labels={"Arié", "Perla"}, shot="scène", score=0.7)
        out = curate.add_extra_takes(self.heroes, {0: [weak], 1: [strong]}, {0: 5, 1: 40}, 3)
        self.assertEqual({c.file_id for c in out} - {"id1", "id2"}, {"id11"})


class ChapterTests(unittest.TestCase):
    def days(self, n_days=3, per_day=8):
        out = []
        for d in range(n_days):
            for k in range(per_day):
                out.append(photo(d * 100 + k, T0 + timedelta(days=d, minutes=25 * k), vec_l=A, note="Piscine"))
        return out

    def test_cuts_at_the_big_gaps(self):
        chapters = curate.cut_chapters(self.days(), target=3)
        self.assertEqual([len(c) for c in chapters], [8, 8, 8])
        self.assertEqual([c[0].when.day for c in chapters], [7, 8, 9])

    def test_number_of_chapters_follows_the_size_of_the_album(self):
        items = []
        for i in range(31):  # 31 photos sur 12 jours
            items.append(photo(i, T0 + timedelta(days=i * 12 // 31, hours=i % 3 * 2), vec_l=A))
        chapters = curate.cut_chapters(items)
        self.assertEqual(len(chapters), 6)
        self.assertTrue(all(3 <= len(c) <= 8 for c in chapters), [len(c) for c in chapters])
        self.assertEqual(sorted((c for ch in chapters for c in ch), key=lambda c: c.when),
                         sorted(items, key=lambda c: c.when))

    def test_a_scene_change_cuts_when_times_are_even(self):
        items = [photo(i, T0 + timedelta(minutes=30 * i), vec_l=A if i < 6 else B) for i in range(12)]
        chapters = curate.cut_chapters(items, target=2)
        self.assertEqual([len(c) for c in chapters], [6, 6])

    def test_small_albums_have_one_chapter(self):
        self.assertEqual(len(curate.cut_chapters(self.days(1, 7))), 1)
        self.assertEqual(curate.cut_chapters([]), [])

    def test_titles(self):
        d = lambda *a: datetime(*a)
        self.assertEqual(curate.date_label(d(2025, 10, 7, 9), d(2025, 10, 7, 18)), "Mardi 7 octobre")
        self.assertEqual(curate.date_label(d(2025, 10, 1, 9), d(2025, 10, 3, 18)), "Du 1er au 3 octobre")
        self.assertEqual(curate.date_label(d(2025, 9, 28), d(2025, 10, 2)), "Du 28 septembre au 2 octobre")
        self.assertEqual(curate.date_label(d(2025, 3, 2), d(2025, 5, 20)), "Mars à mai 2025")
        self.assertEqual(curate.date_label(d(2024, 11, 2), d(2025, 2, 20)), "Novembre 2024 – février 2025")
        group = [photo(i, T0 + timedelta(minutes=i), note=n) for i, n in enumerate(["Piscine", "Piscine", "Portrait"])]
        self.assertEqual(curate.chapter_title(group), "Mardi 7 octobre · Piscine")
        lone = [photo(1, T0, note="Piscine"), photo(2, T0, note="Plage")]
        self.assertEqual(curate.chapter_title(lone), "Mardi 7 octobre")

    def test_a_scene_names_the_chapter_only_when_it_is_half_of_it(self):
        notes = lambda *names: [photo(i, T0 + timedelta(minutes=i), note=n) for i, n in enumerate(names)]
        mixed = notes("Piscine", "Piscine", "Rue", "Toboggan", "Plage", "Câlin")      # 2 sur 6 : le titre mentirait
        self.assertEqual(curate.chapter_title(mixed), "Mardi 7 octobre")
        half = notes("Piscine", "Piscine", "Rue", "Toboggan")                         # 2 sur 4 : la moitié
        self.assertEqual(curate.chapter_title(half), "Mardi 7 octobre · Piscine")
        # « Portrait », « En famille »… sont des genres de photos, jamais le thème d'un chapitre
        generic = notes("Portrait", "Portrait", "Portrait", "Piscine")
        self.assertEqual(curate.chapter_title(generic), "Mardi 7 octobre")


class CoverAndFamilyTests(unittest.TestCase):
    def test_the_cover_is_a_group_facing_the_camera(self):
        solo = photo(1, T0, 0.8, labels={"Eydel"}, frontal=1.0, smile=1.0, beauty=0.9, sharpness=0.9)
        group = photo(2, T0 + timedelta(hours=1), 0.7, labels={"Eydel", "Arié", "Hillel"}, frontal=0.9,
                      smile=0.8, beauty=0.8, sharpness=0.8)
        self.assertEqual(curate.pick_cover([solo, group], ["Eydel", "Arié", "Hillel"]), "id2")
        self.assertIsNone(curate.pick_cover([]))

    def test_the_cover_shows_the_children_not_only_the_parents(self):
        family = ["Eydel", "Arié", "Perla", "Nathanaël"]
        parents = photo(1, T0, 0.9, labels={"Perla", "Nathanaël"}, frontal=1.0, smile=1.0, beauty=0.9, sharpness=0.9)
        parents.meta["family_area"] = 0.05
        kids = photo(2, T0 + timedelta(hours=1), 0.7, labels={"Eydel", "Arié"}, frontal=0.8, smile=0.8, beauty=0.7,
                     sharpness=0.7)
        kids.meta["family_area"] = 0.03
        self.assertEqual(curate.pick_cover([parents, kids], family, kids=["Eydel", "Arié"]), "id2")

    def test_the_cover_is_never_a_selfie_a_meal_or_tiny_faces_when_something_else_exists(self):
        family = ["Eydel", "Arié", "Perla"]
        plain = photo(1, T0, 0.5, labels={"Eydel"}, frontal=0.6, smile=0.6)
        plain.meta["family_area"] = 0.02
        for flaw in ("selfie", "adult_only", "meal", "tiny"):
            flawed = photo(2, T0 + timedelta(hours=1), 0.95, labels={"Eydel", "Arié", "Perla"}, frontal=1.0, smile=1.0,
                           beauty=1.0, sharpness=1.0)
            flawed.meta.update({flaw: True, "family_area": 0.05})
            self.assertEqual(curate.pick_cover([plain, flawed], family, kids=["Eydel", "Arié"]), "id1", flaw)
        only = [flawed]
        self.assertEqual(curate.pick_cover(only, family), "id2")        # rien d'autre : on prend quand même

    def pool(self):
        t = T0
        out = []
        who = [{"Eydel", "Arié"}, {"Eydel"}, {"Arié"}, {"Eydel", "Arié", "Hillel", "Perla"}, {"Hillel"},
               {"Hillel"}, {"Perla"}, {"Eydel", "Perla"}, {"Arié", "Hillel"}, {"Eydel"}, {"Arié"}, {"Hillel"}]
        for i, w in enumerate(who):
            c = photo(i, t + timedelta(hours=3 * i), 0.9 - 0.02 * i,
                      vec_l=unit(*[1.0 if j == i else 0.0 for j in range(16)]), labels=w)
            c.meta["moment"] = i
            out.append(c)
        out.append(photo(50, t + timedelta(hours=60), 0.1, vec_l=unit(*[1.0 if j == 14 else 0.0 for j in range(16)]),
                         labels={"Nathanaël"}))  # médiocre
        out[-1].meta["moment"] = 50
        return out

    def test_everyone_is_present_and_a_group_photo_comes_first(self):
        family = ["Eydel", "Arié", "Hillel", "Perla"]
        pool = self.pool()
        seeds = curate.coverage_seeds(pool, family, babies=["Hillel"], limit=8)
        chosen = [c for c in pool if c.file_id in seeds]
        self.assertIn("id3", seeds)                                    # la photo de groupe (4 de la famille)
        for person, need in (("Eydel", 2), ("Arié", 2), ("Hillel", 3), ("Perla", 2)):
            self.assertGreaterEqual(sum(person in c.labels for c in chosen), need, person)
        self.assertNotIn("id50", seeds)                                # une photo médiocre n'est jamais imposée
        self.assertLessEqual(len(seeds), 8)

    def test_seeds_respect_moments_poses_selfies_and_limit(self):
        pool = self.pool()
        pool[1].meta["moment"] = pool[0].meta["moment"] = 0            # même moment que la photo 0
        pool[2].meta["selfie"] = True
        seeds = curate.coverage_seeds(pool, ["Eydel", "Arié"], limit=8)
        self.assertFalse({"id0", "id1"} <= seeds)
        self.assertNotIn("id2", seeds)
        self.assertLessEqual(len(curate.coverage_seeds(pool, ["Eydel", "Arié", "Hillel", "Perla"], limit=2)), 2)
        self.assertEqual(curate.coverage_seeds(pool, []), set())


class ReserveAndLayoutTests(unittest.TestCase):
    def setUp(self):
        self.a = photo(1, T0, 0.9, vec_l=A)
        self.a2 = photo(2, T0 + timedelta(minutes=2), 0.8, vec_l=A)
        self.a3 = photo(3, T0 + timedelta(minutes=3), 0.7, vec_l=A)
        self.b = photo(4, T0 + timedelta(days=2), 0.85, vec_l=B)
        self.c = photo(5, T0 + timedelta(days=2, hours=2), 0.6, vec_l=unit(0, 0, 1))
        self.pool_all = [self.a, self.a2, self.a3, self.b, self.c]
        self.groups = curate.find_moments(self.pool_all)
        self.pool, self.others = curate.heroes(self.groups)

    def test_reserve_starts_with_other_takes_of_the_chosen_moments(self):
        selected = [c for c in self.pool if c.file_id in ("id1", "id4")]
        reserve = curate.build_reserve(selected, self.others, self.pool, limit=5)
        self.assertEqual([c.file_id for c in reserve], ["id2", "id3", "id5"])

    def test_layout_gives_chapters_cover_and_places_the_reserve(self):
        selected = [photo(i, T0 + timedelta(days=i // 8, minutes=20 * i), vec_l=A) for i in range(24)]
        spare = photo(99, T0 + timedelta(days=2, hours=5), vec_l=A)
        out = curate.layout(selected, [spare], ["Eydel"])
        self.assertEqual(len(out["chapters"]), 4)
        self.assertIn(out["cover"], {c.file_id for c in selected})
        self.assertEqual(out["of"]["id0"]["chapter"], 0)
        last = len(out["chapters"]) - 1
        self.assertEqual(out["of"]["id99"]["chapter"], last)        # prise le dernier jour : dernier chapitre
        self.assertEqual({k for k in out["chapters"][0]}, {"title", "start", "end"})


class CompareTests(unittest.TestCase):
    def test_compare_counts_same_photos_and_same_moments(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / ".cache").mkdir()
            vec = unit(1, 0, 0)
            (work / ".cache" / "editor.json").write_text(json.dumps({f"p{i}": {"vec": vec} for i in range(6)}))
            (work / ".cache" / "features.json").write_text(json.dumps(
                {f"p{i}": {"feat": {"dhash": format(i * 0x0F0F, "016x"), "w": 10, "h": 10, "faces": []}} for i in range(6)}))
            item = lambda i, minutes: {"id": f"p{i}", "name": f"IMG_{i}", "date": (T0 + timedelta(minutes=minutes)).isoformat(),
                                       "score": 0.5, "parts": {}, "people": [], "note": "", "shot": ""}
            auto = {"photos": [item(0, 0), item(1, 120), item(2, 300)], "reserve": [item(3, 2)]}
            ref = {"photos": [item(0, 0), item(3, 2), item(4, 600), item(5, 0)],
                   "review": {"ids": ["p0", "p3", "p4"], "removed": []}}
            (work / "auto.json").write_text(json.dumps(auto))
            (work / "ref.json").write_text(json.dumps(ref))
            lines = []
            out = curate.compare(work / "auto.json", work / "ref.json", work, out=lines.append)
        self.assertEqual(out["exact"], ["p0"])
        self.assertEqual(out["same_moment"], ["p3"])      # p3 est une autre prise du moment de p0
        self.assertEqual(out["missing"], ["p4"])
        self.assertEqual(sorted(out["extra"]), ["p1", "p2"])
        self.assertIn("Sélection à la main : 3 photos", lines[0])

    def test_compare_tells_why_a_wanted_photo_is_missing_and_where_the_moments_are_too_wide(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / ".cache").mkdir()
            (work / ".cache" / "editor.json").write_text(json.dumps({
                "p0": {"vec": unit(1, 0, 0)}, "p1": {"vec": unit(1, 0.4, 0)}, "p2": {"vec": unit(0, 0, 1)},
                "p3": {"vec": unit(0, 1, 0)}}))
            (work / ".cache" / "features.json").write_text(json.dumps(
                {f"p{i}": {"feat": {"dhash": format(i * 0x0F0F, "016x"), "w": 10, "h": 10, "faces": []}} for i in range(4)}))
            (work / ".cache" / "rejected.json").write_text(json.dumps({
                "p3": {"name": "IMG_3", "why": "sans intérêt", "scene": "Rue", "gap": 0.031, "face": 0.012},
                "p2": {"name": "IMG_2", "why": "pas choisie", "scene": "Valises", "score": 0.35,
                       "twin": ["IMG_9", 0.72], "blocked": ["scène Valises 2/2", "plans larges 7/7"]}}))
            (work / ".cache" / "moments.json").write_text(json.dumps([
                {"start": "2025-10-07T10:00", "end": "2025-10-07T10:07", "takes": 7, "removed_before": 3,
                 "ids": ["p0", "p1"], "chosen": ["p0"]},
                {"start": "2025-10-07T20:00", "end": "2025-10-07T20:00", "takes": 1, "removed_before": 0,
                 "ids": ["p3"], "chosen": []}]))
            item = lambda i, minutes: {"id": f"p{i}", "name": f"IMG_{i}", "date": (T0 + timedelta(minutes=minutes)).isoformat(),
                                       "score": 0.5, "parts": {}, "people": [], "note": "", "shot": ""}
            auto = {"photos": [item(2, 300)], "reserve": []}
            ref = {"photos": [item(0, 0), item(1, 7), item(3, 600)],
                   "review": {"ids": ["p0", "p1", "p3"], "removed": []}}
            (work / "auto.json").write_text(json.dumps(auto))
            (work / "ref.json").write_text(json.dumps(ref))
            lines = []
            out = curate.compare(work / "auto.json", work / "ref.json", work, out=lines.append)
        text = "\n".join(lines)
        self.assertIn("IMG_3", text)
        self.assertIn("sans intérêt (scène Rue, écart +0.031, visage 1.2%)", text)
        self.assertEqual(len(out["merged"]), 1)                   # p0 et p1 : un seul moment pour Albumia, deux pour la famille
        self.assertEqual(out["merged"][0]["ids"], ["p0", "p1"])
        self.assertAlmostEqual(out["merged"][0]["span_min"], 7)
        self.assertLess(out["merged"][0]["min_cos"], 0.95)
        self.assertIn("IMG_0, IMG_1", text)
        self.assertIn("ressemble à IMG_9 (0.72)", text)
        self.assertIn("plafonds : scène Valises 2/2, plans larges 7/7", text)
        self.assertIn("7 prises (3 retirées avant), à la main 2, album automatique 0", text)

    def test_the_table_of_strong_moments(self):
        moments = [{"start": "a", "end": "b", "takes": 9, "removed_before": 4, "ids": ["x", "y", "z"], "chosen": ["x"]},
                   {"start": "c", "end": "d", "takes": 2, "removed_before": 0, "ids": ["u", "v"], "chosen": []},
                   {"start": "e", "end": "f", "takes": 6, "removed_before": 0, "ids": ["w"], "chosen": []}]
        table = curate.moments_table(moments, ["x", "y", "u"], ["x"])
        self.assertEqual([(r["takes"], r["ref"], r["auto"]) for r in table], [(9, 2, 1)])   # 6 prises mais personne ne l'a gardé
        self.assertEqual(curate.moments_table({}, [], []), [])                            # ancien format ou absent



class TakesReportTests(unittest.TestCase):
    def test_pairs_of_kept_and_duplicate_takes(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / ".cache").mkdir()
            (work / ".cache" / "editor.json").write_text(json.dumps({f"p{i}": {"vec": unit(1, 0, 0)} for i in range(8)}))
            (work / ".cache" / "features.json").write_text("{}")
            item = lambda i, minutes, **parts: {"id": f"p{i}", "name": f"IMG_{i}", "date": (T0 + timedelta(minutes=minutes)).isoformat(),
                                                "score": 0.5, "parts": parts, "people": [], "note": "", "shot": ""}
            # deux moments : la prise gardée a les yeux ouverts et regarde l'appareil, la retirée est plus nette
            photos = [item(0, 0, frontal=1.0, eyes=1.0, sharpness=0.5), item(1, 2, frontal=0.1, eyes=0.2, sharpness=0.9),
                      item(2, 120, frontal=0.9, eyes=0.9, sharpness=0.4), item(3, 122, frontal=0.2, eyes=0.1, sharpness=0.8),
                      item(4, 124, frontal=0.3, eyes=0.5, sharpness=0.7)]
            plan = {"photos": photos, "reserve": [],
                    "review": {"ids": ["p0", "p2"], "removed": [{"id": "p1", "reason": "doublon"},
                                                                  {"id": "p3", "reason": "doublon"},
                                                                  {"id": "p4", "reason": "pas-belle"}]}}
            (work / "plan.json").write_text(json.dumps(plan))
            lines = []
            out = curate.takes_report([work / "plan.json"], work, out=lines.append, min_pairs=1)
        self.assertEqual(out["pairs"], 2)                     # p4 n'est pas un doublon : pas de paire
        self.assertEqual(out["accuracy"]["frontal"], 1.0)
        self.assertEqual(out["accuracy"]["sharpness"], 0.0)   # la prise gardée est la moins nette
        self.assertGreater(out["weights"]["frontal"], 0)
        self.assertGreater(out["accuracy"]["take_value"], 0.5)

    def plan_with(self, work, photos, removed, kept):
        (work / ".cache").mkdir(exist_ok=True)
        (work / ".cache" / "editor.json").write_text(json.dumps(
            {p["id"]: {"vec": unit(1, 0, 0) if p["id"] != "p9" else unit(0, 1, 0)} for p in photos}))
        (work / ".cache" / "features.json").write_text("{}")
        plan = {"photos": photos, "reserve": [], "review": {"ids": kept, "removed": removed}}
        (work / "plan.json").write_text(json.dumps(plan))
        return work / "plan.json"

    def test_a_criterion_measured_on_one_side_only_is_not_compared(self):
        item = lambda i, minutes, **parts: {"id": f"p{i}", "name": f"IMG_{i}", "score": 0.5, "parts": parts, "people": [],
                                            "date": (T0 + timedelta(minutes=minutes)).isoformat(), "note": "", "shot": ""}
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            # la prise gardée a un critère de l'œil précis, la retirée n'a jamais été notée avec lui : pas de verdict
            photos = [item(0, 0, frontal=1.0, beauty_l=0.9), item(1, 2, frontal=0.1),
                      item(2, 120, frontal=0.9, beauty_l=0.8), item(3, 122, frontal=0.2)]
            path = self.plan_with(work, photos, [{"id": "p1", "reason": "doublon"}, {"id": "p3", "reason": "doublon"}],
                                  ["p0", "p2"])
            out = curate.takes_report([path], work, out=lambda *_: None, min_pairs=1)
        self.assertIn("frontal", out["accuracy"])
        self.assertNotIn("beauty_l", out["accuracy"])
        self.assertNotIn("beauty_l", out["weights"])

    def test_too_few_pairs_say_nothing_about_a_criterion(self):
        item = lambda i, minutes, **parts: {"id": f"p{i}", "name": f"IMG_{i}", "score": 0.5, "parts": parts, "people": [],
                                            "date": (T0 + timedelta(minutes=minutes)).isoformat(), "note": "", "shot": ""}
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            photos = [item(0, 0, frontal=1.0), item(1, 2, frontal=0.1)]
            path = self.plan_with(work, photos, [{"id": "p1", "reason": "doublon"}], ["p0"])
            out = curate.takes_report([path], work, out=lambda *_: None)
        self.assertEqual(out["accuracy"].get("frontal"), None)

    def test_presets_are_compared_on_the_reviewed_pairs(self):
        item = lambda i, minutes, **parts: {"id": f"p{i}", "name": f"IMG_{i}", "score": 0.5, "parts": parts, "people": [],
                                            "date": (T0 + timedelta(minutes=minutes)).isoformat(), "note": "", "shot": ""}
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            # la prise gardée a l'émotion et les gestes, la retirée les yeux ouverts, le sourire et la netteté
            photos, kept, removed = [], [], []
            for m in range(6):
                photos += [item(2 * m, 120 * m, emotion_l=0.9, interaction=0.8, eyes=0.2, smile=0.2, sharpness=0.4),
                           item(2 * m + 1, 120 * m + 2, emotion_l=0.3, interaction=0.3, eyes=1.0, smile=1.0, sharpness=0.9)]
                kept.append(f"p{2 * m}")
                removed.append({"id": f"p{2 * m + 1}", "reason": "doublon"})
            path = TakesReportTests.plan_with(self, work, photos, removed, kept)
            lines = []
            out = curate.takes_report([path], work, out=lines.append)
        self.assertEqual(out["pairs"], 6)
        self.assertEqual(out["presets"]["actuel"], 1.0)
        self.assertLess(out["presets"]["avant"], 1.0)                      # l'ancien réglage payait yeux et netteté
        self.assertGreater(out["weights"]["emotion_l"], 0)
        self.assertLess(out["weights"]["eyes"], 0)
        self.assertTrue(any("Réglages" in line for line in lines))

    def test_removed_duplicates_not_grouped_with_the_kept_photo_are_counted(self):
        item = lambda i, minutes: {"id": f"p{i}", "name": f"IMG_{i}", "score": 0.5, "parts": {}, "people": [],
                                   "date": (T0 + timedelta(minutes=minutes)).isoformat(), "note": "", "shot": ""}
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            photos = [item(0, 0), item(9, 40)]     # même pose, mais 40 minutes plus tard : deux moments pour Albumia
            path = self.plan_with(work, photos, [{"id": "p9", "reason": "doublon"}], ["p0"])
            lines = []
            curate.takes_report([path], work, out=lines.append)
        self.assertTrue(any("1 doublons retirés que je n'ai pas regroupés" in line and "40 min" in line
                            for line in lines), lines)


class PairsReportTests(unittest.TestCase):
    def test_pairs_of_look_alikes_and_what_the_family_did(self):
        item = lambda i, minutes, **kw: {"id": f"p{i}", "name": f"IMG_{i}", "score": 0.5, "parts": {}, "people": [],
                                         "date": (T0 + timedelta(minutes=minutes)).isoformat(), "note": "", **kw}
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / ".cache").mkdir()
            vec = {"p0": unit(1, 0, 0), "p1": unit(1, 0.4, 0), "p2": unit(1, 0.1, 0), "p3": unit(0, 1, 0), "p4": unit(1, 0.5, 0)}
            (work / ".cache" / "editor.json").write_text(json.dumps({k: {"vec": v} for k, v in vec.items()}))
            (work / ".cache" / "features.json").write_text("{}")
            plan = {"photos": [item(0, 0, shot="plan moyen"), item(1, 3, shot="scène"), item(3, 600, shot="scène")],
                    "reserve": [item(2, 1, shot="plan moyen"), item(4, 2, shot="plan moyen")],
                    "review": {"ids": ["p0", "p1", "p3"], "removed": [{"id": "p2", "reason": "doublon"},
                                                                        {"id": "p4", "reason": "pas-belle"}]}}
            (work / "plan.json").write_text(json.dumps(plan))
            lines = []
            rows = curate.pairs_report([work / "plan.json"], work, out=lines.append)
        kinds = {(r["a"], r["b"]): r["roles"] for r in rows}
        self.assertEqual(kinds[("IMG_0", "IMG_1")], "gardée+gardée")        # il garde les deux (autre plan)
        self.assertEqual(kinds[("IMG_0", "IMG_2")], "doublon+gardée")
        self.assertNotIn(("IMG_0", "IMG_3"), kinds)                         # trop différentes
        self.assertFalse(any("IMG_4" in key for key in kinds))              # retirée pour une autre raison : ignorée
        both = next(r for r in rows if r["roles"] == "gardée+gardée")
        self.assertFalse(both["same_shot"])
        self.assertTrue(any(line.startswith("doublon+gardée : 2 paires") or line.startswith("doublon+gardée : 1 paires")
                            for line in lines), lines)


class OpenerTests(unittest.TestCase):
    def setUp(self):
        self.basis = lambda i: unit(*[1.0 if j == i else 0.0 for j in range(128)])

    def album(self):
        out = []
        for d in range(3):                        # 3 jours de 8 photos
            for k in range(8):
                c = photo(d * 10 + k, T0 + timedelta(days=d, minutes=30 * k), 0.7 - 0.01 * k, vec_l=self.basis(d * 10 + k),
                          shot="plan moyen")
                c.meta["moment"] = d * 10 + k
                out.append(c)
        return out

    def wide(self, i, when, score=0.65, note=""):
        c = photo(i, when, score, vec_l=self.basis(i), shot="scène", note=note)
        c.meta["moment"] = i
        return c

    def test_a_chapter_without_a_place_shot_gets_one(self):
        selected = self.album()
        opener = self.wide(100, T0 + timedelta(days=1, hours=-1), note="Le Kotel")
        out = curate.add_openers(selected, selected + [opener], 24)
        self.assertEqual(len(out), 24)                       # jamais plus que demandé
        self.assertIn("id100", {c.file_id for c in out})
        second_day = [c for c in curate.cut_chapters(out) if c[0].when.day == 8][0]
        self.assertTrue(any(c.file_id == "id100" for c in second_day))

    def test_without_room_the_weakest_photo_of_a_full_chapter_gives_way(self):
        selected = self.album()
        weakest = min((c for c in selected if c.when.day == 8), key=lambda c: c.score)
        opener = self.wide(100, T0 + timedelta(days=1, hours=-1))
        out = curate.add_openers(selected, selected + [opener], 24, protected={weakest.file_id})
        self.assertIn(weakest.file_id, {c.file_id for c in out})   # photo protégée : jamais retirée
        self.assertIn("id100", {c.file_id for c in out})
        self.assertEqual(len(out), 24)

    def test_no_opener_when_the_chapter_already_has_one_or_nothing_fits(self):
        selected = self.album()
        selected[0].meta["shot"] = "scène"
        far = self.wide(100, T0 + timedelta(days=1, hours=-8))                 # trop tôt
        weak = self.wide(101, T0 + timedelta(days=1, hours=-1), score=0.1)     # trop faible
        twin = photo(102, T0 + timedelta(days=1, hours=-1), 0.7, vec_l=selected[9].meta["vec_l"], shot="scène")
        twin.meta["moment"] = 102                                              # même pose qu'une photo choisie
        out = curate.add_openers(selected, selected + [far, weak, twin], 30)
        self.assertEqual({c.file_id for c in out}, {c.file_id for c in selected})

    def test_openers_never_go_past_the_wide_shot_limit(self):
        selected = self.album()
        for c in selected:
            c.meta["shot"] = "plan moyen"
        selected[0].meta["shot"] = "scène"                                  # un plan large déjà là : plafond de 1
        spare = [self.wide(100 + d, T0 + timedelta(days=d, hours=-1)) for d in (1, 2)]
        free = curate.add_openers(selected, selected + spare, 30)
        capped = curate.add_openers(selected, selected + spare, 30, wide_limit=1)
        self.assertEqual(sum(c.meta["shot"] == "scène" for c in free), 3)
        self.assertEqual(sum(c.meta["shot"] == "scène" for c in capped), 1)
        self.assertEqual(len(capped), len(selected))

    def test_small_albums_are_left_alone(self):
        few = self.album()[:6]
        self.assertEqual(curate.add_openers(few, few + [self.wide(100, T0)], 6), sorted(few, key=lambda c: c.when))


if __name__ == "__main__":
    unittest.main()
