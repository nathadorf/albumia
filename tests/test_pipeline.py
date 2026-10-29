"""Test de bout en bout de `cmd_build` et `cmd_apply` avec un faux Drive et un faux détecteur de visages."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None


class FakeEngine:
    """Détecte 'un visage' au centre si l'image est assez claire (suffisant pour tester le flux)."""

    def detect_faces(self, img, with_embedding=True):
        h, w = img.shape[:2]
        x, y, fw, fh = w // 3, h // 4, w // 3, h // 2
        crop = cv2.cvtColor(img[y : y + fh, x : x + fw], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        lap = float(cv2.Laplacian(crop, cv2.CV_64F).var())  # vraie netteté, comme le vrai moteur
        f = {"box": [x, y, fw, fh], "score": 0.9, "lap": lap, "smile": 1.0, "eyes": 1.0}
        if with_embedding:
            f["emb"] = [1.0, 0.0]
        return [f]


class FakeClient:
    def __init__(self, files):
        self.files = files
        self.created = []

    def iter_images(self, folder_ids, recursive=True):
        return iter(self.files)

    def fetch_thumbnail(self, f, size=1024):
        rng = np.random.default_rng(abs(hash(f.id)) % (2**32))
        img = rng.integers(0, 255, (300, 400, 3)).astype(np.uint8)
        if f.name.startswith("BLUR"):
            img = cv2.GaussianBlur(img, (0, 0), 8)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def create_folder(self, name, parent_id=None):
        self.created.append(("folder", name, parent_id))
        return {"id": "folder1", "webViewLink": "https://example/folder1"}

    def add_shortcut(self, target_id, name, parent_id):
        self.created.append(("shortcut", target_id, name))
        return {"id": "s"}


@unittest.skipIf(cv2 is None, "OpenCV non installé")
class PipelineTests(unittest.TestCase):
    def make_files(self):
        from album_builder.drive import DriveFile

        files = []
        for i in range(60):
            when = datetime(2026, 1 + i % 12, 1 + i // 12, 10, 0, 0)
            name = ("BLUR_" if i % 10 == 0 else "IMG_") + f"{i:04d}.CR2"
            files.append(DriveFile(f"f{i}", name, "image/x-canon-cr2", "http://t", when, "exif", f"mod{i}"))
        # une photo hors période
        files.append(DriveFile("old", "IMG_old.CR2", "image/x", "http://t", datetime(2021, 8, 6), "exif", "m"))
        return files

    def test_build_then_apply(self):
        from album_builder import cli

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            client = FakeClient(self.make_files())
            argv = ["--name", "Test 2026", "--count", "12", "--folder", "A" * 12, "--year", "2026",
                    "--workdir", str(tmp), "--out", str(tmp / "out"), "--workers", "2"]
            with mock.patch.object(cli, "_client", return_value=client), \
                 mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()):
                cli.main(argv)

            plan_path = tmp / "out" / "test-2026" / "plan.json"
            plan = json.loads(plan_path.read_text())
            photos = plan["photos"]
            self.assertEqual(len(photos), 12)
            self.assertNotIn("old", {p["id"] for p in photos})              # filtre de période
            self.assertFalse(any(p["name"].startswith("BLUR") for p in photos))  # floues écartées
            dates = [p["date"] for p in photos]
            self.assertEqual(dates, sorted(dates))                           # ordre chronologique
            self.assertTrue((plan_path.parent / "plan.html").exists())
            self.assertTrue(any((plan_path.parent / "thumbs").iterdir()))

            # second passage : tout vient du cache, même résultat
            with mock.patch.object(cli, "_client", return_value=client), \
                 mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()), \
                 mock.patch.object(client, "fetch_thumbnail", side_effect=AssertionError("cache ignoré")):
                cli.main(argv)
            self.assertEqual(json.loads(plan_path.read_text())["photos"], photos)

            # création dans Drive
            with mock.patch.object(cli, "_client", return_value=client):
                cli.main(["--apply-plan", str(plan_path), "--workdir", str(tmp)])
            kinds = [c[0] for c in client.created]
            self.assertEqual(kinds.count("folder"), 1)
            self.assertEqual(kinds.count("shortcut"), 12)
            self.assertTrue(client.created[1][2].startswith("2026-"))
            self.assertRegex(client.created[1][2], r"^2026-\d\d-\d\d_10h00_(IMG|BLUR)_\d{4}\.CR2$")  # avec l'heure


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(cv2 is None, "OpenCV non installé")
class PipelineWithPeopleTests(unittest.TestCase):
    def test_build_with_person_and_birth_dates_references(self):
        from album_builder import cli, people

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            refs = tmp / "refs"
            img = np.random.default_rng(1).integers(0, 255, (300, 400, 3)).astype(np.uint8)
            people.save_face(refs, "Eydel", cv2.imencode(".jpg", img)[1].tobytes())
            people.set_birth(refs, "Eydel", "2025-01-01")
            client = FakeClient(PipelineTests().make_files())
            argv = ["--name", "Eydel 2026", "--count", "8", "--folder", "A" * 12, "--year", "2026",
                    "--must", "Eydel", "--refs-dir", str(refs), "--workdir", str(tmp), "--out", str(tmp / "out")]
            with mock.patch.object(cli, "_client", return_value=client), \
                 mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()):
                cli.main(argv)
            plan = json.loads(next((tmp / "out").rglob("plan.json")).read_text(encoding="utf-8"))
            self.assertTrue(plan["photos"])
            # la référence a reçu la date de la photo analysée qui lui ressemble
            self.assertTrue(people.photo_dates(refs)["Eydel"])


class FakeJudge:
    def judge(self, img):
        return {"beauty": 0.5, "emotion": 0.5, "not_album": 0.99 if img.mean() > 127.6 else 0.1}

    def scene(self, vec):
        return "repas" if vec[0] > vec[1] * 3 else "dehors"

    def judge_with_vector(self, img):
        v = np.array([img.mean(), img.std(), 1.0])
        return self.judge(img), (v / np.linalg.norm(v)).round(4).tolist()


@unittest.skipIf(cv2 is None, "OpenCV non installé")
class PipelineEditorTests(unittest.TestCase):
    def test_editor_eye_rejects_and_caches(self):
        from album_builder import aesthetic, cli

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            client = FakeClient(PipelineTests().make_files())
            argv = ["--name", "Oeil", "--count", "10", "--folder", "A" * 12, "--year", "2026",
                    "--workdir", str(tmp), "--out", str(tmp / "out")]
            with mock.patch.object(cli, "_client", return_value=client), \
                 mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()), \
                 mock.patch.object(aesthetic, "load_judge", return_value=FakeJudge()) as load:
                cli.main(argv)
                editor = json.loads((tmp / ".cache" / "editor.json").read_text(encoding="utf-8"))
                self.assertEqual(len(editor), 60)
                cli.main(argv)  # deuxième fois : tout est en cache, le modèle n'est pas rechargé
                self.assertEqual(load.call_count, 1)
            plan = json.loads(next((tmp / "out").rglob("plan.json")).read_text(encoding="utf-8"))
            self.assertTrue(plan["photos"])

    def run_theme(self, tmp, scores=None):
        from album_builder import aesthetic, cli, scoring

        def fake_scores(vecs, days, t, **kw):  # une photo sur deux est « dehors »
            return {fid: ((0.9, True) if i % 2 else (0.1, False)) for i, fid in enumerate(sorted(vecs))}

        client = FakeClient(PipelineTests().make_files())
        argv = ["--name", "Theme", "--count", "10", "--folder", "A" * 12, "--year", "2026",
                "--theme", "dehors", "--theme-strict", "--workdir", str(tmp), "--out", str(tmp / "out")]
        with mock.patch.object(cli, "_client", return_value=client), \
             mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()), \
             mock.patch.object(aesthetic, "load_judge", return_value=FakeJudge()), \
             mock.patch.object(aesthetic, "theme_vectors",
                               side_effect=lambda _d, prompts, *a, **k: [np.array([0.0, 1.0, 0.0])] * len(prompts)), \
             mock.patch.object(scoring, "theme_found", return_value=True), \
             mock.patch.object(scoring, "theme_scores", side_effect=scores or fake_scores):
            cli.main(argv)
        return json.loads(next((tmp / "out").rglob("plan.json")).read_text(encoding="utf-8"))

    def test_strict_theme_keeps_only_matching_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.run_theme(Path(tmp))
        self.assertEqual(plan["settings"]["theme"], "dehors")
        self.assertTrue(plan["settings"]["theme_strict"])
        self.assertEqual(len(plan["photos"]), 10)
        self.assertTrue(all(p["parts"]["theme"] == 0.9 for p in plan["photos"]))

    def test_rare_strict_theme_becomes_a_preference(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.run_theme(Path(tmp), scores=lambda vecs, days, t, **kw: {f: (0.2, False) for f in vecs})
        self.assertFalse(plan["settings"]["theme_strict"])
        self.assertEqual(len(plan["photos"]), 10)

    def test_without_model_album_still_works(self):
        from album_builder import aesthetic, cli

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            client = FakeClient(PipelineTests().make_files())
            argv = ["--name", "Sans", "--count", "10", "--folder", "A" * 12, "--year", "2026",
                    "--workdir", str(tmp), "--out", str(tmp / "out")]
            with mock.patch.object(cli, "_client", return_value=client), \
                 mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()), \
                 mock.patch.object(aesthetic, "load_judge", return_value=None):
                cli.main(argv)
            self.assertTrue(list((tmp / "out").rglob("plan.json")))


class FakeLarge:
    """L'œil plus précis (ViT-L/14) : un vecteur calculé sur l'image, comme FakeJudge."""

    def vector(self, img):
        v = np.array([img.mean(), img.std(), 1.0])
        return (v / np.linalg.norm(v)).round(4).tolist()


@unittest.skipIf(cv2 is None, "OpenCV non installé")
class PipelinePhotographerTests(unittest.TestCase):
    def argv(self, tmp, name="Pro", count=24):
        return ["--name", name, "--count", str(count), "--folder", "A" * 12, "--year", "2026",
                "--workdir", str(tmp), "--out", str(tmp / "out")]

    def build(self, tmp, **kw):
        from album_builder import cli

        client = FakeClient(PipelineTests().make_files())
        with mock.patch.object(cli, "_client", return_value=client), \
             mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()):
            cli.main(self.argv(tmp, **kw))
        return json.loads(next((tmp / "out").rglob("plan.json")).read_text(encoding="utf-8"))

    def test_plan_has_chapters_cover_and_moments(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build(Path(tmp))
        photos = plan["photos"]
        self.assertEqual(len(photos), 24)
        self.assertGreaterEqual(len(plan["chapters"]), 3)
        self.assertTrue(all({"title", "start", "end"} <= set(c) for c in plan["chapters"]))
        self.assertIn(plan["cover"], {p["id"] for p in photos})
        self.assertTrue(all("chapter" in p and "moment" in p for p in photos + plan["reserve"]))
        sizes = [sum(1 for p in photos if p["chapter"] == k) for k in range(len(plan["chapters"]))]
        self.assertTrue(all(n >= 3 for n in sizes), sizes)
        self.assertEqual(plan["stats"]["moments"], plan["stats"]["after_dedupe"])

    def test_the_log_says_why_each_absent_photo_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = self.build(tmp, count=12)
            log = json.loads((tmp / ".cache" / "rejected.json").read_text(encoding="utf-8"))
        chosen = {p["id"] for p in plan["photos"]}
        self.assertFalse(chosen & set(log))                       # une photo choisie n'a pas de motif d'absence
        blurry = {k: v for k, v in log.items() if v["name"].startswith("BLUR")}
        self.assertEqual({v["why"] for v in blurry.values()}, {"floue"})
        self.assertEqual(len(blurry), 6)
        left_out = [v for v in log.values() if v["why"] == "pas choisie"]
        self.assertGreaterEqual(len(left_out), 30)                # 54 moments pour 12 places
        record = left_out[0]
        self.assertLessEqual({"name", "why", "date", "scene", "gap", "people", "face", "shot", "kind", "size", "score",
                              "moment"}, set(record))

    def test_moments_log_counts_the_takes_and_what_was_chosen(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = self.build(tmp, count=12)
            moments = json.loads((tmp / ".cache" / "moments.json").read_text(encoding="utf-8"))
        chosen = {p["id"] for p in plan["photos"]}
        self.assertEqual({i for m in moments for i in m["chosen"]}, chosen)
        self.assertTrue(all(m["takes"] >= 1 and m["start"] <= m["end"] for m in moments))
        self.assertEqual(len(moments), plan["stats"]["moments"])

    def test_removed_photos_are_rated_like_the_others_but_never_offered(self):
        from album_builder import aesthetic, cli

        keys = ("beauty", "emotion", "interaction", "eye_level", "light", "tidy", "face_seen", "moment")
        crit = {k: (np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])) for k in keys}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / ".cache").mkdir()
            (tmp / ".cache" / "excluded.json").write_text(json.dumps({"f3": "pas-belle", "f4": "doublon"}))
            (tmp / ".cache" / "taste.json").write_text(json.dumps({"pos": {}, "neg": {"f3": {}}}))
            client = FakeClient(PipelineTests().make_files())
            with mock.patch.object(cli, "_client", return_value=client), \
                 mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()), \
                 mock.patch.object(aesthetic, "load_judge", return_value=FakeJudge()), \
                 mock.patch.object(aesthetic, "load_large", return_value=FakeLarge()), \
                 mock.patch.object(aesthetic, "criteria_large", return_value=crit), \
                 mock.patch.object(aesthetic, "theme_vectors",
                                   side_effect=lambda _d, prompts, *a, **k: [np.array([0.0, 1.0, 0.0])] * len(prompts)):
                cli.main(self.argv(tmp, count=12))
            plan = json.loads(next((tmp / "out").rglob("plan.json")).read_text(encoding="utf-8"))
            parts = json.loads((tmp / ".cache" / "parts.json").read_text(encoding="utf-8"))
            log = json.loads((tmp / ".cache" / "rejected.json").read_text(encoding="utf-8"))
        offered = {p["id"] for p in plan["photos"] + plan["reserve"]}
        self.assertFalse({"f3", "f4"} & offered)                  # retirées à une relecture : jamais proposées
        self.assertEqual({log["f3"]["why"], log["f4"]["why"]}, {"retirée avant"})
        # … mais notées comme les autres, avec l'œil plus précis quand il les a vues (f3 est un contre-exemple)
        self.assertIn("beauty_l", parts["f3"])
        self.assertIn("sharpness", parts["f4"])
        self.assertEqual(set(parts["f3"]), set(parts[plan["photos"][0]["id"]]))

    def test_second_look_with_the_precise_eye(self):
        from album_builder import aesthetic, cli

        keys = ("beauty", "emotion", "interaction", "eye_level", "light", "tidy", "face_seen", "moment")
        crit = {k: (np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])) for k in keys}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            client = FakeClient(PipelineTests().make_files())
            with mock.patch.object(cli, "_client", return_value=client), \
                 mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()), \
                 mock.patch.object(aesthetic, "load_judge", return_value=FakeJudge()), \
                 mock.patch.object(aesthetic, "load_large", return_value=FakeLarge()) as load, \
                 mock.patch.object(aesthetic, "criteria_large", return_value=crit), \
                 mock.patch.object(aesthetic, "theme_vectors",
                                   side_effect=lambda _d, prompts, *a, **k: [np.array([0.0, 1.0, 0.0])] * len(prompts)):
                cli.main(self.argv(tmp, count=12))
                editor = json.loads((tmp / ".cache" / "editor.json").read_text(encoding="utf-8"))
                seen = [v for v in editor.values() if v.get("vec_l")]
                self.assertGreaterEqual(len(seen), 50)  # toutes sauf les 6 floues, écartées avant
                self.assertTrue(all(v.get("vl") == aesthetic.LARGE_VERSION for v in seen))
                cli.main(self.argv(tmp, count=12))  # deuxième fois : rien à regarder de nouveau
                self.assertEqual(load.call_count, 1)
            plan = json.loads(next((tmp / "out").rglob("plan.json")).read_text(encoding="utf-8"))
        self.assertTrue(all("beauty_l" in p["parts"] and "interaction" in p["parts"] for p in plan["photos"]))
        self.assertEqual(len(plan["photos"]), 12)


class ShortlistAndFlagsTests(unittest.TestCase):
    def test_shortlist_keeps_the_best_of_each_moment_and_every_day(self):
        from datetime import date
        from album_builder import pipeline
        from tests.test_scoring import cand

        cands = []
        for d in range(10):
            for k in range(30):  # 10 jours de 30 photos, 20 minutes entre deux : un moment chacune
                i = d * 100 + k
                c = cand(i, datetime(2025, 10, 1 + d, 8) + timedelta(minutes=20 * k), 0.95 - 0.002 * (d * 30 + k),
                         dhash=format((i * 0x9E3779B97F4A7C15) % 2**64, "016x"))
                c.meta = {}
                cands.append(c)
        weak = cands[-1]
        short = pipeline.pick_shortlist(cands, 5, protected={weak.file_id})
        ids = {c.file_id for c in short}
        self.assertLess(len(short), len(cands))
        self.assertGreaterEqual(len(short), pipeline.SHORTLIST_FLOOR)
        self.assertIn(weak.file_id, ids)                                  # protégée
        self.assertTrue({c.when.date() for c in cands} <= {c.when.date() for c in short})  # chaque jour
        self.assertIn(cands[0].file_id, ids)
        self.assertEqual(len(pipeline.pick_shortlist(cands[:50], 5)), 50)  # petit dossier : tout passe

    def test_left_out_photos_say_which_chosen_photo_they_look_like(self):
        from album_builder import pipeline
        from tests.test_curate import photo, unit

        v = unit(1, 0, 0)
        chosen = photo(1, datetime(2025, 10, 7, 10), 0.8, vec_l=v)
        twin = photo(2, datetime(2025, 10, 7, 14), 0.7, vec_l=unit(1, 0.1, 0))   # quatre heures plus tard, même pose
        other = photo(3, datetime(2025, 10, 8, 10), 0.6, vec_l=unit(0, 0, 1))
        take = photo(4, datetime(2025, 10, 7, 10, 1), 0.5, vec_l=v)
        for c, m in ((chosen, 0), (twin, 1), (other, 2), (take, 0)):
            c.meta["moment"] = m
        out = pipeline.left_out([chosen], [chosen, twin, other], {0: [take], 1: [], 2: []})
        self.assertNotIn(chosen.file_id, out)
        self.assertEqual(out[take.file_id]["why"], "autre prise du moment")
        self.assertEqual(out[twin.file_id]["why"], "pas choisie")
        self.assertEqual(out[twin.file_id]["twin"][0], chosen.name)
        self.assertGreater(out[twin.file_id]["twin"][1], 0.9)
        self.assertLess(out[other.file_id]["twin"][1], 0.1)

    def test_family_removals_come_from_the_other_family_albums_only(self):
        from album_builder import pipeline

        def plan(root, slug, family, removed):
            (root / slug).mkdir()
            (root / slug / "plan.json").write_text(json.dumps({
                "settings": {"family": family}, "review": {"removed": [{"id": i, "reason": r} for i, r in removed]}}))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan(root, "octobre", True, [("papa", "personnes"), ("flou", "pas-belle")])
            plan(root, "eydel", False, [("cousin", "personnes")])          # pas un album en famille
            plan(root, "souccot", True, [("voisin", "personnes")])         # cet album-ci : relu à part
            (root / "casse").mkdir()
            (root / "casse" / "plan.json").write_text("{")
            self.assertEqual(pipeline.family_removals(root, "souccot"), {"papa"})
            # un album relu avant qu'on note « en famille » : il l'est s'il nommait les mêmes personnes
            (root / "ancien").mkdir()
            (root / "ancien" / "plan.json").write_text(json.dumps({
                "settings": {"must": [], "bonus": ["Eydel", "Arié", "Hillel", "Perla", "Nathanaël"]},
                "review": {"removed": [{"id": "inconnu", "reason": "personnes"}, {"id": "flou2", "reason": "pas-belle"}]}}))
            (root / "duo").mkdir()
            (root / "duo" / "plan.json").write_text(json.dumps({
                "settings": {"bonus": ["Eydel", "Arié"]}, "review": {"removed": [{"id": "dos", "reason": "personnes"}]}}))
            family = {"Eydel", "Arié", "Hillel", "Perla", "Nathanaël"}
            self.assertEqual(pipeline.family_removals(root, "souccot", family), {"papa", "inconnu"})
            self.assertEqual(pipeline.family_removals(root, "souccot", {"Eydel", "Arié"}), {"papa"})

    def test_copy_of_names_the_photo_that_absorbed_a_copy(self):
        from album_builder import pipeline
        from tests.test_curate import photo

        kept = photo(1, datetime(2025, 10, 12, 9, 36), 0.8, dhash="0" * 16)
        copy = photo(2, datetime(2025, 10, 12, 9, 36, 1), 0.7, dhash="000000000000000f")
        far = photo(3, datetime(2025, 10, 13, 9, 36), 0.7, dhash="f" * 16)
        self.assertEqual(pipeline.copy_of(copy, [far, kept]), [kept.name, 4, 1])
        self.assertEqual(pipeline.copy_of(copy, []), [])

    def test_left_out_says_which_limits_were_reached(self):
        from album_builder import pipeline
        from tests.test_curate import photo, unit

        base = lambda i: unit(*[1.0 if j == i else 0.0 for j in range(16)])
        chosen = [photo(i, datetime(2025, 10, 7 + i, 10), 0.8, vec_l=base(i), shot="scène", note="Jeux gonflables")
                  for i in range(3)]
        meal = photo(10, datetime(2025, 10, 12, 10), 0.6, vec_l=base(10), note="Repas")
        meal.meta["meal"] = True
        wide = photo(11, datetime(2025, 10, 13, 10), 0.6, vec_l=base(11), shot="scène", note="Jeux gonflables")
        same = photo(12, datetime(2025, 10, 14, 10), 0.6, vec_l=unit(1, 0.05, *[0] * 14), shot="plan moyen", note="Portrait")
        for k, c in enumerate(chosen + [meal, wide, same]):
            c.meta["moment"] = k
        out = pipeline.left_out(chosen, chosen + [meal, wide, same], {},
                                {"scene": 2, "wide": 3, "meal": 0, "selfie": 1, "decor": None})
        self.assertEqual(out["id10"]["blocked"], ["repas 0/0"])
        self.assertEqual(out["id11"]["blocked"], ["plans larges 3/3", "scène Jeux gonflables 3/2"])
        self.assertTrue(any(b.startswith("même pose que") for b in out["id12"]["blocked"]))
        self.assertEqual(out["id12"]["twin"][0], chosen[0].name)

    def test_a_small_folder_gets_every_moment_before_the_rules_leave_places_empty(self):
        from album_builder import curate, pipeline, scenes
        from tests.test_curate import photo, unit

        base = lambda i: unit(*[1.0 if j == i else 0.0 for j in range(16)])
        pool = []
        for i in range(10):                              # 10 moments, tous la même scène, tous des plans larges
            c = photo(i, datetime(2025, 10, 7 + i, 10), 0.8 - 0.01 * i, vec_l=base(i), shot="scène", note="Jeux dehors",
                      labels={"Eydel"})
            c.meta["moment"] = i
            pool.append(c)
        strict, strict_limits = pipeline.select_moments(pool, 10, expected=2, pinned=set(), rules={}, cap=2, wide=2)
        self.assertEqual(len(strict), 2)                 # si l'on se contente de 2, les plafonds suffisent
        self.assertEqual(strict_limits, {"scene": 2, "wide": 2})
        full, full_limits = pipeline.select_moments(pool, 10, expected=10, pinned=set(), rules={}, cap=2, wide=2)
        self.assertEqual(len(full), 10)                  # sinon on assouplit puis on lève, sans rien perdre
        self.assertEqual(full_limits, {"scene": None, "wide": None})   # la dernière passe n'avait plus de plafond
        middle, middle_limits = pipeline.select_moments(pool, 10, expected=4, pinned=set(), rules={}, cap=2, wide=2)
        self.assertEqual(len(middle), 4)                 # 2e passe : deux photos de plus par activité et par plan large
        self.assertEqual(middle_limits, {"scene": 4, "wide": 4})
        self.assertTrue({c.file_id for c in strict} <= {c.file_id for c in middle})

    def test_adult_alone_and_tiny_subject(self):
        from datetime import date
        from album_builder import pipeline, scenes

        births = {"Eydel": date(2021, 7, 14)}
        feat = {"w": 1000, "h": 750}
        big, small = {"box": [0, 0, 200, 200]}, {"box": [300, 300, 50, 50]}
        alone = lambda faces, labels, b=births: pipeline.is_adult_alone(faces, labels, b, date(2025, 10, 7), feat)
        self.assertTrue(alone([big], ["Perla"]))                 # papa ou maman seul
        self.assertFalse(alone([big], ["Eydel"]))                # un enfant seul est le cœur de l'album
        self.assertFalse(alone([big], [None]))                   # inconnu : on ne sait pas
        self.assertFalse(alone([big, big], ["Perla", "Eydel"]))
        self.assertTrue(alone([big, small], ["Perla", None]))    # un visage minuscule ne compte pas
        self.assertFalse(alone([big], ["Perla"], {}))            # aucune date de naissance : personne n'est « adulte »

        class C:
            meta, note = {"subject_area": 0.01}, "Jeux dehors"

        self.assertTrue(pipeline.is_tiny(C))
        C.note = "Le Kotel"
        self.assertEqual(scenes.KIND[C.note], scenes.PLACE)
        self.assertFalse(pipeline.is_tiny(C))                    # un lieu qui compte, même loin
        C.meta, C.note = {"subject_area": 0.0}, "Jeux dehors"
        self.assertFalse(pipeline.is_tiny(C))                    # sans visage : ce n'est pas un sujet minuscule


class ReportTests(unittest.TestCase):
    def test_contact_sheet_shows_chapters_and_cover(self):
        from album_builder import report
        from tests.test_scoring import cand

        photos = [cand(i, datetime(2025, 10, 7 + i // 3, 10, i % 3), 0.5) for i in range(6)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            layout = {"chapters": [{"title": "Mardi 7 octobre", "start": "", "end": ""},
                                   {"title": "Mercredi 8 octobre · Piscine", "start": "", "end": ""}],
                      "cover": "id4", "of": {c.file_id: {"chapter": i // 3} for i, c in enumerate(photos)}}
            path = report.write_plan(tmp / "out", "Test", {}, photos, tmp / "thumbs", {}, [], layout)
            plan = json.loads(path.read_text(encoding="utf-8"))
            sheet = (path.parent / "plan.html").read_text(encoding="utf-8")
        self.assertEqual(plan["cover"], "id4")
        self.assertEqual([p["chapter"] for p in plan["photos"]], [0, 0, 0, 1, 1, 1])
        self.assertIn("Mercredi 8 octobre · Piscine", sheet)
        self.assertEqual(sheet.count("couverture"), 1)
