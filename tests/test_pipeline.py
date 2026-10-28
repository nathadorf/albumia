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
        return {"beauty": 0.5, "emotion": 0.5, "not_album": 0.9 if img.mean() > 127.5 else 0.1}


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
