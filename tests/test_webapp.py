"""Parcours complet de la page web avec un faux Drive : description → aperçu → album créé."""
import importlib
import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import cv2
    import flask  # noqa: F401
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None

from tests.test_pipeline import FakeClient, FakeEngine


class WebFakeClient(FakeClient):
    def get_folder(self, folder_id):
        return {"id": folder_id, "name": "PHOTOS EYDEL"} if folder_id.startswith("A") else None

    def whoami(self):
        return "test@example.com"


@unittest.skipIf(cv2 is None, "OpenCV ou Flask non installé")
class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ALBUMIA_HOME"] = self.tmp.name
        from album_builder import webapp

        self.webapp = importlib.reload(webapp)
        self.client = WebFakeClient(PipelineFiles.make())
        patches = [
            mock.patch.object(self.webapp, "drive", return_value=self.client),
            mock.patch("album_builder.vision.get_engine", return_value=FakeEngine()),
            mock.patch("album_builder.vision.build_references", return_value={}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.http = self.webapp.app.test_client()

    def tearDown(self):
        os.environ.pop("ALBUMIA_HOME", None)
        self.tmp.cleanup()

    def wait(self, job):
        for _ in range(200):
            if job["state"] != "running":
                return job
            time.sleep(0.05)
            job = self.http.get(f"/api/jobs/{job['id']}").get_json()
        self.fail("tâche trop longue")

    def test_home_page_and_status(self):
        self.assertIn(b"Albumia", self.http.get("/").data)
        self.assertIn(b"albumia", self.http.get("/static/app.js").data.lower())
        status = self.http.get("/api/status").get_json()
        self.assertTrue(status["connected"])
        self.assertFalse(status["credentials"])

    def test_folder_check(self):
        ok = self.http.post("/api/folder", json={"link": "https://drive.google.com/drive/folders/" + "A" * 12})
        self.assertEqual(ok.get_json()["name"], "PHOTOS EYDEL")
        bad = self.http.post("/api/folder", json={"link": "bonjour"})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.get_json()["code"], "bad_folder")

    def test_faces_then_people(self):
        img = np.random.default_rng(1).integers(0, 255, (400, 400, 3)).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        res = self.http.post("/api/faces", data={"files": (io.BytesIO(buf.tobytes()), "famille.jpg")},
                             content_type="multipart/form-data").get_json()
        self.assertEqual(len(res[0]["faces"]), 1)
        token = res[0]["faces"][0]["token"]
        saved = self.http.post("/api/faces/save", json={"items": [{"token": token, "name": "Eydel"}]}).get_json()
        self.assertEqual(saved["saved"], 1)
        self.assertEqual(saved["people"][0]["name"], "Eydel")
        photo = saved["people"][0]["photos"][0]
        self.assertEqual(self.http.get(photo).status_code, 200)
        it = self.http.post("/api/interpret", json={"prompt": "20 photos d'eydel en 2026"}).get_json()
        self.assertEqual((it["count"], it["must"]), (20, ["Eydel"]))
        self.assertEqual(self.http.delete("/api/people/Eydel").get_json(), [])

    def test_build_review_apply(self):
        body = {"prompt": "12 photos de 2026", "folder": "A" * 12, "name": "Test 2026", "count": 12,
                "date_from": "2026-01-01", "date_to": "2026-12-31"}
        job = self.wait(self.http.post("/api/build", json=body).get_json())
        self.assertEqual(job["state"], "done", job)
        slug = job["result"]["slug"]
        plan = self.http.get(f"/api/plans/{slug}").get_json()
        self.assertEqual(len(plan["photos"]), 12)
        self.assertTrue(plan["reserve"])
        first = plan["photos"][0]["id"]
        self.assertEqual(self.http.get(f"/plans/{slug}/thumbs/{first}.jpg").status_code, 200)

        # chaque ✕ est retenu tout de suite, sans créer l'album ; « Annuler » le défait
        from album_builder import taste
        home = Path(self.tmp.name)
        gone = plan["photos"][0]["id"]
        review = {"ids": [p["id"] for p in plan["photos"][1:]], "removed": [{"id": gone, "reason": "gout"}]}
        self.assertEqual(self.http.put(f"/api/plans/{slug}/review", json=review).status_code, 200)
        self.assertIn(gone, taste.excluded(home))
        self.assertIn(gone, taste.load_examples(home)["neg"])
        self.assertEqual(self.http.get(f"/api/plans/{slug}").get_json()["review"]["removed"][0]["id"], gone)
        self.http.put(f"/api/plans/{slug}/review", json={"ids": [p["id"] for p in plan["photos"]], "removed": []})
        self.assertNotIn(gone, taste.excluded(home))

        # la personne retire une photo et prend une remplaçante
        ids = [p["id"] for p in plan["photos"][1:]] + [plan["reserve"][0]["id"], "inconnu"]
        job = self.wait(self.http.post("/api/apply", json={"slug": slug, "ids": ids, "name": "Mon album"}).get_json())
        self.assertEqual(job["state"], "done", job)
        self.assertEqual(job["result"]["count"], 12)
        kinds = [c[0] for c in self.client.created]
        self.assertEqual(kinds.count("folder"), 2)  # « Albums (Albumia) » puis l'album
        self.assertEqual(kinds.count("shortcut"), 12)
        albums = self.http.get("/api/albums").get_json()
        self.assertEqual(albums[0]["name"], "Mon album")
        self.assertTrue(albums[0]["drive_url"])

    def test_build_errors_are_friendly(self):
        job = self.wait(self.http.post("/api/build", json={"folder": "A" * 12, "date_from": "2030-01-01"}).get_json())
        self.assertEqual(job["error"]["code"], "empty_period")

    def test_cancel(self):
        slow = self.client.fetch_thumbnail

        def fetch(f, size=1024):
            time.sleep(0.05)
            return slow(f, size)

        with mock.patch.object(self.client, "fetch_thumbnail", side_effect=fetch):
            job = self.http.post("/api/build", json={"folder": "A" * 12}).get_json()
            time.sleep(0.2)
            self.http.post(f"/api/jobs/{job['id']}/cancel")
            self.assertEqual(self.wait(job)["state"], "cancelled")


class PipelineFiles:
    @staticmethod
    def make():
        from tests.test_pipeline import PipelineTests

        return PipelineTests.make_files(None)


if __name__ == "__main__":
    unittest.main()
