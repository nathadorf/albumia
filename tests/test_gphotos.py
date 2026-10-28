"""Photos choisies dans Google Photos (sélecteur de Google), sans réseau."""
import tempfile
import unittest
from pathlib import Path

from album_builder import gphotos, pipeline


class FakePicker:
    def __init__(self, items):
        self.items, self.deleted = items, []

    def picked_items(self, session_id):
        return iter(self.items)

    def download(self, base_url, size=1024):
        return None if "broken" in base_url else b"\xff\xd8jpeg"

    def delete_session(self, session_id):
        self.deleted.append(session_id)


def item(i, when="2025-07-14T10:00:00Z", url=None, kind="PHOTO"):
    return {"id": i, "createTime": when, "type": kind,
            "mediaFile": {"baseUrl": url or f"https://x/{i}", "mimeType": "image/jpeg", "filename": f"IMG_{i}.jpg",
                          "mediaFileMetadata": {"width": 4000, "height": 3000, "cameraMake": "Google",
                                                "cameraModel": "Pixel 8"}}}


class GooglePhotosTests(unittest.TestCase):
    def test_selections_add_up_and_feed_the_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p1 = FakePicker([item("a"), item("b", url="https://x/broken"), item("v", kind="VIDEO")])
            r1 = gphotos.import_session(p1, "s1", tmp)
            self.assertEqual((r1["added"], r1["failed"], r1["count"]), (1, 1, 1))
            self.assertEqual(p1.deleted, ["s1"])
            r2 = gphotos.import_session(FakePicker([item("a"), item("c", when="2025-10-07T18:00:00Z")]), "s2", tmp)
            self.assertEqual((r2["added"], r2["count"], r2["from"], r2["to"]), (1, 2, "2025-07-14", "2025-10-07"))

            src = gphotos.GooglePhotosSource(tmp)
            files = sorted(src.iter_images(["gphotos"]), key=lambda f: f.id)
            self.assertEqual([f.id for f in files], ["gp_a", "gp_c"])
            self.assertEqual((files[0].when.year, files[0].camera, files[0].width), (2025, "Google Pixel 8", 4000))
            self.assertEqual(src.fetch_thumbnail(files[0]), b"\xff\xd8jpeg")

    def test_gphotos_is_a_valid_source(self):
        self.assertEqual(pipeline.parse_folder_id("gphotos"), "gphotos")
