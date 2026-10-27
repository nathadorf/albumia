import unittest
from pathlib import Path

from album_builder.errors import AlbumError

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV non installé")
class VisionTests(unittest.TestCase):
    def setUp(self):
        from album_builder import vision

        self.vision = vision
        rng = np.random.default_rng(0)
        self.sharp = (rng.integers(0, 255, (480, 640, 3))).astype(np.uint8)
        self.blurry = cv2.GaussianBlur(self.sharp, (0, 0), 6)

    def test_sharp_image_has_higher_laplacian_than_blurry(self):
        a = self.vision.analyze(self.sharp, None)
        b = self.vision.analyze(self.blurry, None)
        self.assertGreater(a["lap_center"], 20 * b["lap_center"])

    def test_dhash_is_stable_and_discriminating(self):
        g1 = cv2.cvtColor(self.sharp, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(self.blurry, cv2.COLOR_BGR2GRAY)
        self.assertEqual(self.vision.dhash(g1), self.vision.dhash(g1.copy()))
        self.assertEqual(len(self.vision.dhash(g1)), 16)
        gradient = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
        self.assertNotEqual(self.vision.dhash(g1), self.vision.dhash(gradient))

    def test_exposure_stats(self):
        black = np.zeros((100, 100, 3), np.uint8)
        f = self.vision.analyze(black, None)
        self.assertAlmostEqual(f["clip_low"], 1.0)
        self.assertAlmostEqual(f["mean"], 0.0)

    def test_resize_max(self):
        out = self.vision.resize_max(self.sharp, 320)
        self.assertEqual(max(out.shape[:2]), 320)

    def test_face_engine_on_blank_image(self):
        # nécessite les modèles ONNX ; ignoré s'ils ne sont pas téléchargeables
        model_dir = Path(__file__).resolve().parent.parent / "models"
        try:
            engine = self.vision.FaceEngine(model_dir)
        except AlbumError:
            self.skipTest("modèles ONNX indisponibles (pas de réseau ?)")
        blank = np.full((480, 640, 3), 127, np.uint8)
        self.assertEqual(engine.detect_faces(blank), [])


if __name__ == "__main__":
    unittest.main()
