import unittest
from datetime import date

from album_builder.intent import interpret, normalize

TODAY = date(2026, 10, 27)
PEOPLE = ["Eydel", "Frères", "Noam"]


class IntentTests(unittest.TestCase):
    def test_example_from_readme(self):
        it = interpret("50 photos d'Eydel et ses frères en 2026", PEOPLE, today=TODAY)
        self.assertEqual(it.count, 50)
        self.assertEqual((it.date_from, it.date_to), ("2026-01-01", "2026-12-31"))
        self.assertEqual(it.must, ["Eydel"])
        self.assertEqual(it.bonus, ["Frères"])
        self.assertEqual(it.name, "Eydel & Frères 2026")

    def test_together_makes_everyone_required(self):
        it = interpret("Eydel et Noam ensemble", PEOPLE, today=TODAY)
        self.assertEqual(it.must, ["Eydel", "Noam"])
        self.assertEqual(it.bonus, [])

    def test_order_follows_text_not_list(self):
        it = interpret("noam avec eydel", PEOPLE, today=TODAY)
        self.assertEqual(it.must, ["Noam"])
        self.assertEqual(it.bonus, ["Eydel"])

    def test_selected_people_without_mention(self):
        it = interpret("les plus belles photos", PEOPLE, selected=["Noam"], today=TODAY)
        self.assertEqual(it.must, ["Noam"])

    def test_no_false_person_match_inside_word(self):
        it = interpret("photos de Noamie", PEOPLE, today=TODAY)
        self.assertEqual(it.must, [])

    def test_season(self):
        it = interpret("une trentaine de photos de l'été 2025", PEOPLE, today=TODAY)
        self.assertEqual(it.count, 30)
        self.assertEqual((it.date_from, it.date_to), ("2025-06-01", "2025-08-31"))
        self.assertEqual(it.period_label, "Été 2025")

    def test_winter_spans_two_years(self):
        it = interpret("hiver 2026", PEOPLE, today=TODAY)
        self.assertEqual((it.date_from, it.date_to), ("2025-12-01", "2026-02-28"))

    def test_month_without_year_is_last_occurrence(self):
        it = interpret("nos vacances en décembre", PEOPLE, today=TODAY)
        self.assertEqual((it.date_from, it.date_to), ("2025-12-01", "2025-12-31"))
        it = interpret("juillet et août", PEOPLE, today=TODAY)
        self.assertEqual((it.date_from, it.date_to), ("2026-07-01", "2026-08-31"))

    def test_year_range(self):
        it = interpret("de 2023 à 2025", PEOPLE, today=TODAY)
        self.assertEqual((it.date_from, it.date_to), ("2023-01-01", "2025-12-31"))

    def test_smiles_portraits_and_quoted_name(self):
        it = interpret('Album « Mamie 80 ans », des portraits joyeux', PEOPLE, today=TODAY)
        self.assertTrue(it.favor_smiles)
        self.assertTrue(it.faces_only)
        self.assertEqual(it.name, "Mamie 80 ans")

    def test_defaults(self):
        it = interpret("", PEOPLE, today=TODAY)
        self.assertEqual(it.count, 50)
        self.assertIsNone(it.date_from)
        self.assertEqual(it.name, "Album du 27/10/2026")

    def test_normalize(self):
        self.assertEqual(normalize("Été’s FRÈRES"), "ete's freres")


if __name__ == "__main__":
    unittest.main()
