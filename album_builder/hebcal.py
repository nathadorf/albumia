"""Dates des fêtes juives pour n'importe quelle année, sans réseau (calendrier hébraïque calculé).

Algorithme de Dershowitz et Reingold (« Calendrical Calculations ») : on calcule le nouvel an
(Roch Hachana) de chaque année hébraïque, puis la longueur des mois de l'année.
"""
from __future__ import annotations

from datetime import date, timedelta

_EPOCH = -1373427  # 1 Tichri de l'an 1, en jours depuis le 1/1/1 (date.toordinal)


def _elapsed(y: int) -> int:
    months = (235 * y - 234) // 19
    parts = 12084 + 13753 * months
    day = 29 * months + parts // 25920
    return day + 1 if (3 * (day + 1)) % 7 < 3 else day


def _correction(y: int) -> int:
    ny0, ny1, ny2 = _elapsed(y - 1), _elapsed(y), _elapsed(y + 1)
    if ny2 - ny1 == 356:
        return 2
    if ny1 - ny0 == 382:
        return 1
    return 0


def new_year(y: int) -> date:
    """1 Tichri (Roch Hachana) de l'année hébraïque `y`."""
    return date.fromordinal(_EPOCH + _elapsed(y) + _correction(y))


def leap(y: int) -> bool:
    return (7 * y + 1) % 19 < 7


def _months(y: int) -> list[int]:
    """Longueurs des mois depuis Tichri : Tichri, Hechvan, Kislev, Tevet, Chevat, [Adar I], Adar, Nissan, Iyar, Sivan…"""
    length = (new_year(y + 1) - new_year(y)).days
    hechvan = 30 if length % 10 == 5 else 29
    kislev = 29 if length % 10 == 3 else 30
    return [30, hechvan, kislev, 29, 30] + ([30] if leap(y) else []) + [29, 30, 29, 30]


def _day(y: int, month_index: int, day: int) -> date:
    return new_year(y) + timedelta(days=sum(_months(y)[:month_index]) + day - 1)


def holiday(name: str, year: int) -> tuple[date, date]:
    """Période de la fête pendant l'année civile `year`, veille comprise (repas du soir)."""
    fall = year + 3761    # Tichri et Kislev tombent à l'automne de l'année civile
    spring = year + 3760  # Adar, Nissan, Sivan : au printemps
    adar = 6 if leap(spring) else 5
    if name == "roch-hachana":
        d = _day(fall, 0, 1)
        return d - timedelta(days=1), d + timedelta(days=1)
    if name == "kippour":
        d = _day(fall, 0, 10)
        return d - timedelta(days=1), d
    if name == "souccot":  # construction de la souccah (4 jours avant) jusqu'à Sim'hat Torah
        d = _day(fall, 0, 15)
        return d - timedelta(days=5), d + timedelta(days=8)
    if name == "hanouka":
        d = _day(fall, 2, 25)
        return d - timedelta(days=1), d + timedelta(days=7)
    if name == "pourim":
        d = _day(spring, adar, 14)
        return d - timedelta(days=1), d + timedelta(days=1)
    if name == "pessah":
        d = _day(spring, adar + 1, 15)
        return d - timedelta(days=1), d + timedelta(days=7)
    if name == "chavouot":
        d = _day(spring, adar + 3, 6)
        return d - timedelta(days=1), d + timedelta(days=1)
    raise KeyError(name)
