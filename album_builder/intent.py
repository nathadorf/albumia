"""Comprendre la description libre d'un album, en français, sans IA ni réseau.

« 50 photos d'Eydel et ses frères pendant l'été 2025, des moments joyeux »
  → 50 photos, du 01/06/2025 au 31/08/2025, Eydel obligatoire, Frères en bonus, sourires favorisés.

Le résultat est une proposition : l'interface l'affiche et la personne peut tout corriger.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
    "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}
# saison -> (mois de début, mois de fin) ; l'hiver commence en décembre de l'année précédente
SEASONS = {"printemps": (3, 5), "ete": (6, 8), "automne": (9, 11), "hiver": (12, 2)}
SEASON_LABEL = {"printemps": "Printemps", "ete": "Été", "automne": "Automne", "hiver": "Hiver"}
APPROX = {"dizaine": 10, "quinzaine": 15, "vingtaine": 20, "trentaine": 30, "quarantaine": 40,
          "cinquantaine": 50, "soixantaine": 60, "centaine": 100}
NUMBER_WORDS = {"dix": 10, "quinze": 15, "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50,
                "soixante": 60, "cent": 100}
TOGETHER = ("ensemble", "tous", "toutes", "reunis", "les deux", "les trois", "en famille")
SMILES = ("sourire", "souriant", "rire", "rires", "joie", "joyeu", "heureu", "fou rire", "complice", "rigol")
PORTRAITS = ("portrait", "visage", "gros plan")
DEFAULT_COUNT = 50
MAX_COUNT = 500


def normalize(text: str) -> str:
    """minuscules, sans accents, apostrophes simplifiées."""
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _person_pattern(name: str) -> re.Pattern:
    words = normalize(name).split()
    if not words:
        return re.compile(r"(?!x)x")
    stem = r"\s+".join(re.escape(w.rstrip("sx")) for w in words)
    return re.compile(rf"(?<![a-z0-9]){stem}[sx]?(?![a-z0-9])")


@dataclass
class Intent:
    name: str = ""
    count: int = DEFAULT_COUNT
    date_from: str | None = None   # AAAA-MM-JJ, inclus
    date_to: str | None = None     # AAAA-MM-JJ, inclus
    period_label: str = ""
    must: list[str] = field(default_factory=list)
    bonus: list[str] = field(default_factory=list)
    faces_only: bool = False
    favor_smiles: bool = False
    understood: list[str] = field(default_factory=list)  # phrases courtes affichées à l'écran

    def to_dict(self) -> dict:
        return asdict(self)


def _count(t: str) -> int | None:
    m = re.search(r"(\d{1,4})\s*(?:photos?|images?|cliches?|souvenirs?|pages?)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:une|un)\s+(" + "|".join(APPROX) + r")", t)
    if m:
        return APPROX[m.group(1)]
    m = re.search(r"\b(" + "|".join(NUMBER_WORDS) + r")\s+(?:photos?|images?)", t)
    if m:
        return NUMBER_WORDS[m.group(1)]
    return None


def _end_of_month(year: int, month: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def _period(t: str, today: date) -> tuple[date | None, date | None, str]:
    years = sorted({int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", t)})
    if not years:
        if re.search(r"cette annee|de l'annee\b", t):
            years = [today.year]
        elif re.search(r"l'annee (?:derniere|passee)|annee derniere", t):
            years = [today.year - 1]

    season = next((s for s in SEASONS if re.search(rf"\b{s}\b", t)), None)
    months = [MONTHS[m] for m in re.findall(r"\b(" + "|".join(MONTHS) + r")\b", t)]

    def default_year(month: int) -> int:
        # « en juillet » sans année : le dernier mois de juillet passé
        return today.year if month <= today.month else today.year - 1

    if season:
        start_m, end_m = SEASONS[season]
        year = years[-1] if years else default_year(end_m)
        if season == "hiver":
            return date(year - 1, 12, 1), _end_of_month(year, 2), f"Hiver {year - 1}-{year}"
        return date(year, start_m, 1), _end_of_month(year, end_m), f"{SEASON_LABEL[season]} {year}"
    if months:
        year = years[-1] if years else default_year(max(months))
        lo, hi = min(months), max(months)
        names = [k.capitalize() for k, v in MONTHS.items() if v in (lo, hi)]
        label = names[0] if lo == hi else f"{names[0]}-{names[-1]}"
        label = {"Fevrier": "Février", "Aout": "Août", "Decembre": "Décembre"}.get(label, label)
        return date(year, lo, 1), _end_of_month(year, hi), f"{label} {year}"
    if years:
        lo, hi = years[0], years[-1]
        return date(lo, 1, 1), date(hi, 12, 31), str(lo) if lo == hi else f"{lo}-{hi}"
    return None, None, ""


def _album_name(text: str, people: list[str], period_label: str, today: date) -> str:
    m = re.search(r"[«\"“]\s*([^»\"”]{2,80}?)\s*[»\"”]", text)
    if m:
        return m.group(1).strip()
    who = " & ".join(people[:3])
    name = " ".join(x for x in (who, period_label) if x)
    return name or f"Album du {today.strftime('%d/%m/%Y')}"


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 février
        return d.replace(year=d.year + years, day=28)


AGE_RE = re.compile(r"premiere annee|1(?:re|ere)? annee|(\d+|deux|trois|quatre|cinq) ans?\b")
AGE_WORDS = {"deux": 2, "trois": 3, "quatre": 4, "cinq": 5}


def interpret(text: str, people: list[str], selected: list[str] | None = None,
              today: date | None = None, births: dict[str, date] | None = None) -> Intent:
    """`people` : toutes les personnes connues ; `selected` : celles cochées dans l'interface ;
    `births` : dates de naissance connues (« la première année d'Eydel », « Arié à 2 ans »)."""
    today = today or date.today()
    t = normalize(text or "")
    it = Intent()

    n = _count(t)
    if n:
        it.count = max(1, min(MAX_COUNT, n))
        it.understood.append(f"{it.count} photos")
    lo, hi, label = _period(t, today)
    if lo:
        it.date_from, it.date_to, it.period_label = lo.isoformat(), hi.isoformat(), label
        it.understood.append(f"période : {label}")

    # personnes citées, dans l'ordre où elles apparaissent dans le texte
    found = []
    for p in people:
        m = _person_pattern(p).search(t)
        if m:
            found.append((m.start(), p))
    mentioned = [p for _, p in sorted(found)]
    targets = mentioned + [p for p in (selected or []) if p not in mentioned]
    if targets:
        if len(targets) == 1 or any(w in t for w in TOGETHER):
            it.must = targets
        else:
            it.must, it.bonus = targets[:1], targets[1:]
        it.understood.append("avec " + ", ".join(it.must)
                             + (f" (et si possible {', '.join(it.bonus)})" if it.bonus else ""))

    # « la première année d'Eydel », « Arié à 2 ans » : période calculée depuis la naissance
    main = next((p for p in it.must if births and p in births), None)
    m = AGE_RE.search(t) if main else None
    if m and not re.search(r"\b(19|20)\d{2}\b", t):
        birth = births[main]
        if m.group(1) is None:
            years, label = 0, "La première année"
        else:
            years = int(AGE_WORDS.get(m.group(1), m.group(1)))
            label = f"{years} an{'s' if years > 1 else ''}"
        lo = _add_years(birth, years)
        hi = _add_years(birth, years + 1) - timedelta(days=1)
        it.date_from, it.date_to = lo.isoformat(), hi.isoformat()
        of = "d'" if normalize(main)[:1] in "aeiouyh" else "de "
        it.period_label = f"{label} {of}{main}" if years == 0 else f"{main} à {label}"
        it.understood = [u for u in it.understood if not u.startswith("période")]
        it.understood.append(f"période : {it.period_label}")

    if any(w in t for w in PORTRAITS):
        it.faces_only = True
        it.understood.append("des portraits")
    if any(w in t for w in SMILES):
        it.favor_smiles = True
        it.understood.append("priorité aux sourires")

    it.name = _album_name(text or "", targets, label, today)
    if it.period_label and it.period_label != label:  # période définie par l'âge
        it.name = _album_name(text or "", [], it.period_label, today)
    return it
