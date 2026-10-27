"""Lance Albumia : prépare l'environnement la première fois, puis ouvre la page dans le navigateur.

N'utilise que la bibliothèque standard de Python, pour pouvoir tourner avant toute installation.
Usage : `python lancer.py` (VS Code et les fichiers Albumia.bat / Albumia.command le font pour toi).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / ".albumia-installed"
URL = "http://127.0.0.1:8765/"


def say(text: str) -> None:
    print(f"  {text}", flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def already_running() -> bool:
    try:
        with urllib.request.urlopen(URL + "api/ping", timeout=1) as resp:
            return b"albumia" in resp.read()
    except OSError:
        return False


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def ensure_environment() -> None:
    if not venv_python().exists():
        say("Première utilisation : je prépare Albumia (2 à 5 minutes, une seule fois)…")
        venv.EnvBuilder(with_pip=True).create(VENV)
    if STAMP.exists() and STAMP.read_text().strip() == requirements_hash():
        return
    say("Installation des composants nécessaires…")
    cmd = [str(venv_python()), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", str(REQUIREMENTS)]
    if subprocess.call(cmd) != 0:
        say("")
        say("L'installation n'a pas abouti. Vérifie la connexion Internet puis relance Albumia.")
        sys.exit(1)
    STAMP.write_text(requirements_hash())


def main() -> None:
    try:
        sys.stdout.reconfigure(errors="replace")  # vieilles consoles Windows sans emoji
    except AttributeError:
        pass
    print()
    say("📸 Albumia")
    if sys.version_info < (3, 10):
        say(f"Albumia a besoin de Python 3.10 ou plus récent (tu as {sys.version.split()[0]}).")
        say("Installe-le depuis https://www.python.org/downloads/ puis relance.")
        sys.exit(1)
    if already_running():
        say("Albumia est déjà lancé : j'ouvre la page.")
        webbrowser.open(URL)
        return
    ensure_environment()
    env = {**os.environ, "ALBUMIA_HOME": str(ROOT), "PYTHONUTF8": "1"}
    try:
        sys.exit(subprocess.call([str(venv_python()), "-m", "album_builder.webapp"], cwd=ROOT, env=env))
    except KeyboardInterrupt:
        say("Albumia est arrêté. À bientôt !")


if __name__ == "__main__":
    main()
