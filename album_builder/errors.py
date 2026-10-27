"""Erreurs « métier » : un message clair, en français, destiné à la personne qui utilise l'outil."""
from __future__ import annotations


class AlbumError(Exception):
    """Erreur attendue (dossier vide, modèle introuvable…). `code` permet à l'interface de proposer une solution."""

    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code


class Cancelled(Exception):
    """La personne a annulé la préparation de l'album."""
