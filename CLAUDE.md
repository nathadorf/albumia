# Albumia

Application locale (page web servie par Flask sur 127.0.0.1:8765) qui sélectionne les meilleures
photos d'un dossier Google Drive et crée un album (dossier de raccourcis). Utilisatrice cible :
non technique. Tout texte visible est en français, au tutoiement, sans jargon.

## Règles du projet
- Confidentialité d'abord : jamais les originaux, seulement les miniatures Drive ; analyse locale.
- Aucun appel payant par défaut. Claude (`claude_rerank.py`) reste opt-in, caché sans ANTHROPIC_API_KEY.
- La description libre est interprétée localement (`intent.py`), sans IA ni réseau.
- Rien n'est écrit dans Drive avant validation de l'aperçu.
- Ne jamais committer `credentials.json`, `token*.json`, `refs/`, `.cache/`, `out/`, `models/`.
- `lancer.py` n'utilise que la bibliothèque standard (il tourne avant l'installation).

## Structure
- `lancer.py` / `Albumia.bat` / `.vscode/tasks.json` (runOn folderOpen) : lancement en un clic
- `album_builder/webapp.py` + `static/` : serveur et page (JS sans framework)
- `album_builder/pipeline.py` : build_plan / apply_plan (page et CLI)
- `album_builder/intent.py`, `people.py`, `drive.py`, `vision.py`, `scoring.py`, `report.py`, `cli.py`

## Tests
`python -m unittest discover -s tests -t .`

## À vérifier / améliorer en priorité
- Constantes de netteté (`SHARP_RANGE_*` dans scoring.py) à calibrer sur de vraies photos.
- Seuil de reconnaissance pour les enfants (visages qui changent vite).
