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
- Les photos de la famille ne sortent pas de son PC, vignettes comprises : ne les regarde pas dans une session
  Claude (pas de planche-contact, pas d'outil d'image) sauf demande écrite de Nathanaël. On travaille sur des
  nombres, des identifiants et des dates (`curate --compare`, `curate --takes`, `taste --eval`).

## Structure
- `lancer.py` / `Albumia.bat` / `.vscode/tasks.json` (runOn folderOpen) : lancement en un clic
- `album_builder/webapp.py` + `static/` : serveur et page (JS sans framework)
- `album_builder/pipeline.py` : build_plan / apply_plan (page et CLI). 1re passe rapide (CLIP ViT-B/32) sur
  toutes les photos, 2e passe précise (ViT-L/14) sur les meilleures candidates seulement, puis le tri de photographe
- `album_builder/curate.py` : tri de photographe (un moment = une photo, la prise la plus naturelle, jamais deux
  fois la même pose, 2ᵉ prise des moments forts si le dossier est petit, chapitres, plan large d'ouverture, couverture,
  photos imposées pour que chacun soit présent) ;
  `python -m album_builder.curate --compare auto/plan.json main/plan.json`, `--takes plan.json…` (jeux de poids
  `TAKE_PRESETS` + poids ajustés) et `--pairs plan.json…` (paires proches gardées / retirées ; règle « même pose »)
  (`.cache/rejected.json` : pourquoi chaque photo est absente d'un album, en chiffres ; `--compare` le lit ;
  `.cache/moments.json` : les moments trouvés)
- `album_builder/scoring.py` : critères, `select_spread` (répartition par jour/mois, plafonds, même pose bloquée)
- `album_builder/scenes.py` (description de chaque photo, « sans intérêt »), `aesthetic.py` (CLIP), `taste.py`
  (goût appris : `python -m album_builder.taste --eval`), `compose.py`, `learn.py`, `hebcal.py`, `gphotos.py`
- `album_builder/intent.py`, `people.py`, `drive.py`, `vision.py`, `report.py`, `cli.py`

## Tests
`python -m unittest discover -s tests -t .`

## À vérifier / améliorer en priorité
- Constantes du tri de photographe (`curate.py` : `MOMENT_COS`, `POSE_COS`, `TAKE_WEIGHTS`, `SCENE_SHARE`…) à
  régler sur ses relectures avec `--compare` (sa sélection faite à la main : 19 % de retraits contre 29 à 43 %
  pour les albums automatiques précédents) et `--takes` (quelle prise du moment il garde).
- Poids des critères de photographe (`interaction`, `eye_level`, `light`, `tidy`, `face_seen`, `moment`, à 0 dans
  `Weights`) : à fixer d'après `taste --eval` (AUC de chaque critère seul).
- Le goût appris ne généralise pas encore mieux que les règles (AUC 0,86 contre 0,81 sur des journées jamais vues).
- Constantes de netteté (`SHARP_RANGE_*` dans scoring.py) à calibrer sur de vraies photos.
- Seuil de reconnaissance pour les enfants (visages qui changent vite).
