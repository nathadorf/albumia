# 📸 Albumia

Albumia choisit les plus belles photos d'un dossier Google Drive et les range dans un album,
sans rien déplacer ni dupliquer. Tu lui montres **qui** tu veux voir, tu **décris** l'album
avec tes mots, tu colles le **lien du dossier**, et il s'occupe du reste.

> « 50 photos d'Eydel et ses frères en 2026, plein de sourires »

---

## Mode d'emploi (au quotidien)

1. **Ouvre Albumia.** Dans VS Code : *Fichier › Ouvrir le dossier…* › `albumia`. La page s'ouvre
   toute seule dans le navigateur. (Sans VS Code : double-clic sur `Albumia.bat`.)
2. **Qui ?** Touche les personnes à retrouver. La première fois, clique sur **+ Ajouter**, dépose
   une photo où l'on voit bien les visages (une photo de famille marche très bien) et écris le
   prénom sous chaque visage.
3. **Décris l'album** comme tu parles : combien de photos, quelle période, quelle ambiance.
   Albumia affiche juste en dessous ce qu'il a compris ; tu peux tout corriger.
4. **Colle le lien du dossier Drive** (ouvre le dossier dans Drive, copie l'adresse en haut).
5. **Préparer mon album.** Albumia regarde chaque photo, puis te montre sa sélection, rangée par
   chapitres (« Mardi 7 octobre · Piscine ») avec une photo de couverture. Le ✕ retire une photo et
   la remplace par une autre ; le ⇄ propose les autres prises du même moment.
6. **Créer l'album dans mon Drive.** Il apparaît dans *Mon Drive › Albums (Albumia)*, les photos
   dans l'ordre (date et heure en début de nom).

Un guide en une page pour utiliser Albumia au quotidien : [GUIDE.md](GUIDE.md).

Rien n'est écrit dans Drive avant l'étape 6. L'album contient des **raccourcis** : tes
originaux ne bougent pas et ça ne prend aucune place en plus.

**Ça ne marche pas comme prévu ?**

| Ce que tu vois | Quoi faire |
|---|---|
| « Je n'ai reconnu personne » | Accepte de réessayer « moins strict », ou ajoute 2-3 photos récentes de la personne (les enfants changent vite). |
| Une personne est mal reconnue | Clique sur le nombre de photos sous son visage et retire les visages qui ne lui ressemblent plus. |
| La page dit qu'Albumia n'est pas lancé | Ferme VS Code et rouvre le dossier, ou double-clic sur `Albumia.bat`. |
| C'est long | La première analyse d'un gros dossier prend du temps. Les fois suivantes, c'est beaucoup plus rapide. |

---

## Installation (une seule fois, environ 15 minutes)

### 1. Python et VS Code

- **Python 3.10 ou plus récent** : [python.org/downloads](https://www.python.org/downloads/).
  Sous Windows, coche **« Add python.exe to PATH »** pendant l'installation.
- **VS Code** : [code.visualstudio.com](https://code.visualstudio.com/). Accepte l'extension
  *Python* qu'il propose (utile pour la touche F5, pas obligatoire).

### 2. Récupérer Albumia

Dans VS Code : *Afficher › Palette de commandes* › **Git: Cloner** › colle
`https://github.com/nathadorf/albumia` › choisis `C:\Users\…\projects` comme destination.

À l'ouverture du dossier, VS Code demande deux autorisations : **faire confiance aux auteurs**
du dossier, puis **autoriser les tâches automatiques**. Accepte les deux : c'est ce qui lance
Albumia tout seul. La toute première fois, l'installation des composants prend 2 à 5 minutes.

Pour lancer à la main : `Ctrl+Maj+B`, ou **F5**, ou double-clic sur `Albumia.bat`
(`Albumia.command` sur Mac).

### 3. L'accès Google Drive

Albumia se connecte à Drive avec **ton propre** projet Google Cloud : aucun intermédiaire.

1. Va sur [console.cloud.google.com](https://console.cloud.google.com) et crée un projet,
   par exemple « albumia ».
2. **APIs & Services › Library** : cherche **Google Drive API** et clique sur **Enable**.
3. **APIs & Services › OAuth consent screen** : choisis **External**, donne un nom (« Albumia »)
   et ton adresse comme contact. Dans **Audience**, ajoute comme **utilisateurs de test**
   l'adresse Gmail de chaque personne qui utilisera Albumia (toi et ta femme). Pas besoin de
   publier ni de faire valider l'application.
4. **APIs & Services › Credentials › Create credentials › OAuth client ID**, type
   **Desktop app**. Télécharge le fichier JSON.
5. Ouvre Albumia : la page te demande ce fichier. Dépose-le dedans (il est enregistré sous
   `credentials.json`, jamais envoyé sur GitHub).
6. Clique sur **Se connecter à Google** avec le compte qui contient les photos. L'écran
   « Google n'a pas validé cette application » est normal : **Paramètres avancés › Accéder à Albumia**.

C'est fini : Albumia se souvient de la connexion.

---

## Comment Albumia choisit (comme un photographe)

Albumia fait ce que ferait un photographe à qui tu confierais tes photos :

- **Une photo par moment.** Les prises faites à quelques minutes d'écart sur la même scène sont
  un seul moment ; il garde la plus naturelle (l'enfant de face, les yeux ouverts, plusieurs de la
  famille), pas forcément la plus nette. Il en garde deux seulement pour un moment fort.
- **Jamais deux fois la même pose**, même à des heures d'écart.
- **Tout le monde est là** : chacun sur au moins deux photos, une photo de groupe, les bébés
  un peu plus. Pas d'adulte seul, au plus quelques selfies, repas et photos d'une même activité.
- **Des plans variés** : pas plus d'un quart de plans larges, mais un plan large au début de
  chaque chapitre pour situer le lieu (le Kotel, la salle de jeux).
- **Un séjour se raconte en chapitres**, coupés aux grands écarts de temps ou de scène, avec une
  photo de groupe en couverture. Un album court et fort vaut mieux qu'un album rempli.
- **Il apprend de toi.** Chaque photo retirée (pas belle, doublon, pas les bonnes personnes) et
  chaque photo gardée à la relecture comptent pour les albums suivants.

La première fois, un regard plus précis (un modèle de 430 Mo, téléchargé une fois) est posé sur
les meilleures candidates seulement : compte une dizaine de minutes pour un gros dossier, puis
c'est gardé en mémoire. Tout reste sur l'ordinateur.

## Confidentialité et coût

- **Gratuit.** Albumia n'utilise aucune IA payante par défaut.
- **L'analyse est 100 % locale.** OpenCV et deux petits modèles de visages tournent sur
  l'ordinateur. Seules les miniatures Drive (~1024 px) sont lues, jamais les RAW de 30 Mo.
- **La description est comprise sur place**, par des règles simples en français : aucun texte
  ne part sur Internet.
- **Autorisations minimales** : lecture du Drive + écriture limitée à ce qu'Albumia crée
  (`drive.file`). Si Google refuse la création des raccourcis avec cette permission,
  Albumia propose de demander l'autorisation complète, une seule fois.
- **Les visages de référence** restent dans `refs/` sur l'ordinateur et ne sont jamais versionnés.
- **Option Claude (désactivée, cachée par défaut).** Elle n'apparaît que si la variable
  d'environnement `ANTHROPIC_API_KEY` est définie et `pip install -r requirements-claude.txt`
  fait. Elle envoie environ 150 miniatures à l'API Anthropic pour un tri plus fin, et c'est
  **payant** (clé API séparée de l'abonnement Claude). Vérifie la politique de conservation de
  l'API avant de l'activer.

## Ce qui n'a pas pu être vérifié

Le code est testé automatiquement (plus de 150 tests, avec un faux Drive et un faux détecteur de
visages), et le tri a été mesuré sur les albums de la famille, mais pas sur tous les cas de vrai
compte Drive :

- **Modèles de visages** : téléchargés au premier usage depuis GitHub, avec Hugging Face en
  secours. S'ils ne se téléchargent pas, le message indique où les déposer à la main (`models/`).
- **Miniatures des RAW (.CR2)** : Drive devrait en fournir ; une photo sans miniature est ignorée
  et comptée.
- **Raccourcis avec la permission restreinte** : non testé ; le repli « autorisation complète »
  est prévu.
- **Seuils de netteté** : valeurs de départ (`SHARP_RANGE_*` dans `album_builder/scoring.py`).
- **Seuils du tri de photographe** (`album_builder/curate.py`) : réglés sur un voyage et quatre
  relectures de la famille ; à recontrôler sur de nouveaux albums.

---

## Pour bricoler (développeur)

```
lancer.py               prépare .venv puis lance la page (bibliothèque standard uniquement)
album_builder/
  webapp.py             serveur Flask local (127.0.0.1:8765) + API utilisée par la page
  static/               la page : index.html, app.js, style.css (sans framework)
  intent.py             description libre -> nombre, période, personnes, ambiance
  people.py             refs/<prénom>/ : visages de référence
  pipeline.py           build_plan / apply_plan, communs à la page et à la ligne de commande
  curate.py             le tri d'un photographe : moments, prises, chapitres, couverture
  scoring.py            critères, répartition (jours, mois, âges), plafonds
  scenes.py, aesthetic.py, taste.py, compose.py, learn.py   ce que montre chaque photo, goût appris
  hebcal.py, gphotos.py fêtes juives, photos choisies dans Google Photos
  drive.py, vision.py, report.py, claude_rerank.py, cli.py
```

Tests : `Ctrl+Maj+P › Tasks: Run Test Task`, ou

```bash
.venv/bin/python -m unittest discover -s tests -t .      # Windows : .venv\Scripts\python.exe
```

Comparer un album automatique à une sélection faite à la main, et régler le choix de la prise :

```bash
python -m album_builder.curate --compare out/auto/plan.json out/main/plan.json
python -m album_builder.curate --takes out/main/plan.json
python -m album_builder.taste --eval        # ce que le goût appris sait prédire
```

La ligne de commande d'origine fonctionne toujours :

```bash
python -m album_builder --name "Eydel & frères 2026" --count 50 \
  --folder "https://drive.google.com/drive/folders/…" --year 2026 --must Eydel --bonus Frères
python -m album_builder --apply-plan out/eydel-freres-2026/plan.json
```

Options utiles : `--from/--to AAAA-MM-JJ`, `--match-threshold 0.38` (reconnaissance moins
stricte), `--faces-only`, `--no-recursive`, `--copy` (copies au lieu de raccourcis : consomme du
quota), `--use-claude --describe "…"`.
