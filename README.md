# My Content — comparaison de deux recommandations

Deux algorithmes distincts : **filtrage collaboratif article–article par cosinus** et **recommandation basée sur le contenu par cosinus sur embeddings**. La popularité sert de repère et de repli. Les scores des deux approches ne sont pas mélangés.

## Tester en local

Dans PowerShell, depuis ce dossier :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Ouvrir http://localhost:8501. Les commandes utilisent directement Python dans l'environnement virtuel, sans activation PowerShell.

Sur une nouvelle installation avec Python 3.11 :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

L'artefact entraîné est dans `models/recommender.npz`. Il n'est pas versionné dans Git : lancer le notebook pour le recréer sur une nouvelle machine.

L'interface propose quatre onglets : recommandations, comparaison des deux méthodes sur le même utilisateur, exemples testés et résultats. Elle affiche cinq articles, leurs catégories, scores et l'origine de chaque suggestion. La case « Tester un nouvel utilisateur » permet de commencer sans historique, puis avec un article apprécié.

## Notebook et entraînement

[Notebook exécuté](notebooks/comparaison_approches.ipynb) : préparation, deux modèles, validation, test, exemples, décision et export.

```powershell
$env:PATH = (Join-Path (Get-Location) '.venv\Scripts') + ';' + $env:PATH
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 notebooks/comparaison_approches.ipynb
```

Le même pipeline est accessible en ligne de commande :

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Les sources Globo.com doivent être présentes dans `data/brut/` : `clicks/*.csv`, `articles_metadata.csv` et `articles_embeddings.pickle`. Les fichiers d'origine sont conservés. La préparation retient 2 977 561 clics sur 2 988 181. Les contrôles retirent 7 331 clics hors session de 24 heures et 3 289 antérieurs à la publication ; aucun doublon exact ni ligne incomplète.

Pour un contrôle rapide avec le petit échantillon, utiliser `--sample --output-dir .runtime/sample`. Ce chemin préserve les modèles complets ; les performances de l'échantillon ne sont pas représentatives.

## Résultats sur les mêmes 2 000 cas de test

| Approche | Cibles retrouvées | Hit Rate@5 | MRR@5 | Catégories moyennes / 5 | Suggestions sans clic personnalisées |
|---|---:|---:|---:|---:|---:|
| Collaboratif | 40 | 2,00 % | 0,01063 | 4,20 | 0,00 % |
| Contenu | 25 | 1,25 % | 0,00625 | 2,22 | 5,10 % |
| Popularité de référence | 32 | 1,60 % | 0,00401 | 4,27 | 0,00 % |


Les modèles sont reconstruits toutes les six heures avec les clics passés. Validation de 48 heures puis test distinct de 48 heures. Candidats, cas, dates et exclusions sont communs ; 98,45 % des cibles de test sont disponibles dans les candidats.

**Choix métier déclaré avant le test :** pouvoir personnaliser un article sans clic, lorsqu'un embedding compatible est disponible. Le contenu est retenu pour ce besoin. Le collaboratif obtient la meilleure pertinence mesurée et serait le choix de cette comparaison si l'objectif prioritaire était la récupération du prochain clic sur le catalogue déjà consulté.

La performance absolue reste faible : cette preuve de concept démontre des capacités et des compromis, sans établir une qualité suffisante pour un lancement. Les scores des anciennes versions supprimées ne sont pas directement comparables à ce nouveau protocole et à ces deux modèles purs.

## Exemples expliqués

- **Concentration :** pour l'utilisateur 1956, le contenu propose 1 catégorie(s) parmi cinq, contre 5 pour le collaboratif. En moyenne sur le test : 2,22 contre 4,20. Ce sont des indicateurs descriptifs du risque de répétition, pas une preuve d'effet de bulle à long terme.
- **Article sans clic :** sur les 402 cibles de test sans consultation d'entraînement, le contenu en retrouve 6 et le collaboratif 0. Un exemple réel de validation, son rang et les deux listes sont enregistrés dans `models/scenarios.json`.
- **Nouvel utilisateur :** sans préférence, repli identique pour les deux. Avec un premier article apprécié, le contenu utilise son embedding ; le collaboratif peut exploiter des voisins s'ils existent.

Les exemples, leurs limites et les résultats globaux figurent dans l'interface et le notebook.

## Dates de simulation

La dernière activité du jeu est la création du 13 mars 2018 ; elle correspond à aujourd'hui dans l'affichage. Les consultations retenues s'arrêtent en octobre 2017. Le catalogue récent de mars ne possède donc pas de clics : le collaboratif y affiche le repli.

L'option **« Fin des consultations »** rejoue le catalogue à la fin des clics pour observer les deux modèles avec des articles connus. Les deux approches utilisent toujours la même date sélectionnée. Les dates affichées reçoivent toutes le même décalage ; les fenêtres de validation et de test conservent leurs dates historiques.

## Architecture et barème

- [Architecture actuelle et cible Azure](docs/architecture.md) : schémas, fonctions des briques et mises à jour des bases Utilisateurs et Articles.
- [Suivi du barème](docs/bareme.md) : preuves présentes et points restant à réaliser.
- [Comparaison détaillée](docs/comparaison.md) : formules, résultats, exemples et limites.
- [Carte des modèles](models/README.md) : contenu de l'artefact et contrat de prédiction.

La cible est **Streamlit → Azure Functions serverless → Blob Storage privé**, avec des bases Utilisateurs et Articles actualisées par ingestion. Le moteur NumPy et son contrat sont réutilisables. Le déploiement Azure et le test HTTP réel sont prévus pour la phase suivante.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Les tests vérifient les deux algorithmes séparément, les articles sans clic, les nouveaux utilisateurs, les publications futures, l'absence de données futures dans l'apprentissage, la conservation après export et les parcours Streamlit.

## Organisation

- `src/data.py` : lecture, contrôles, chronologie et ACP.
- `src/train.py` : matrice collaborative, voisins et profils de contenu.
- `src/recommender.py` : moteur de prédiction NumPy commun aux deux approches.
- `src/evaluate.py` : cas partagés, mesures et décision sur validation.
- `src/experiment.py` : chaîne complète, exemples et sauvegarde des résultats.
- `app/app.py` : interface locale.
- `tests/` : vérifications automatisées.
- `models/metrics.json` et `models/scenarios.json` : résultats et preuves lisibles.

Le jeu ne fournit pas l'encodeur d'origine : un véritable nouvel article nécessite un embedding compatible ou une migration de représentation. La simulation d'une préférence dans l'interface n'écrit pas encore dans une base Utilisateurs.

## Version GitHub

Le code, le notebook exécuté et les rapports sont versionnés sur la branche [codex/comparaison-recommandations](https://github.com/CaNaLLx/-OpenClassrooms_projet_10/tree/codex/comparaison-recommandations). Les données brutes, l'environnement virtuel, les caches et le modèle binaire restent locaux et sont exclus de Git.

Vérifications de cette version : neuf tests automatisés passent ; les deux méthodes ont été contrôlées sur 200 utilisateurs à la fin des consultations et à la référence globale. Les cinq articles sont distincts, non lus et déjà publiés à la date choisie.
