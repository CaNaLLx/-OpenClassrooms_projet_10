# Modèles de recommandation My Content

## Artefact

`recommender.npz` contient deux approches indépendantes :

- **Collaboratif article–article :** cosinus des colonnes d'une matrice binaire utilisateurs × articles, construite avec les 20 dernières consultations de chaque utilisateur. Les 50 meilleurs voisins sont conservés par article.
- **Contenu :** embeddings réduits par ACP à 64 dimensions et profil pondéré des dix dernières lectures. Le classement utilise leur produit scalaire après normalisation, équivalent au cosinus.

Les deux utilisent les mêmes candidats et les mêmes exclusions. Un repli commun complète les places sans score personnalisé. Les sources sont indiquées dans chaque réponse.

L'artefact inclut le catalogue, dates et catégories, lecteurs connus, profils, historiques, voisins et paramètres d'ACP. Les tableaux sont enregistrés en float32 ou entiers, sans objets pickle. Il fait environ 166,4 Mio. Son format est propre à cette nouvelle version du projet.

## Utilisation

```python
from src.recommender import Recommender

model = Recommender.load("models/recommender.npz")
result = model.recommend(1956)
other = model.recommend(1956, method="collaborative")
new_user = model.recommend(999999, method="content", seed_articles=[283764])
```

La réponse contient cinq identifiants si le catalogue comporte assez d'articles non lus, leurs catégories et dates, leurs scores, les préférences utilisées et le nombre de suggestions de repli. Un score n'est pas une probabilité et ne se compare pas directement entre les deux méthodes.

## Décision et mesures

Le choix est figé sur validation, avant le test. L'hypothèse métier retenue est de pouvoir personnaliser des articles sans clic disposant d'un embedding. Le contenu est sélectionné pour cette capacité. Le collaboratif obtient la meilleure pertinence mesurée.

Sur 2 000 cas communs : CF 40 cibles retrouvées (2,00 %), contenu 25 (1,25 %), repère populaire 32 (1,60 %). Le contenu propose en moyenne 2,22 catégories parmi cinq, contre 4,20 pour le CF. Il fournit 5,10 % de suggestions personnalisées d'articles sans clic.

Ces résultats restent faibles ; ils justifient une comparaison et des compromis, sans démontrer une qualité suffisante pour un lancement. Le fichier `metrics.json` est la source exacte des chiffres. `scenarios.json` contient les situations concrètes, et `evaluation_cases.jsonl` les cas individuels des deux périodes.

## Temps et nouveaux articles

La dernière création ou consultation est la référence considérée comme aujourd'hui. Les données donnent le 13 mars 2018, alors que les consultations retenues s'arrêtent en octobre 2017. Le catalogue récent de la référence globale est donc sans clics d'apprentissage : le CF utilise le repli, tandis que le contenu peut personnaliser avec les embeddings disponibles.

La vue « Fin des consultations » dans l'interface permet de comparer les approches à la fin de l'historique. Les périodes de validation et de test gardent leur chronologie réelle ; les scores d'octobre ne sont pas des mesures de satisfaction à la référence de mars.

Un nouvel utilisateur sans préférence reçoit le repli. Avec un premier article apprécié, le contenu construit un profil ; le CF peut utiliser des voisins si cet article en possède. Un article réellement nouveau demande des métadonnées et un embedding compatible. Les fichiers Globo.com ne fournissent pas l'encodeur d'origine ni les textes historiques.

## Exécution future dans Azure

`src/recommender.py` utilise uniquement NumPy pour la prédiction. La même classe pourra être appelée dans une Azure Function HTTP après chargement d'une version de l'artefact depuis Blob Storage privé. Le pipeline scientifique, SciPy et scikit-learn restent séparés de la requête de recommandation.
