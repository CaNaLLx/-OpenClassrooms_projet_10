# Suivi du barème

Les preuves locales sont distinguées des tâches cloud restantes. Un test du moteur n'est pas un test de fonction serverless déployée.

## Sélectionner l'architecture logicielle

| Critère | Preuve | État |
|---|---|---|
| CE1 — Identifier les briques | Tableaux des composants actuels et cibles dans `docs/architecture.md` | Documenté |
| CE2 — Schématiser les briques | Deux schémas Mermaid : local et cible Azure | Documenté |
| CE3 — Décrire chaque brique | Entrées, fonctions et sorties, puis responsabilités cibles | Documenté |
| CE4 — Actualiser Utilisateurs et Articles | Bases nommées dans le schéma et opérations de création, mise à jour et retrait décrites | Cible documentée ; ingestion réelle à réaliser |

## Concevoir la chaîne IA

| Critère | Preuve | État |
|---|---|---|
| CE1 — Entraîner et tester | Notebook exécuté `notebooks/comparaison_approches.ipynb`, pipeline, métriques et tests | Entraînement et évaluation complets réalisés |
| CE2 — Tester CF et contenu | Deux scores indépendants, cas et protocole communs, résultats séparés et exemples | Réalisé dans le notebook et les rapports |
| CE3 — Fonction serverless appelée par l'application | Contrat défini et classe de prédiction réutilisable | Déploiement Azure et test HTTP réel à réaliser ensuite |
| CE4 — Fonctions et classes | Modules `data`, `train`, `evaluate`, `experiment`, classe `Recommender` | Implémenté |
| CE5 — Git et GitHub | Historique Git préservé | Commit et publication de cette nouvelle version à vérifier |

## Comparaisons à présenter

- Hit Rate@5, MRR@5 et NDCG@5 sur les mêmes clics futurs.
- Article sans clic avant apprentissage : possibilité de score personnalisé et résultat constaté.
- Nouvel utilisateur : sans préférence, puis après un premier choix.
- Concentration des catégories et similarité des suggestions : indicateurs descriptifs, sans prétendre prouver un effet de bulle à long terme.
- Différence entre meilleure pertinence globale et respect de l'exigence d'articles entrants.

Les supports de soutenance seront reconstruits à partir de cette version validée dans une phase ultérieure.
