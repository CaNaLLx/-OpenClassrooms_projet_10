# Comparer le filtrage collaboratif et le contenu

## Question métier

My Content veut proposer cinq articles pertinents et intégrer de nouveaux lecteurs et articles. La pertinence sur le prochain clic et la capacité à personnaliser un article sans clic sont deux objectifs distincts. La comparaison conserve ces deux dimensions.

L'hypothèse métier de cette preuve de concept est que l'ajout d'articles sans attendre leurs premières consultations est essentiel. Elle est déclarée avant l'évaluation. La décision peut changer si l'on donne la priorité à la récupération du prochain clic sur un catalogue déjà consulté.

## Deux algorithmes identifiables

### 1. Collaboratif article–article

Une matrice binaire utilisateurs × articles est construite avec les 20 dernières consultations de chaque lecteur, après déduplication des paires. La similarité entre deux articles est le cosinus entre leurs colonnes : nombre de lecteurs communs divisé par la racine du produit de leurs nombres de lecteurs. On conserve 50 voisins par article.

Le score d'un candidat est la somme des similarités avec les dix derniers articles lus, pondérées selon leur rang de récence. Les embeddings, catégories et textes ne participent pas à ce score. Les dates déterminent seulement l'admissibilité des candidats, comme pour le contenu.

**Atouts :** exploite les comportements collectifs, peut faire découvrir des articles aux contenus différents, se passe de textes. **Limites :** nouveaux articles sans voisins, interactions rares, biais vers les articles souvent consultés et actualisation nécessaire.

Référence : [Sarwar et al., 2001](https://files.grouplens.org/papers/www10_sarwar.pdf).

### 2. Contenu par profil et cosinus

Les embeddings de 250 dimensions sont réduits à 64 par ACP. Celle-ci est ajustée seulement sur les articles consultés et publiés avant la validation. La moyenne normalisée des dix derniers embeddings lus, pondérés par 0,8 à la puissance de leur rang, forme le profil utilisateur.

Le score est la similarité cosinus entre le profil et le candidat. Aucun score collaboratif ni popularité n'y est ajouté. Un article peut recevoir un score personnalisé sans avoir été consulté, si son embedding existe.

**Atouts :** démarrage avec peu de préférences, articles sans clic, personnalisation indépendante des autres utilisateurs. **Limites :** qualité des embeddings, répétition des thèmes, profil impossible sans préférence, encodeur compatible nécessaire à un nouvel article réel.

Références : [Pazzani et Billsus, 2007](https://doi.org/10.1007/978-3-540-72079-9_10), [cosinus scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

## Protocole et résultats

385 fichiers fournissent 2 988 181 clics. La règle de session de 24 heures en exclut 7 331 et le contrôle de publication 3 289 ; 2 977 561 restent disponibles. Les 48 dernières heures servent au test et les 48 précédentes à la validation. Chaque fenêtre de six heures reconstruit les modèles avec les clics strictement passés. L'ACP reste fixe.

Un cas est le premier clic inédit dans une fenêtre d'un utilisateur déjà connu. Deux mille cas communs sont échantillonnés par période avec la graine 42. Les candidats sont publiés avant la requête, pendant les sept derniers jours ; la fenêtre est élargie si moins de cinq articles sont disponibles. Les articles déjà lus sont retirés.

| Test | Collaboratif | Contenu | Repère populaire |
|---|---:|---:|---:|
| Cibles retrouvées sur 2 000 | 40 | 25 | 32 |
| Hit Rate@5 | 2,00 % | 1,25 % | 1,60 % |
| MRR@5 | 0,01063 | 0,00625 | 0,00401 |
| Couverture des candidats | 17,67 % | 35,18 % | 0,31 % |
| Catégories moyennes parmi cinq | 4,20 | 2,22 | 4,27 |
| Concentration dans une catégorie | 34,53 % | 71,87 % | 33,14 % |
| Similarité moyenne entre suggestions | 0,113 | 0,666 | 0,039 |
| Articles sans clic personnalisés | 0,00 % | 5,10 % | 0,00 % |
| Places de repli | 0,86 % | 0,00 % | 100,00 % |


Sur les 402 cibles de test sans consultation dans l'apprentissage, le contenu en retrouve 6 et le CF 0. Ces effectifs restent faibles pour conclure à une performance robuste sur ce groupe. Le repère populaire utilise les lecteurs observés dans l'entraînement, sans mélange avec les scores personnalisés.

Le collaboratif a le meilleur Hit Rate@5. Les scores absolus restent faibles. La diversité seule ne suffit pas à justifier un modèle si sa pertinence ne répond pas aux attentes des utilisateurs.

## Situations concrètes reproductibles

### Concentration et risque de répétition

L'exemple de validation de l'utilisateur 1956 montre 1 catégorie(s) parmi cinq pour le contenu, contre 5 pour le collaboratif. Les listes et similarités sont enregistrées dans `models/scenarios.json` et affichées dans le notebook et l'interface.

Cet exemple est sélectionné pour illustrer un contraste. Les moyennes du test sont : 2,22 catégories pour le contenu contre 4,20 pour le CF. Cela décrit une concentration dans les suggestions ; prouver un effet de bulle à long terme demanderait des expositions, des lectures longitudinales et une comparaison de leur évolution. Le collaboratif peut également enfermer dans des comportements dominants.

Référence : [Nguyen et al., 2014](https://doi.org/10.1145/2566486.2568012).

### Démarrage à froid d'un article

L'article 202652 n'avait aucune consultation avant la coupure. Pour l'utilisateur 99736, le contenu le propose au rang 4 ; il est absent des cinq résultats collaboratifs. Il correspond à la prochaine lecture réellement observée dans ce cas de validation. L'historique et les deux listes sont conservés dans `models/scenarios.json`.

Cette démonstration montre que le contenu peut exploiter un embedding sans attendre les interactions. Elle ne signifie pas que tout article sans clic sera pertinent, comme le faible score sur ce groupe le rappelle.

### Nouvel utilisateur

| Information disponible | Collaboratif | Contenu |
|---|---|---|
| Aucune préférence | Popularité, puis fraîcheur | Même repli ; aucune personnalisation |
| Premier article apprécié avec interactions connues | Utilise les voisins appris de cet article | Construit un profil avec son embedding |
| Premier article apprécié sans interactions | Pas de voisins : repli | Peut personnaliser avec son embedding |

La préférence est simulée dans l'interface ; la base Utilisateurs cible devra la conserver pour les visites suivantes.

## Décision et limites

L'approche **contenu** est retenue sous l'exigence déclarée de personnaliser les articles entrants. Le modèle **collaboratif** reste le meilleur de cette comparaison sur le prochain clic et une alternative si l'on privilégie les lecteurs et articles déjà connus. Le choix n'est pas présenté comme une victoire du contenu sur la pertinence.

La référence globale du catalogue est mars 2018, après la fin des consultations d'octobre 2017. L'interface conserve cette référence comme aujourd'hui et propose aussi de rejouer la fin des consultations. Le catalogue récent de mars est sans interactions : le CF utilise alors le repli. Les performances historiques d'octobre ne sont pas attribuées à cette période de mars.

Les embeddings sont fournis et leur entraînement d'origine n'est pas reproduit. Les titres et textes sont absents ; ajouter un article réel exige un encodeur compatible ou une migration de représentation. Les clics sont des signaux implicites et plusieurs fenêtres peuvent concerner le même lecteur. Une validation en ligne et des données nouvelles restent nécessaires avant toute conclusion de qualité produit.
