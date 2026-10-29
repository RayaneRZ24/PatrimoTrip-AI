# PatrimoTrip AI — dossier du démonstrateur

## Problème traité

Un utilisateur dispose d'une photo de monument et souhaite comprendre ce lieu puis préparer une découverte culturelle. Une réponse générative seule peut inventer une identité, une date ou une référence. Le projet sépare donc proposition d'identité, recherche documentaire, reproduction des preuves, organisation du parcours et confirmation de l'enregistrement.

## Périmètre livré

Le démonstrateur couvre dix monuments de Paris. Il associe une modalité image, des demandes textuelles, un corpus documentaire, un agent à actions bornées et des outils locaux MCP. L'application web réunit ces fonctions et conserve les parcours confirmés sur le PC.

Le scénario principal est réalisable localement : photo → proposition ou abstention → accord de deux analyses → extraits historiques consultables → promenade extérieure estimée → confirmation → sauvegarde et export. Les informations pratiques du jour ne sont pas déduites de l'histoire du monument.

## Méthode

La reconnaissance utilise un modèle de caractéristiques visuelles et une comparaison aux références, puis une analyse multimodale indépendante. Une règle figée permet de s'abstenir avant la génération historique. La ville, le pays et le nom canonique sont aussi contrôlés. Les modèles sont chargés successivement pour respecter les limites de mémoire de la machine.

La recherche photo combine BM25 et embeddings par fusion de rangs, avec un filtre de site en amont. Les demandes composées sont séparées en volets. La présentation conserve une introduction du site et les réponses affichent des phrases exactes des sources, sélectionnées par le modèle mais reproduites par le programme.

L'agent choisit une action structurée, dont les arguments sont validés. Les outils sont exécutés par le client MCP. L'optimisation de promenade est déterministe et ne demande pas au modèle d'inventer des distances. L'enregistrement est une action distincte, autorisée par la confirmation utilisateur.

## Données et reproductibilité

Le corpus historique, les images de référence et les ensembles d'évaluation sont figés par empreintes. Leurs métadonnées, sources et licences sont conservées. Les photographies de référence ne sont pas utilisées comme requêtes de calibration ou de test. Les photographes et sources sont séparés entre les deux derniers ensembles.

Le pilote de calibration choisit une règle avant les inférences du test réservé. Après examen des résultats, une nouvelle variante du modèle, des références ou des seuils nécessite un nouveau test réservé. Un résultat sur quelques images ne doit pas être présenté comme une précision générale. Les rapports locaux sont conservés hors Git et peuvent être joints à la présentation après relecture.

## Vérification logicielle

Les tests couvrent les empreintes, la séparation des données, l'abstention, les décisions en cas de désaccord, l'exclusion des images des rapports, les citations exactes et leur contexte, les appels MCP, le refus des écritures non confirmées, le respect des budgets de temps et l'optimisation sur de petits exemples comparés exhaustivement.

L'interface est vérifiée en navigateur sur les flux de préparation, confirmation, sauvegarde et export. Les écrans d'inférence y utilisent des réponses simulées, pour distinguer une vérification d'interface d'une évaluation des modèles. Le contrôle `verify_release.py --models` permet de vérifier le chargement réel et le nouveau contrat documentaire sur le PC cible.

## Limites et prolongements

L'accord des modèles n'est pas une preuve d'identité. Une citation exacte n'est pas une garantie de vérité ou de pertinence. Les refus réduisent la couverture. Les parcours ne suivent pas les rues et n'intègrent pas de disponibilités ou horaires vérifiés. Les informations pratiques nécessitent des sources officielles actualisées ou des API fiables avant d'étendre le planificateur aux visites intérieures et budgets d'entrée.

L'application n'est pas un service public déployé : elle écoute uniquement en local, sans comptes distants ni réservation. Elle ne prétend pas intégrer toutes les technologies récentes disponibles. Les choix sont liés à l'objectif pédagogique, aux licences, à la qualité des données et aux ressources du PC.

L'énoncé original du cours n'était plus accessible pendant la phase d'intégration. La correspondance à d'éventuelles exigences supplémentaires — autres modalités, livrables ou critères de notation — doit être vérifiée avec ce document. Ce dossier décrit ce qui est effectivement implémenté, sans lui attribuer des fonctions absentes.
