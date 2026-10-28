# Recherche et réponses par sous-question

Cette étape compare la méthode documentaire précédente avec `facet_coverage_v1` sur six questions : trois questions historiques composées, deux demandes actuelles absentes, une question mixte sur l'événement de construction et le prix actuel. Chaque question est exécutée avec les deux méthodes, soit douze générations.

```powershell
py .\evaluate_retrieval.py --take-over-test-server
```

Cette commande est lancée depuis `hybrid-v1`. Les sites sont fournis pour isoler la recherche et la rédaction : ce n'est pas une mesure de reconnaissance photo et elle ne corrige pas les confusions visuelles comme Invalides/Panthéon. Une partie des formulations et faits a déjà servi au développement. Les contrôles de statut et les citations connues ne prouvent pas la véracité des réponses.

## Méthode

La décomposition est un ensemble borné de règles de syntaxe française, sans appel génératif supplémentaire : conjonction suivie d'une interrogation, demandes séparées par point d'interrogation ou point-virgule, quelques formulations d'usage/origine. Deux ou trois volets sont pris en charge. Une formulation non reconnue ou plus de trois volets conserve la question entière au lieu de supprimer des demandes. Ce n'est pas un analyseur linguistique général ou multilingue.

Les requêtes de chaque volet sont complétées par des termes de recherche neutres, selon leur intention (événement, commémoration, conception, date, usage, tarif, horaire). Aucune date, valeur, identité d'architecte ou réponse attendue n'est injectée. Une commémoration peut reprendre le contexte du volet événement pour ses embeddings ; les termes lexicaux restent propres au volet. Les développements de requête servent seulement à la recherche et ne sont pas présentés comme des faits au rédacteur.

La recherche BM25 et dense reste filtrée sur le site. Les termes du nom du site et certains mots génériques sont écartés de la requête lexicale, puisque l'identité est déjà filtrée. Les rangs sont fusionnés par volet puis les candidats sont réunis en alternance. Le contexte conserve au maximum **quatre passages et 4 800 caractères**, avec déduplication des recouvrements. Pour une question composée, les quatre passages peuvent provenir du même document si leurs sections sont distinctes ; la limite de la référence reste deux passages par document. Les preuves rattachées à chaque volet dans les traces sont des candidats, pas une attestation de soutien sémantique.

Le rédacteur retourne un statut et une phrase pour chaque volet. Un volet `answered` exige une citation récupérée. Un volet `insufficient_sources` peut signaler une absence sans citation. Le programme vérifie chaque sous-réponse, refuse une référence invalide ou une sous-question oubliée, puis agrège les réponses dans l'ordre demandé. Il peut donc présenter un fait historique sourcé et signaler séparément l'absence d'un tarif actuel, sans faire passer la seconde partie pour un fait documenté. Les identifiants de volets ne sont pas affichés comme éléments d'interface.

Les questions simples conservent la méthode antérieure, notamment les deux contrôles de prix et d'horaire. Le module conserve les règles d'identité v3 du parcours photo. La nouvelle rédaction composée est versionnée `photo_facets_v1`, distincte de la consigne simple `photo_partial_answers_v2`. Une relance documentaire d'anciens extraits avec `--replay-latest` garde son objet expérimental : elle ne refait pas la recherche et n'évalue pas cette nouvelle sélection.

## Évaluation et lecture

Le modèle d'embeddings est chargé une fois pour préparer les requêtes manquantes, puis arrêté avant de charger le modèle texte sans projecteur. Les vecteurs existants sont réutilisés dans le même namespace. Les deux méthodes utilisent le même modèle, mais la nouvelle méthode change **à la fois la recherche et le format de rédaction** : le résultat mesure cette version de chaîne documentaire, pas l'effet isolé d'une seule modification.

Le plafond de sortie est de 512 tokens pour la référence et les questions simples, et 768 tokens pour le format composé qui répète les statuts par volet. Les autres réglages du modèle sont conservés. Le test utilise une seule passe dans l'ordre référence puis méthode par volet. Les temps de chaque méthode utilisent un modèle texte déjà chargé et des requêtes pré-encodées ; ils n'incluent pas un changement de modèle par question. La durée totale, `model-events.json` et les mesures de mémoire incluent la préparation du lot. Ces chronométrages ne représentent pas une latence interactive complète ni un benchmark statistique de vitesse.

`cases.json` contient les statuts attendus par volet et des extraits justificatifs connus, y compris plusieurs preuves alternatives lorsqu'elles ont été annotées. Ces annotations sont utilisées pour les scores après l'inférence et ne sont pas transmises au modèle. Elles restent partielles, et leur présence dans un contexte ne prouve pas que la réponse les utilise correctement.

Le rapport produit dans `results/retrieval-facets-<date-identifiant>/report.zip` conserve :

- Les requêtes et plans, classements par volet, contextes et réponses des deux méthodes.
- La présence des preuves connues par volet documentaire, les statuts et contrôles de références.
- Les statuts explicites des volets d'information absente, lorsqu'ils sont disponibles. La référence n'expose pas de statut par volet pour une question composée : les dénominateurs correspondants diffèrent et ne se comparent pas directement.
- `review.json` avec chaque phrase et ses extraits pour une relecture humaine ; les valeurs restent `not_reviewed` avant lecture.
- Les temps, versions et empreintes ainsi que les journaux RAM/GPU globaux à une seconde.

`complete` signifie que les douze exécutions ont abouti techniquement, pas que les réponses sont correctes. Le programme n'écrit aucune note automatique de véracité. À la fin, les processus démarrés sont arrêtés et un seul chemin `report.zip` est affiché. Aucune nouvelle photo ni nouveau modèle n'est nécessaire avec l'installation préparée.

## Vérification du code

Les tests couvrent le découpage borné sans injection des faits attendus, les budgets et le filtrage de site, les réponses partielles, les citations invalides, les volets dupliqués ou oubliés et les sorties tronquées. Une simulation du comparatif complet vérifie les fichiers et scores techniques, avec des vecteurs et réponses factices : elle ne valide pas les performances réelles. Le gain doit encore être mesuré sur le PC utilisateur.
