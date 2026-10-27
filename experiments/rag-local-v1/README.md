# PatrimoTrip AI — première chaîne RAG locale

Cette étape établit une **référence documentaire légère**, avec le modèle 4B déjà chargé dans llama.cpp. Elle ne charge aucun autre modèle et n'utilise aucune API distante. Elle permet de tester les réponses à partir des 840 passages du corpus plutôt que les seules connaissances du modèle.

Ce n'est pas encore la recherche hybride cible du projet. Le moteur actuel est une recherche lexicale SQLite FTS5/BM25, avec routage explicite par nom de site et ajout d'une introduction encyclopédique pour les sites nommés. Les alias sont définis dans le code pour les dix sites. Les sources bibliographiques sont écartées de la sélection ; deux passages au maximum sont retenus par document, avec suppression de recouvrements importants. Une question sans nom de site ne bénéficie pas de ce routage. Ces choix constituent une référence à comparer, pas un modèle sémantique.

## Pourquoi cette étape

Le PC dispose de 16 Go de RAM, mais environ 1 Go était libre avec le modèle 4B chargé lors du relevé utilisateur. Ajouter maintenant un modèle d'embeddings chargé en FP32 et une nouvelle pile de services ferait courir le risque de pagination et de forte latence.

Le lanceur utilise Python standard et SQLite. Il n'a besoin ni de pip, ni d'un serveur de base de données supplémentaire. Le cache SQLite vise 8 MiB ; ce paramètre ne plafonne pas toute la mémoire du processus Python. Le coût réel reste à mesurer sous Windows. Le modèle 4B, déjà chargé, reste le principal consommateur de ressources.

L'architecture cible conserve recherche hybride, embeddings multilingues, reranking, vision et MCP. La suite devra comparer une variante compacte quantifiée ou des chargements successifs, sans garder tous les modèles en RAM/GPU simultanément. Cette première chaîne sert à mesurer ce que ces composants apporteront.

## Installation et démarrage Windows

Python 3.13 convient à ce lanceur. Il a été vérifié sous Python 3.12 dans l'environnement de développement ; la génération réelle avec le 4B Windows n'a pas été exécutée ici.

Garder llama-server ouvert sur `http://127.0.0.1:8080`, avec le 4B et les réglages déjà testés. Dans une autre fenêtre PowerShell, se placer dans le dossier extrait :

```powershell
py .\rag_local.py doctor
py .\rag_local.py index
py .\rag_local.py ask "Pourquoi la tour Eiffel a-t-elle été construite ?"
```

`doctor` vérifie Python, SQLite FTS5, les empreintes du corpus et la disponibilité du serveur local. `index` crée l'index une seule fois ; les exécutions suivantes le réutilisent si le corpus est inchangé. Une reconstruction explicite avec un corpus modifié sauvegarde l'ancien index.

La commande `ask` affiche la réponse, les liens de ses sources, le temps total et l'emplacement du fichier de résultat local. Aucun identifiant de compte ni secret n'est requis. Les requêtes restent sur une adresse de boucle locale, avec les proxys désactivés uniquement pour ce client local.

Pour examiner les extraits sans appeler le modèle :

```powershell
py .\rag_local.py search "Où se trouve le Panthéon de Paris ?"
```

Pour vérifier les composants sans modèle :

```powershell
py -m unittest -v
```

## Ce qui est contrôlé

- Les empreintes du corpus, les identifiants des passages et leur correspondance au texte normalisé.
- Au maximum quatre extraits, représentant au maximum 4 800 caractères de texte source ; ce n'est pas un compteur exact de tokens. Titres, instructions et question ajoutent du contexte. La sortie est plafonnée à 512 tokens.
- Une sortie JSON contrainte contenant statut, réponse et citations.
- Des citations choisies parmi les extraits réellement récupérés.
- La correspondance entre les marqueurs `[S1]` dans le texte et la liste de citations.
- Le rejet des références inconnues, des listes incohérentes et des réponses tronquées, sans réparation silencieuse.
- Une absence de résultat lexical produit une réponse documentaire insuffisante sans appeler le modèle.

**Ces contrôles n'établissent pas que la source soutient chaque affirmation.** La fidélité, l'absence d'hallucinations et la bonne gestion des contradictions doivent être évaluées en lisant les extraits et les réponses. Une erreur de référence est rejetée, mais une réponse fausse munie d'une référence existante peut encore passer les contrôles techniques.

Les horaires et prix actuels ne sont pas établis par ce corpus. Une présence d'informations historiques ou touristiques ne doit pas servir à promettre des conditions de visite actualisées. Les préférences, parcours, outils MCP et analyse de nouvelles photos ne sont pas encore implémentés dans ce lanceur textuel.

## Premier essai conseillé

1. Demander pourquoi la tour Eiffel a été construite : vérifier Exposition universelle de 1889, et absence de référence à Galilée.
2. Demander où se trouve le Panthéon : vérifier le 5e arrondissement ou une localisation exacte soutenue par les extraits, sans la place du Carrousel.
3. Demander qui a conçu le palais Garnier : vérifier Charles Garnier et sa source.
4. Demander un tarif actuel non établi par les extraits : vérifier que le modèle signale l'information manquante.

Ces questions de développement ne constituent pas un test indépendant : elles ont servi au réglage du moteur. Un jeu de validation et de test séparé sera nécessaire avant d'affirmer une performance générale.

## Résultats et confidentialité

Chaque demande est conservée dans `runs/<horodatage-identifiant>/result.json` avec les passages, leurs sources, la réponse brute, le résultat des vérifications et les temps. Ce dossier et les index générés sont exclus de Git. Les résultats ne sont pas envoyés vers GitHub ou un outil de télémétrie. Le lanceur ne publie rien automatiquement.

Éviter de saisir des données personnelles inutiles. Le présent corpus contient des sources publiques. Le serveur d'inférence doit lui aussi être lié à `127.0.0.1` comme dans les commandes déjà utilisées.

## Corpus et licences

Les 10 sites, 30 documents et 840 passages sont ceux du corpus Paris v1. Les textes Wikipédia sont sous CC BY-SA 4.0, avec liens vers les révisions. Les notices Mérimée et fiches DATAtourisme portent leurs attributions et dates dans les métadonnées. Consulter `corpus/ATTRIBUTIONS.md` et les champs `source` des documents. L'archive de ce lanceur ne contient pas les images mentionnées dans la fiche d'attributions originale.

## Validation effectuée

Huit contrôles automatisés passent : construction/réutilisation de l'index, extraits réels pour la tour Eiffel, localisation du Panthéon et architecte Garnier, traitement des requêtes comme données SQL, absence de résultat, vérification des références, rejet des destinations externes et flux HTTP local avec gestion d'une réponse tronquée.

Les commandes doctor, index et ask ont aussi été exécutées avec un faux serveur local. Ces vérifications portent sur le code, les données et le transport. Elles ne sont pas des tests du modèle 4B ni des mesures de performance sur le PC personnel.


## Lanceur 1.1 : références associées à chaque phrase

Le script source courant utilise un contrat JSON `status` + `claims`. Chaque élément de `claims` contient `text` et sa liste `citations`, avec au maximum trois phrases. Le modèle n'écrit pas de marqueur dans le texte ; le code valide les références puis affiche les marqueurs à la fin de la phrase concernée. La sortie normalisée enregistrée garde les champs `status`, `answer` et `citations` pour faciliter la relecture.

Ce changement évite de demander au modèle de maintenir deux représentations synchronisées des références. Il ne supprime pas le contrôle précédent : les références inconnues, les phrases factuelles sans source et les sorties tronquées restent rejetées. Les réponses anciennes ne sont pas réparées rétroactivement. La vérification du soutien réel des faits reste manuelle.

Le prompt limite aussi la réponse aux informations demandées, pour réduire les ajouts annexes. Les contradictions du corpus ne sont pas corrigées automatiquement par ce changement.

L'archive initiale `rag-local-v1.zip` reste inchangée. Pour essayer la version courante en conservant le premier script, télécharger le fichier source `rag_local.py` sous le nom `rag_local_claims.py` dans le même dossier, puis lancer :

```powershell
py .\rag_local_claims.py ask "Où se trouve le Panthéon de Paris ?"
```

Dix contrôles automatisés passent dans l'environnement de développement, dont le cas du Panthéon sans marqueur généré, la vérification des sources phrase par phrase et la conservation du rejet des anciennes réponses incohérentes. Le contrat HTTP est testé avec un faux serveur ; la génération réelle du nouveau format doit être validée sur le PC utilisateur.
