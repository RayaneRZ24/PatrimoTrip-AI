# PatrimoTrip AI — évaluation hybride locale v1

Un lancement prépare les embeddings, compare la recherche lexicale et dense, teste la fusion et le reranking, puis archive les résultats dans **un seul `report.zip`**. Python standard et SQLite FTS5 suffisent : aucun paquet pip, PyTorch ni service de base de données n'est ajouté.

## Comparer les profils texte et vision

Cette comparaison ne télécharge rien et n'utilise pas le modèle d'embeddings. Elle réutilise les fichiers 4B existants et fixe les mêmes extraits documentaires avant d'exécuter les deux profils :

```powershell
py .\compare_profiles.py --take-over-test-server
```

Le profil `vision_baseline` charge le projecteur comme lors de l'expérience précédente ; le profil `text_only` le retire. Le nombre de couches GPU, le contexte, les batches, les paramètres de génération et les six questions restent identiques. Douze réponses sont produites au total. Le test mesure la mémoire globale après le chargement de chaque profil, pendant les questions ; le temps de chargement est enregistré séparément. Les extraits viennent de la référence avec routage et sont partagés à l'identique : ce test isole le profil de génération, sans refaire la comparaison des moteurs de recherche.

Le résultat est dans `results/profiles-<date-identifiant>/report.zip`. L'archive contient les deux séries de réponses et leurs contextes, les minimums de RAM disponible, la médiane de RAM disponible, les pics GPU et les temps de réponse, ainsi que les journaux bruts. La conformité des citations et les statuts attendus sont contrôlés ; la qualité factuelle doit encore être comparée en lisant les réponses. Les mesures sont globales, à une seconde et sur un seul passage vision puis texte ; les autres applications et l'ordre de passage peuvent influer. Aucun gain mémoire n'est présumé.

Les règles de remplacement du serveur sont celles décrites ci-dessous. À la fin, le serveur **texte** reste prêt sur 8080. Il faut recharger le profil vision pour accepter des photos ; le basculement automatique par type de message n'est pas encore implémenté. Si une étape échoue, le rapport et les résultats déjà obtenus sont conservés, et les processus lancés sont arrêtés.

Pour utiliser explicitement le profil léger dans le lanceur hybride existant :

```powershell
py .\run_local.py ask "Qui a conçu le palais Garnier ?" --text-only --take-over-test-server
```

Le mode sans `--text-only` conserve le profil vision précédent pour permettre les comparaisons et éviter de changer silencieusement sa configuration.

## Lancement sur le PC préparé

Depuis ce dossier extrait, avec Python 3.13, llama.cpp Vulkan et les deux fichiers 4B déjà téléchargés :

```powershell
py .\run_local.py run --take-over-test-server
```

Le chemin attendu des fichiers existants est `$HOME\PatrimoTripAI\models\qwen3.5-4b\Qwen3.5-4B-Q4_K_M.gguf` et `mmproj-F16.gguf`. Les options `--generation-model` et `--projector` permettent de fournir d'autres chemins. Le profil de génération reprend le test utilisateur : Vulkan0, 20 couches GPU, contexte 4096, une requête à la fois, batch 256 / microbatch 128, projecteur déporté, budget de raisonnement zéro. Référence de compatibilité : llama.cpp b11889 / d854d98ba.

Le lanceur télécharge un GGUF de **Qwen3-Embedding-0.6B**, uniquement auprès d'une liste d'éditeurs identifiés, avec modèle de base et licence Apache-2.0 déclarés. Il préfère Q8_0, puis Q4_K_M, fixe la révision obtenue et vérifie le SHA-256 LFS avant utilisation. Le dépôt, la révision, la licence et l'empreinte figurent dans le rapport. Les fichiers 4B précédemment vérifiés sont réutilisés ; leurs empreintes sont enregistrées, sans nouveau téléchargement.

Le modèle d'embeddings et le 4B sont chargés **successivement**. Le premier lancement encode 840 passages et 32 questions ; cela peut prendre plusieurs minutes. Une progression est affichée. Les vecteurs sont conservés dans SQLite après chaque élément pour reprendre un travail interrompu. Le namespace dépend du modèle, du corpus et du format d'entrée. Si le démarrage GPU des embeddings échoue, un seul essai CPU est effectué, avec le 4B arrêté. Un échec d'encodage ultérieur est rapporté sans substitution silencieuse.

Avec `--take-over-test-server`, le lanceur peut arrêter le serveur de test Windows déjà sur 8080 **seulement** si son processus est llama-server.exe et sa commande contient le chemin exact du 4B attendu, `--host 127.0.0.1` et `--port 8080`. Il refuse de remplacer un autre service. Le port 8081 doit être libre. Les nouveaux processus qu'il démarre sont suivis individuellement. À la fin d'une exécution réussie, le 4B reste accessible sur 8080 ; lors d'une erreur, les processus démarrés par le lanceur sont arrêtés. L'ancien serveur de test arrêté n'est pas restauré automatiquement en cas d'erreur.

Les inférences restent sur 127.0.0.1 ; les appels locaux ignorent les proxys. Internet sert à l'acquisition initiale du modèle public. Après cette acquisition et la création du cache, une reprise peut fonctionner hors connexion. Rien n'est téléversé automatiquement. Les poids, caches et résultats sont exclus de Git.

## Comparaison effectuée

| Méthode | Fonctionnement |
|---|---|
| `bm25` | SQLite FTS5, sans ajout automatique d'introduction par alias |
| `routed_current` | Référence existante : recherche lexicale, routage des noms connus, introduction |
| `dense` | Embeddings 1024 dimensions normalisés, similarité cosinus sur les 840 passages |
| `hybrid_rrf` | Fusion des rangs BM25 et dense, poids égaux, constante 60, profondeur 50 |
| `hybrid_llm_rerank` | Reclassement des huit premiers candidats hybrides par le 4B existant |

L'instruction de tâche est ajoutée **aux questions seulement**, conformément au format Qwen3-Embedding. Les passages encodent titre, section et texte. Le reranking est un essai listwise avec le modèle génératif 4B ; ce n'est pas un modèle spécialisé entraîné pour le reranking. Il reçoit des extraits limités à 700 caractères et doit retourner une permutation JSON des identifiants des huit candidats. Une sortie incorrecte est conservée comme échec, sans réparation ni repli caché.

Les quatre méthodes de recherche sont évaluées sur toutes les questions positives. Le reranking est testé sur un sous-ensemble fixé de six questions : cinq positives et une demande de tarif actuel absent. Pour chacune, trois réponses sont générées (`routed_current`, hybride, hybride avec reranking), soit 18 essais lorsque les étapes réussissent. La sélection de contexte applique les mêmes limites : quatre passages, 4 800 caractères, déduplication et au maximum deux passages par document.

## Jeu annoté et limites d'interprétation

`cases.json` contient **32 questions de développement** : dix avec le nom du site, vingt paraphrases, une contradiction de sources et une demande sans réponse actuelle disponible. Pour 31 questions, des citations exactes ancrent les passages justificatifs connus. Les autres passages pouvant répondre ne sont pas tous annotés : les métriques sont donc une présence d'**éléments justificatifs connus**, pas un rappel exhaustif ni une précision générale. Le cas négatif est exclu de ces moyennes ; le statut d'abstention attendu est contrôlé dans les essais de génération.

Ce jeu a servi à développer l'outil et ne constitue pas un test indépendant. Il faudra ensuite ajouter des questions nouvelles, des sources alternatives pertinentes et une lecture humaine des réponses. Les variantes se comparent sur le même sous-ensemble : le tableau des cinq cas positifs de reranking ne doit pas être comparé directement au tableau des 31 questions positives.

La question du concours Garnier conserve la contradiction observée : Wikipédia donne une victoire en 1861 et la notice Mérimée PA00089004 indique 1860. Les deux groupes de preuves sont annotés. Le corpus d'origine n'est pas corrigé silencieusement. Les références et attributions du corpus sont dans `corpus/ATTRIBUTIONS.md` ; les contrôles d'intégrité existants sont conservés.

Un marqueur de source valide ne prouve pas qu'une affirmation est soutenue par le passage. Le rapport distingue contrôles de références, statut attendu pour l'absence de tarif, et **relecture factuelle toujours en attente**. Le statut `complete` signifie que l'exécution et ces contrôles ont réussi, pas que l'application est validée.

## Temps, mémoire et fichiers produits

Le benchmark encode les questions à l'avance puis exécute les générations en lot. Ses durées de réponse avec requêtes déjà encodées **n'incluent pas** le changement de modèle par question. `model-events.json` donne les chargements du lot et `summary.json` la durée totale, préparation comprise. Il ne faut pas présenter ces durées de génération comme une latence interactive complète.

La mémoire physique disponible est relevée chaque seconde dans `memory.jsonl`. Si nvidia-smi est présent, `gpu.csv` relève la mémoire et l'utilisation globales GPU chaque seconde. Ces mesures incluent les autres applications et peuvent manquer un pic bref ; elles ne sont pas une allocation précise du processus Python.

Chaque exécution crée `results/<date-identifiant>/report.zip`, y compris en cas d'erreur après initialisation. L'archive contient synthèse, classements, extraits et réponses, journaux de chargement et mesures de ressources. Elle ne contient ni poids ni cache vectoriel. Elle peut contenir des chemins locaux dans les journaux ; son partage reste manuel. `rapport.md` permet de lire la synthèse avant de partager l'archive.

Pour une question interactive ultérieure (le temps total inclut cette fois les changements de modèle nécessaires) :

```powershell
py .\run_local.py ask "Qui a conçu le palais Garnier ?" --take-over-test-server
```

Pour refaire seulement la référence lexicale, sans modèle ni réseau :

```powershell
py .\run_local.py lexical
```

## Vérification du code

```powershell
py -m unittest discover -s . -p "test_*.py" -v
```

Quatorze tests couvrent les annotations réelles, le format de l'API d'embeddings simulée, les vecteurs, l'isolation du cache, la fusion des rangs, le rejet d'un reranking invalide, la sélection des processus, les changements de modèle simulés et le flux complet avec modèles factices. Les tests des profils vérifient aussi les commandes sans projecteur, les contextes identiques, la collecte des ressources et la conservation du résultat initial si le démarrage suivant échoue. Les dix tests du RAG d'origine ont aussi été exécutés. La recherche lexicale sur le vrai corpus a été exécutée dans l'environnement de développement.

L'expérience hybride initiale a depuis été exécutée sur le PC Windows de l'utilisateur : 872 encodages, six rerankings et 18 réponses ont abouti. Le rapport a confirmé une RAM libre descendant à environ 278 Mio pendant la génération. Il reste privé et n'est pas intégré à ce paquet. Le nouveau comparatif **avec/sans projecteur** est vérifié avec des processus et réponses simulés ; son gain mémoire réel doit être mesuré sur ce PC. Aucun gain n'est annoncé avant cette mesure.

Sources techniques : [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [llama.cpp au commit d854d98ba](https://github.com/ggml-org/llama.cpp/tree/d854d98ba), et corpus Paris v1 dans le dépôt. Aucun fine-tuning n'est effectué. Cette expérience évalue la recherche documentaire ; l'intégration MCP et le planificateur de voyage restent des étapes distinctes du projet.
