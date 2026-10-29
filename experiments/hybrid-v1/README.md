# PatrimoTrip AI — évaluation hybride locale v1

## Application locale intégrée

```powershell
py .\verify_release.py --models --open
```

Ce lancement vérifie le logiciel, les outils MCP et le parcours photo sur une image connue, écrit un rapport unique puis ouvre l'application à `http://127.0.0.1:8765`. Garder PowerShell ouvert. Pour les lancements suivants : `py .\app_local.py --open`.

L'application propose l'analyse photo avec abstention, un agent de questions culturelles utilisant MCP, la préparation de promenades extérieures, la confirmation de sauvegarde, l'export et l'impression. Le nouveau parcours documentaire reproduit des phrases sources exactes : le modèle sélectionne des identifiants, pas des paraphrases factuelles libres. Les données pratiques actuelles et les routes dans les rues ne sont pas vérifiées ; les parcours restent des estimations.

Documentation : [guide utilisateur](docs/USER_GUIDE.md), [architecture](docs/ARCHITECTURE.md), [installation](docs/SETUP.md), [dossier de projet](docs/PROJECT_REPORT.md). Les sections suivantes conservent l'historique des expériences ; les anciens contrats de génération ne décrivent pas tous le nouveau mode assisté.

Un lancement prépare les embeddings, compare la recherche lexicale et dense, teste la fusion et le reranking, puis archive les résultats dans **un seul `report.zip`**. Python standard et SQLite FTS5 suffisent : aucun paquet pip, PyTorch ni service de base de données n'est ajouté.

## Parcours photo assisté

Le parcours recommandé est désormais `py .\photo_assisted.py --pick --take-over-test-server` : comparaison aux références avec la règle figée 0,70/0,15, analyse indépendante de Qwen, puis explication sourcée seulement si les deux proposent le même site. Un refus ou désaccord demande une précision avant toute génération historique. Le processus CPU se ferme avant Qwen et les serveurs gérés s'arrêtent à la fin. Voir [visual-policy-v1/README.md](visual-policy-v1/README.md) pour les confirmations explicites, relances, limites et fichiers produits. Les anciens scripts restent disponibles pour reproduire les expériences antérieures.

## Essai séparé de comparaison visuelle

`py .\evaluate_visual_refs.py` compare les dix images d'évaluation aux dix références originales avec DINOv2-small sur CPU. Ce nouvel essai crée son propre environnement avec NumPy, Pillow et ONNX Runtime ; les outils précédents restent en Python standard. Il fournit des candidats visuels, sans confirmer d'identité, sans seuil hors catalogue et sans modifier le parcours historique. Voir [visual-references-v1/README.md](visual-references-v1/README.md) pour les téléchargements vérifiés, les licences et les limites. Les résultats réels du modèle restent à mesurer sur le PC utilisateur.

Pour comparer une et trois vues par site : `py .\evaluate_visual_refs.py --expanded-references`. La [banque enrichie](visual-references-v2/README.md) ajoute vingt photos attribuées et vérifiées visuellement. Les anciens poids, vecteurs et résultats restent utilisables ; le rapport aligne les classements avant/après sur les mêmes dix photos déjà connues. La galerie locale [gallery.html](visual-references-v2/gallery.html) présente les trente références. Aucun gain de précision ni rejet hors catalogue n'est présumé avant la comparaison.

Le [protocole pilote d'abstention](abstention-v1/README.md) utilise quarante autres images, vingt pour choisir une règle score/écart et vingt pour mesurer cette règle figée : `py .\evaluate_visual_refs.py --calibrate-abstention`. Les photographes, sources et fichiers sont séparés entre ces deux ensembles. La règle est enregistrée avant les inférences du test, sans modifier le parcours historique. Les refus et mauvaises propositions sont rapportés séparément ; zéro erreur de calibration ne garantit pas zéro erreur future.

## Serveur existant et port 8080

L'évaluation démarre et arrête ses propres modèles. Il n'est pas nécessaire de lancer `llama-server` dans une deuxième fenêtre. Si un serveur manuel est encore ouvert, on peut le fermer avec Ctrl+C avant de lancer l'évaluation. L'option `--take-over-test-server` conserve le contrôle strict du processus attendu.

L'arrêt du serveur reconnu attend désormais jusqu'à 30 secondes la libération du port (six auparavant). Si le port reste occupé, le journal `model-events.json` conserve le nom et le PID du propriétaire observé, sans sa ligne de commande complète. Le lanceur s'arrête avec un message explicite et n'arrête pas un nouveau propriétaire du port. Ce changement améliore l'attente et le diagnostic ; sans rapport complémentaire, il ne permet pas d'affirmer pourquoi un port particulier est resté occupé.

## Recherche documentaire par sous-question

La chaîne propose maintenant `facet_coverage_v2` pour les formulations composées reconnues : recherche par volet, sélection équilibrée des extraits et statut de réponse distinct pour chaque volet. Le contexte reste limité à quatre passages et 4 800 caractères. La version v2 réserve de la place aux derniers extraits et termine leurs préfixes à une ponctuation de phrase, afin de limiter les preuves coupées. Les questions simples conservent la méthode précédente. Une réponse partielle peut établir un fait historique et signaler séparément qu'un tarif actuel manque.

Le comparatif documentaire utilise six questions avec les deux méthodes, sans réexécuter la vision :

```powershell
py .\evaluate_retrieval.py --take-over-test-server
```

Les sites sont fournis pour isoler la recherche et la rédaction. Les photos, poids et vecteurs existants restent conservés. Les règles, limites, mesures et fichiers produits sont décrits dans [retrieval-v2/README.md](retrieval-v2/README.md). Les tests automatisés vérifient le logiciel ; le gain réel sur les réponses doit être mesuré sur le PC utilisateur.

## Contrôle de l'identité canonique — v3

Le modèle visuel ne produit plus d'identifiant Wikidata. Il choisit un **nom canonique qualifié par ville et pays** ; le programme résout ce nom dans le dictionnaire du corpus. L'analyse fournit séparément la ville, le pays, les observations et l'hypothèse. Le champ `identity_check` conserve ces valeurs, la version `canonical_location_v3` et les éventuelles contradictions. Aucune confiance numérique calibrée ni vérification indépendante n'est ajoutée.

Si la ville ou le pays sont inconnus ou différents de ceux du site choisi, ou si l'analyse cite le nom d'un autre monument du corpus, la proposition devient `uncertain` / `unknown`. Le programme demande une précision et ne lance ni recherche documentaire ni génération historique. La détection de noms utilise les alias du corpus : elle est conservatrice et peut aussi arrêter une analyse qui faisait une comparaison légitime. Cette règle ne reconnaît pas tous les lieux du monde et ne remplace pas une bonne reconnaissance visuelle.

Un nom faux accompagné d'une ville, d'un pays et d'une hypothèse cohérents peut toujours passer. Le contrôle ne garantit donc pas de corriger une confusion visuelle Invalides/Panthéon. Les documents historiques n'attestent jamais à eux seuls l'identité de la photo. Les indications explicites `--confirm-site` restent enregistrées comme fournies par l'utilisateur, sans prétendre à une vérification indépendante.

Une relance documentaire issue d'une ancienne identification visuelle sans ce contrôle demande désormais une nouvelle analyse de photo au lieu de reproduire silencieusement l'identification ancienne. Une identité fournie par l'utilisateur reste utilisable ; une identité v3 est recontrôlée contre le nom et le site documentaire lors de sa relance.

Pour mesurer cette correction sur **six cas déjà examinés** :

```powershell
py .\evaluate_photo.py --identity-regression --take-over-test-server
```

Le sous-ensemble comprend Eiffel et Panthéon comme contrôles, Garnier, Invalides, le Panthéon de Rome et le mur sans contexte. Les photos, questions et réponses attendues d'origine sont conservées ; aucun label attendu n'est transmis à l'inférence. Le manifeste et les images figés ne sont pas modifiés. Le résultat est une régression de développement, pas une nouvelle mesure indépendante de précision.

Si le rapport complet antérieur de douze cas est présent dans `results/` et correspond au même jeu, `identity-before-after.json` compare les mêmes cas avant et après, en indiquant le dossier source. Les premiers résultats restent intacts. Les proportions de ce sous-ensemble ne doivent pas être comparées directement à celles des douze scénarios initiaux. La régression v3 initiale ciblait l'identité. La version documentaire par sous-question est évaluée séparément ; les versions d'identité, de recherche et de rédaction sont consignées dans les nouveaux rapports.

## Évaluer le parcours sur des images supplémentaires

Un jeu de **dix nouvelles images et douze scénarios** est inclus dans `evaluation-v1/`, avec leurs attributions et des questions documentées ou sans information actuelle disponible. Le lanceur exécute le parcours complet, sans transmettre les réponses attendues au modèle :

```powershell
py .\evaluate_photo.py --take-over-test-server
```

Un seul `report.zip` rassemble les sorties et les mesures. Les détails du jeu, les limites des scores et la grille de lecture sont dans [evaluation-v1/README.md](evaluation-v1/README.md). Les contrôles automatiques ne prétendent pas mesurer la véracité : `review.json` conserve les affirmations et les extraits pour une lecture humaine. Ce lancement peut prendre plusieurs minutes et arrête ses modèles à la fin.

## Photo → monument proposé → explication sourcée

Le lanceur `photo_local.py` relie désormais la vision, les embeddings et la génération textuelle. Pour sélectionner une photo avec la boîte de dialogue Windows :

```powershell
py .\photo_local.py --pick --take-over-test-server
```

Vous pouvez aussi indiquer un chemin et une question :

```powershell
py .\photo_local.py "C:\Photos\monument.jpg" --question "Quelle était sa fonction d'origine ?" --take-over-test-server
```

Ce premier profil accepte JPEG/PNG, au maximum 8 Mio et 4 millions de pixels. Il ne redimensionne pas les originaux et ne traite pas les HEIC. Les dimensions sont contrôlées avant de remplacer un serveur ; le décodage complet de l'image reste assuré par llama.cpp. Une erreur de fichier ou d'inférence est archivée.

Le lanceur charge la vision pour obtenir une proposition JSON, la décharge, charge les embeddings si la question ou des passages du site ne sont pas encore en cache, puis charge le **profil texte sans projecteur** pour rédiger l'explication. Un seul modèle lourd est conservé à la fois. Le cache et son namespace sont les mêmes que dans l'expérience hybride : les 840 passages déjà préparés n'ont pas à être réencodés. Les poids déjà vérifiés dans `models/model.json` sont réutilisés hors connexion ; s'ils manquent, la même acquisition publique vérifiée est proposée par le lanceur existant. La première exécution sur un dossier sans cache peut donc demander une préparation plus longue.

L'identification est toujours présentée comme une **proposition visuelle non vérifiée**, sans score de confiance inventé. Les observations du modèle sont séparées de son hypothèse et ne sont pas transmises comme preuves à la génération documentaire. L'explication concerne le site proposé ; ses références historiques ne prouvent pas l'identité de la photo. Un modèle peut encore proposer le mauvais site avec assurance : une validation sur de nouvelles photos reste nécessaire.

Si la sortie visuelle est incohérente, tronquée ou invalide, aucune explication n'est générée. Si le modèle indique `uncertain`, le programme demande un nom ou une meilleure vue ; s'il indique `out_of_scope`, il signale que l'image n'a pas été identifiée parmi les dix sites du corpus. Dans ces deux cas, aucune recherche ni génération historique n'a lieu et le modèle vision est arrêté. Le programme affiche les identifiants des sites pour permettre une correction explicite, sans sélectionner automatiquement le plus proche.

Pour fournir vous-même une identité connue ou corriger une proposition :

```powershell
py .\photo_local.py --pick --confirm-site Q188856 --take-over-test-server
```

Cet exemple fournit explicitement « Panthéon » ; le programme ne le déduit pas de la photo et n'appelle pas la vision. L'origine utilisateur est conservée dans le résultat, sans prétendre à une vérification indépendante.

La recherche hybride BM25 + dense + RRF est limitée **aux passages rattachés au site choisi**, avant le classement des candidats. Elle conserve quatre extraits au maximum et les contrôles de citations par affirmation existants. Le reranking génératif n'est pas ajouté à ce parcours : les essais précédents ne justifient pas son coût systématique. Les sources peuvent mentionner d'autres édifices dans leur contenu ; le filtre porte sur le rattachement documentaire, pas sur l'absence de toute comparaison.

Les observations proposées, l'explication, les liens des références et le chemin du `report.zip` sont affichés dans la console. L'archive se trouve dans `results/photo-<date-identifiant>/` et contient notamment `vision.json`, `retrieval.json`, `answer.json` si ces étapes ont été atteintes, `result.json`, les journaux, les empreintes et les mesures RAM/GPU. La photo originale et la requête image en base64 ne sont pas copiées dans ces fichiers de résultat. Les métadonnées enregistrent le nom, les dimensions et l'empreinte du fichier. Aucun résultat n'est envoyé automatiquement.

Après une réponse acceptée, le serveur texte reste prêt sur 8080. Après une abstention ou une erreur, les processus lancés sont arrêtés ; aucun redémarrage implicite du serveur précédent n'est garanti. La règle de reprise du serveur existant est la même que pour les autres lanceurs.

Les tests du parcours utilisent des réponses et processus simulés : ils couvrent l'isolation des sources, les interruptions, les images incompatibles, la réutilisation du cache, les deux origines d'identité et les réponses rejetées. Un premier parcours sur une photo Eiffel a été exécuté sur le PC Windows. La relance documentaire v2 a corrigé une abstention sur ce cas, tout en laissant un détail non étayé à relever. Le nouveau jeu d'évaluation doit encore être exécuté ; aucune précision générale de reconnaissance n'est annoncée.

## Relancer seulement une explication documentaire

```powershell
py .\photo_local.py --replay-latest --take-over-test-server
```

Ce mode reprend le dernier rapport `photo-*` contenant une recherche documentaire. Il conserve sa question, son identification et ses extraits ; il ne relance ni la vision, ni les embeddings, ni la recherche. Le corpus est vérifié et les textes sont contrôlés contre les passages d'origine avant d'être transmis au modèle. Les liens et métadonnées documentaires sont reconstruits depuis le corpus local. Les nouveaux fichiers sont écrits dans un autre dossier, sans modifier le rapport précédent. Le rapport conserve `replayed_from` et précise la portée de cette relance.

La consigne photo `photo_partial_answers_v2` autorise une réponse partielle lorsque certains aspects demandés sont documentés. Une destination explicite de construction peut répondre à la question de la fonction d'origine ; aucun fait spécifique à un monument n'est codé en dur. La question par défaut des nouvelles photos devient « Présente brièvement ce monument et explique pourquoi il a été construit. ». Une relance garde toutefois la question originale, pour comparer la rédaction à contexte identique. La consigne des expériences hybrides et des comparaisons de profils reste inchangée.

Un rapport utilisateur a montré une abstention erronée malgré des extraits donnant le but de construction. Les nouveaux tests vérifient la relance et la conservation d'une abstention véritable ; ils ne prouvent pas à eux seuls que la nouvelle consigne élimine ce défaut du modèle. Une nouvelle génération réelle est nécessaire pour la vérifier.

Le résultat distingue maintenant le statut documentaire `answered`, `insufficient_sources` ou `needs_clarification` du contrôle technique des références. Une abstention au format valide n'est plus désignée comme une identification avec réponse historique. L'en-tête des liens de sources n'est affiché que lorsqu'il existe des citations. La progression des embeddings distingue les éléments déjà en cache des encodages réellement calculés.

## Comparer les profils texte et vision

Cette comparaison ne télécharge rien et n'utilise pas le modèle d'embeddings. Elle réutilise les fichiers 4B existants et fixe les mêmes extraits documentaires avant d'exécuter les deux profils :

```powershell
py .\compare_profiles.py --take-over-test-server
```

Le profil `vision_baseline` charge le projecteur comme lors de l'expérience précédente ; le profil `text_only` le retire. Le nombre de couches GPU, le contexte, les batches, les paramètres de génération et les six questions restent identiques. Douze réponses sont produites au total. Le test mesure la mémoire globale après le chargement de chaque profil, pendant les questions ; le temps de chargement est enregistré séparément. Les extraits viennent de la référence avec routage et sont partagés à l'identique : ce test isole le profil de génération, sans refaire la comparaison des moteurs de recherche.

Le résultat est dans `results/profiles-<date-identifiant>/report.zip`. L'archive contient les deux séries de réponses et leurs contextes, les minimums de RAM disponible, la médiane de RAM disponible, les pics GPU et les temps de réponse, ainsi que les journaux bruts. La conformité des citations et les statuts attendus sont contrôlés ; la qualité factuelle doit encore être comparée en lisant les réponses. Les mesures sont globales, à une seconde et sur un seul passage vision puis texte ; les autres applications et l'ordre de passage peuvent influer. Aucun gain mémoire n'est présumé.

Les règles de remplacement du serveur sont celles décrites ci-dessous. À la fin, le serveur **texte** reste prêt sur 8080. Le lanceur `photo_local.py` gère le rechargement successif des profils pour les photos ; ce comparatif seul ne reçoit pas d'images. Si une étape échoue, le rapport et les résultats déjà obtenus sont conservés, et les processus lancés sont arrêtés.

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

Quarante-six tests couvrent les annotations réelles, le format de l'API d'embeddings simulée, les vecteurs, l'isolation du cache, la fusion des rangs, le rejet d'un reranking invalide, la sélection des processus, les changements de modèle simulés et le flux complet avec modèles factices. Les tests des profils vérifient aussi les commandes sans projecteur, les contextes identiques, la collecte des ressources et la conservation du résultat initial si le démarrage suivant échoue. Les dix tests du RAG d'origine ont aussi été exécutés. La recherche lexicale sur le vrai corpus a été exécutée dans l'environnement de développement.

L'expérience hybride initiale a depuis été exécutée sur le PC Windows de l'utilisateur : 872 encodages, six rerankings et 18 réponses ont abouti. Le rapport a confirmé une RAM libre descendant à environ 278 Mio pendant la génération. Il reste privé et n'est pas intégré à ce paquet. Le comparatif **avec/sans projecteur** a depuis été exécuté : pic GPU global de 2 873 Mio avec vision et 2 008 Mio en texte, soit 865 Mio d'écart observé, avec les six réponses identiques. Aucun gain significatif de RAM ni de vitesse n'a été établi sur cet essai unique. Ces résultats ne valident pas la fiabilité de nouvelles photos.

Sources techniques : [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [llama.cpp au commit d854d98ba](https://github.com/ggml-org/llama.cpp/tree/d854d98ba), et corpus Paris v1 dans le dépôt. Aucun fine-tuning n'est effectué. Cette expérience évalue la recherche documentaire ; l'intégration MCP et le planificateur de voyage restent des étapes distinctes du projet.
