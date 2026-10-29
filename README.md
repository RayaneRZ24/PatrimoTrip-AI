# PatrimoTrip AI

Un démonstrateur local pour découvrir dix monuments parisiens à partir d'une photo, consulter des extraits historiques et composer une promenade culturelle. Les inférences tournent sur le PC ; aucune clé d'API payante n'est nécessaire.

## Lancer l'application

L'environnement déjà préparé utilise Windows, Python 3.13, llama.cpp Vulkan, Qwen3.5-4B Q4_K_M, son projecteur multimodal, Qwen3-Embedding-0.6B et DINOv2-small sur CPU. Les scripts conservent les modèles, caches et résultats hors Git.

Depuis le dépôt :

```powershell
Set-Location .\experiments\hybrid-v1
py .\verify_release.py --models --open
```

Depuis l'archive déjà extraite dans le dossier utilisateur :

```powershell
Set-Location (Join-Path $HOME "PatrimoTripAI\hybrid-v1")
py .\verify_release.py --models --open
```

Ce lancement vérifie le logiciel et les outils, exécute un contrôle réel sur la photo connue de la tour Eiffel, produit un `report.zip`, puis ouvre **http://127.0.0.1:8765**. Garder PowerShell ouvert ; Ctrl+C arrête l'application et ses processus d'inférence gérés. Un contrôle sur une image connue ne constitue pas une nouvelle mesure de précision.

Pour les lancements suivants :

```powershell
py .\app_local.py --open
```

Si le port est occupé, choisir par exemple `--port 8766`. Aucun serveur llama.cpp à lancer dans une deuxième fenêtre. Le catalogue et les parcours fonctionnent sans charger les modèles ; l'écran « État local » indique les fichiers et programmes repérés. Les modèles et leur chargement sont contrôlés au moment de l'analyse.

## Fonctions disponibles

- **Photo** : choix d'une image, comparaison aux références avec abstention, analyse indépendante de Qwen et réponse documentaire seulement si les deux proposent le même site.
- **Question culturelle** : Qwen choisit une action bornée — sources historiques, parcours, liens pratiques ou demande de précision — puis le programme exécute les outils via MCP.
- **Réponse sourcée** : le modèle sélectionne des phrases ; le programme reproduit exactement les extraits et lie chaque citation à ses positions dans le passage. Aucune paraphrase factuelle libre n'est acceptée dans le nouveau parcours assisté.
- **Promenade** : choix du point de départ, des monuments, du temps disponible et du temps sur place ; optimisation du nombre d'étapes puis de la distance estimée.
- **Enregistrement** : prévisualisation et confirmation explicite, liste locale des parcours, export JSON et impression depuis le navigateur.
- **Traçabilité** : rapports par analyse, références consultables, états d'abstention et journaux d'outils.

L'interface, les vignettes et les schémas de parcours sont locaux et n'utilisent pas de tuiles cartographiques distantes. Les pages sources et liens pratiques s'ouvrent uniquement à la demande dans le navigateur.

## Ce que cette version ne garantit pas

L'identification reste une **proposition**, même si les deux analyses sont d'accord. Une photo connue peut être refusée. La règle visuelle 0,70/0,15 et les références sont figées ; les petits essais pilotes ne prouvent pas un taux d'erreur général.

Les citations exactes vérifient la fidélité de la reproduction. Elles ne garantissent pas la pertinence de la sélection ni l'exactitude de la source. Les introductions sont prioritaires pour présenter un monument, afin de ne pas lui attribuer une mesure concernant un projet concurrent. Les sources restent affichées pour relecture.

Les promenades sont **extérieures et indicatives**. La marche est calculée sur des distances géographiques majorées, pas sur un réseau de rues. Les temps sur place sont des hypothèses choisies par l'utilisateur. Horaires du jour, tarifs, accès intérieurs, travaux, disponibilité et réservations ne sont pas vérifiés. Le logiciel n'effectue aucun achat ni réservation.

Cette livraison réunit le périmètre connu du projet en une première application. L'énoncé original du cours n'était plus accessible pendant cette phase : la conformité à toutes ses exigences doit être confrontée au document, sans la présumer.

## Documentation et tests

- [Guide et scénario de démonstration](experiments/hybrid-v1/docs/USER_GUIDE.md)
- [Architecture et choix techniques](experiments/hybrid-v1/docs/ARCHITECTURE.md)
- [Dossier de projet et limites](experiments/hybrid-v1/docs/PROJECT_REPORT.md)
- [Installation des modèles et chemins attendus](experiments/hybrid-v1/docs/SETUP.md)
- [Références du corpus](experiments/hybrid-v1/corpus/ATTRIBUTIONS.md)
- [Calibration et test de l'abstention](experiments/hybrid-v1/abstention-v1/README.md)

```powershell
py -m unittest discover -s experiments/hybrid-v1 -v
```

Les tests logiciels vérifient les contrats, pas une précision générale des modèles. Les vérifications de navigateur portent sur les interactions de l'interface ; les réponses d'inférence y sont simulées. Les parcours, sauvegardes et exports y utilisent les vrais outils locaux. Le contrôle des modèles réels doit être effectué sur la machine cible.

## MCP pour les outils locaux

Le serveur stdio s'exécute avec Python standard :

```text
command: py
args: ["C:\\chemin\\PatrimoTrip-AI\\experiments\\hybrid-v1\\mcp_server.py"]
```

Outils : `catalog_search`, `heritage_evidence`, `plan_walk`, `tourism_reference_links`, `list_trips`, `save_trip`. Le client de l'application effectue une initialisation MCP puis des appels JSON-RPC. L'agent n'a pas d'action d'enregistrement : celle-ci passe par le bouton de confirmation utilisateur. Ce serveur implémente le transport stdio, pas un service MCP HTTP public.

## Organisation

`experiments/hybrid-v1` contient l'application et ses moteurs ; `benchmarks` et les anciens lanceurs conservent les expériences reproductibles. Les photographies et corpus ont leurs propres licences et attributions. `.app-state/`, `models/`, `results/` et les caches ne sont pas versionnés. Les rapports utilisateur ne sont pas publiés dans le dépôt.
