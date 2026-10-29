# Parcours photo assisté avec abstention et accord visuel

```powershell
py .\photo_assisted.py --pick --take-over-test-server
```

Pour un premier contrôle du parcours sur une image déjà utilisée en développement :

```powershell
py .\photo_assisted.py .\evaluation-v1\images\a01.jpg --take-over-test-server
```

Cette commande prend une photo, propose éventuellement un monument, puis cherche des sources et rédige une explication. Elle produit un seul `report.zip`. Ce contrôle de fonctionnement sur une image connue ne mesure pas une nouvelle précision indépendante.

## Ordre des étapes

1. DINOv2-small compare les pixels aux trente références. La règle figée exige une similarité d'au moins **0,70** et un écart d'au moins **0,15** entre les deux premiers sites. Si elle s'abstient, aucun modèle Qwen ni recherche documentaire n'est lancé : le programme demande le nom/la ville ou une autre photo.
2. Le processus CPU de DINOv2 se termine avant le chargement de Qwen. Qwen analyse la photo avec la liste du catalogue, sans recevoir le candidat ni les scores de DINOv2. Son nom canonique, sa ville et son pays sont contrôlés comme auparavant.
3. Qwen doit proposer le même site que la comparaison aux références. Un désaccord, une abstention ou une contradiction géographique arrête le parcours avant la recherche et la génération historique.
4. En cas d'accord, la vision Qwen s'arrête avant les embeddings et le profil de génération textuelle. La recherche reste filtrée sur le site proposé, avec réponses par volet pour les questions composées. Les citations sont vérifiées formellement ; elles ne sont pas une preuve automatique du soutien de chaque affirmation.

L'accord de deux analyses n'est pas une preuve d'identité. Le résultat reste un **monument proposé**, avec `identity_verified: false`. Les observations visuelles ne deviennent pas des sources historiques. Une information actuelle absente du corpus doit rester signalée comme manquante.

## Règle figée et provenance

`policy.json` contient la configuration nécessaire au parcours, pas des journaux utilisateur. Les seuils sont issus du protocole pilote de calibration déjà défini, sans nouvel ajustement sur le test. La configuration lie la règle aux empreintes des deux manifestes de références, du catalogue, du protocole et à l'espace de noms exact du modèle/prétraitement. Son empreinte approuvée est contrôlée par `visual_guard.py`. Une modification de la règle, des références ou du descripteur provoque un arrêt, sans solution de repli vers Qwen seul.

La règle peut refuser une bonne photo d'un monument connu. Un refus ne signifie donc pas « hors catalogue ». Sa couverture reste limitée, et le petit test pilote ne garantit pas un taux d'erreur général. L'ajout de l'accord Qwen constitue une nouvelle composition du parcours : les performances de l'ensemble restent à mesurer, sans réutiliser les résultats déjà examinés comme nouvelle estimation indépendante.

## Confirmation explicite et relance

Si l'utilisateur connaît l'identité, il peut la fournir explicitement, par exemple pour la tour Eiffel à Paris :

```powershell
py .\photo_assisted.py --pick --confirm-site Q243 --take-over-test-server
```

Ce chemin enregistre `identity_origin: user_provided` et `visual_guard.status: bypassed_user_identity`. Il ne revendique aucune validation visuelle et ne charge pas les modèles de vision. Il reste limité aux dix sites du catalogue, dont les villes/pays sont affichés avec les identifiants.

`--replay-latest` relance seulement la partie documentaire avec la question et les extraits du rapport. Pour une origine visuelle, le mode assisté exige un ancien résultat lié à la même photo, à la règle actuelle et à un accord cohérent ; un ancien rapport sans ce contrôle demande une nouvelle analyse photo. La relance ne refait pas d'analyse visuelle et ne constitue pas une nouvelle confirmation. Une identité explicitement fournie par l'utilisateur reste enregistrée comme telle.

## Ressources et rapport

Le parent reste en Python standard. Il appelle `.visual-venv` pour le modèle CPU, puis attend sa fin avant les modèles llama.cpp. Les poids, dépendances et vecteurs existants sont réutilisés. Si l'environnement de vision est absent, lancer d'abord `py .\evaluate_visual_refs.py` pour le préparer. Les modèles restent séquentiels ; leur coût réel sur le PC doit être vérifié avec les rapports.

Le mode assisté arrête ses propres serveurs en fin de traitement. Aucun serveur à démarrer manuellement. L'option de reprise du serveur conserve les contrôles stricts existants ; aucun processus non reconnu n'est arrêté.

Le rapport contient `visual-guard.json`, les éventuels résultats de vision Qwen, la décision d'accord, les extraits et la réponse s'il y a lieu. La photo originale passe par un répertoire temporaire hors du rapport, supprimé après le processus CPU. Les requêtes contenant la photo en base64 ne sont pas enregistrées. Aucune image originale n'est ajoutée au ZIP.

Les tests logiciels couvrent les liaisons photo/modèle/règle, les refus, désaccords, erreurs du processus CPU, relances, confirmations humaines, séparation des processus et exclusion de l'image temporaire. Ils utilisent des modèles simulés : le parcours réel reste à vérifier sur le PC utilisateur.

`photo_local.py` et les anciens benchmarks conservent leur chemin historique pour la reproductibilité. Utiliser **`photo_assisted.py`** pour ce nouveau parcours ; les anciens résultats de reconnaissance ne doivent pas être attribués à ce mode.
