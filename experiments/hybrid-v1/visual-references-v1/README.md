# Comparer les images à des références avec un modèle dédié

```powershell
py .\evaluate_visual_refs.py
```

L'expérience s'exécute séparément du parcours photo et ne le modifie pas. Le lanceur crée `.visual-venv`, installe des roues CPU versionnées de NumPy 2.2.6, Pillow 11.3.0 et ONNX Runtime 1.23.2, puis lance le worker dans cet environnement. Ces trois versions disposent de roues Windows x64 pour Python 3.13. Aucun PyTorch, CUDA, abonnement ou appel d'inférence distant n'est nécessaire. Le premier lancement requiert Internet pour PyPI et Hugging Face. Les lancements suivants réutilisent les dépendances et les fichiers vérifiés locaux.

## Modèle et traitement

Le candidat est DINOv2-small (21 millions de paramètres), conversion ONNX publique `Xenova/dinov2-small` du modèle amont `facebook/dinov2-small`. Le champ API de licence de la conversion peut être absent : il ne doit pas être présenté comme une déclaration Apache-2.0 de son auteur. C'est un modèle de caractéristiques visuelles, distinct du générateur Qwen. Il est choisi comme essai compatible avec la machine, pas comme modèle le plus récent ni comme solution déjà validée sur les monuments. Les descripteurs CLS comportent 384 dimensions. ONNX Runtime n'utilise que `CPUExecutionProvider`, avec deux threads internes et un thread inter-op ; aucun modèle génératif n'est lancé. Fermer un serveur manuel encore ouvert permet de libérer sa RAM, mais ce lanceur ne l'arrête pas.

La métadonnée du dépôt est résolue une fois, puis sa révision est figée dans `models/dinov2-small-onnx/model-lock.json`. Le lanceur vérifie l'architecture attendue, les tailles, les SHA-256 LFS des poids et les empreintes Git des configurations. Si la conversion déclare Apache-2.0, cette déclaration est conservée. Si son champ de licence est vide, le lanceur vérifie l'empreinte Git de sa fiche README à la révision figée, exige un lien explicite vers `facebook/dinov2-small`, puis contrôle la déclaration Apache-2.0 du dépôt amont et conserve sa révision. Une licence différente explicitement déclarée n'est jamais remplacée par celle du modèle amont. Le verrou distingue `conversion_declared_license`, `license_basis` et `license_evidence` ; une licence amont n'est pas présentée comme une déclaration de licence de la conversion. Les métadonnées publiques et la fiche vérifiée sont ajoutées au rapport, y compris si la validation échoue. Aucun code distant n'est exécuté. Le fichier float32 `onnx/model.onnx` est sélectionné explicitement, sans choix automatique d'une autre quantification. Le réseau n'est utilisé que pour télécharger les fichiers ; les images restent sur le PC.

Le prétraitement vérifie et applique la configuration du modèle : RGB, orientation EXIF, petit côté 256, interpolation déclarée, recadrage central 224, facteur 1/255 et moyenne/écart-type déclarés. Une configuration ou sortie ONNX non reconnue provoque un arrêt explicite. Le recadrage central peut éliminer des parties utiles d'une photo ; cet essai ne résout pas tous les angles et détails. Les vecteurs sont mis en cache par empreinte d'image et espace de noms incluant modèle, configuration, sortie et version. Un redémarrage réutilise les calculs achevés.

## Références et évaluation

Les dix photos sont les références originales du projet, une par site, avec leurs empreintes et attributions. Elles sont distinctes des dix images `evaluation-v1`. Les photos hors catalogue et le mur ne sont jamais ajoutés à l'index. Les labels des requêtes ne sont consultés qu'après le classement pour les scores ; le calcul utilise uniquement les pixels. Le script vérifie les empreintes, l'absence de doublons exacts entre index et requêtes et la correspondance avec le catalogue.

Le classement utilise la similarité cosinus de vecteurs normalisés. Il retourne les trois meilleurs candidats et l'écart entre les deux premiers. **Une similarité n'est pas une probabilité ni une identité confirmée.** Aucun seuil n'est calibré sur ces données. Même Rome ou un mur reçoit un voisin du catalogue ; cela ne constitue donc pas un détecteur hors catalogue. Il faudra des données de calibration distinctes et davantage d'angles de référence avant de proposer un éventuel filtre automatique.

Les dix requêtes ont déjà servi au développement : il s'agit d'un essai de régression avec un autre modèle, pas d'une mesure indépendante de précision. Les résultats standards (5) et détails (2) sont séparés. Les deux photos hors catalogue et le mur restent à examiner, sans score de rejet inventé. Un rapport photo complet antérieur est comparé si son empreinte de jeu correspond ; aucun nouveau lancement de Qwen n'est nécessaire. Leur accord n'est pas une preuve d'identité, et leur désaccord ne dit pas automatiquement quel modèle a raison.

Le test doit notamment établir si les Invalides se rapprochent de leur propre référence plutôt que du Panthéon, et si le Panthéon de Rome reste un faux candidat parisien. Une seule référence par monument peut être insuffisante. L'intégration au parcours historique dépendra des résultats : ce script ne remplace aucun identifiant et ne génère aucune explication historique.

## Rapport et limites de validation

Un seul chemin `results/visual-refs-.../report.zip` est affiché. Le rapport inclut les candidats et scores, les empreintes, versions et paramètres du modèle, les métadonnées d'attribution, et les erreurs éventuelles, y compris d'installation. Il exclut les images et les poids. Les temps par image peuvent être des lectures du cache ; ils ne mesurent pas une vitesse d'inférence générale. Aucun gain de RAM ou de précision n'est présumé.

Les tests automatisés vérifient les contrats de téléchargement, empreintes, classement, cache, absence de confirmation abusive et séparation des images. Les tests de flux utilisent des vecteurs simulés. L'environnement cloud n'a pas pu accéder à Hugging Face ; le chargement des vrais poids et le résultat sur les monuments restent à vérifier sur le PC utilisateur. Le lanceur conserve un rapport exploitable si l'accès réseau, les fichiers distants ou le contrat de prétraitement diffèrent de ce qui est attendu.
