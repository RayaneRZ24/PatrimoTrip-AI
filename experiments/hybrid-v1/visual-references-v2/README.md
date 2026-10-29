# Banque visuelle à trois vues par monument

Cette banque ajoute **vingt photographies**, soit deux nouvelles vues pour chacun des dix sites. Les dix références originales restent inchangées dans `visual-references-v1` : trente images sont utilisées au total. Les fichiers de test `evaluation-v1` sont conservés et ne deviennent pas des références.

```powershell
py .\evaluate_visual_refs.py --expanded-references
```

À lancer depuis `hybrid-v1`. Le même modèle DINOv2-small ONNX reste sur CPU. Les dépendances, poids et vecteurs déjà vérifiés sont réutilisés ; seules les images sans vecteur sont encodées. Aucun serveur Qwen ni téléchargement de poids supplémentaire n'est nécessaire si le premier essai visuel a abouti.

## Sélection et provenance

Les vingt ajouts proviennent de Wikimedia Commons. Les licences, auteurs, liens sources, URLs originales, empreintes des miniatures, dimensions et vues décrites sont consignés dans `references.json` et `ATTRIBUTIONS.md`. Les miniatures de Commons sont conservées sans retouche locale. Les sources CC BY et CC BY-SA concernent différentes versions : conserver la licence propre à chaque fichier et son attribution lors de la redistribution.

La sélection a été faite manuellement pour varier les angles et contextes, avant toute exécution de DINOv2 sur cette nouvelle banque. Elle est informée par les erreurs des essais précédents : c'est du développement, pas une sélection aveugle. Les ajouts comprennent notamment une façade nord et le dôme sud des Invalides, un portique oblique et le dôme du Panthéon, le côté Seine de Notre-Dame, les côtés ouest/nord de Garnier, l'entrée et les verrières extérieures de la Sainte-Chapelle, et la Cour Carrée du Louvre. Les vues générales, contre-plongées et détails ne sont pas interchangeables. Quelques images conservent des éléments temporaires, tels que l'exposition devant le dôme des Invalides ; ces limites sont indiquées dans les descriptions.

La revue visuelle a écarté un dessin du Panthéon, une ancienne photographie partielle de l'Arc, une image trop centrée sur une gargouille et une photo du Louvre avec des personnes au premier plan. Les tests automatiques vérifient les empreintes, les trente fichiers distincts, deux ajouts par site et l'absence de sources Commons ou fichiers de requête dans les références. Un contrôle visuel et un contrôle perceptuel simple ont aussi servi à repérer d'éventuelles copies évidentes. Cela ne prouve ni l'absence de toutes les images dérivées ni l'absence de recouvrement avec le préentraînement du modèle.

`gallery.html` permet de consulter les trois vues de chaque site dans le navigateur. La première vue vient de la banque initiale ; les suivantes sont les ajouts. Aucun script ni service distant n'est nécessaire pour afficher cette galerie locale.

## Comparaison sans confirmation automatique

Le classement utilise le maximum des similarités avec les références d'un site. Les trois candidats sont trois **sites distincts**, jamais trois photos du même site. Le nombre de vues est identique entre sites pour éviter qu'un monument dispose de beaucoup plus d'occasions de produire un score élevé.

Le rapport compare les dix mêmes photos avec une puis trois références par site. `baseline-cases.json` conserve les résultats de base, `cases.json` ceux de la banque enrichie, et `bank-comparison.json` les aligne image par image. Les groupes standard et détail restent séparés. Les vecteurs sont partagés et les appels peuvent lire le cache : ce comparatif n'est pas une mesure de vitesse d'inférence.

Ajouter des références peut aider les vues difficiles, mais peut aussi augmenter les faux rapprochements, notamment pour des monuments ressemblants hors catalogue. Le mur, le Colisée et le Panthéon de Rome reçoivent toujours des candidats : aucun seuil de rejet n'est calibré ici et aucune identité n'est confirmée. L'écart entre scores n'est pas une probabilité. Le parcours photo/histoire n'est pas modifié.

Les résultats restent à mesurer sur le PC utilisateur. Ensuite seulement, un jeu de calibration distinct et un jeu de test réservé permettront d'étudier une règle d'abstention. Les photos déjà utilisées ici ne doivent pas être présentées comme ce futur test indépendant.
