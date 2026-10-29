# Abstention visuelle : calibration et test séparés

```powershell
py .\evaluate_visual_refs.py --calibrate-abstention
```

Le lanceur réutilise `.visual-venv`, DINOv2-small sur CPU, les trente références et les vecteurs déjà calculés. Les quarante nouvelles images sont incluses dans l'archive ; elles ne sont pas ajoutées aux références. Le premier essai visuel réussi doit avoir préparé les dépendances et les poids, sinon le mécanisme de téléchargement vérifié reste disponible. Aucun serveur Qwen n'est lancé.

## Données figées avant les résultats du modèle

Quarante photographies Wikimedia Commons sont réparties manuellement en deux ensembles de vingt :

| Ensemble | Monuments du catalogue | Hors catalogue | Images ambiguës |
|---|---:|---:|---:|
| Calibration | 10, une photo par site | 8 | 2 |
| Test réservé (`heldout`) | 10, une autre photo par site | 8 | 2 |

Les images ont été sélectionnées et revues avant toute exécution des descripteurs sur ce jeu. Les manifestes conservent les auteurs, licences, sources, dates disponibles, groupes de photographes, dimensions et empreintes. Aucun fichier ni page Commons des références ou anciens essais n'est repris. Aucun groupe de photographe déclaré ne traverse calibration et test ; les alias NikonZ7II/Paris Orlando sont regroupés. Les sites connus apparaissent volontairement dans les deux ensembles, avec des photos différentes. Les monuments hors catalogue sont différents entre les ensembles.

La calibration comprend un nouveau Panthéon de Rome, la cathédrale Saint-Paul de Londres, des églises, un arc et des places. Le test comporte notamment la basilique Saint-Paul-hors-les-Murs, le British Museum, le palais de Westminster et d'autres bâtiments hors catalogue. Les nuages et murs sont des cas sans contexte suffisant. Des vues de détail, reflets, photos de nuit et une photo monochrome rendent aussi les exemples connus difficiles : leur refus peut être raisonnable et doit être compté comme une baisse de couverture, pas caché derrière la précision des seuls cas acceptés.

Les labels connus reposent sur la provenance Commons et la revue visuelle, pas sur une identification garantie à partir des seuls pixels. Une vue de rue dont la Sainte-Chapelle était masquée par le Palais de justice a été écartée, avant calcul des descripteurs, au profit d'une photo montrant clairement son intérieur. Un contrôle perceptuel simple a complété les vérifications de fichiers et sources. Cela ne prouve ni l'absence de toutes les images dérivées ni l'absence de ces images dans le préentraînement.

Les échecs précédents ont motivé le choix de ressemblances difficiles. Cet ensemble est un **petit protocole pilote**, non un échantillon représentatif des photos touristiques ni un benchmark externe. `manifest.json` fige les empreintes des deux listes, des exclusions et de la banque de références. Voir [ATTRIBUTIONS.md](ATTRIBUTIONS.md) pour les crédits.

## Règle choisie seulement sur la calibration

Le modèle reste inchangé. Pour chaque image, le score d'un site est le maximum de ses trois similarités de référence, puis les sites sont classés. La règle candidate propose le premier site si les deux conditions sont satisfaites : similarité au-dessus d'un seuil et écart avec le deuxième site au-dessus d'un autre seuil. Sinon elle s'abstient. Les scores ne sont pas des probabilités.

La grille est fixée dans le code : similarité de 0 à 1 par pas de 0,05 ; écart de 0 à 1 par pas de 0,025 ; une règle qui refuse tout est également disponible. Une mauvaise proposition est soit un mauvais site connu, soit une proposition de site du catalogue pour une image hors catalogue/ambiguë. Parmi les règles ne faisant aucune erreur **observée sur la calibration**, le programme maximise le nombre de bonnes propositions. En cas d'égalité il préfère le seuil de similarité le plus élevé, puis le seuil d'écart le plus élevé. Si aucune règle non triviale ne conserve une bonne proposition sans erreur observée, le refus de tout reste le résultat valide.

Cette contrainte empirique n'est pas une garantie statistique. Les seuils peuvent surajuster vingt exemples, et la règle score/écart peut être insuffisante face aux ressemblances architecturales.

## Séparation entre sélection et mesure

L'ordre du worker est imposé :

1. Vérifier les fichiers, sources et séparation des photographes.
2. Calculer uniquement les résultats de calibration.
3. Choisir la règle et écrire `policy.json`, incluant l'empreinte des résultats de calibration et l'espace de noms du descripteur.
4. Calculer les descripteurs et classements du test réservé, puis appliquer la règle inchangée.
5. Vérifier que `policy.json` n'a pas changé et écrire les métriques du test.

Les labels et noms de sujets ne sont pas fournis à l'encodeur ni à la fonction de décision. Le réglage reçoit les labels de calibration seulement. Les labels du test servent à compter les résultats après les décisions ; ils ne servent pas à modifier les seuils. Les tests logiciels vérifient aussi qu'une erreur sur le test reste rapportée au lieu de provoquer un nouveau réglage.

## Rapport et interprétation

Le rapport unique `results/visual-abstention-.../report.zip` contient `policy.json`, `calibration-results.json`, `heldout-results.json`, les métriques séparées dans `summary.json`, les métadonnées de provenance et les erreurs éventuelles. Il n'inclut pas les photos, poids ni vecteurs. Les métriques distinguent propositions correctes, mauvaises propositions, refus de sites connus et fausses propositions pour les cas hors catalogue/ambigus. Si aucune image n'est proposée, la précision parmi les propositions est `null`, pas 100 %.

Les temps peuvent inclure des lectures du cache et ne sont pas une mesure générale de vitesse. Le calcul réel des seuils et les résultats DINOv2 restent à vérifier sur le PC utilisateur ; les tests de protocole utilisent des classements simulés.

Ce script ne modifie pas le parcours photo/histoire et ne confirme aucune identité. Un refus ne signifie pas automatiquement « hors catalogue » : une mauvaise vue d'un site connu peut aussi être refusée. Une proposition acceptée reste une proposition.

**Après lecture des résultats du test, toute modification des seuils, du modèle ou des références fait de ce test un jeu de développement.** Un nouveau jeu réservé sera nécessaire pour mesurer la nouvelle version. Relancer ce même protocole ne crée pas une nouvelle évaluation indépendante.
