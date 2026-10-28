# Évaluation supplémentaire du parcours photo — v1

Jeu fixé le 28 octobre 2026 avant tout lancement de ces images sur le modèle utilisateur. Dix images et douze scénarios, exécutés par `../evaluate_photo.py`. Les images ont été examinées pour vérifier leur sujet et leur lisibilité ; ce choix n'a pas été fait à partir des résultats du modèle évalué.

## Composition

- Cinq vues générales : tour Eiffel partiellement masquée par des arbres, Notre-Dame, palais Garnier, Panthéon avec échafaudage, Invalides au crépuscule.
- Deux vues de détail : dessous de l'Arc de Triomphe et façade partielle du Panthéon. Elles ont des indices naturels, notamment des inscriptions ; ce ne sont pas des images déclarées impossibles à reconnaître. Une bonne identification ou une abstention prudente sont distinguées dans le rapport.
- Deux sites hors corpus : Colisée et Panthéon de Rome. Le second est un exemple de façade classique pouvant être confondue avec le Panthéon parisien.
- Une image sans contexte suffisant : mur de briques en gros plan, qui ne permet pas d'établir l'identité d'un monument.
- Cinq questions historiques documentées, deux questions sur un tarif ou un horaire actuel absent du corpus, cinq demandes associées aux détails ou aux rejets.

Les deux questions d'information actuelle réutilisent les images Eiffel et Panthéon. Cela fait douze exécutions complètes pour dix fichiers différents. La première tentative par image compte dans les scores visuels ; les répétitions et leurs éventuels désaccords sont indiqués séparément.

## Provenance et limites

`assets.json` conserve source, auteur, licence, URL de téléchargement, dimensions, empreinte SHA-256 et note de lecture de chaque image. Les fichiers sources sont différents de ceux du premier benchmark visuel ; les empreintes sont également différentes. `ATTRIBUTIONS.md` accompagne les images redistribuées sous leur licence. Ce sont des miniatures fournies par Wikimedia, sans retouche locale.

Il s'agit de nouvelles images pour nos essais, **pas d'une garantie d'absence dans les données d'entraînement du modèle**. Le corpus, les faits historiques et certains types de questions ont déjà été utilisés pendant le développement. Les images viennent de dix fichiers sélectionnés, pas d'un échantillon aléatoire représentant toutes les conditions touristiques. Les textes présents naturellement dans les images sont conservés. Le jeu n'est pas une preuve d'exactitude générale.

Les identifiants attendus, les statuts attendus et les citations annotées ne sont pas transmis au modèle. Le parcours reçoit seulement la photo, la question et la liste des dix sites autorisés, comme dans l'application. Aucune identité n'est forcée par `--confirm-site` pendant ce test.

## Mesures et relecture

Les métriques automatiques distinguent :

- Identification exacte sur les cinq vues générales.
- Identification exacte et comportement prudent sur les deux détails.
- Rejet sans génération historique parisienne sur les deux lieux hors corpus ; le classement explicite `out_of_scope` est compté séparément de `uncertain`.
- Demande de précision pour le mur sans contexte.
- Présence des citations annotées dans le contexte pour les cinq questions documentées.
- Conformité des références et statut documentaire attendu **après identification correcte**.
- Abstention sur le tarif/horaire actuel, uniquement si le bon site a d'abord été identifié.
- Durée par scénario incluant changements de modèles et calculs nécessaires ; RAM disponible et mémoire GPU globales échantillonnées à une seconde.

Les citations annotées sont des preuves connues partielles : d'autres passages peuvent aussi répondre. Leur absence dans le contexte ne suffit pas à déclarer la réponse fausse. Une génération au format valide n'est jamais comptée comme une preuve de véracité.

`review.json` rassemble chaque affirmation, les extraits qu'elle cite et tous les extraits fournis. La notation reste `not_reviewed` jusqu'à lecture : renseigner `support` avec `supported`, `partial`, `unsupported` ou `contradicted`, et `answers_question` avec `yes`, `partial` ou `no`. Examiner notamment les dates, les superlatifs non justifiés, les approximations, la pertinence de chaque citation et les abstentions injustifiées. Les observations visuelles doivent être comparées aux photos séparément.

`complete` signifie que les cas ont été exécutés sans exception ; cela ne veut pas dire qu'ils sont tous réussis, ni que les réponses ont été validées factuellement. Les erreurs ne sont pas retirées des dénominateurs. Si une image est répétée, un échec de sa première tentative ne disparaît pas grâce à une tentative suivante réussie.

## Exécution Windows

Depuis le dossier parent `hybrid-v1` :

```powershell
py .\evaluate_photo.py --take-over-test-server
```

Les données sont déjà incluses dans le paquet. Les modèles et vecteurs existants sont réutilisés ; de nouvelles questions peuvent nécessiter des encodages. Un seul modèle lourd est chargé à la fois, avec les mêmes réglages que le parcours photo actuel. Ce test utilise l'état courant du cache et ne mesure pas un démarrage sur un PC neuf.

L'exécution peut prendre plusieurs minutes. Les résultats sont enregistrés après chaque scénario. Un échec isolé est conservé et les cas suivants continuent. À la fin, le lanceur arrête les processus qu'il a démarrés et affiche le chemin d'un unique `results/evaluation-<date-identifiant>/report.zip`. Aucune image ni donnée personnelle de l'utilisateur n'est nécessaire ; aucun résultat n'est envoyé automatiquement. L'archive contient les journaux et les sorties, pas les poids ni le cache vectoriel.

Les tests de ce lanceur couvrent les empreintes et annotations, le refus d'une image altérée, la séparation des labels et de l'inférence, les rejets, les dénominateurs et la poursuite après erreur avec un modèle simulé. Les performances réelles doivent être mesurées sur le PC Windows.
