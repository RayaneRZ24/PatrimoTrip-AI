# PatrimoTrip AI — premier test visuel local

Ce test compare des configurations sur **les mêmes 12 images**, sans API distante. Il ne mesure pas encore la précision générale du système : les dix images parisiennes proviennent du corpus de référence, et Notre-Dame a déjà été essayée manuellement. Aucune image nouvelle indépendante n'est revendiquée.

## Contenu et périmètre

- 8 vues standards : tour Eiffel, palais du Louvre (vue partielle), Notre-Dame, Arc de Triomphe, Panthéon, Sacré-Cœur, palais Garnier et Sainte-Chapelle.
- 2 vues éloignées : Invalides et place de la Concorde.
- 2 sites hors des dix identités autorisées : tour de Londres et Tower Bridge.

Les vues éloignées ne sont pas nécessairement impossibles à identifier. Une abstention peut y être justifiée : elle doit être revue manuellement et n'est pas amalgamée au score des huit vues standards. Reconnaître un monument londonien est permis, mais le classer comme un site parisien est une erreur.

Le corpus ne contient pas encore de photos floues, occultées ou indépendantes. Il faudra les ajouter pour une véritable évaluation. Les descriptions attendues du manifeste sont des repères de relecture, pas une annotation exhaustive indépendante.

## Exécution sur Windows

Garder le serveur llama.cpp ouvert sur `127.0.0.1:8080`, avec la configuration précédente. Fermer les autres applications lourdes pour rendre les essais plus comparables. Le script réalise douze requêtes successives, sans télécharger de modèle ni installer de dépendance.

Depuis le dossier de ce test :

```powershell
powershell.exe -NoProfile -File .\Run-VisionTest.ps1
```

Si une politique d'exécution empêche de lancer ce fichier, examiner le script, débloquer uniquement cette copie téléchargée et appliquer RemoteSigned uniquement au nouveau processus :

```powershell
Unblock-File -LiteralPath .\Run-VisionTest.ps1
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File .\Run-VisionTest.ps1
```

Ces commandes ne modifient pas la politique permanente de Windows. Aucun lancement administrateur n'est nécessaire.

Si le serveur utilise un autre port local :

```powershell
powershell.exe -NoProfile -File .\Run-VisionTest.ps1 -BaseUrl http://127.0.0.1:8081
```

Le script n'accepte qu'une adresse HTTP de boucle locale. Les photos sont envoyées en base64, sans leur nom de fichier et sans la réponse attendue. Le prompt fournit les dix identités possibles : il s'agit d'une reconnaissance dans un ensemble connu, avec abstention et rejet hors périmètre, et non d'une reconnaissance libre mondiale.

## Sorties

Un sous-dossier unique `results/<date-identifiant>/` contient :

- `results.csv` : identités, statuts, réponses structurées, temps et erreurs ;
- `responses.jsonl` : textes bruts, réponses analysées, temps fournis par le serveur et erreurs ;
- `summary.json` : compteurs par catégorie ;
- `configuration.json` : paramètres du client et empreinte du manifeste ;
- `gpu.csv` et `gpu-errors.txt` si nvidia-smi a pu être lancé sous Windows.

Les résultats restent locaux et sont exclus de Git. Aucun token ni compte n'est nécessaire. Ne pas publier automatiquement les résultats personnels.

Le relevé GPU couvre l'utilisation globale du GPU, y compris le bureau et d'autres logiciels, pas uniquement le modèle. Il est échantillonné une fois par seconde et peut manquer un pic bref. Le processus de relevé démarré par le test est arrêté à sa fin. La RAM du modèle n'est pas mesurée automatiquement.

## Lecture des résultats

- Un appel en erreur n'est pas une identification réussie.
- Une réponse non JSON ou contraire au schéma n'est pas corrigée silencieusement et n'est pas comptée comme correcte.
- `identity_correct` compare l'identité au corrigé ; `status_correct` vérifie l'identification ou le rejet attendu.
- Les descriptions restent `pending` jusqu'à relecture humaine, même si l'identité est correcte.
- Pour Notre-Dame, vérifier notamment trois portails, deux tours, la rosace, la galerie des statues au-dessus des portails et l'absence de dates ajoutées.
- Les temps englobent l'appel HTTP local, le traitement visuel et la génération. Ils ne sont pas équivalents aux seuls tokens/s de génération.
- Les requêtes sont séquentielles, température 0, graine 42, plafond de sortie 256 tokens et cache de prompt désactivé. Le cache système et l'état initial du serveur ne sont pas contrôlés. Une graine ne garantit pas des sorties identiques sur tous les backends.
- Le premier cas peut inclure des coûts de première utilisation. Douze images ne permettent pas d'établir une latence p95 représentative ni une précision générale robuste.

Pour comparer deux modèles, conserver les images et le prompt, redémarrer le serveur avec le modèle et ses réglages, puis renseigner `-RunLabel`. Cette étiquette n'est pas une détection automatique des réglages : noter séparément le contexte, les couches GPU et l'offload visuel. Ne pas charger le 4B avant d'avoir analysé le premier essai du 2B.

## Vérifications effectuées sur le lanceur

Le lanceur a été exécuté sous PowerShell 7.4.6 sur Linux, avec un faux serveur HTTP local, sans modèle et sans GPU. Le test a vérifié les réponses correctes et incorrectes, les JSON invalides, les erreurs HTTP, l'absence de réponse, le rejet d'une destination externe et d'une image corrompue. Le code utilise des constructions compatibles avec Windows PowerShell 5.1 ; son exécution sur le PC Windows de l'utilisateur reste à effectuer.

Ces vérifications valident la collecte et le comptage, pas la performance ou la qualité d'un modèle. Les résultats simulés ne sont pas livrés comme résultats d'évaluation.

## Provenance et licences

Chaque image possède sa page source, son auteur, sa licence et son empreinte dans `cases.json`. Voir également `ATTRIBUTIONS.md`. Les miniatures Commons sont conservées telles que téléchargées ; les conditions varient par fichier. La provenance publique ne remplace pas la vérification des droits applicables.
