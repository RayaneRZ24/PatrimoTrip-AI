# Utiliser PatrimoTrip AI

## Démarrage

Ouvrir PowerShell dans `C:\Users\<utilisateur>\PatrimoTripAI\hybrid-v1`, puis exécuter `py .\app_local.py --open`. Le navigateur ouvre l'application. Garder PowerShell ouvert ; Ctrl+C ferme le serveur local et les processus d'inférence qu'il a lancés. Si un ancien serveur de test reconnu occupe le port llama.cpp, les analyses utilisent les contrôles stricts de reprise existants. Aucun processus non reconnu n'est arrêté.

## Explorer une photo

Choisir un JPEG ou PNG de 8 Mio et 4 millions de pixels maximum. L'aperçu reste dans le navigateur ; la copie reçue est utilisée localement puis supprimée du dossier de travail. Le programme compare la photo aux trente références, applique la règle d'abstention figée et, si elle passe, demande une analyse indépendante de Qwen. L'explication n'est produite qu'en cas d'accord.

Le résultat peut être un monument proposé, une demande de précision, une information documentaire manquante ou un échec technique. Un refus ne signifie pas automatiquement que le lieu est hors catalogue. Si l'identité est réellement connue de l'utilisateur, il peut la sélectionner explicitement ; cette provenance est enregistrée et ne devient pas une validation visuelle.

Les textes historiques du nouveau parcours sont des citations exactes. Déplier chaque source pour lire le contexte, sa provenance et sa licence. La pertinence des extraits reste à apprécier. Le bouton de rapport télécharge un ZIP sans la photo originale.

## Poser une question

Le sélecteur de contexte est facultatif. Une question peut nommer le monument, par exemple « Qui a conçu le palais Garnier ? ». Choisir un contexte aide les questions comme « Pourquoi a-t-il été construit ? ». En cas de contradiction entre le lieu choisi et celui compris dans la question, l'assistant demande une précision.

L'agent choisit une consultation du corpus, un parcours, des liens pratiques ou une clarification. Les horaires et prix du jour ne sont pas inventés à partir des documents historiques. Les liens pratiques sont des candidats à consulter, pas des résultats de réservation ou de disponibilité.

## Composer une promenade

Choisir un point de départ, une durée, un nombre maximal d'étapes et un temps sur place. Les cases permettent de limiter les lieux envisagés ; le point de départ est toujours inclus. Le programme cherche le plus grand nombre d'étapes réalisable selon ses hypothèses, puis réduit la marche estimée. Il ne promet pas que toutes les cases cochées seront visitées.

Les positions sont montrées sur un schéma, sans réseau de rues. Les estimations utilisent 4,5 km/h et une majoration de 30 % de la distance à vol d'oiseau. Les visites sont extérieures ; vérifier l'accès, les horaires et les éventuels tarifs avant un déplacement réel.

Relire le parcours, lui donner un nom, puis cliquer sur « Confirmer et enregistrer ». Il apparaît dans les parcours enregistrés. Il peut être rouvert, exporté en JSON ou imprimé. Les fichiers restent sur cet ordinateur, dans `.app-state/trips/`.

## Scénario de démonstration pour la présentation

1. Montrer le catalogue et expliquer que cette version se limite à dix monuments de Paris.
2. Analyser `evaluation-v1/images/a01.jpg`, avec une question sur la présentation et la construction du monument. Montrer les extraits et leur source.
3. Montrer qu'une photo refusée mène à une demande de précision, sans histoire générée automatiquement.
4. Demander « Prépare une promenade de trois heures à partir de la tour Eiffel ». Montrer le parcours, ses estimations et les limites d'accès non vérifié.
5. Modifier les paramètres dans le formulaire, confirmer le parcours et exporter le fichier JSON.
6. Montrer le serveur MCP, un rapport de calibration et la séparation entre vérifications logicielles et évaluation des modèles.

Les images de démonstration déjà utilisées ne doivent pas être présentées comme un nouveau test indépendant.

## Dépannage

- **Application inaccessible** : garder PowerShell ouvert et utiliser l'URL affichée. Si le port est occupé, lancer `py .\app_local.py --port 8766 --open`.
- **Moteur ou modèle absent** : consulter « État local » et [SETUP.md](SETUP.md). Le catalogue et le formulaire de promenade restent disponibles.
- **Analyse déjà en cours** : attendre sa fin. Les modèles lourds sont volontairement utilisés successivement.
- **Photo trop grande** : fournir une copie JPEG/PNG de moins de 4 millions de pixels, sans modifier l'original.
- **Refus d'identification** : donner une vue plus nette ou une identité que vous connaissez. Ne choisir aucun nom au hasard pour contourner le refus.
- **Réponse arrêtée** : télécharger le rapport de diagnostic lorsqu'il est disponible. Les résultats détaillés se trouvent dans `results/` ; les journaux de l'application dans `.app-state/jobs/`.
- **Règle ou modèle incompatible** : ne pas modifier les seuils manuellement. Une modification des références, du modèle ou du prétraitement exige une nouvelle évaluation.
