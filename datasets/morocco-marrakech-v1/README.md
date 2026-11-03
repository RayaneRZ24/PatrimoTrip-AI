# Pilote documentaire Maroc — Marrakech v1

Ce dossier prépare l'extension marocaine de PatrimoTrip AI. Il n'est pas encore un corpus historique final ni une banque de reconnaissance calibrée. L'application parisienne n'est pas modifiée par ce dossier.

## Première sélection

| Monument | Identifiant stable | Texte français à acquérir |
|---|---|---|
| Mosquée Koutoubia | Q1137533 | Mosquée Koutoubia |
| Palais El Bahia | Q2465115 | Palais de la Bahia |
| Médersa Ben Youssef | Q1560540 | Médersa Ben Youssef |

Les trois entités Wikidata ont un pays déclaré Maroc, des coordonnées et un lien d'article français. Les révisions et réponses brutes sont figées dans `sources/`. Les dates de construction, auteurs et dynasties contenus dans les déclarations Wikidata ne sont pas automatiquement traités comme des faits validés : leurs qualificatifs et références doivent être examinés.

Trois originaux JPEG Commons sont téléchargés, vérifiés par SHA1 par rapport à l'API source et par SHA256 dans ce paquet. Ils ont été examinés visuellement et sont cohérents avec les métadonnées des monuments. Cette inspection n'est pas une authentification indépendante. Un original dépasse le profil photo de l'application (4 millions de pixels) ; une copie de travail correctement créditée sera nécessaire avant intégration.

Ouvrir `index.html` pour consulter la galerie et les crédits. Les images sont utilisées sans modification dans ce paquet. Voir `ATTRIBUTIONS.md` pour les auteurs et les conditions de réutilisation. La licence CC0 des données structurées ne couvre pas les photos ou les textes Wikipédia.

## Acquérir les textes — Python standard, sans modèle ni clé

Depuis ce dossier, exécuter :

```powershell
py .\validate.py
py .\collect_texts.py
```

Le collecteur demande un article à la fois, vérifie son identifiant Wikidata, la présence du texte, la révision, la licence du site et l'empreinte du contenu. Les réponses brutes et les instantanés sont conservés dans `text-snapshots/`. Un instantané déjà valide est repris sans nouvelle requête ; un fichier corrompu est signalé et jamais remplacé silencieusement. Il respecte un délai entre requêtes et les limitations HTTP 429/503, sans désactiver TLS.

Le fichier à transmettre pour revue est `results/marrakech-textes.zip`. Il contient les trois textes lorsqu'ils sont acquis, leurs sources, leurs empreintes et un bilan. Une collecte partielle reste explicitement `incomplete` et termine avec un code d'échec. `--offline` contrôle les instantanés existants sans accès réseau. Aucun paquet pip, aucun chargement d'IA et aucune modification des modèles ne sont nécessaires.

Les instantanés générés et bilans locaux sont exclus de Git ; ils ne deviennent pas automatiquement une nouvelle version publique du corpus.

## Audit avant intégration

Pour chaque monument :

1. Relire les paragraphes sur l'origine, la fonction et les campagnes de construction, en distinguant commanditaire, constructeur et restaurateur.
2. Extraire les preuves exactes pour des questions de contrôle : « Pourquoi ce lieu a-t-il été construit ? », « À quelle période ? », « Qui est associé à sa construction ? ». Ne pas compléter les informations absentes.
3. Chercher une source indépendante complémentaire dont les droits sont connus. Les pages UNESCO et les portails marocains sont des pistes de vérification, pas des textes déjà acquis ou redistribuables.
4. Découper les textes en passages avec provenance et empreintes, puis mesurer la recherche documentaire.
5. Ajouter plusieurs photos distinctes par monument, séparer références et photos d'évaluation et refaire une calibration avant de proposer une identification marocaine dans l'application.
6. Adapter le catalogue, les alias, les villes et les parcours afin de ne jamais relier Paris à Marrakech dans une promenade.

Les horaires, tarifs actuels, entrées et coordonnées d'accès restent non vérifiés. Les champs correspondants sont nuls. Les coordonnées sont des positions de monuments déclarées, pas des entrées visiteurs vérifiées.

## Vérifications logicielles

```powershell
py -m unittest discover -v
```

Les fixtures de test sont synthétiques et ne certifient pas les textes réels. Le validateur examine aussi les fichiers publics du pilote, les révisions, les empreintes et les licences.
