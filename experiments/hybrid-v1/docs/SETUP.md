# Installation et configuration

## Installation déjà réalisée dans cette session

Sur le PC utilisateur, le code se trouve dans `$HOME\PatrimoTripAI\hybrid-v1`. Les versions observées pendant la préparation sont Python 3.13.2 et llama.cpp b11889, avec Vulkan0 sur la GTX 1650 Ti. Les commandes de mise à jour d'archive conservent `models/`, `results/`, `vectors.sqlite3`, `.visual-venv/` et `.app-state/`.

L'application se lance avec `py .\app_local.py --open` ; le contrôle complet avec `py .\verify_release.py --models --open`. Pour ouvrir le code dans VS Code :

```powershell
code (Join-Path $HOME "PatrimoTripAI\hybrid-v1")
```

## Chemins attendus

```text
PatrimoTripAI/
  models/qwen3.5-4b/
    Qwen3.5-4B-Q4_K_M.gguf
    mmproj-F16.gguf
  hybrid-v1/
    app_local.py
    models/
      model.json
      Qwen3-Embedding-0.6B.Q8_0.gguf
      dinov2-small-onnx/model-lock.json
      dinov2-small-onnx/onnx/model.onnx
      visual-descriptor-cache/
    .visual-venv/
    corpus/
    results/
    .app-state/trips/
```

Les poids Qwen4B et projecteur ont été téléchargés et vérifiés pendant la configuration du PC. Leur provenance est `unsloth/Qwen3.5-4B-GGUF`, modèle de base `Qwen/Qwen3.5-4B`, Apache-2.0. Empreintes utilisées dans cette configuration :

- Qwen4B : `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
- Projecteur : `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864`.

L'acquisition vérifiée du modèle d'embeddings reste intégrée dans `runtime.py`. L'environnement CPU et le modèle DINOv2 sont préparés par `py .\evaluate_visual_refs.py`, si nécessaire. Le contrôle de licence DINOv2 distingue la déclaration absente du dépôt de conversion de la preuve de licence et de provenance du modèle amont. Les fichiers déjà vérifiés sont réutilisés hors connexion.

## Nouvelle machine

1. Installer Python 3.13 et llama.cpp Vulkan (`winget install --id ggml.llamacpp --exact --source winget` sur Windows).
2. Fermer puis rouvrir PowerShell après l'installation ; vérifier `py --version`, `llama-cli --version` et `llama-cli --list-devices`.
3. Télécharger les deux fichiers GGUF attendus depuis la provenance indiquée, vérifier les SHA-256 ci-dessus et les placer dans le dossier attendu.
4. Préparer la vision CPU avec `py .\evaluate_visual_refs.py`. Ce lanceur crée son environnement isolé et télécharge les fichiers ONNX avec vérification de taille, empreinte et révision.
5. Lancer le contrôle complet. Le modèle d'embeddings sera préparé si le parcours photo en a besoin et s'il manque encore.

Il n'est pas nécessaire de lancer un serveur llama.cpp à la main. Internet est nécessaire pour les premières installations et acquisitions ; les inférences sont locales. Un cache ou verrou de modèle modifié n'est pas remplacé silencieusement par une autre version.

## Sauvegarder son travail

Conserver `.app-state/trips/` pour les parcours, `results/` pour les rapports, `models/` pour éviter les téléchargements, et `vectors.sqlite3` pour les embeddings déjà calculés. Les mises à jour de l'archive ne doivent pas effacer ces dossiers. Aucun de ces fichiers privés n'est publié dans Git.
