"""Frozen descriptor policy evaluated in a short-lived CPU subprocess."""
import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import traceback

import evaluate_abstention as abstention
import visual_refs as visual
from run_local import ROOT, save

VERSION = 'assisted_visual_policy_v1'
POLICY_SHA256 = 'cb6ee7296d2bc65062bd44ed8f6621d9093cd9217fe922c1dd9e73cdaf76e3ec'


def load_policy(root=ROOT):
    root = pathlib.Path(root)
    path = root/'visual-policy-v1/policy.json'
    if visual.sha(path) != POLICY_SHA256:
        raise ValueError('La règle visuelle figée a été modifiée ; une nouvelle validation est nécessaire.')
    policy = json.loads(path.read_text(encoding='utf-8'))
    if policy['version'] != VERSION or visual.sha(root/'corpus/sites.json') != policy['catalog_sha256']:
        raise ValueError('Le catalogue ne correspond pas à la règle visuelle.')
    for name, digest in policy['reference_manifests_sha256'].items():
        if visual.sha(root/name) != digest:
            raise ValueError('Les références ne correspondent pas à la règle visuelle figée.')
    if visual.sha(root/'abstention-v1/manifest.json') != policy['calibration_protocol_sha256']:
        raise ValueError('Le protocole de calibration ne correspond pas à la règle figée.')
    return policy


def validate_record(record, image_sha, sites, root=ROOT):
    policy = load_policy(root)
    if (record.get('status') != 'complete' or record.get('version') != VERSION or
            record.get('policy_sha256') != POLICY_SHA256 or record.get('image_sha256') != image_sha or
            record.get('descriptor_namespace') != policy['descriptor_namespace'] or
            record.get('reference_manifests_sha256') != policy['reference_manifests_sha256'] or
            record.get('identity_confirmed') is not False):
        raise ValueError('Résultat de comparaison incompatible avec la photo ou la règle figée.')
    ranking = record.get('top3_candidates', [])
    if len(ranking) != 3 or len({r['site_id'] for r in ranking}) != 3:
        raise ValueError('Classement visuel invalide.')
    for r in ranking:
        score = r.get('cosine_similarity')
        if (r.get('site_id') not in sites or not isinstance(score, (float, int)) or isinstance(score, bool) or
                not math.isfinite(score) or not -1.000001 <= score <= 1.000001 or r.get('references_for_site') != 3):
            raise ValueError('Candidat ou score visuel invalide.')
    if any(ranking[i]['cosine_similarity'] < ranking[i+1]['cosine_similarity'] for i in range(2)):
        raise ValueError('Ordre des candidats visuels invalide.')
    feature = abstention.features(record)
    decision = abstention.decide(feature, policy['rule'])
    if record.get('features') != feature or any(record.get(k) != v for k, v in decision.items()):
        raise ValueError('La décision ne correspond pas aux scores et à la règle figée.')
    return record


def run_guard(data, metadata, output):
    """Keep original photo bytes out of report directories and release ORT before Qwen."""
    output = pathlib.Path(output)
    image_sha = hashlib.sha256(data).hexdigest()
    if metadata.get('sha256') != image_sha:
        raise ValueError('Empreinte de photo incohérente avant comparaison.')
    python = ROOT/'.visual-venv'/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.is_file():
        raise ValueError('Environnement de vision absent. Lancez d’abord py evaluate_visual_refs.py pour le préparer.')
    record_path = output/'visual-guard.json'
    print('Comparaison de la photo aux références visuelles...', flush=True)
    # The temporary original is outside the recursively archived report folder.
    with tempfile.TemporaryDirectory(prefix='patrimotrip-photo-') as temporary:
        image = pathlib.Path(temporary)/('input.png' if metadata['mime'] == 'image/png' else 'input.jpg')
        image.write_bytes(data)
        with (output/'visual-guard.log').open('w', encoding='utf-8') as log:
            process = subprocess.run([str(python), str(ROOT/'visual_guard.py'), '--image', str(image),
                                      '--image-sha256', image_sha, '--output', str(record_path.resolve())],
                                     stdout=log, stderr=subprocess.STDOUT, timeout=240, check=False)
    if process.returncode != 0 or not record_path.is_file():
        detail = ''
        if record_path.is_file():
            detail = json.loads(record_path.read_text(encoding='utf-8')).get('error', '')
        raise ValueError('Comparaison visuelle indisponible ; aucune explication générée. '+detail)
    record = json.loads(record_path.read_text(encoding='utf-8'))
    sites = {s['site_id'] for s in json.loads((ROOT/'corpus/sites.json').read_text(encoding='utf-8'))}
    return validate_record(record, image_sha, sites)


def worker(image, image_sha, output):
    record = {'version': VERSION, 'status': 'failed', 'identity_confirmed': False}
    try:
        # Read-only image validation shared with the normal photo command.
        from photo_local import load_image
        _, metadata = load_image(image)
        if metadata['sha256'] != image_sha:
            raise ValueError('La photo temporaire ne correspond pas à son empreinte.')
        policy = load_policy()
        bank, _ = visual.expanded_bank(ROOT/'visual-references-v1', ROOT/'visual-references-v2', ROOT/'evaluation-v1')
        spec = visual.fetch_model(ROOT/'models/dinov2-small-onnx')
        encoder = visual.Encoder(ROOT/'models/dinov2-small-onnx', spec)
        if encoder.namespace != policy['descriptor_namespace']:
            raise ValueError('Le modèle ou son prétraitement a changé ; la règle ne peut pas être réutilisée.')
        cache = ROOT/'models/visual-descriptor-cache'
        references = []
        for row, path in bank:
            vector, _ = visual.cached_encode(encoder, path, cache)
            references.append({'site_id': row['site_id'], 'sha256': row['sha256'], 'vector': vector})
        vector, _ = visual.cached_encode(encoder, image, cache)
        ranking = visual.rank_vector(vector, references)[:3]
        if visual.sha(image) != image_sha:
            raise ValueError('La photo a changé pendant le calcul visuel.')
        feature = abstention.features({'top3_candidates': ranking})
        record.update(status='complete', image_sha256=image_sha, policy_sha256=POLICY_SHA256,
                      descriptor_namespace=encoder.namespace, reference_manifests_sha256=policy['reference_manifests_sha256'],
                      top3_candidates=ranking, features=feature, **abstention.decide(feature, policy['rule']))
        validate_record(record, image_sha, {row['site_id'] for row, _ in bank})
    except Exception as error:
        record.update(status='failed', error=str(error))
        (output.parent/'visual-guard-error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        save(output, record)
    return 0 if record['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=pathlib.Path, required=True)
    parser.add_argument('--image-sha256', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    raise SystemExit(worker(args.image, args.image_sha256, args.output))
