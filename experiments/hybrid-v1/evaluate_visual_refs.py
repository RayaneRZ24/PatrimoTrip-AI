"""Isolated CPU experiment: candidate image retrieval, no automatic identity gate."""
import argparse
import datetime as dt
import importlib.metadata
import json
import pathlib
import subprocess
import sys
import time
import traceback
import uuid
import venv
import zipfile

import visual_refs as visual

ROOT = pathlib.Path(__file__).resolve().parent
DEPENDENCIES = {'numpy': '2.2.6', 'Pillow': '11.3.0', 'onnxruntime': '1.23.2'}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def archive(output):
    with zipfile.ZipFile(output/'report.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob('*'):
            if p.is_file() and p.name != 'report.zip': z.write(p, p.relative_to(output))
    print('Rapport local : '+str(output/'report.zip'), flush=True)


def prior_predictions(results, assets_sha):
    for p in sorted(results.glob('evaluation-*/summary.json'), reverse=True):
        try:
            summary = json.loads(p.read_text(encoding='utf-8'))
            if summary.get('status') != 'complete' or summary.get('cases_planned') != 12 or summary.get('assets_sha256') != assets_sha:
                continue
            rows = json.loads((p.parent/'cases-summary.json').read_text(encoding='utf-8'))
            by_asset = {}
            for row in rows: by_asset.setdefault(row['asset_id'], row.get('predicted_site_id'))
            return p.parent.name, by_asset
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None, {}


def worker(output):
    started = time.perf_counter()
    summary = {'status': 'failed', 'version': visual.VERSION,
               'scope': 'Previously seen development photos; independent descriptor model, not independent test data.',
               'inference_provider': 'CPUExecutionProvider', 'automatic_identity_gate': False,
               'outside_detection': 'not_implemented', 'similarities_are_probabilities': False,
               'original_photo_pipeline_modified': False,
               'script_sha256': {n: visual.sha(ROOT/n) for n in ['visual_refs.py', 'evaluate_visual_refs.py']}}
    try:
        summary['dependencies'] = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
        refs_root = ROOT/'visual-references-v1'; queries_root = ROOT/'evaluation-v1'
        refs_manifest = json.loads((refs_root/'references.json').read_text(encoding='utf-8'))
        query_rows = json.loads((queries_root/'assets.json').read_text(encoding='utf-8'))
        references = visual.checked_assets(refs_root, refs_manifest['references'])
        queries = visual.checked_assets(queries_root, query_rows, 'expected_site_id')
        if len(references) != 10 or len(queries) != 10 or len({r['site_id'] for r, _ in references}) != 10:
            raise ValueError('Unexpected reference/query counts.')
        if {r['sha256'] for r, _ in references} & {r['sha256'] for r, _ in queries}:
            raise ValueError('Reference and query image leakage.')
        sites = {r['site_id'] for r in json.loads((ROOT/'corpus/sites.json').read_text(encoding='utf-8'))}
        if {r['site_id'] for r, _ in references} != sites:
            raise ValueError('Reference sites differ from the catalog.')
        model_dir = ROOT/'models/dinov2-small-onnx'
        spec = visual.fetch_model(model_dir)
        save(output/'model-lock.json', spec)
        print('Chargement du modèle visuel sur CPU ; aucun serveur génératif lancé.', flush=True)
        encoder = visual.Encoder(model_dir, spec)
        summary['descriptor_namespace'] = encoder.namespace
        rows = visual.evaluate(encoder, references, queries, ROOT/'models/visual-descriptor-cache')
        previous, predictions = prior_predictions(ROOT/'results', visual.sha(queries_root/'assets.json'))
        for row in rows:
            old = predictions.get(row['asset_id'])
            row['previous_pipeline_site_id'] = old
            row['different_from_previous_pipeline'] = row['top3_candidates'][0]['site_id'] != old if old and old != 'unknown' else None
        save(output/'cases.json', rows)
        save(output/'reference-metadata.json', refs_manifest)
        save(output/'query-metadata.json', query_rows)
        summary.update(status='complete', model=spec, previous_pipeline_run=previous,
                       query_assets_sha256=visual.sha(queries_root/'assets.json'),
                       reference_manifest_sha256=visual.sha(refs_root/'references.json'),
                       cases=len(rows), known_site_top1={kind: {'correct': sum(r['top1_exact'] is True for r in rows if r['kind'] == kind),
                                                           'total': sum(r['kind'] == kind for r in rows)} for kind in ['standard', 'detail']},
                       outside_and_ambiguous_review_required=sum(r['kind'] in ['outside', 'ambiguous'] for r in rows))
        print('\nCandidats visuels :', flush=True)
        for row in rows:
            top = row['top3_candidates'][0]
            print(row['asset_id']+' -> '+top['site_id']+' ; similarité '+format(top['cosine_similarity'], '.3f')+
                  ' ; identité non confirmée', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        summary['error'] = str(error) or 'Interrupted'
        (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('Essai arrêté : '+summary['error'], flush=True)
    finally:
        summary['total_wall_seconds'] = time.perf_counter()-started
        save(output/'summary.json', summary)
        (output/'rapport.md').write_text(
            '# Comparaison visuelle avec références\n\n'
            'Les scores cosinus ne sont pas des probabilités. Un meilleur voisin est toujours retourné, même pour Rome ou un mur. '
            'Aucun seuil hors catalogue, aucune identité confirmée et aucune réponse historique automatique ne sont produits.\n\n'
            'Les images de test sont déjà connues du développement, mais distinctes des dix références. Un seul angle par site limite le rappel. '
            'Le modèle CPU et ses erreurs peuvent différer du modèle génératif ; leur accord ne prouve pas une identité. '
            'Voir cases.json pour Invalides, Panthéon de Paris/Rome et les écarts avec le parcours précédent, si disponible.\n\n'
            'Aucune photo, aucun poids ni vecteur complet ne sont inclus dans ce rapport. '
            'Les durées par image peuvent réutiliser le cache ; le temps total inclut les chargements et téléchargements du worker.\n', encoding='utf-8')
        archive(output)
    return 0 if summary['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=pathlib.Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.output is None or not args.output.resolve().is_relative_to((ROOT/'results').resolve()):
            parser.error('Invalid output directory.')
        raise SystemExit(worker(args.output))
    output = ROOT/'results'/('visual-refs-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    try:
        env_dir = ROOT/'.visual-venv'
        python = env_dir/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        if not python.exists():
            print('Création d’un environnement Python isolé pour la vision CPU...', flush=True)
            venv.EnvBuilder(with_pip=True).create(env_dir)
        check = subprocess.run([str(python), '-c', 'import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in '+repr(list(DEPENDENCIES))+'}))'],
                               capture_output=True, text=True)
        if check.returncode != 0 or json.loads(check.stdout) != DEPENDENCIES:
            print('Installation isolée : NumPy, Pillow et ONNX Runtime CPU (premier lancement)...', flush=True)
            with (output/'installation.log').open('w', encoding='utf-8') as log:
                subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--only-binary=:all:']+
                               [k+'=='+v for k, v in DEPENDENCIES.items()], stdout=log, stderr=subprocess.STDOUT, check=True)
        code = subprocess.call([str(python), str(pathlib.Path(__file__).resolve()), '--worker', '--output', str(output)])
        if not (output/'report.zip').is_file():
            raise RuntimeError('Worker stopped before writing its report; exit code '+str(code))
        raise SystemExit(code)
    except (Exception, KeyboardInterrupt) as error:
        save(output/'summary.json', {'status': 'bootstrap_failed', 'error': str(error) or 'Interrupted',
                                    'dependencies_requested': DEPENDENCIES, 'automatic_identity_gate': False})
        (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('Installation ou lancement arrêté : '+str(error), flush=True)
        archive(output)
        raise SystemExit(1)


if __name__ == '__main__': main()
