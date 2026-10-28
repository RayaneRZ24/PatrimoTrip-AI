"""Compare documentary retrieval and answers with supplied identities; no photo inference."""
import argparse
import datetime as dt
import json
import pathlib
import statistics
import subprocess
import time
import traceback
import zipfile

import facet_rag
import hybrid
import photo_local as photo
import rag_local as rag
import runtime
from evaluate_photo import review_entries
from run_local import ROOT, embedding_cache_key, query_id, save

CASES = ROOT/'retrieval-v2/cases.json'


def load_cases(corpus, sites):
    manifest = json.loads(CASES.read_text(encoding='utf-8'))
    ids = set()
    for case in manifest['cases']:
        if case['case_id'] in ids or not case['case_id'].isalnum() or case['site_id'] not in sites:
            raise ValueError('Invalid documentary case.')
        ids.add(case['case_id'])
        plan = facet_rag.plan_question(case['question'], sites[case['site_id']]['name'])
        if [f['facet_id'] for f in plan['facets']] != [f['facet_id'] for f in case['facets']]:
            raise ValueError('Question plan no longer matches the versioned evaluation facets.')
        for facet in case['facets']:
            for annotation in facet['known_evidence']:
                p = corpus.passages[annotation['passage_id']]
                if p['site_id'] != case['site_id'] or annotation['quote'] not in p['text']:
                    raise ValueError('Invalid documentary evidence annotation.')
    return manifest


def prepare(corpus, sites, manifest, manager, output, monitor):
    model_path, metadata = runtime.download_embedding(ROOT/'models')
    cache = hybrid.VectorCache(ROOT/'vectors.sqlite3', embedding_cache_key(metadata, corpus))
    site_ids = {c['site_id'] for c in manifest['cases']}
    items = [('document', p['passage_id'], hybrid.document_input(p, corpus.sources[p['document_id']]))
             for p in corpus.passages.values() if p['site_id'] in site_ids]
    queries = {}
    for case in manifest['cases']:
        plan = facet_rag.plan_question(case['question'], sites[case['site_id']]['name'])
        for q in [plan['base_query']]+facet_rag.extra_queries(plan):
            queries[query_id(q)] = hybrid.query_input(q)
    items += [('query', key, text) for key, text in queries.items()]
    missing = sum(cache.get(role, key, text) is None for role, key, text in items)
    save(output/'embedding.json', {'model': metadata, 'items': len(items), 'missing': missing, 'cache_namespace': cache.model_key})
    print(f'Cache : {len(items)-missing} éléments réutilisables ; {missing} à encoder.', flush=True)
    if missing:
        monitor.phase = 'embedding'
        try:
            url = manager.start('embedding', model_path)
        except runtime.ModelStartupError:
            manager.stop()
            url = manager.start('embedding', model_path, embedding_cpu=True)
        encoder = hybrid.Embeddings(url)
        count = 0
        for role, key, text in items:
            _, reused, _ = cache.encode(encoder, role, key, text)
            if not reused:
                count += 1
                if count % 25 == 0 or count == missing:
                    print(f'Nouveaux encodages : {count}/{missing}', flush=True)
        manager.stop()
    return cache


def score(case, result, evidence):
    actual = result.get('answer', {}).get('status')
    statuses = result.get('facet_statuses')
    if statuses is None and len(case['facets']) == 1:
        statuses = {'F1': actual}
    rows = []
    for facet in case['facets']:
        hit = any(e['passage_id'] == a['passage_id'] and a['quote'] in e['text'] for a in facet['known_evidence'] for e in evidence) if facet['known_evidence'] else None
        rows.append({'facet_id': facet['facet_id'], 'known_evidence_in_context': hit, 'expected_status': facet['expected_status'],
                     'model_status': statuses.get(facet['facet_id']) if statuses else None,
                     'status_matches': statuses.get(facet['facet_id']) == facet['expected_status'] if statuses else None})
    return {'answer_status': actual, 'expected_aggregate_status_matches': actual == case['expected_status'],
            'reference_checks_passed': result.get('reference_checks_passed', False), 'facets': rows,
            'factual_review': 'pending', 'automatic_factual_accuracy': 'not_measured'}


def compare(corpus, sites, manifest, cache, client, output):
    rows, review = [], []
    for case in manifest['cases']:
        plan = facet_rag.plan_question(case['question'], sites[case['site_id']]['name'])
        for method in ['baseline', 'per_facet']:
            folder = output/'cases'/case['case_id']/method
            folder.mkdir(parents=True)
            started = time.perf_counter()
            print(case['case_id']+' : '+method, flush=True)
            row = {'case_id': case['case_id'], 'method': method}
            try:
                if method == 'per_facet' and plan['compound']:
                    evidence, rankings = facet_rag.retrieve(corpus, case['site_id'], sites[case['site_id']]['name'], plan, cache)
                else:
                    evidence, rankings = photo.scoped_retrieval(corpus, case['site_id'], plan['base_query'], cache)
                row['retrieval_seconds'] = time.perf_counter()-started
                save(folder/'retrieval.json', {'site_id': case['site_id'], 'plan': plan, 'rankings': rankings, 'evidence': evidence})
                if method == 'per_facet' and plan['compound']:
                    result = facet_rag.ask(plan['base_query'], plan, evidence, client, photo.PHOTO_SYSTEM)
                else:
                    result = rag.ask(corpus.database, plan['base_query'], client,
                                     retrieval_fn=lambda database, question: evidence,
                                     retrieval_name='site_scoped_hybrid_rrf', system_prompt=photo.PHOTO_SYSTEM)
                save(folder/'answer.json', result)
                row.update(score(case, result, evidence))
                review.extend({**r, 'method': method} for r in review_entries(case['case_id'], folder))
            except Exception as error:
                row['error'] = str(error)
                (folder/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            row['method_seconds'] = time.perf_counter()-started
            rows.append(row)
            save(output/'comparison.json', rows)
            save(output/'review.json', review)
    summary = {}
    for method in ['baseline', 'per_facet']:
        selected = [r for r in rows if r['method'] == method]
        facets = [f for r in selected for f in r.get('facets', [])]
        known = [f for f in facets if f['known_evidence_in_context'] is not None]
        missing = [f for f in facets if f['expected_status'] == 'insufficient_sources' and f['model_status'] is not None]
        summary[method] = {'cases': len(selected), 'errors': sum(bool(r.get('error')) for r in selected),
                           'reference_checks_passed': sum(bool(r.get('reference_checks_passed')) for r in selected),
                           'expected_aggregate_status_matches': sum(bool(r.get('expected_aggregate_status_matches')) for r in selected),
                           'documented_facets_with_known_evidence': sum(f['known_evidence_in_context'] for f in known),
                           'documented_facets_scored': len(known),
                           'missing_facets_with_expected_status': sum(bool(f['status_matches']) for f in missing),
                           'missing_facets_with_explicit_status': len(missing),
                           'median_warm_method_seconds': statistics.median(r['method_seconds'] for r in selected if not r.get('error')) if any(not r.get('error') for r in selected) else None}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take-over-test-server', action='store_true')
    models = pathlib.Path.home()/'PatrimoTripAI/models/qwen3.5-4b'
    parser.add_argument('--generation-model', type=pathlib.Path, default=models/'Qwen3.5-4B-Q4_K_M.gguf')
    args = parser.parse_args()
    output = ROOT/'results'/('retrieval-facets-'+dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+runtime.uuid_suffix())
    output.mkdir(parents=True)
    report = {'status': 'started', 'scope': 'Documentary-only development comparison with supplied identities, not photo recognition accuracy.',
              'retrieval_version': facet_rag.VERSION, 'factual_review': 'pending', 'utc_started': dt.datetime.now(dt.timezone.utc).isoformat()}
    manager, monitor, started, success = None, None, time.perf_counter(), False
    try:
        corpus = hybrid.Corpus(ROOT)
        sites = {s['site_id']: s for s in json.loads((ROOT/'corpus/sites.json').read_text(encoding='utf-8'))}
        manifest = load_cases(corpus, sites)
        with args.generation_model.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise ValueError('Invalid generation GGUF file.')
        report.update(cases_planned=len(manifest['cases']), corpus_fingerprint=corpus.fingerprint,
                      generation_sha256=rag.sha_file(args.generation_model), cases_sha256=rag.sha_file(CASES),
                      generation_settings={'gpu_layers': 20, 'context': 4096, 'batch': 256, 'ubatch': 128, 'parallel': 1,
                                           'reasoning_budget': 0, 'vision': False, 'baseline_max_tokens': 512, 'compound_max_tokens': 768},
                      script_sha256={p.name: rag.sha_file(p) for p in ROOT.glob('*.py')})
        save(output/'evaluation-cases.json', manifest)
        manager = runtime.Models(args.generation_model.resolve(), models/'mmproj-F16.gguf', output, args.take_over_test_server)
        version = subprocess.run([manager.executable, '--version'], capture_output=True, text=True, errors='replace', timeout=20, env=manager.environment)
        report['llama_version'] = (version.stdout+version.stderr)[:2000]
        manager.prepare_ports()
        monitor = runtime.Resources(output, manager.environment); monitor.start()
        cache = prepare(corpus, sites, manifest, manager, output, monitor)
        monitor.phase = 'text_generation_start'
        url = manager.start('generation', generation_vision=False)
        monitor.phase = 'warm_comparison'
        report['comparison'] = compare(corpus, sites, manifest, cache, rag.LocalModel(url), output)
        success = all(m['errors'] == 0 for m in report['comparison'].values())
        report['status'] = 'complete' if success else 'complete_with_case_errors'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=str(error) or 'Interrupted')
        (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('Comparaison arrêtée : '+report['error'], flush=True)
    finally:
        if manager:
            manager.stop(); save(output/'model-events.json', manager.events)
        if monitor:
            monitor.stop()
        report['total_wall_seconds'] = time.perf_counter()-started
        report['server_left_running'] = False
        save(output/'summary.json', report)
        (output/'rapport.md').write_text('# Comparaison documentaire par sous-question\n\n'+
            'Voir summary.json et comparison.json. Les présences de preuves connues ne mesurent ni rappel exhaustif ni véracité.\n\n'+
            'Les identités sont fournies pour isoler la recherche et la rédaction ; aucune reconnaissance photo dans ce test.\n\n'+
            'Les temps par méthode utilisent des requêtes pré-encodées et un modèle texte déjà chargé. Le temps total et model-events.json incluent la préparation et les chargements du lot.\n\n'+
            'Relire review.json : support = supported / partial / unsupported / contradicted ; answers_question = yes / partial / no. Garder not_reviewed avant lecture.\n', encoding='utf-8')
        with zipfile.ZipFile(output/'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in output.rglob('*'):
                if path.is_file() and path.name != 'report.zip':
                    archive.write(path, path.relative_to(output))
        print('Rapport local : '+str(output/'report.zip'), flush=True)
        if not success:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
