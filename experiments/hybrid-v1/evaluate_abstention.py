"""Pilot abstention calibration with a rule frozen before held-out inference."""
import json
import importlib.metadata
import math
import pathlib
import time
import traceback
import urllib.parse

import visual_refs as visual
from evaluate_visual_refs import ROOT, save, archive

VERSION = 'visual_abstention_pilot_v1'
SCORE_GRID = tuple(i/20 for i in range(21))
GAP_GRID = tuple(i/40 for i in range(41))


def page_key(row):
    return urllib.parse.unquote(row['source_page']).replace('_', ' ').casefold()


def verify_splits(dataset, bank):
    dataset = pathlib.Path(dataset)
    manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != VERSION:
        raise ValueError('Unexpected abstention dataset version.')
    for name, digest in manifest['metadata_sha256'].items():
        if name not in {'calibration.json', 'heldout.json', 'excluded-sources.json'} or visual.sha(dataset/name) != digest:
            raise ValueError('Modified split/exclusion metadata.')
    if set(manifest['metadata_sha256']) != {'calibration.json', 'heldout.json', 'excluded-sources.json'}:
        raise ValueError('Incomplete split metadata checksums.')
    for relative, digest in manifest['reference_manifests_sha256'].items():
        if relative not in {'visual-references-v1/references.json', 'visual-references-v2/references.json'} or visual.sha(dataset.parent/relative) != digest:
            raise ValueError('Reference bank differs from the frozen calibration protocol.')
    if len(manifest['reference_manifests_sha256']) != 2:
        raise ValueError('Incomplete reference bank fingerprints.')
    excluded = json.loads((dataset/'excluded-sources.json').read_text(encoding='utf-8'))
    excluded_hashes = {r['sha256'] for r in excluded} | {r['sha256'] for r, _ in bank}
    excluded_pages = {page_key(r) for r in excluded} | {page_key(r) for r, _ in bank}
    sites = {r['site_id'] for r, _ in bank}
    splits, all_hashes, all_pages, all_ids, groups = {}, set(), set(), set(), {}
    for split in ['calibration', 'heldout']:
        rows = json.loads((dataset/(split+'.json')).read_text(encoding='utf-8'))
        assets = visual.checked_assets(dataset, rows, 'expected_site_id')
        if (len(rows) != 20 or sum(r['kind'] == 'known' for r in rows) != 10 or
                sum(r['kind'] == 'outside' for r in rows) != 8 or sum(r['kind'] == 'ambiguous' for r in rows) != 2 or
                {r['expected_site_id'] for r in rows if r['kind'] == 'known'} != sites):
            raise ValueError('Each split must contain ten catalog sites, eight outside subjects and two ambiguous images.')
        for row in rows:
            if (row['split'] != split or row['asset_id'] in all_ids or
                    row['sha256'] in all_hashes | excluded_hashes or page_key(row) in all_pages | excluded_pages):
                raise ValueError('Duplicate, old query, reference source or split leakage.')
            if row['kind'] != 'known' and row['expected_site_id'] != 'unknown':
                raise ValueError('Outside/ambiguous images must not have a catalog label.')
            if not row.get('author') or not row.get('license') or not row.get('photographer_group') or row.get('visual_review') != 'reviewed_before_descriptor_run':
                raise ValueError('Missing attribution, photographer group or image review.')
            group = row['photographer_group']
            if group in groups and groups[group] != split:
                raise ValueError('Photographer group straddles calibration and held-out data.')
            groups[group] = split
            all_hashes.add(row['sha256']); all_pages.add(page_key(row)); all_ids.add(row['asset_id'])
        splits[split] = assets
    calibration_subjects = {r['subject'] for r, _ in splits['calibration'] if r['kind'] == 'outside'}
    heldout_subjects = {r['subject'] for r, _ in splits['heldout'] if r['kind'] == 'outside'}
    if calibration_subjects & heldout_subjects:
        raise ValueError('Outside monument subjects straddle calibration and held-out data.')
    return manifest, splits


def features(row):
    ranked = row['top3_candidates']
    if len(ranked) < 2:
        raise ValueError('Two different candidates are required for the score gap.')
    first, second = ranked[:2]
    score, runner = first['cosine_similarity'], second['cosine_similarity']
    if (not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) and -1.000001 <= x <= 1.000001 for x in [score, runner])
            or first['site_id'] == second['site_id'] or score < runner):
        raise ValueError('Invalid candidate scores or order.')
    return {'candidate_site_id': first['site_id'], 'similarity': score, 'gap': score-runner}


def decide(feature, rule):
    # No query label, filename, subject name or source metadata is accepted here.
    accepted = not rule['reject_all'] and feature['similarity'] >= rule['minimum_similarity'] and feature['gap'] >= rule['minimum_gap']
    return {'decision': 'propose' if accepted else 'abstain',
            'proposed_site_id': feature['candidate_site_id'] if accepted else None,
            'identity_confirmed': False}


def fit_rule(calibration):
    if not calibration or not any(r['expected_site_id'] == 'unknown' for r in calibration) or not any(r['expected_site_id'] != 'unknown' for r in calibration):
        raise ValueError('Calibration requires both catalog and unknown images.')
    samples = [(features(r), r['expected_site_id']) for r in calibration]
    best = {'reject_all': True, 'minimum_similarity': 1.000001, 'minimum_gap': 0.0}
    best_key = (0, best['minimum_similarity'], best['minimum_gap'])
    admissible = 1
    for threshold in SCORE_GRID:
        for margin in GAP_GRID:
            rule = {'reject_all': False, 'minimum_similarity': threshold, 'minimum_gap': margin}
            correct, wrong = 0, 0
            for feature, expected in samples:
                if decide(feature, rule)['decision'] == 'propose':
                    if expected != 'unknown' and feature['candidate_site_id'] == expected: correct += 1
                    else: wrong += 1
            if wrong:
                continue
            admissible += 1
            key = (correct, threshold, margin)
            if key > best_key:
                best, best_key = rule, key
    return {'version': VERSION, 'rule': best, 'calibration_cases': len(calibration),
            'calibration_correct_proposals': best_key[0], 'calibration_wrong_proposals': 0,
            'grid': {'similarity': list(SCORE_GRID), 'gap': list(GAP_GRID), 'reject_all_candidate': True},
            'admissible_rules': admissible,
            'selection': 'Maximize correct proposals subject to zero observed calibration errors; ties prefer higher similarity then gap. Reject-all remains available.',
            'limitation': 'Zero observed calibration errors is not a statistical guarantee; this small development sample may overfit.'}


def apply_and_score(rows, policy):
    output = []
    for row in rows:
        feature = features(row)
        decision = decide(feature, policy['rule'])
        # Truth is consulted only after making the decision.
        expected = row['expected_site_id']
        accepted = decision['decision'] == 'propose'
        correct = accepted and expected != 'unknown' and decision['proposed_site_id'] == expected
        output.append({**row, **decision, 'features': feature, 'proposal_correct': correct if accepted else None})
    known = [r for r in output if r['expected_site_id'] != 'unknown']
    unknown = [r for r in output if r['expected_site_id'] == 'unknown']
    accepted = [r for r in output if r['decision'] == 'propose']
    correct = sum(r['proposal_correct'] is True for r in accepted)
    metrics = {'cases': len(output), 'known_cases': len(known), 'unknown_cases': len(unknown),
               'proposals': len(accepted), 'correct_proposals': correct, 'wrong_proposals': len(accepted)-correct,
               'empirical_precision_among_proposals': correct/len(accepted) if accepted else None,
               'known_correct_proposal_rate': correct/len(known) if known else None,
               'known_abstentions': sum(r['decision'] == 'abstain' for r in known),
               'unknown_false_proposals': sum(r['decision'] == 'propose' for r in unknown),
               'unknown_false_proposal_rate': sum(r['decision'] == 'propose' for r in unknown)/len(unknown) if unknown else None,
               'outside_false_proposals': sum(r['decision'] == 'propose' and r['kind'] == 'outside' for r in output),
               'ambiguous_false_proposals': sum(r['decision'] == 'propose' and r['kind'] == 'ambiguous' for r in output)}
    return output, metrics


def run_protocol(encoder, bank, splits, cache_dir, output):
    # Test descriptors are not requested until policy.json has been written.
    calibration = visual.evaluate(encoder, bank, splits['calibration'], cache_dir)
    save(output/'calibration-ranking.json', calibration)
    policy = fit_rule(calibration)
    policy.update(descriptor_namespace=encoder.namespace,
                  calibration_ranking_sha256=visual.sha(output/'calibration-ranking.json'),
                  heldout_used_for_selection=False)
    save(output/'policy.json', policy)
    policy_sha = visual.sha(output/'policy.json')
    cal_scored, cal_metrics = apply_and_score(calibration, policy)
    save(output/'calibration-results.json', cal_scored)
    print('Règle enregistrée avant le test : '+json.dumps(policy['rule']), flush=True)
    heldout = visual.evaluate(encoder, bank, splits['heldout'], cache_dir)
    if visual.sha(output/'policy.json') != policy_sha:
        raise ValueError('Policy changed during held-out inference.')
    test_scored, test_metrics = apply_and_score(heldout, policy)
    save(output/'heldout-results.json', test_scored)
    return {'policy_sha256': policy_sha, 'rule': policy['rule'], 'calibration': cal_metrics,
            'heldout': test_metrics, 'rule_frozen_before_heldout_inference': True}


def worker(output):
    started = time.perf_counter()
    summary = {'status': 'failed', 'version': VERSION, 'automatic_identity_gate': False,
               'inference_provider': 'CPUExecutionProvider', 'reference_count': 30,
               'original_photo_pipeline_modified': False, 'scope': 'Small fixed pilot calibration/test split; no general accuracy guarantee.',
               'script_sha256': {n: visual.sha(ROOT/n) for n in ['evaluate_abstention.py', 'visual_refs.py', 'evaluate_visual_refs.py']}}
    try:
        dataset = ROOT/'abstention-v1'
        bank, _ = visual.expanded_bank(ROOT/'visual-references-v1', ROOT/'visual-references-v2', ROOT/'evaluation-v1')
        manifest, splits = verify_splits(dataset, bank)
        save(output/'dataset-manifest.json', manifest)
        for split in ['calibration', 'heldout']:
            save(output/(split+'-metadata.json'), [row for row, _ in splits[split]])
        summary['dependencies'] = {name: importlib.metadata.version(name) for name in ['numpy', 'Pillow', 'onnxruntime']}
        spec = visual.fetch_model(ROOT/'models/dinov2-small-onnx', evidence_dir=output)
        save(output/'model-lock.json', spec)
        encoder = visual.Encoder(ROOT/'models/dinov2-small-onnx', spec)
        result = run_protocol(encoder, bank, splits, ROOT/'models/visual-descriptor-cache', output)
        summary.update(status='complete', descriptor_namespace=encoder.namespace,
                       dataset_manifest_sha256=visual.sha(dataset/'manifest.json'), **result)
        print('Résultats du test réservé : '+json.dumps(result['heldout'], ensure_ascii=False), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        summary['error'] = str(error) or 'Interrupted'
        (output/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('Évaluation arrêtée : '+summary['error'], flush=True)
    finally:
        summary['total_wall_seconds'] = time.perf_counter()-started
        save(output/'summary.json', summary)
        (output/'rapport.md').write_text(
            '# Abstention visuelle — essai pilote\n\n'
            'La règle score/écart est choisie seulement sur la calibration, puis enregistrée et figée avant les inférences du test. '
            'Elle peut refuser toutes les images si aucune règle utile ne satisfait la contrainte sur la calibration. '
            'Zéro erreur observée sur un petit échantillon ne garantit pas zéro erreur future.\n\n'
            'Lire policy.json, calibration-results.json et heldout-results.json. Un refus ne prouve pas que le monument est hors catalogue. '
            'Une proposition n’est pas une identité confirmée. Le parcours photo/histoire reste inchangé. '
            'Si les seuils, références ou le modèle sont modifiés après lecture du test, ce test devient un jeu de développement ; '
            'un nouveau jeu réservé sera nécessaire. Aucune image ni aucun poids n’est inclus dans le rapport.\n', encoding='utf-8')
        archive(output)
    return 0 if summary['status'] == 'complete' else 1
