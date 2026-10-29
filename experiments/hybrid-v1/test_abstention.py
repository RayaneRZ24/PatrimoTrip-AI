import copy
import json
import pathlib
import shutil
import tempfile
import unittest
import zipfile
from unittest import mock

import evaluate_abstention as abst
import visual_refs as visual

ROOT = pathlib.Path(__file__).resolve().parent


def row(asset, expected, first='A', score=0.8, runner=0.2, kind='known'):
    return {'asset_id': asset, 'expected_site_id': expected, 'kind': kind,
            'top3_candidates': [{'site_id': first, 'cosine_similarity': score},
                                {'site_id': 'B' if first != 'B' else 'A', 'cosine_similarity': runner}]}


class AbstentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, _ = visual.expanded_bank(ROOT/'visual-references-v1', ROOT/'visual-references-v2', ROOT/'evaluation-v1')

    def test_real_splits_are_disjoint_from_old_data_and_photographer_groups(self):
        _, splits = abst.verify_splits(ROOT/'abstention-v1', self.bank)
        self.assertEqual([len(splits[k]) for k in ['calibration', 'heldout']], [20, 20])
        groups = [{r['photographer_group'] for r, _ in splits[s]} for s in ['calibration', 'heldout']]
        self.assertFalse(groups[0] & groups[1])

    def copy_dataset(self, temp):
        root = pathlib.Path(temp)
        dataset = root/'abstention-v1'; shutil.copytree(ROOT/'abstention-v1', dataset)
        for folder in ['visual-references-v1', 'visual-references-v2']:
            (root/folder).mkdir(); shutil.copyfile(ROOT/folder/'references.json', root/folder/'references.json')
        return dataset

    def update_split(self, dataset, split, rows):
        abst.save(dataset/(split+'.json'), rows)
        manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
        manifest['metadata_sha256'][split+'.json'] = visual.sha(dataset/(split+'.json'))
        abst.save(dataset/'manifest.json', manifest)

    def test_validation_rejects_source_and_photographer_leakage_even_with_new_metadata_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = self.copy_dataset(temp)
            calibration = json.loads((dataset/'calibration.json').read_text(encoding='utf-8'))
            original = json.loads((dataset/'heldout.json').read_text(encoding='utf-8'))
            for problem in ['photographer', 'source', 'old_source', 'label', 'outside_subject']:
                rows = copy.deepcopy(original)
                if problem == 'photographer': rows[0]['photographer_group'] = calibration[0]['photographer_group']
                elif problem == 'source': rows[0]['source_page'] = calibration[0]['source_page']
                elif problem == 'old_source': rows[0]['source_page'] = self.bank[0][0]['source_page']
                elif problem == 'label': rows[-1]['expected_site_id'] = 'Q243'
                else: rows[10]['subject'] = calibration[10]['subject']
                self.update_split(dataset, 'heldout', rows)
                with self.subTest(problem=problem), self.assertRaises(ValueError): abst.verify_splits(dataset, self.bank)

    def test_rule_uses_calibration_errors_and_keeps_both_known_and_unknown_samples(self):
        calibration = [row('c1', 'A'), row('c2', 'unknown', score=0.7, runner=0.2, kind='outside')]
        policy = abst.fit_rule(calibration)
        scored, metrics = abst.apply_and_score(calibration, policy)
        self.assertEqual(metrics['correct_proposals'], 1)
        self.assertEqual(metrics['wrong_proposals'], 0)
        self.assertEqual(scored[1]['decision'], 'abstain')
        self.assertTrue(all(not r['identity_confirmed'] for r in scored))
        with self.assertRaises(ValueError): abst.fit_rule([])
        with self.assertRaises(ValueError): abst.fit_rule([calibration[0]])

    def test_reject_all_is_retained_when_wrong_candidate_dominates_correct_candidate(self):
        calibration = [row('good', 'A', score=0.7, runner=0.3),
                       row('wrong', 'unknown', score=0.95, runner=0.05, kind='outside')]
        policy = abst.fit_rule(calibration)
        self.assertTrue(policy['rule']['reject_all'])
        _, metrics = abst.apply_and_score(calibration, policy)
        self.assertEqual(metrics['proposals'], 0)
        self.assertIsNone(metrics['empirical_precision_among_proposals'])
        self.assertEqual(metrics['known_abstentions'], 1)

    def test_bad_scores_are_not_treated_as_safe_abstention_or_confidence(self):
        for value in [float('nan'), float('inf'), True, 1.2]:
            with self.subTest(value=value), self.assertRaises(ValueError): abst.features(row('bad', 'A', score=value))
        with self.assertRaises(ValueError): abst.features(row('bad', 'A', score=0.2, runner=0.7))
        bad = row('bad', 'A'); bad['top3_candidates'][1]['site_id'] = 'A'
        with self.assertRaises(ValueError): abst.features(bad)

    def test_policy_exists_before_heldout_inference_and_is_not_retuned_on_failure(self):
        calibration = [row('c1', 'A'), row('c2', 'unknown', score=0.2, runner=0.1, kind='outside')]
        heldout = [row('t1', 'unknown', score=0.99, runner=0.01, kind='outside')]
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp); snapshot = {}
            def inference(encoder, bank, assets, cache):
                if assets == ['calibration']: return calibration
                self.assertEqual(assets, ['heldout'])
                self.assertTrue((output/'policy.json').is_file())
                snapshot['hash'] = visual.sha(output/'policy.json')
                return heldout
            encoder = mock.Mock(namespace='synthetic-only')
            with mock.patch.object(abst.visual, 'evaluate', side_effect=inference):
                result = abst.run_protocol(encoder, [], {'calibration': ['calibration'], 'heldout': ['heldout']}, output/'cache', output)
            self.assertEqual(result['policy_sha256'], snapshot['hash'])
            self.assertEqual(result['heldout']['wrong_proposals'], 1)
            self.assertEqual(result['calibration']['wrong_proposals'], 0)
            self.assertTrue(result['rule_frozen_before_heldout_inference'])
            self.assertFalse(json.loads((output/'policy.json').read_text())['heldout_used_for_selection'])

    def test_policy_modification_during_test_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp)
            calibration = [row('c', 'A'), row('u', 'unknown', score=0.1, runner=0, kind='outside')]
            def inference(encoder, bank, assets, cache):
                if assets == ['c']: return calibration
                (output/'policy.json').write_text('{}')
                return [row('test', 'A')]
            with mock.patch.object(abst.visual, 'evaluate', side_effect=inference), self.assertRaises(ValueError):
                abst.run_protocol(mock.Mock(namespace='synthetic'), [], {'calibration': ['c'], 'heldout': ['t']}, output/'cache', output)

    def test_worker_saves_failure_report_without_running_model_on_invalid_data(self):
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp)
            with mock.patch.object(abst, 'verify_splits', side_effect=ValueError('synthetic invalid dataset')), mock.patch.object(abst.visual, 'fetch_model') as fetch:
                self.assertEqual(abst.worker(output), 1)
            fetch.assert_not_called()
            self.assertEqual(json.loads((output/'summary.json').read_text())['status'], 'failed')
            self.assertTrue((output/'report.zip').is_file())

    def test_complete_worker_reports_heldout_errors_and_excludes_image_bytes(self):
        def inference(encoder, bank, assets, cache):
            results = []
            for sample, _ in assets:
                known = sample['expected_site_id'] != 'unknown'
                first = sample['expected_site_id'] if known else 'Q243'
                score = 0.8 if known else 0.2 if sample['split'] == 'calibration' else 0.99
                results.append(row(sample['asset_id'], sample['expected_site_id'], first, score, 0.01, sample['kind']))
            return results
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp)
            with mock.patch.object(abst.visual, 'fetch_model', return_value={'synthetic': True}), \
                 mock.patch.object(abst.visual, 'Encoder', return_value=mock.Mock(namespace='synthetic-only')), \
                 mock.patch.object(abst.visual, 'evaluate', side_effect=inference), \
                 mock.patch.object(abst.importlib.metadata, 'version', return_value='synthetic-test'):
                self.assertEqual(abst.worker(output), 0)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['calibration']['wrong_proposals'], 0)
            self.assertEqual(summary['heldout']['unknown_false_proposals'], 10)
            self.assertTrue(summary['rule_frozen_before_heldout_inference'])
            self.assertFalse(summary['automatic_identity_gate'])
            with zipfile.ZipFile(output/'report.zip') as z:
                self.assertIn('heldout-results.json', z.namelist())
                self.assertFalse(any(n.endswith(('.jpg', '.onnx')) for n in z.namelist()))


if __name__ == '__main__': unittest.main()
