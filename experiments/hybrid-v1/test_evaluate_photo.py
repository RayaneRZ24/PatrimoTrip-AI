import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import evaluate_photo as evaluation
import hybrid
from run_local import save

ROOT = pathlib.Path(__file__).resolve().parent


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = hybrid.Corpus(ROOT)
        cls.manifest, cls.assets = evaluation.verify_dataset(evaluation.DATASET, cls.corpus)
        cls.cases = {c['case_id']: c for c in cls.manifest['cases']}
        cls.sites = {s['site_id']: s for s in json.loads((ROOT/'corpus/sites.json').read_text(encoding='utf-8'))}

    def test_frozen_assets_annotations_and_corruption_detection(self):
        self.assertEqual(len(self.assets), 10)
        self.assertEqual(len(self.manifest['cases']), 12)
        self.assertTrue(set(a['sha256'] for a in self.assets.values()).isdisjoint(self.manifest['previous_image_sha256']))
        with tempfile.TemporaryDirectory() as temp:
            folder = pathlib.Path(temp)/'dataset'
            shutil.copytree(evaluation.DATASET, folder)
            image = folder/self.assets['a01']['image']
            data = image.read_bytes(); image.write_bytes(data[:-1]+bytes([data[-1]^1]))
            with self.assertRaises(ValueError):
                evaluation.verify_dataset(folder, self.corpus)

    def test_rejection_requires_no_history_and_wrong_identity_cannot_pass_price_test(self):
        case, asset = self.cases['q10'], self.assets['a08']
        result = {'identification': {'site_id': 'unknown', 'status': 'out_of_scope'}, 'historical_answer_generated': False}
        self.assertTrue(evaluation.score_case(case, asset, result, [])['safe_rejection_without_history'])
        result['historical_answer_generated'] = True
        self.assertFalse(evaluation.score_case(case, asset, result, [])['safe_rejection_without_history'])
        wrong = {'identification': {'site_id': 'Q28785', 'status': 'identified'},
                 'answer_status': 'insufficient_sources', 'reference_checks_passed': True}
        score = evaluation.score_case(self.cases['q06'], self.assets['a01'], wrong, [])
        self.assertFalse(score['format_and_expected_status_on_correct_identity'])
        # Even the right passage identifier does not suffice if its supporting quote was truncated.
        a = self.cases['q01']['known_evidence'][0]
        score = evaluation.score_case(self.cases['q01'], self.assets['a01'], wrong,
                                      [{'passage_id': a['passage_id'], 'text': 'Unrelated truncated fragment'}])
        self.assertFalse(score['known_evidence_in_context'])

    def test_summary_uses_first_image_attempt_and_keeps_failed_case_denominators(self):
        base = {'case_id': 'q01', 'asset_id': 'a01', 'asset_kind': 'standard', 'case_kind': 'documented',
                'predicted_site_id': 'Q243', 'visual_status': 'identified', 'identity_exact': True, 'seconds': 1}
        rows = [base, {**base, 'case_id': 'q06', 'case_kind': 'missing_current', 'predicted_site_id': 'Q28785', 'identity_exact': False},
                {'case_id': 'q02', 'asset_id': 'a02', 'asset_kind': 'standard', 'case_kind': 'documented', 'error': 'failed', 'seconds': 1}]
        report = evaluation.summarize(rows)
        self.assertEqual(report['standard_images'], 2)
        self.assertEqual(report['standard_identity_exact'], 1)
        self.assertEqual(report['case_errors'], 1)
        self.assertEqual(report['repeated_image_prediction_disagreements'], 1)
        self.assertEqual(report['automatic_factual_accuracy'], 'not_measured')

    def test_regression_selection_preserves_original_manifest(self):
        original = json.dumps(self.manifest, sort_keys=True)
        selected = evaluation.select_cases(self.manifest, True)
        self.assertEqual([c['case_id'] for c in selected['cases']], ['q01', 'q03', 'q04', 'q05', 'q11', 'q12'])
        self.assertEqual(json.dumps(self.manifest, sort_keys=True), original)
        self.assertEqual(len(self.manifest['cases']), 12)

    def test_before_after_comparison_requires_matching_dataset_and_asset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            baseline = root/'evaluation-20261028'; baseline.mkdir()
            output = root/'evaluation-identity-new'; output.mkdir()
            save(baseline/'summary.json', {'status': 'complete', 'cases_planned': 12,
                                          'manifest_sha256': 'manifest', 'assets_sha256': 'assets'})
            before = [{'case_id': c['case_id'], 'asset_id': c['asset_id'], 'predicted_site_id': 'Q188856',
                       'visual_status': 'identified'} for c in self.manifest['cases']]
            save(baseline/'cases-summary.json', before)
            after = [{'case_id': 'q03', 'asset_id': 'a03', 'predicted_site_id': 'Q187840', 'visual_status': 'identified'}]
            self.assertIsNone(evaluation.previous_comparison(root, output, after, 'wrong-manifest', 'assets'))
            self.assertEqual(evaluation.previous_comparison(root, output, after, 'manifest', 'assets'), baseline.name)
            comparison = json.loads((output/'identity-before-after.json').read_text(encoding='utf-8'))
            self.assertEqual(comparison['cases'][0]['before']['predicted_site_id'], 'Q188856')
            self.assertEqual(comparison['cases'][0]['after']['predicted_site_id'], 'Q187840')

    def test_full_suite_simulation_isolates_gold_continues_errors_and_leaves_review_pending(self):
        calls = []
        def fake_pipeline(corpus, sites, data, metadata, question, manager, folder, resources):
            # Signature intentionally excludes confirmed_site/gold evidence; this is a simulation, not model accuracy.
            aid = pathlib.Path(metadata['filename']).stem
            case = next(c for c in self.manifest['cases'] if c['asset_id'] == aid and c['question'] == question)
            calls.append(case['case_id'])
            if case['case_id'] == 'q03':
                raise RuntimeError('Simulated model failure')
            asset = self.assets[aid]
            if asset['kind'] in {'outside', 'detail', 'ambiguous'}:
                return {'status': 'out_of_scope' if asset['kind'] == 'outside' else 'needs_clarification',
                        'identification': {'site_id': 'unknown', 'status': 'out_of_scope' if asset['kind'] == 'outside' else 'uncertain'},
                        'historical_answer_generated': False}
            pid = case['known_evidence'][0]['passage_id'] if case['known_evidence'] else next(
                pid for pid, p in corpus.passages.items() if p['site_id'] == asset['expected_site_id'])
            evidence = corpus.context([pid])
            status = case['expected_answer_status']
            citations = ['S1'] if status == 'answered' else []
            save(folder/'retrieval.json', {'evidence': evidence})
            save(folder/'answer.json', {'reference_checks_passed': True, 'evidence': evidence,
                 'answer': {'status': status}, 'claims_response': {'claims': [{'text': 'Simulation technique sans validation factuelle.', 'citations': citations}]}})
            return {'status': 'proposed_identification_with_answer',
                    'identification': {'site_id': asset['expected_site_id'], 'status': 'identified'},
                    'answer_status': status, 'reference_checks_passed': True, 'historical_answer_generated': True}
        manager, monitors = mock.Mock(), mock.Mock()
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(evaluation.photo, 'pipeline', side_effect=fake_pipeline):
            folder = pathlib.Path(temp)
            result = evaluation.run_suite(self.corpus, self.sites, self.manifest, self.assets, evaluation.DATASET,
                                          manager, folder, resource_factory=monitors)
            self.assertEqual(calls, [c['case_id'] for c in self.manifest['cases']])
            self.assertEqual(result['cases_attempted'], 12)
            self.assertEqual(result['case_errors'], 1)
            self.assertEqual(result['standard_identity_exact'], 4)
            self.assertEqual(result['standard_images'], 5)
            self.assertEqual(result['missing_current_abstentions_on_correct_identity'], 2)
            review = json.loads((folder/'review.json').read_text(encoding='utf-8'))
            self.assertEqual(len(review), 6)
            self.assertTrue(all(row['support'] == 'not_reviewed' for row in review))
            self.assertEqual(monitors.return_value.stop.call_count, 12)
            manager.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
