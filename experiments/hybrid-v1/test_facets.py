import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import evaluate_retrieval
import facet_rag as facets
import hybrid
import photo_local as photo
from run_local import query_id

ROOT = pathlib.Path(__file__).resolve().parent


class FacetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = hybrid.Corpus(ROOT)
        cls.sites = {s['site_id']: s for s in json.loads((ROOT/'corpus/sites.json').read_text(encoding='utf-8'))}
        cls.manifest = evaluate_retrieval.load_cases(cls.corpus, cls.sites)

    def test_split_is_bounded_and_does_not_inject_expected_facts(self):
        case = self.manifest['cases'][0]
        plan = facets.plan_question(case['question'], 'tour Eiffel')
        self.assertEqual([f['kind'] for f in plan['facets']], ['event', 'commemoration'])
        queries = ' '.join(f['query'] for f in plan['facets'])
        for fact in ['1889', 'Révolution', 'centenaire']:
            self.assertNotIn(fact, queries)
        self.assertFalse(facets.plan_question('Quel est le tarif aujourd’hui ?', 'tour Eiffel')['compound'])
        overflow = 'Qui a conçu le monument ? Quand a-t-il été inauguré ? Où est-il ? Quel est son prix ?'
        # Four demands are kept together, never silently truncated to three.
        fallback = facets.plan_question(overflow, 'tour Eiffel')
        self.assertFalse(fallback['compound'])
        self.assertEqual(fallback['question'], overflow)
        self.assertEqual(len(self.manifest['cases']), 6)

    def cache(self, folder):
        cache = hybrid.VectorCache(folder/'vectors.sqlite3', 'SYNTHETIC-NOT-REAL-EMBEDDINGS', dimensions=3)
        for pid in self.corpus.passages:
            cache.put('document', pid, 'test', [1, 0, 0], 0)
        for case in self.manifest['cases']:
            plan = facets.plan_question(case['question'], self.sites[case['site_id']]['name'])
            for q in [plan['base_query']]+facets.extra_queries(plan):
                cache.put('query', query_id(q), hybrid.query_input(q), [1, 0, 0], 0)
        return cache

    def test_per_facet_candidates_remain_scoped_bounded_and_keep_a_known_commemoration(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = self.cache(pathlib.Path(temp))
            case = self.manifest['cases'][0]
            plan = facets.plan_question(case['question'], 'tour Eiffel')
            evidence, trace = facets.retrieve(self.corpus, 'Q243', 'tour Eiffel', plan, cache)
            self.assertLessEqual(len(evidence), 4)
            self.assertLessEqual(sum(len(e['text']) for e in evidence), 4800)
            self.assertTrue(all(e['site_id'] == 'Q243' for e in evidence))
            self.assertTrue(any('centenaire de la Révolution française' in e['text'] for e in evidence))
            self.assertEqual(len(trace['per_facet']), 2)

    def test_partial_answer_requires_sources_for_answered_part_and_explicit_missing_status(self):
        case = self.manifest['cases'][-1]
        plan = facets.plan_question(case['question'], 'tour Eiffel')
        evidence = self.corpus.context([case['facets'][0]['known_evidence'][0]['passage_id']])
        draft = {'answers': [
            {'facet_id': 'F1', 'status': 'answered', 'claims': [{'text': 'Réponse de test.', 'citations': ['S1']}]},
            {'facet_id': 'F2', 'status': 'insufficient_sources', 'claims': [{'text': 'Tarif actuel absent des extraits.', 'citations': []}]}]}
        answer, statuses, review = facets.render_answers(draft, plan, evidence)
        self.assertEqual(answer['status'], 'answered')
        self.assertEqual(statuses, {'F1': 'answered', 'F2': 'insufficient_sources'})
        self.assertEqual(answer['citations'], ['S1'])
        self.assertEqual(review[1]['facet_status'], 'insufficient_sources')
        for change in ['no_fact_source', 'duplicate_facet', 'missing_facet', 'unknown_source']:
            altered = copy.deepcopy(draft)
            if change == 'no_fact_source': altered['answers'][0]['claims'][0]['citations'] = []
            elif change == 'duplicate_facet': altered['answers'][1]['facet_id'] = 'F1'
            elif change == 'missing_facet': altered['answers'].pop()
            else: altered['answers'][0]['claims'][0]['citations'] = ['S99']
            with self.subTest(change=change), self.assertRaises(ValueError):
                facets.render_answers(altered, plan, evidence)

    def test_facet_api_keeps_questions_in_user_data_and_rejects_truncation(self):
        plan = facets.plan_question('Qui a conçu USER_SENTINEL et quand a-t-il été inauguré ?', 'palais Garnier')
        pid = self.manifest['cases'][2]['facets'][0]['known_evidence'][0]['passage_id']
        evidence = self.corpus.context([pid])
        client = mock.Mock()
        client.request.side_effect = [{'data': [{'id': 'fake'}]}, {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}]
        result = facets.ask(plan['base_query'], plan, evidence, client, photo.PHOTO_SYSTEM)
        self.assertFalse(result['reference_checks_passed'])
        payload = client.request.call_args_list[-1].args[1]
        self.assertNotIn('USER_SENTINEL', payload['messages'][0]['content'])
        self.assertIn('USER_SENTINEL', payload['messages'][1]['content'])
        self.assertEqual(payload['response_format']['json_schema']['schema']['properties']['answers']['minItems'], 2)

    def test_comparison_flow_with_synthetic_vectors_and_fake_model_does_not_claim_accuracy(self):
        def request(path, payload=None):
            if path == '/v1/models': return {'data': [{'id': 'fake'}]}
            props = payload['response_format']['json_schema']['schema']['properties']
            if 'answers' in props:
                user = json.loads(payload['messages'][1]['content'])
                answers = []
                for part in user['aspects']:
                    missing = facets.facet_kind(part['question']) in {'price', 'hours'}
                    answers.append({'facet_id': part['facet_id'], 'status': 'insufficient_sources' if missing else 'answered',
                                    'claims': [{'text': 'Simulation technique uniquement.', 'citations': [] if missing else ['S1']}]})
                value = {'answers': answers}
            else:
                question = payload['messages'][1]['content'].split('\nEXTRAITS DOCUMENTAIRES')[0]
                missing = 'aujourd' in question
                value = {'status': 'insufficient_sources' if missing else 'answered',
                         'claims': [{'text': 'Simulation technique uniquement.', 'citations': [] if missing else ['S1']}]}
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        with tempfile.TemporaryDirectory() as temp:
            folder = pathlib.Path(temp)
            cache = self.cache(folder)
            client = mock.Mock(); client.request.side_effect = request
            summary = evaluate_retrieval.compare(self.corpus, self.sites, self.manifest, cache, client, folder)
            self.assertEqual(summary['baseline']['cases'], 6)
            self.assertEqual(summary['per_facet']['cases'], 6)
            self.assertEqual(summary['per_facet']['reference_checks_passed'], 6)
            self.assertEqual(summary['per_facet']['missing_facets_with_expected_status'], 3)
            self.assertEqual(summary['per_facet']['missing_facets_with_explicit_status'], 3)
            rows = json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
            self.assertTrue(all(r['automatic_factual_accuracy'] == 'not_measured' for r in rows))
            review = json.loads((folder/'review.json').read_text(encoding='utf-8'))
            self.assertEqual(len(review), 16)
            self.assertTrue(all(r['support'] == 'not_reviewed' for r in review))

    def test_photo_compound_path_prepares_facet_queries_and_uses_sequential_profiles(self):
        label = next(k for k, v in photo.canonical_sites(self.sites).items() if v == 'Q243')
        raw = {'site_name': label, 'city': 'Paris', 'country': 'France', 'status': 'identified',
               'observations': ['Structure visible.'], 'hypothesis': 'Tour Eiffel.'}
        case = self.manifest['cases'][-1]
        evidence = self.corpus.context([case['facets'][0]['known_evidence'][0]['passage_id']])
        client, manager = mock.Mock(), mock.Mock()
        manager.start.return_value = 'http://127.0.0.1:8080'
        def request(path, payload=None):
            if path == '/v1/models': return {'data': [{'id': 'fake'}]}
            if 'site_name' in payload['response_format']['json_schema']['schema']['properties']:
                value = raw
            else:
                value = {'answers': [
                    {'facet_id': 'F1', 'status': 'answered', 'claims': [{'text': 'Simulation sourcée.', 'citations': ['S1']}]},
                    {'facet_id': 'F2', 'status': 'insufficient_sources', 'claims': [{'text': 'Tarif absent.', 'citations': []}]}]}
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        client.request.side_effect = request
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo.rag, 'LocalModel', return_value=client), \
             mock.patch.object(photo, 'prepare_vectors') as prepare, mock.patch.object(photo.facet_rag, 'retrieve', return_value=(evidence, {})):
            result = photo.pipeline(self.corpus, self.sites, b'fake-image', {'mime': 'image/png'}, case['question'],
                                    manager, pathlib.Path(temp), mock.Mock())
            self.assertEqual(manager.start.call_args_list, [mock.call('generation', generation_vision=True),
                                                          mock.call('generation', generation_vision=False)])
            manager.stop.assert_called_once()
            self.assertEqual(len(prepare.call_args.kwargs['extra_queries']), 2)
            self.assertEqual(result['facet_statuses']['F2'], 'insufficient_sources')
            self.assertFalse(result['all_facets_answered'])
            self.assertTrue(result['reference_checks_passed'])


if __name__ == '__main__':
    unittest.main()
