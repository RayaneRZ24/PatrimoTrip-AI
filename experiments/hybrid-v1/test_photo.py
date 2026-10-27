import base64
import json
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

import hybrid
import photo_local as photo
import rag_local as rag
import run_local
import runtime

ROOT = pathlib.Path(__file__).resolve().parent
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQmcAAAAASUVORK5CYII=')


class PhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = hybrid.Corpus(ROOT)
        cls.sites = {s['site_id']: s for s in json.loads((ROOT/'corpus/sites.json').read_text(encoding='utf-8'))}

    def proposal(self, status='identified', site_id='Q243'):
        return {'site_id': site_id, 'status': status, 'observations': ['Structure métallique visible.'], 'hypothesis': 'Une tour métallique.'}

    def test_image_dimensions_limits_and_supported_headers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp)/'photo.png'
            path.write_bytes(PNG)
            data, metadata = photo.load_image(path)
            self.assertEqual((metadata['width'], metadata['height'], metadata['mime']), (1, 1, 'image/png'))
            self.assertEqual(data, PNG)
            # JPEG SOF header parsing; the inference server remains responsible for full decoding.
            path.write_bytes(b'\xff\xd8\xff\xc0'+struct.pack('>HBHHB', 8, 8, 600, 800, 1))
            _, metadata = photo.load_image(path)
            self.assertEqual((metadata['width'], metadata['height']), (800, 600))
            for invalid in [b'not an image', b'\xff\xd8\xff\xc0\x00\x01',
                            PNG[:16]+struct.pack('>II', 4000, 3000)+PNG[24:]]:
                path.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    photo.load_image(path)
            path.write_bytes(PNG)
            with mock.patch.object(photo, 'MAX_BYTES', 10), self.assertRaises(ValueError):
                photo.load_image(path)

    def test_prediction_rejects_inconsistent_unknown_and_bad_shapes(self):
        self.assertEqual(photo.validate_prediction(self.proposal(), self.sites)['site_id'], 'Q243')
        for change in [{'status': 'uncertain'}, {'site_id': 'unknown'}, {'site_id': 'Q999'}, {'observations': []},
                       {'observations': ['a']*4}, {'hypothesis': ''}, {'extra': 'field'}, {'observations': ['x'*251]}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                photo.validate_prediction({**self.proposal(), **change}, self.sites)
        for status in ['uncertain', 'out_of_scope']:
            self.assertEqual(photo.validate_prediction(self.proposal(status, 'unknown'), self.sites)['status'], status)

    def test_scoped_hybrid_retrieval_cannot_include_other_monuments(self):
        corpus = self.corpus
        question = 'À propos de la tour Eiffel : pourquoi la construire ?'
        with tempfile.TemporaryDirectory() as temp:
            cache = hybrid.VectorCache(pathlib.Path(temp)/'vectors.sqlite3', 'fake', dimensions=3)
            for pid, passage in corpus.passages.items():
                cache.put('document', pid, 'fixture', [1, 0, 0], 0)
            cache.put('query', run_local.query_id(question), hybrid.query_input(question), [1, 0, 0], 0)
            evidence, rankings = photo.scoped_retrieval(corpus, 'Q243', question, cache)
            self.assertTrue(evidence)
            self.assertTrue(all(e['site_id'] == 'Q243' for e in evidence))
            self.assertTrue(all(corpus.passages[pid]['site_id'] == 'Q243' for r in rankings.values() for pid in r))
            self.assertEqual(corpus.bm25(question, site_id='nonexistent'), [])

    def test_complete_photo_path_switches_profiles_and_separates_visual_claims(self):
        corpus = self.corpus
        evidence = corpus.context(corpus.bm25('tour Eiffel', site_id='Q243'))
        client, manager, resources = mock.Mock(), mock.Mock(), mock.Mock()
        manager.start.return_value = 'http://127.0.0.1:8080'
        requests = []
        def request(path, payload=None):
            if path == '/v1/models':
                return {'data': [{'id': 'fake-Qwen3.5-4B'}]}
            requests.append(payload)
            props = payload['response_format']['json_schema']['schema']['properties']
            if 'site_id' in props:
                value = self.proposal()
            else:
                value = {'status': 'answered', 'claims': [{'text': 'Réponse simulée uniquement.', 'citations': ['S1']}]}
            return {'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}]}
        client.request.side_effect = request
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo.rag, 'LocalModel', return_value=client), \
             mock.patch.object(photo, 'prepare_vectors') as prepare, \
             mock.patch.object(photo, 'scoped_retrieval', return_value=(evidence, {'hybrid_rrf': []})):
            folder = pathlib.Path(temp)
            result = photo.pipeline(corpus, self.sites, PNG, {'mime': 'image/png'}, 'Pourquoi ce monument ?', manager, folder, resources)
            self.assertEqual(manager.start.call_args_list, [mock.call('generation', generation_vision=True),
                                                          mock.call('generation', generation_vision=False)])
            manager.stop.assert_called_once()
            prepare.assert_called_once()
            self.assertEqual(result['status'], 'proposed_identification_with_answer')
            self.assertFalse(result['identity_verified'])
            self.assertEqual(result['identity_origin'], 'visual_proposal_not_verified')
            self.assertTrue(result['reference_checks_passed'])
            self.assertTrue(requests[0]['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
            self.assertNotIn('Structure métallique visible.', json.dumps(requests[1], ensure_ascii=False))
            self.assertNotIn('data:image', (folder/'vision.json').read_text())

    def test_abstention_and_out_of_scope_do_not_retrieve_or_generate(self):
        for status in ['uncertain', 'out_of_scope']:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp, \
                 mock.patch.object(photo, 'identify', return_value=self.proposal(status, 'unknown')), \
                 mock.patch.object(photo, 'prepare_vectors') as prepare, mock.patch.object(photo.rag, 'ask') as ask:
                manager = mock.Mock()
                manager.start.return_value = 'http://127.0.0.1:8080'
                result = photo.pipeline(self.corpus, self.sites, PNG, {}, 'Question', manager, pathlib.Path(temp), mock.Mock())
                self.assertFalse(result['historical_answer_generated'])
                self.assertEqual(result['status'], 'needs_clarification' if status == 'uncertain' else 'out_of_scope')
                prepare.assert_not_called(); ask.assert_not_called(); manager.stop.assert_called_once()
                manager.start.assert_called_once_with('generation', generation_vision=True)

    def test_invalid_vision_stops_worker_and_never_retrieves(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'identify', side_effect=ValueError('bad JSON')), \
             mock.patch.object(photo, 'prepare_vectors') as prepare:
            manager = mock.Mock()
            manager.start.return_value = 'http://127.0.0.1:8080'
            with self.assertRaises(ValueError):
                photo.pipeline(self.corpus, self.sites, PNG, {}, 'Question', manager, pathlib.Path(temp), mock.Mock())
            manager.stop.assert_called_once(); prepare.assert_not_called()

    def test_user_identity_bypasses_vision_but_does_not_claim_verification(self):
        evidence = self.corpus.context(self.corpus.bm25('Panthéon', site_id='Q188856'))
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'identify') as identify, \
             mock.patch.object(photo, 'prepare_vectors'), mock.patch.object(photo, 'scoped_retrieval', return_value=(evidence, {})), \
             mock.patch.object(photo.rag, 'ask', return_value={'answer': {'status': 'insufficient_sources', 'answer': 'Information absente.', 'citations': []},
                                                            'reference_checks_passed': True, 'model_called': True}):
            manager = mock.Mock()
            manager.start.return_value = 'http://127.0.0.1:8080'
            result = photo.pipeline(self.corpus, self.sites, PNG, {}, 'Tarif actuel ?', manager, pathlib.Path(temp), mock.Mock(), 'Q188856')
            identify.assert_not_called()
            manager.start.assert_called_once_with('generation', generation_vision=False)
            self.assertEqual(result['identity_origin'], 'user_provided')
            self.assertFalse(result['identity_verified'])
            self.assertEqual(result['answer']['status'], 'insufficient_sources')

    def test_truncated_visual_response_is_rejected_before_identification(self):
        client = mock.Mock()
        client.request.side_effect = [{'data': [{'id': 'fake'}]}, {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}]
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            photo.identify(client, PNG, {'mime': 'image/png'}, self.sites, pathlib.Path(temp))

    def test_rejected_answer_is_not_exposed_as_a_valid_explanation(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'prepare_vectors'), \
             mock.patch.object(photo, 'scoped_retrieval', return_value=([], {})), \
             mock.patch.object(photo.rag, 'ask', return_value={'reference_checks_passed': False,
                 'answer': {'answer': 'Invalid response must stay hidden.'}, 'validation_error': 'Bad citations'}):
            manager = mock.Mock()
            manager.start.return_value = 'http://127.0.0.1:8080'
            result = photo.pipeline(self.corpus, self.sites, PNG, {}, 'Question', manager, pathlib.Path(temp), mock.Mock(), 'Q243')
            self.assertEqual(result['status'], 'answer_rejected')
            self.assertIsNone(result['answer'])
            self.assertEqual(result['error'], 'Bad citations')

    def test_embedding_namespace_preserves_existing_user_cache(self):
        metadata = {'sha256': 'e40d4a708d0b324fc133bae547bd64dec79edb885410ef262c1e5c2efb2050b3'}
        self.assertEqual(run_local.embedding_cache_key(metadata, self.corpus),
                         'e532ad62605bf6867ca19fdda868a6ef342781a8f2354f6c53a823ee91a6988b')

    def test_vector_preparation_reuses_index_and_loads_only_missing_query(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = pathlib.Path(temp)
            corpus = mock.Mock(fingerprint='corpus', passages={'p': {'site_id': 'Q243', 'passage_id': 'p', 'document_id': 'd', 'section_path': [], 'text': 'Passage'}},
                               sources={'d': {'title': 'Titre'}})
            metadata = {'sha256': 'modelsha'}
            key = run_local.embedding_cache_key(metadata, corpus)
            cache = hybrid.VectorCache(folder/'vectors.sqlite3', key)
            doc = hybrid.document_input(corpus.passages['p'], corpus.sources['d'])
            vector = [1.0]+[0.0]*1023
            cache.put('document', 'p', doc, vector, 0)
            manager, resources, encoder = mock.Mock(), mock.Mock(), mock.Mock()
            encoder.encode.return_value = vector
            with mock.patch.object(photo, 'ROOT', folder), mock.patch.object(photo.runtime, 'download_embedding', return_value=(folder/'model.gguf', metadata)), \
                 mock.patch.object(photo.hybrid, 'Embeddings', return_value=encoder):
                photo.prepare_vectors(corpus, 'Q243', 'Question', manager, folder, resources)
                encoder.encode.assert_called_once_with(hybrid.query_input('Question'))
                manager.start.assert_called_once_with('embedding', folder/'model.gguf')
                manager.stop.assert_called_once()
                manager.reset_mock(); encoder.reset_mock()
                photo.prepare_vectors(corpus, 'Q243', 'Question', manager, folder, resources)
                encoder.encode.assert_not_called(); manager.start.assert_not_called()

    def replay_fixture(self, folder):
        pid = next(pid for pid, p in self.corpus.passages.items() if p['site_id'] == 'Q243' and p['start'] == 0 and p['source_type'] == 'wikipedia')
        evidence = self.corpus.context([pid])
        question = "À propos du monument « tour Eiffel » : Présente brièvement ce monument et sa fonction d'origine."
        record = {'site_id': 'Q243', 'question': question, 'evidence': evidence, 'rankings': {}}
        previous = {'image': {'filename': 'fixture.png'}, 'question': "Présente brièvement ce monument et sa fonction d'origine.",
                    'identification': self.proposal(), 'identity_origin': 'visual_proposal_not_verified'}
        for name, value in [('retrieval.json', record), ('result.json', previous), ('summary.json', {'corpus_fingerprint': self.corpus.fingerprint})]:
            (folder/name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return record

    def test_replay_uses_original_question_and_excerpts_without_vision_or_embeddings(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'identify') as identify, \
             mock.patch.object(photo, 'prepare_vectors') as prepare, mock.patch.object(photo.rag, 'LocalModel') as factory:
            source = pathlib.Path(temp)/'source'; source.mkdir()
            output = pathlib.Path(temp)/'new'; output.mkdir()
            record = self.replay_fixture(source)
            client = factory.return_value
            client.request.side_effect = [{'data': [{'id': 'fake'}]}, {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                'status': 'answered', 'claims': [{'text': "La tour Eiffel a été construite pour l'Exposition universelle de Paris de 1889.", 'citations': ['S1']}]})}}]}]
            manager = mock.Mock(); manager.start.return_value = 'http://127.0.0.1:8080'
            result = photo.replay_documentary(self.corpus, self.sites, source, manager, output, mock.Mock())
            identify.assert_not_called(); prepare.assert_not_called()
            manager.start.assert_called_once_with('generation', generation_vision=False)
            saved = json.loads((output/'answer.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['question'], record['question'])
            self.assertEqual(saved['evidence'], record['evidence'])
            self.assertEqual(result['status'], 'proposed_identification_with_answer')
            payload = client.request.call_args_list[-1].args[1]
            self.assertIn('réponse est partielle', payload['messages'][0]['content'])
            self.assertEqual(saved['answer']['citations'], ['S1'])

    def test_replay_rejects_modified_excerpt_before_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            source = pathlib.Path(temp)/'source'; source.mkdir()
            output = pathlib.Path(temp)/'new'; output.mkdir()
            record = self.replay_fixture(source)
            record['evidence'][0]['text'] = 'Invented fact not in the corpus.'
            (source/'retrieval.json').write_text(json.dumps(record), encoding='utf-8')
            manager = mock.Mock()
            with self.assertRaises(ValueError):
                photo.replay_documentary(self.corpus, self.sites, source, manager, output, mock.Mock())
            manager.start.assert_not_called()

    def test_valid_abstention_is_not_reported_as_a_documentary_answer(self):
        result = photo.attach_answer({}, {'reference_checks_passed': True, 'model_called': True,
            'answer': {'status': 'insufficient_sources', 'answer': 'Le tarif actuel est absent.', 'citations': []}}, 'tour Eiffel')
        self.assertEqual(result['status'], 'proposed_identification_insufficient_sources')
        self.assertEqual(result['answer_status'], 'insufficient_sources')
        self.assertEqual(result['answer']['citations'], [])


if __name__ == '__main__':
    unittest.main()
