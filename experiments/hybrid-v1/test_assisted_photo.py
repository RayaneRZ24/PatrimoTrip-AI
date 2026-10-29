import copy
import hashlib
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import hybrid
import photo_local as photo
import visual_guard as guard
import evaluate_abstention as abstention

ROOT = pathlib.Path(__file__).resolve().parent
IMAGE = b'synthetic image bytes for subprocess-boundary tests only'
IMAGE_SHA = hashlib.sha256(IMAGE).hexdigest()


def gate_record(accepted=True):
    policy = guard.load_policy()
    score = 0.8 if accepted else 0.4
    ranking = [{'site_id': site, 'cosine_similarity': value, 'references_for_site': 3}
               for site, value in [('Q243', score), ('Q2981', 0.2), ('Q187840', 0.1)]]
    feature = abstention.features({'top3_candidates': ranking})
    return {'status': 'complete', 'version': guard.VERSION, 'image_sha256': IMAGE_SHA,
            'policy_sha256': guard.POLICY_SHA256, 'descriptor_namespace': policy['descriptor_namespace'],
            'reference_manifests_sha256': policy['reference_manifests_sha256'],
            'top3_candidates': ranking, 'features': feature, **abstention.decide(feature, policy['rule'])}


class AssistedPhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = hybrid.Corpus(ROOT)
        cls.sites = {s['site_id']: s for s in json.loads((ROOT/'corpus/sites.json').read_text(encoding='utf-8'))}

    def prediction(self, site='Q243'):
        label = next(k for k, v in photo.canonical_sites(self.sites).items() if v == site)
        return photo.resolve_identity({'site_name': label, 'status': 'identified', 'city': 'Paris', 'country': 'France',
                                       'observations': ['Éléments visibles simulés.'], 'hypothesis': self.sites[site]['name']}, self.sites)

    def test_frozen_policy_matches_bank_and_keeps_pilot_thresholds(self):
        policy = guard.load_policy()
        self.assertEqual(policy['rule'], {'reject_all': False, 'minimum_similarity': 0.7, 'minimum_gap': 0.15})
        self.assertEqual(policy['version'], guard.VERSION)

    def test_changed_policy_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); (root/'visual-policy-v1').mkdir()
            policy = guard.load_policy(); policy['rule']['minimum_similarity'] = 0.1
            (root/'visual-policy-v1/policy.json').write_text(json.dumps(policy))
            with self.assertRaises(ValueError): guard.load_policy(root)

    def test_child_record_is_bound_to_photo_model_policy_and_computed_decision(self):
        record = gate_record()
        guard.validate_record(record, IMAGE_SHA, self.sites)
        for problem in ['photo', 'policy', 'model', 'decision', 'score', 'candidate']:
            changed = copy.deepcopy(record)
            if problem == 'photo': changed['image_sha256'] = 'other'
            elif problem == 'policy': changed['policy_sha256'] = 'other'
            elif problem == 'model': changed['descriptor_namespace'] = 'other'
            elif problem == 'decision': changed['decision'] = 'abstain'
            elif problem == 'score': changed['top3_candidates'][0]['cosine_similarity'] = float('nan')
            else: changed['top3_candidates'][2]['site_id'] = 'unknown'
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                guard.validate_record(changed, IMAGE_SHA, self.sites)

    def test_subprocess_temporary_image_is_outside_report_and_removed_before_return(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); output = root/'report'; output.mkdir()
            python = root/'.visual-venv'/('Scripts/python.exe' if guard.sys.platform == 'win32' else 'bin/python')
            python.parent.mkdir(parents=True); python.write_bytes(b'fake executable')
            (root/'corpus').mkdir(); shutil.copyfile(ROOT/'corpus/sites.json', root/'corpus/sites.json')
            observed = {}
            def run(command, **kwargs):
                image = pathlib.Path(command[command.index('--image')+1])
                self.assertEqual(image.read_bytes(), IMAGE)
                self.assertFalse(image.is_relative_to(output))
                observed['temporary'] = image
                pathlib.Path(command[command.index('--output')+1]).write_text(json.dumps(gate_record()))
                return mock.Mock(returncode=0)
            with mock.patch.object(guard, 'ROOT', root), mock.patch.object(guard.subprocess, 'run', side_effect=run):
                result = guard.run_guard(IMAGE, {'sha256': IMAGE_SHA, 'mime': 'image/jpeg'}, output)
            self.assertEqual(result['decision'], 'propose')
            self.assertFalse(observed['temporary'].exists())
            self.assertFalse(any(p.suffix in ['.jpg', '.png'] for p in output.rglob('*')))

    def test_reference_abstention_does_not_start_qwen_or_retrieval(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'identify') as identify, \
             mock.patch.object(photo, 'prepare_vectors') as prepare, mock.patch.object(photo, 'documentary_answer') as generate:
            manager = mock.Mock()
            result = photo.pipeline(self.corpus, self.sites, IMAGE, {'sha256': IMAGE_SHA}, 'Question',
                                    manager, pathlib.Path(temp), mock.Mock(), visual_gate=lambda *args: gate_record(False))
            self.assertEqual(result['status'], 'needs_clarification')
            self.assertFalse(result['historical_answer_generated'])
            self.assertEqual(result['identification']['site_id'], 'unknown')
            manager.start.assert_not_called(); identify.assert_not_called(); prepare.assert_not_called(); generate.assert_not_called()

    def test_disagreement_stops_before_documentary_generation(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'identify', return_value=self.prediction('Q188856')), \
             mock.patch.object(photo, 'prepare_vectors') as prepare, mock.patch.object(photo, 'documentary_answer') as generate:
            manager = mock.Mock(); manager.start.return_value = 'http://127.0.0.1:8080'
            result = photo.pipeline(self.corpus, self.sites, IMAGE, {'sha256': IMAGE_SHA}, 'Question', manager,
                                    pathlib.Path(temp), mock.Mock(), visual_gate=lambda *args: gate_record())
            self.assertEqual(result['status'], 'needs_clarification')
            self.assertFalse(result['visual_agreement']['matched'])
            self.assertFalse(result['historical_answer_generated'])
            prepare.assert_not_called(); generate.assert_not_called()
            manager.start.assert_called_once_with('generation', generation_vision=True)

    def test_agreement_keeps_independent_qwen_input_and_allows_only_scoped_sources(self):
        events = []
        evidence = self.corpus.context(self.corpus.bm25('tour Eiffel', site_id='Q243'))
        def gate_callback(*args): events.append('descriptor_child_exited'); return gate_record()
        def identify(*args): events.append('qwen_vision'); return self.prediction()
        answer = {'reference_checks_passed': True, 'model_called': True,
                  'answer': {'status': 'answered', 'answer': 'Simulation de texte sourcé. [S1]', 'citations': ['S1']}}
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'identify', side_effect=identify) as vision, \
             mock.patch.object(photo, 'prepare_vectors'), mock.patch.object(photo, 'scoped_retrieval', return_value=(evidence, {})) as retrieve, \
             mock.patch.object(photo, 'documentary_answer', return_value=answer):
            manager = mock.Mock(); manager.start.return_value = 'http://127.0.0.1:8080'
            manager.stop.side_effect = lambda: events.append('stop_owned')
            result = photo.pipeline(self.corpus, self.sites, IMAGE, {'sha256': IMAGE_SHA}, 'Pourquoi ce monument ?', manager,
                                    pathlib.Path(temp), mock.Mock(), visual_gate=gate_callback)
            self.assertEqual(events[:3], ['stop_owned', 'descriptor_child_exited', 'qwen_vision'])
            self.assertEqual(vision.call_args.args[1], IMAGE)
            self.assertEqual(len(vision.call_args.args), 5)  # No descriptor candidate in Qwen inputs.
            self.assertEqual(retrieve.call_args.args[1], 'Q243')
            self.assertTrue(result['visual_agreement']['matched'])
            self.assertFalse(result['identity_verified'])
            self.assertEqual(result['status'], 'proposed_identification_with_answer')

    def test_unavailable_reference_worker_never_falls_back_to_qwen(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'identify') as identify:
            manager = mock.Mock()
            with self.assertRaises(ValueError):
                photo.pipeline(self.corpus, self.sites, IMAGE, {'sha256': IMAGE_SHA}, 'Question', manager,
                               pathlib.Path(temp), mock.Mock(), visual_gate=mock.Mock(side_effect=ValueError('failed child')))
            identify.assert_not_called(); manager.start.assert_not_called()

    def test_explicit_user_identity_is_reported_as_bypass_without_visual_inference(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'identify') as identify, \
             mock.patch.object(photo, 'prepare_vectors'), mock.patch.object(photo, 'scoped_retrieval', return_value=([], {})), \
             mock.patch.object(photo, 'documentary_answer', return_value={'reference_checks_passed': True, 'model_called': False,
                 'answer': {'status': 'insufficient_sources', 'answer': 'Aucune source.', 'citations': []}}):
            callback = mock.Mock()
            result = photo.pipeline(self.corpus, self.sites, IMAGE, {}, 'Question', mock.Mock(), pathlib.Path(temp),
                                    mock.Mock(), confirmed_site='Q243', visual_gate=callback)
            callback.assert_not_called(); identify.assert_not_called()
            self.assertEqual(result['identity_origin'], 'user_provided')
            self.assertEqual(result['visual_guard']['status'], 'bypassed_user_identity')
            self.assertFalse(result['identity_verified'])

    def replay_fixture(self, folder):
        pid = next(pid for pid, p in self.corpus.passages.items() if p['site_id'] == 'Q243' and p['start'] == 0 and p['source_type'] == 'wikipedia')
        evidence = self.corpus.context([pid])
        previous = {'image': {'filename': 'synthetic.png', 'sha256': IMAGE_SHA}, 'question': 'Pourquoi ce monument ?',
                    'identification': self.prediction(), 'identity_origin': 'reference_and_qwen_proposal_not_verified',
                    'visual_guard': gate_record(), 'visual_agreement': {'matched': True, 'reference_site_id': 'Q243', 'qwen_site_id': 'Q243'}}
        for name, value in [('result.json', previous), ('summary.json', {'corpus_fingerprint': self.corpus.fingerprint}),
                            ('retrieval.json', {'site_id': 'Q243', 'question': 'À propos du monument « tour Eiffel » : Pourquoi ce monument ?', 'evidence': evidence})]:
            (folder/name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        return previous

    def test_assisted_replay_requires_prior_gate_and_agreement(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'documentary_answer') as generate:
            source = pathlib.Path(temp)/'source'; source.mkdir()
            output = pathlib.Path(temp)/'output'; output.mkdir()
            previous = self.replay_fixture(source)
            previous.pop('visual_guard')
            (source/'result.json').write_text(json.dumps(previous), encoding='utf-8')
            result = photo.replay_documentary(self.corpus, self.sites, source, mock.Mock(), output, mock.Mock(), require_visual_gate=True)
            self.assertEqual(result['status'], 'needs_clarification')
            self.assertFalse(result['historical_answer_generated'])
            generate.assert_not_called()

    def test_assisted_replay_preserves_bound_gate_without_rerunning_visual_models(self):
        answer = {'reference_checks_passed': True, 'model_called': True,
                  'answer': {'status': 'answered', 'answer': 'Simulation sourcée. [S1]', 'citations': ['S1']}}
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'documentary_answer', return_value=answer) as generate, \
             mock.patch.object(guard, 'run_guard') as gate, mock.patch.object(photo, 'identify') as vision:
            source = pathlib.Path(temp)/'source'; source.mkdir()
            output = pathlib.Path(temp)/'output'; output.mkdir()
            self.replay_fixture(source)
            result = photo.replay_documentary(self.corpus, self.sites, source, mock.Mock(), output, mock.Mock(), require_visual_gate=True)
            self.assertEqual(result['status'], 'proposed_identification_with_answer')
            self.assertTrue(result['visual_agreement']['matched'])
            self.assertFalse(result['identity_verified'])
            gate.assert_not_called(); vision.assert_not_called()
            self.assertEqual(generate.call_args.kwargs['plan']['version'], photo.facet_rag.VERSION)

    def test_assisted_replay_rejects_changed_agreement_or_image_binding(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(photo, 'documentary_answer') as generate:
            source = pathlib.Path(temp)/'source'; source.mkdir()
            output = pathlib.Path(temp)/'output'; output.mkdir()
            original = self.replay_fixture(source)
            for problem in ['agreement', 'photo']:
                changed = copy.deepcopy(original)
                if problem == 'agreement': changed['visual_agreement']['reference_site_id'] = 'Q2981'
                else: changed['image']['sha256'] = 'changed'
                (source/'result.json').write_text(json.dumps(changed), encoding='utf-8')
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    photo.replay_documentary(self.corpus, self.sites, source, mock.Mock(), output, mock.Mock(), require_visual_gate=True)
            generate.assert_not_called()

    def test_worker_stops_before_encoding_if_descriptor_namespace_changed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); output = root/'record.json'; image = root/'image.png'; image.write_bytes(IMAGE)
            with mock.patch.object(photo, 'load_image', return_value=(IMAGE, {'sha256': IMAGE_SHA})), \
                 mock.patch.object(guard.visual, 'fetch_model', return_value={}), \
                 mock.patch.object(guard.visual, 'Encoder', return_value=mock.Mock(namespace='changed-model')), \
                 mock.patch.object(guard.visual, 'cached_encode') as encode:
                self.assertEqual(guard.worker(image, IMAGE_SHA, output), 1)
            encode.assert_not_called()
            self.assertEqual(json.loads(output.read_text())['status'], 'failed')

    def test_worker_success_uses_real_bank_contract_and_leaves_no_image_in_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); report = root/'report'; report.mkdir()
            image = root/'input.png'; image.write_bytes(IMAGE)
            def encode(encoder, path, cache):
                match = path == image or path.name.startswith('Q243')
                return ([1.0, 0.0]+[0.0]*382 if match else [0.0, 1.0]+[0.0]*382), False
            with mock.patch.object(photo, 'load_image', return_value=(IMAGE, {'sha256': IMAGE_SHA})), \
                 mock.patch.object(guard.visual, 'fetch_model', return_value={'synthetic': True}), \
                 mock.patch.object(guard.visual, 'Encoder', return_value=mock.Mock(namespace=guard.load_policy()['descriptor_namespace'])), \
                 mock.patch.object(guard.visual, 'cached_encode', side_effect=encode):
                self.assertEqual(guard.worker(image, IMAGE_SHA, report/'record.json'), 0)
            record = json.loads((report/'record.json').read_text())
            guard.validate_record(record, IMAGE_SHA, self.sites)
            self.assertEqual(record['proposed_site_id'], 'Q243')
            self.assertFalse(record['identity_confirmed'])
            self.assertFalse(any(p.suffix in ['.jpg', '.png'] for p in report.rglob('*')))


if __name__ == '__main__': unittest.main()
