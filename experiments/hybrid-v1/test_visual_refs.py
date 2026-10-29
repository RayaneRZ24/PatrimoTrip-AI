import copy
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import visual_refs as visual
import evaluate_visual_refs as evaluate

ROOT = pathlib.Path(__file__).resolve().parent


class VisualReferenceTests(unittest.TestCase):
    def test_packaged_reference_bank_is_distinct_from_queries_and_matches_catalog(self):
        refs = json.loads((ROOT/'visual-references-v1/references.json').read_text(encoding='utf-8'))['references']
        queries = json.loads((ROOT/'evaluation-v1/assets.json').read_text(encoding='utf-8'))
        visual.checked_assets(ROOT/'visual-references-v1', refs)
        visual.checked_assets(ROOT/'evaluation-v1', queries, 'expected_site_id')
        self.assertEqual(len(refs), 10)
        self.assertEqual({r['site_id'] for r in refs}, {r['site_id'] for r in json.loads((ROOT/'corpus/sites.json').read_text(encoding='utf-8'))})
        self.assertFalse({r['sha256'] for r in refs} & {r['sha256'] for r in queries})
        self.assertTrue(all(r['author'] and r['license'] and r['source_page'] for r in refs))

    def metadata(self):
        return {'id': visual.MODEL_REPO, 'sha': 'a'*40, 'cardData': {'license': 'apache-2.0'}, 'siblings': [
            {'rfilename': f, 'size': 10, 'blobId': 'b'*40} if not f.endswith('.onnx') else
            {'rfilename': f, 'lfs': {'sha256': 'c'*64, 'size': 100}} for f in visual.FILES]}

    def test_remote_model_spec_requires_public_license_revision_and_digests(self):
        meta = self.metadata()
        visual.validate_spec(visual.model_spec(meta))
        for field in ['license', 'revision', 'digest', 'size', 'repo']:
            changed = copy.deepcopy(meta)
            if field == 'license': changed['cardData']['license'] = 'unknown'
            elif field == 'revision': changed['sha'] = 'main'
            elif field == 'repo': changed['id'] = 'untrusted/model'
            elif field == 'digest': changed['siblings'][0]['blobId'] = None
            else: changed['siblings'][0]['size'] = 500_000_000
            with self.subTest(field=field), self.assertRaises(ValueError): visual.model_spec(changed)
        spec = visual.model_spec(meta); spec['files'][0]['file'] = '../outside'
        with self.assertRaises(ValueError): visual.validate_spec(spec)

    def test_download_digests_cover_lfs_and_git_config_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp)/'config.json'; content = b'{"test":true}'
            path.write_bytes(content)
            entries = [{'algorithm': 'sha256', 'digest': hashlib.sha256(content).hexdigest(), 'size': len(content)},
                       {'algorithm': 'git_blob_sha1', 'digest': hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest(), 'size': len(content)}]
            for e in entries: self.assertTrue(visual.check_file(path, e))
            path.write_bytes(b'x'*len(content))
            for e in entries: self.assertFalse(visual.check_file(path, e))

    def provenance(self):
        metadata = self.metadata(); metadata['cardData'] = {}
        readme = b'ONNX conversion of https://huggingface.co/facebook/dinov2-small\n'
        metadata['siblings'].append({'rfilename': 'README.md', 'size': len(readme),
                                    'blobId': hashlib.sha1(b'blob '+str(len(readme)).encode()+b'\0'+readme).hexdigest()})
        upstream = {'id': visual.UPSTREAM_REPO, 'sha': 'd'*40, 'cardData': {'license': 'apache-2.0'}}
        return metadata, readme, upstream

    def test_empty_conversion_license_requires_verified_upstream_evidence(self):
        metadata, readme, upstream = self.provenance()
        with self.assertRaises(ValueError): visual.model_spec(metadata)
        evidence = visual.upstream_license_evidence(metadata, readme, upstream)
        spec = visual.model_spec(metadata, evidence)
        visual.validate_spec(spec)
        self.assertIsNone(spec['conversion_declared_license'])
        self.assertEqual(spec['license_basis'], 'verified_upstream_card_and_conversion_link')
        for card in [None, {'license': ''}]:
            metadata['cardData'] = card
            visual.validate_spec(visual.model_spec(metadata, evidence))

    def test_fallback_rejects_bad_provenance_license_and_revision(self):
        metadata, readme, upstream = self.provenance()
        with self.assertRaises(ValueError): visual.upstream_license_evidence(metadata, readme+b'changed', upstream)
        other = readme.replace(b'dinov2-small', b'another-model')
        changed = copy.deepcopy(metadata)
        changed['siblings'][-1].update(size=len(other), blobId=hashlib.sha1(b'blob '+str(len(other)).encode()+b'\0'+other).hexdigest())
        with self.assertRaises(ValueError): visual.upstream_license_evidence(changed, other, upstream)
        for alteration in [{'id': 'other/repo'}, {'cardData': {}}, {'cardData': {'license': 'non-commercial'}}, {'sha': 'main'}]:
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                visual.upstream_license_evidence(metadata, readme, {**upstream, **alteration})
        evidence = visual.upstream_license_evidence(metadata, readme, upstream)
        metadata['cardData'] = {'license': 'non-commercial'}
        with self.assertRaises(ValueError): visual.model_spec(metadata, evidence)
        metadata['cardData'] = {}; metadata['sha'] = 'e'*40
        with self.assertRaises(ValueError): visual.model_spec(metadata, evidence)

    def test_cached_lock_cannot_omit_upstream_proof(self):
        metadata, readme, upstream = self.provenance()
        spec = visual.model_spec(metadata, visual.upstream_license_evidence(metadata, readme, upstream))
        for field in ['license_evidence', 'conversion_declared_license', 'license_basis']:
            changed = copy.deepcopy(spec)
            if field in ['license_evidence', 'license_basis']: changed.pop(field)
            else: changed[field] = 'unknown'
            with self.subTest(field=field), self.assertRaises(ValueError): visual.validate_spec(changed)

    def test_ranking_uses_normalized_pixels_descriptors_without_expected_query_label(self):
        x, y = [1.0]+[0.0]*383, [0.0, 1.0]+[0.0]*382
        refs = [{'site_id': 'A', 'sha256': 'a', 'vector': x}, {'site_id': 'B', 'sha256': 'b', 'vector': y}]
        ranked = visual.rank_vector([v*5 for v in y], refs)
        self.assertEqual(ranked[0]['site_id'], 'B')
        self.assertEqual(ranked[0]['cosine_similarity'], 1)
        for bad in [[0.0]*384, [float('nan')]*384, [1.0]*3]:
            with self.assertRaises(ValueError): visual.rank_vector(bad, refs)

    def test_cached_descriptor_is_namespaced_and_does_not_encode_again(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); image = root/'image'; image.write_bytes(b'image')
            encoder = mock.Mock(namespace='version1'); encoder.encode.return_value = [1.0]+[0.0]*383
            self.assertFalse(visual.cached_encode(encoder, image, root/'cache')[1])
            self.assertTrue(visual.cached_encode(encoder, image, root/'cache')[1])
            encoder.encode.assert_called_once()
            encoder.namespace = 'version2'
            self.assertFalse(visual.cached_encode(encoder, image, root/'cache')[1])

    def test_fake_encoder_evaluation_does_not_confirm_nearest_neighbor_for_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            refs, queries = [], []
            for index, site in enumerate(['A', 'B']):
                path = root/str(index); path.write_bytes(str(index).encode())
                refs.append(({'site_id': site, 'sha256': visual.sha(path)}, path))
            path = root/'query'; path.write_bytes(b'outside')
            queries.append(({'asset_id': 'test', 'expected_site_id': 'unknown', 'kind': 'outside', 'sha256': visual.sha(path)}, path))
            encoder = mock.Mock(namespace='synthetic-only'); encoder.encode.return_value = [1.0]+[0.0]*383
            rows = visual.evaluate(encoder, refs, queries, root/'cache')
            self.assertFalse(rows[0]['identity_confirmed'])
            self.assertIsNone(rows[0]['top1_exact'])
            self.assertEqual(rows[0]['outside_detection'], 'not_implemented')
            self.assertEqual(len(rows[0]['top3_candidates']), 2)

    def test_reference_checks_reject_modified_and_escaping_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); path = root/'image.jpg'; path.write_bytes(b'abc')
            row = {'image': 'image.jpg', 'sha256': visual.sha(path), 'site_id': 'A'}
            visual.checked_assets(root, [row])
            with self.assertRaises(ValueError): visual.checked_assets(root, [row, row])
            with self.assertRaises(ValueError): visual.checked_assets(root, [{**row, 'image': '../image.jpg'}])
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): visual.checked_assets(root, [row])

    def test_prior_pipeline_comparison_requires_matching_query_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp); old = root/'evaluation-20260101'; old.mkdir()
            (old/'summary.json').write_text(json.dumps({'status': 'complete', 'cases_planned': 12, 'assets_sha256': 'verified'}))
            (old/'cases-summary.json').write_text(json.dumps([{'asset_id': 'a05', 'predicted_site_id': 'Q188856'}]))
            self.assertEqual(evaluate.prior_predictions(root, 'other'), (None, {}))
            self.assertEqual(evaluate.prior_predictions(root, 'verified')[1]['a05'], 'Q188856')


if __name__ == '__main__': unittest.main()
