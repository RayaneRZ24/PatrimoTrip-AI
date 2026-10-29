"""CPU visual descriptors. Similarity is an uncalibrated candidate, never identity proof."""
import hashlib
import json
import math
import pathlib
import re
import time
import urllib.request

MODEL_REPO = 'Xenova/dinov2-small'
UPSTREAM_REPO = 'facebook/dinov2-small'
FILES = ('config.json', 'preprocessor_config.json', 'onnx/model.onnx')
VERSION = 'dinov2_reference_ranking_v1'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_file(path, entry):
    path = pathlib.Path(path)
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    if entry['algorithm'] == 'sha256':
        return sha(path) == entry['digest']
    if entry['algorithm'] != 'git_blob_sha1':
        raise ValueError('Unknown file digest algorithm.')
    content = path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == entry['digest']


def validate_license_evidence(evidence, revision):
    if (not isinstance(evidence, dict) or evidence.get('upstream_repo') != UPSTREAM_REPO or
            evidence.get('upstream_license') != 'apache-2.0' or
            not re.fullmatch(r'[a-f0-9]{40}', evidence.get('upstream_revision', '')) or
            not re.fullmatch(r'[a-f0-9]{40}', evidence.get('conversion_card_git_blob', '')) or
            not re.fullmatch(r'[a-f0-9]{64}', evidence.get('conversion_card_sha256', '')) or
            evidence.get('conversion_revision') != revision):
        raise ValueError('Missing verified upstream licence/provenance evidence.')


def upstream_license_evidence(metadata, readme, upstream):
    """Verify a pinned conversion card links the exact base model, not a guessed licence."""
    entry = next((e for e in metadata.get('siblings', []) if e.get('rfilename') == 'README.md'), {})
    digest = hashlib.sha1(b'blob '+str(len(readme)).encode()+b'\0'+readme).hexdigest()
    if len(readme) != entry.get('size') or digest != entry.get('blobId'):
        raise ValueError('Conversion README digest/size mismatch.')
    text = readme.decode('utf-8')
    if not re.search(r'https://huggingface\.co/facebook/dinov2-small(?=$|[\s/)#?\]"<>])', text):
        raise ValueError('Conversion card does not link the expected upstream model.')
    if upstream.get('id') != UPSTREAM_REPO or (upstream.get('cardData') or {}).get('license') != 'apache-2.0':
        raise ValueError('Upstream model does not explicitly declare Apache-2.0.')
    evidence = {'upstream_repo': UPSTREAM_REPO, 'upstream_revision': upstream.get('sha'),
                'upstream_license': 'apache-2.0', 'conversion_revision': metadata['sha'],
                'conversion_card_git_blob': digest, 'conversion_card_sha256': hashlib.sha256(readme).hexdigest()}
    validate_license_evidence(evidence, metadata['sha'])
    return evidence


def model_spec(metadata, license_evidence=None):
    if metadata.get('id') != MODEL_REPO:
        raise ValueError('Unexpected conversion repository: '+str(metadata.get('id')))
    revision = metadata.get('sha', '')
    if not re.fullmatch(r'[a-f0-9]{40}', revision):
        raise ValueError('Model revision is not pinned.')
    declared = (metadata.get('cardData') or {}).get('license')
    if declared not in (None, '', 'apache-2.0'):
        raise ValueError('Unexpected explicitly declared conversion licence: '+str(declared))
    if declared != 'apache-2.0':
        validate_license_evidence(license_evidence, revision)
    siblings = {e['rfilename']: e for e in metadata.get('siblings', [])}
    entries = []
    for name in FILES:
        e = siblings.get(name, {})
        lfs = e.get('lfs') or {}
        size = lfs.get('size', e.get('size'))
        algorithm = 'sha256' if lfs else 'git_blob_sha1'
        digest = lfs.get('sha256') if lfs else e.get('blobId')
        if (not isinstance(size, int) or not 0 < size < 200_000_000 or
                not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{'+('64' if lfs else '40')+r'}', digest)):
            raise ValueError('Missing or unexpected size/digest for '+name)
        entries.append({'file': name, 'size': size, 'algorithm': algorithm, 'digest': digest})
    return {'repo': MODEL_REPO, 'revision': revision, 'license': 'apache-2.0', 'files': entries,
            'conversion_declared_license': declared,
            'license_basis': 'conversion_card' if declared == 'apache-2.0' else 'verified_upstream_card_and_conversion_link',
            'license_evidence': license_evidence}


def validate_spec(spec):
    if (spec.get('repo') != MODEL_REPO or spec.get('license') != 'apache-2.0' or
            not re.fullmatch(r'[a-f0-9]{40}', spec.get('revision', '')) or
            [e.get('file') for e in spec.get('files', [])] != list(FILES)):
        raise ValueError('Invalid local model lock.')
    if spec.get('license_basis') == 'verified_upstream_card_and_conversion_link':
        if spec.get('conversion_declared_license') not in (None, ''):
            raise ValueError('Inconsistent conversion licence lock.')
        validate_license_evidence(spec.get('license_evidence'), spec['revision'])
    elif spec.get('license_basis') != 'conversion_card' or spec.get('conversion_declared_license') != 'apache-2.0':
        raise ValueError('Missing or inconsistent conversion licence basis.')
    for entry in spec['files']:
        count = 64 if entry.get('algorithm') == 'sha256' else 40 if entry.get('algorithm') == 'git_blob_sha1' else 0
        if not count or not re.fullmatch(r'[a-f0-9]{'+str(count)+r'}', entry.get('digest', '')) or not 0 < entry['size'] < 200_000_000:
            raise ValueError('Invalid local file lock.')


def fetch_model(folder, evidence_dir=None):
    folder = pathlib.Path(folder); folder.mkdir(parents=True, exist_ok=True)
    lock = folder/'model-lock.json'
    if lock.exists():
        spec = json.loads(lock.read_text(encoding='utf-8')); validate_spec(spec)
    else:
        print('Recherche du modèle visuel public et de sa licence...', flush=True)
        with urllib.request.urlopen('https://huggingface.co/api/models/'+MODEL_REPO+'?blobs=true', timeout=45) as response:
            metadata = json.load(response)
        if evidence_dir:
            (pathlib.Path(evidence_dir)/'conversion-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
        evidence = None
        if metadata.get('id') == MODEL_REPO and not (metadata.get('cardData') or {}).get('license'):
            revision = metadata.get('sha', '')
            if not re.fullmatch(r'[a-f0-9]{40}', revision):
                raise ValueError('Conversion revision is not pinned.')
            print('Licence absente du champ API : vérification de la fiche de conversion et du modèle amont...', flush=True)
            with urllib.request.urlopen('https://huggingface.co/'+MODEL_REPO+'/resolve/'+revision+'/README.md', timeout=45) as response:
                readme = response.read(1_000_001)
            if len(readme) > 1_000_000:
                raise ValueError('Model card exceeds expected size.')
            with urllib.request.urlopen('https://huggingface.co/api/models/'+UPSTREAM_REPO+'?blobs=true', timeout=45) as response:
                upstream = json.load(response)
            if evidence_dir:
                (pathlib.Path(evidence_dir)/'conversion-README.md').write_bytes(readme)
                (pathlib.Path(evidence_dir)/'upstream-metadata.json').write_text(json.dumps(upstream, indent=2)+'\n', encoding='utf-8')
            evidence = upstream_license_evidence(metadata, readme, upstream)
        spec = model_spec(metadata, evidence)
        lock.write_text(json.dumps(spec, indent=2)+'\n', encoding='utf-8')
    for entry in spec['files']:
        dest = folder/entry['file']; dest.parent.mkdir(parents=True, exist_ok=True)
        if check_file(dest, entry):
            continue
        if dest.exists():
            raise ValueError('Un fichier différent existe déjà : '+str(dest))
        print('Téléchargement vérifié : '+entry['file']+' ('+str(round(entry['size']/1024**2))+' Mio)', flush=True)
        url = 'https://huggingface.co/'+MODEL_REPO+'/resolve/'+spec['revision']+'/'+entry['file']
        part = dest.with_suffix(dest.suffix+'.part')
        with urllib.request.urlopen(url, timeout=90) as response, part.open('wb') as stream:
            received = 0
            while True:
                chunk = response.read(1024*1024)
                if not chunk: break
                received += len(chunk)
                if received > entry['size']:
                    raise ValueError('File exceeds pinned size.')
                stream.write(chunk)
        if not check_file(part, entry):
            raise ValueError('Empreinte incorrecte : '+entry['file'])
        part.replace(dest)
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if config.get('model_type') != 'dinov2' or config.get('hidden_size') != 384 or config.get('patch_size') != 14:
        raise ValueError('Unexpected DINOv2-small architecture.')
    return spec


def checked_assets(root, rows, field='site_id'):
    root = pathlib.Path(root).resolve()
    seen = set()
    result = []
    for row in rows:
        path = (root/row['image']).resolve()
        if not path.is_relative_to(root) or sha(path) != row['sha256'] or row['sha256'] in seen:
            raise ValueError('Invalid, duplicate or modified image.')
        if not isinstance(row.get(field), str):
            raise ValueError('Missing site identifier.')
        seen.add(row['sha256'])
        result.append((row, path))
    return result


def rank_vector(vector, references):
    # No query identity/label is accepted here. Only image vectors and catalog labels.
    def normalize(v):
        if len(v) != 384 or any(not math.isfinite(x) for x in v):
            raise ValueError('Invalid visual descriptor.')
        length = math.sqrt(math.fsum(x*x for x in v))
        if length <= 1e-12:
            raise ValueError('Zero descriptor.')
        return [x/length for x in v]
    q = normalize(vector)
    scored = [{'site_id': r['site_id'], 'reference_sha256': r['sha256'],
               'cosine_similarity': math.fsum(a*b for a, b in zip(q, normalize(r['vector'])))} for r in references]
    return sorted(scored, key=lambda r: (-r['cosine_similarity'], r['site_id']))


class Encoder:
    def __init__(self, folder, spec):
        import onnxruntime as ort
        self.folder = pathlib.Path(folder)
        self.processor = json.loads((self.folder/'preprocessor_config.json').read_text(encoding='utf-8'))
        p = self.processor
        if not all(p.get(k) is True for k in ['do_resize', 'do_center_crop', 'do_rescale', 'do_normalize']):
            raise ValueError('Unsupported image processor switches.')
        if p.get('resample') not in [0, 1, 2, 3]:
            raise ValueError('Unsupported resize filter.')
        if p.get('crop_size') != {'height': 224, 'width': 224} or p.get('size') != {'shortest_edge': 256}:
            raise ValueError('Unexpected image size/crop; preprocessing must be reviewed.')
        if (len(p.get('image_mean', [])) != 3 or len(p.get('image_std', [])) != 3 or
                any(x <= 0 for x in p['image_std']) or abs(p.get('rescale_factor', 0)-1/255) > 1e-8):
            raise ValueError('Unexpected image normalization.')
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.folder/'onnx/model.onnx'), options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != 'tensor(float)':
            raise ValueError('Unsupported ONNX image input.')
        self.input_name = inputs[0].name
        outputs = {o.name for o in self.session.get_outputs()}
        self.output_name = 'last_hidden_state' if 'last_hidden_state' in outputs else 'pooler_output' if 'pooler_output' in outputs else None
        if not self.output_name:
            raise ValueError('No recognized ONNX descriptor output.')
        self.namespace = hashlib.sha256(json.dumps({'spec': spec, 'processor': p, 'version': VERSION,
                                                   'output': self.output_name}, sort_keys=True).encode()).hexdigest()

    def encode(self, path):
        import numpy as np
        from PIL import Image, ImageOps
        p = self.processor
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            w, h = image.size
            size = p['size']['shortest_edge']
            new_size = (size, int(size*h/w)) if w <= h else (int(size*w/h), size)
            image = image.resize(new_size, resample=p['resample'])
            w, h = image.size
            left, top = (w-224)//2, (h-224)//2
            image = image.crop((left, top, left+224, top+224))
            pixels = np.asarray(image, dtype=np.float32)*p['rescale_factor']
        pixels = (pixels-np.asarray(p['image_mean'], dtype=np.float32))/np.asarray(p['image_std'], dtype=np.float32)
        values = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
        output = self.session.run([self.output_name], {self.input_name: values})[0]
        if self.output_name == 'last_hidden_state':
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 384:
                raise ValueError('Invalid patch descriptor shape.')
            vector = output[0, 0, :]
        else:
            if output.shape != (1, 384): raise ValueError('Invalid pooled descriptor shape.')
            vector = output[0]
        result = [float(x) for x in vector]
        # Reuse validation; no silent bad vectors in cache.
        rank_vector(result, [])
        return result


def cached_encode(encoder, image, cache_dir):
    cache_dir = pathlib.Path(cache_dir)/encoder.namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir/(sha(image)+'.json')
    if path.exists():
        vector = json.loads(path.read_text(encoding='utf-8'))
        rank_vector(vector, [])
        return vector, True
    vector = encoder.encode(image)
    path.write_text(json.dumps(vector)+'\n', encoding='utf-8')
    return vector, False


def evaluate(encoder, references, queries, cache_dir):
    bank = []
    for row, path in references:
        print('Référence visuelle : '+row['site_id'], flush=True)
        vector, cached = cached_encode(encoder, path, cache_dir)
        bank.append({'site_id': row['site_id'], 'sha256': row['sha256'], 'vector': vector})
    rows = []
    for row, path in queries:
        print('Comparaison visuelle : '+row['asset_id'], flush=True)
        started = time.perf_counter()
        vector, cached = cached_encode(encoder, path, cache_dir)
        ranked = rank_vector(vector, bank)
        # Gold query labels are consulted ONLY AFTER ranking, for scoring.
        expected = row['expected_site_id']
        rows.append({'asset_id': row['asset_id'], 'sha256': row['sha256'], 'kind': row['kind'],
                     'expected_site_id': expected, 'top3_candidates': ranked[:3],
                     'top1_exact': ranked[0]['site_id'] == expected if expected != 'unknown' else None,
                     'score_gap': ranked[0]['cosine_similarity']-ranked[1]['cosine_similarity'],
                     'identity_confirmed': False, 'outside_detection': 'not_implemented',
                     'seconds': time.perf_counter()-started, 'cached': cached})
    return rows
