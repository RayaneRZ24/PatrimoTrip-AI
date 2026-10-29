"""PatrimoTrip AI local web app. Loopback only; serialized inference; no cloud inference."""
import argparse
import base64
import binascii
import hashlib
import http.server
import json
import os
import pathlib
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser

import mcp_client
import photo_local
import runtime
import trip_tools
from run_local import ROOT

MAX_BODY = 12*1024*1024


def stop_owned_process_tree(process):
    if process.poll() is not None: return
    if os.name == 'nt': subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=15)
    else:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass


class Application:
    def __init__(self, root=ROOT):
        self.root = pathlib.Path(root); self.token = secrets.token_urlsafe(32)
        self.jobs = {}; self.drafts = {}; self.lock = threading.Lock(); self.inference_lock = threading.Lock()
        self.process = None; self.closing = False

    def status(self):
        models = pathlib.Path.home()/'PatrimoTripAI/models/qwen3.5-4b'
        env = runtime.windows_environment()
        return {'python': sys.version.split()[0], 'llama_found': bool(shutil.which('llama-server', path=env.get('PATH'))),
                'generation_model_found': (models/'Qwen3.5-4B-Q4_K_M.gguf').is_file(),
                'projector_found': (models/'mmproj-F16.gguf').is_file(),
                'visual_environment_found': (self.root/'.visual-venv'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')).is_file(),
                'busy': self.inference_lock.locked(), 'inference': 'local', 'catalog_sites': 10}

    def add_draft(self, trip):
        identifier = uuid.uuid4().hex
        with self.lock:
            self.drafts[identifier] = trip
            if len(self.drafts) > 100: self.drafts.pop(next(iter(self.drafts)))
        return {'draft_id': identifier, **trip}

    def start_job(self, kind, body):
        question = body.get('question', photo_local.DEFAULT_QUESTION if kind == 'photo' else '')
        if not isinstance(question, str) or not question.strip() or len(question) > 800: raise ValueError('Question requise, 800 caractères maximum.')
        site = body.get('site_id') or None
        if site is not None and site not in {s['site_id'] for s in trip_tools.catalog()}: raise ValueError('Monument hors catalogue.')
        data = None
        if kind == 'photo':
            encoded = body.get('image_base64')
            if not isinstance(encoded, str) or len(encoded) > 12*1024*1024: raise ValueError('Photo JPEG/PNG de 8 Mio maximum.')
            try: data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError): raise ValueError('Photo encodée invalide.')
            if len(data) > photo_local.MAX_BYTES: raise ValueError('Photo trop volumineuse.')
        if self.closing or not self.inference_lock.acquire(blocking=False): raise RuntimeError('Une analyse est déjà en cours. Attendez sa fin.')
        identifier = uuid.uuid4().hex
        job = {'job_id': identifier, 'status': 'running', 'kind': kind, 'phase': 'Préparation de l’analyse locale…'}
        with self.lock: self.jobs[identifier] = job
        threading.Thread(target=self.run_job, args=(identifier, kind, question, site, data), daemon=True).start()
        return {'job_id': identifier}

    def run_job(self, identifier, kind, question, site, data):
        folder = self.root/'.app-state/jobs'/identifier
        image = None
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if kind == 'photo':
                image = folder/('photo.png' if data.startswith(b'\x89PNG') else 'photo.jpg'); image.write_bytes(data)
                photo_local.load_image(image)
                command = [sys.executable, str(self.root/'photo_assisted.py'), str(image), '--question', question, '--take-over-test-server']
                if site: command += ['--confirm-site', site]
            else:
                command = [sys.executable, str(self.root/'agent_local.py'), '--question', question, '--take-over-test-server']
                if site: command += ['--site-id', site]
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            options = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            with self.lock:
                if self.closing: raise RuntimeError('Application en cours de fermeture.')
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', env=env, **options)
                self.process = process
            report = None
            with (folder/'execution.log').open('w', encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line); log.flush()
                    if line.startswith('Rapport local : '): report = pathlib.Path(line[len('Rapport local : '):].strip()).resolve()
                    phase = None
                    if 'Comparaison de la photo' in line: phase = 'Comparaison aux références visuelles…'
                    elif 'embedding' in line.lower(): phase = 'Recherche des extraits documentaires…'
                    elif 'Chargement local' in line: phase = 'Analyse et sélection des sources sur votre ordinateur…'
                    if phase:
                        with self.lock: self.jobs[identifier]['phase'] = phase
            code = process.wait()
            if not report or not report.is_relative_to((self.root/'results').resolve()) or not report.is_file():
                raise RuntimeError('L’analyse n’a pas produit de rapport. Consultez la fenêtre de lancement ou le journal local.')
            with self.lock: self.jobs[identifier]['report_path'] = str(report)
            result_path = report.parent/'result.json'
            result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
            if code: raise RuntimeError(result.get('message') or json.loads((report.parent/'summary.json').read_text()).get('error') or 'Analyse interrompue ; le rapport contient le diagnostic.')
            if (report.parent/'answer.json').exists():
                result['evidence'] = json.loads((report.parent/'answer.json').read_text(encoding='utf-8')).get('evidence', [])
            if result.get('trip'): result['trip'] = self.add_draft(result['trip'])
            with self.lock:
                self.jobs[identifier].update(status='done', phase='Terminé', result=result, report_path=str(report))
        except Exception as error:
            with self.lock: self.jobs[identifier].update(status='failed', error=str(error), phase='Analyse arrêtée')
        finally:
            if image: image.unlink(missing_ok=True)
            with self.lock: self.process = None
            self.inference_lock.release()

    def close(self):
        self.closing = True
        with self.lock: process = self.process
        if process: stop_owned_process_tree(process)


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'PatrimoTripLocal/1.0'
    def log_message(self, *args): pass

    def allowed_host(self):
        port = self.server.server_port
        return self.headers.get('Host') in ['127.0.0.1:'+str(port), 'localhost:'+str(port)]

    def send(self, value, status=200, mime='application/json; charset=utf-8', filename=None):
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode('utf-8')
        try:
            self.send_response(status); self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if filename: self.send_header('Content-Disposition', 'attachment; filename="'+filename+'"')
            self.end_headers(); self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError): pass

    def do_GET(self):
        if not self.allowed_host(): return self.send({'error': 'Hôte local invalide.'}, 403)
        url = urllib.parse.urlsplit(self.path); path = url.path; app = self.server.application
        try:
            if path == '/': return self.send((app.root/'web/index.html').read_bytes(), mime='text/html; charset=utf-8')
            if path == '/favicon.ico': return self.send(b'', 204, mime='image/x-icon')
            if path in ['/app.js', '/styles.css']:
                return self.send((app.root/'web'/path[1:]).read_bytes(), mime='text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
            if re.fullmatch(r'/media/Q\d+\.jpg', path):
                return self.send((app.root/'visual-references-v1/images'/path.rsplit('/', 1)[-1]).read_bytes(), mime='image/jpeg')
            if path == '/api/bootstrap':
                credits = {r['site_id']: r for r in json.loads((app.root/'visual-references-v1/references.json').read_text(encoding='utf-8'))['references']}
                return self.send({'token': app.token, 'catalog': trip_tools.catalog(), 'image_credits': credits, 'status': app.status()})
            if path == '/api/trips': return self.send({'trips': mcp_client.call_tool('list_trips', {})})
            if path.startswith('/api/jobs/'):
                with app.lock: job = dict(app.jobs.get(path.rsplit('/', 1)[-1], {}))
                if not job: return self.send({'error': 'Analyse introuvable.'}, 404)
                job['report_available'] = bool(job.get('report_path'))
                job.pop('report_path', None)
                return self.send(job)
            if path == '/api/report':
                identifier = urllib.parse.parse_qs(url.query).get('id', [''])[0]
                with app.lock: report = app.jobs.get(identifier, {}).get('report_path')
                if not report: return self.send({'error': 'Rapport indisponible.'}, 404)
                return self.send(pathlib.Path(report).read_bytes(), mime='application/zip', filename='patrimotrip-report.zip')
            if path == '/api/trip-export':
                identifier = urllib.parse.parse_qs(url.query).get('id', [''])[0]
                if not re.fullmatch(r'[a-f0-9]{32}', identifier): raise ValueError('Identifiant invalide.')
                return self.send((app.root/'.app-state/trips'/(identifier+'.json')).read_bytes(), filename='patrimotrip-'+identifier+'.json')
            return self.send({'error': 'Ressource introuvable.'}, 404)
        except (OSError, ValueError, RuntimeError) as error: return self.send({'error': str(error)}, 400)

    def do_POST(self):
        app = self.server.application; port = self.server.server_port
        if (not self.allowed_host() or self.headers.get('Origin') not in [None, 'http://127.0.0.1:'+str(port), 'http://localhost:'+str(port)] or
                self.headers.get('X-PatrimoTrip-Token') != app.token): return self.send({'error': 'Requête locale non autorisée.'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send({'error': 'Requête JSON trop volumineuse ou invalide.'}, 413)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict): raise ValueError('Objet JSON attendu.')
            path = urllib.parse.urlsplit(self.path).path
            if path in ['/api/photo', '/api/chat']: return self.send(app.start_job('photo' if path == '/api/photo' else 'chat', body), 202)
            if path == '/api/plan': return self.send(app.add_draft(mcp_client.call_tool('plan_walk', body)))
            if path == '/api/trips/save':
                if body.get('confirmed') is not True: raise ValueError('Confirmez l’enregistrement du parcours affiché.')
                with app.lock: draft = app.drafts.get(body.get('draft_id'))
                if not draft: raise ValueError('Préparez d’abord un parcours.')
                title = body.get('title', 'Promenade culturelle à Paris')
                if not isinstance(title, str) or len(title) > 120: raise ValueError('Titre invalide.')
                return self.send(mcp_client.call_tool('save_trip', {'trip': {**draft, 'title': title}, 'confirmed': True}))
            return self.send({'error': 'Action introuvable.'}, 404)
        except (ValueError, TypeError, KeyError) as error: return self.send({'error': str(error)}, 400)
        except RuntimeError as error: return self.send({'error': str(error)}, 409)


def create_server(port=8765, application=None):
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.application = application or Application(); server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--port', type=int, default=8765); parser.add_argument('--open', action='store_true')
    args = parser.parse_args(argv)
    server = create_server(args.port); url = 'http://127.0.0.1:'+str(server.server_port)
    print('PatrimoTrip AI : '+url+'\nGardez cette fenêtre ouverte. Ctrl+C pour arrêter.', flush=True)
    if args.open: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.application.close(); server.server_close()


if __name__ == '__main__': main()
