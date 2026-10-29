"""One-command local verification, one report ZIP, optional real photo check and app launch."""
import argparse
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile

import app_local
import mcp_client
import rag_local
from run_local import ROOT, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', action='store_true', help='Run a real, already-seen Eiffel photo through the assisted pipeline.')
    parser.add_argument('--open', action='store_true', help='Open the app after verification.')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    output = ROOT/'results'/('release-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+os.urandom(4).hex()); output.mkdir(parents=True)
    started = time.perf_counter(); interrupted = False; summary = {'status':'failed','scope':'Software checks and optional already-seen photo smoke check; not an independent accuracy measurement.'}
    env = {**os.environ, 'PYTHONIOENCODING':'utf-8'}
    try:
        print('Vérification du corpus, des outils et du logiciel…', flush=True)
        summary['environment'] = app_local.Application().status()
        summary['corpus_fingerprint'] = rag_local.verified_corpus(ROOT/'corpus')
        with (output/'tests.log').open('w', encoding='utf-8') as log:
            tests = subprocess.run([sys.executable,'-m','unittest','discover','-q'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=120)
        summary['software_tests_passed'] = tests.returncode == 0
        if tests.returncode: raise ValueError('Des tests logiciels ont échoué ; voir tests.log dans le rapport.')
        trip = mcp_client.call_tool('plan_walk', {'anchor_site_id':'Q243','duration_minutes':240,'max_stops':4})
        save(output/'mcp-trip-check.json', trip)
        summary['mcp_roundtrip_passed'] = bool(trip['stops']) and trip['total_minutes_estimate'] <= 240
        summary['status'] = 'software_verified'
        if args.models:
            status = summary['environment']
            if not all(status[k] for k in ['llama_found','generation_model_found','projector_found','visual_environment_found']):
                summary.update(status='models_not_ready', message='Le catalogue et les parcours restent utilisables. Certains modèles ou programmes sont absents.')
            else:
                print('Contrôle réel sur la photo connue de la tour Eiffel. Les modèles se chargent successivement…', flush=True)
                command = [sys.executable,str(ROOT/'photo_assisted.py'),str(ROOT/'evaluation-v1/images/a01.jpg'),'--take-over-test-server']
                with (output/'photo-check.log').open('w',encoding='utf-8') as log:
                    options = {'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session':True}
                    run = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,**options)
                    try: run.wait(timeout=900)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        app_local.stop_owned_process_tree(run)
                        raise
                lines = (output/'photo-check.log').read_text(encoding='utf-8').splitlines()
                paths = [pathlib.Path(l[len('Rapport local : '):]).resolve() for l in lines if l.startswith('Rapport local : ')]
                if not paths or not paths[-1].is_relative_to((ROOT/'results').resolve()): raise ValueError('Rapport du contrôle photo introuvable.')
                source = paths[-1].parent
                for name in ['result.json','summary.json','answer.json','retrieval.json','visual-guard.json','model-events.json','error.txt']:
                    if (source/name).exists(): (output/'photo').mkdir(exist_ok=True); shutil.copyfile(source/name,output/'photo'/name)
                result = json.loads((source/'result.json').read_text(encoding='utf-8')) if (source/'result.json').exists() else {}
                summary['real_photo_check'] = {'exit_code':run.returncode,'status':result.get('status'),
                    'visual_agreement':result.get('visual_agreement',{}).get('matched'),
                    'exact_quotes_verified':result.get('exact_quotes_verified',False),
                    'all_facets_answered':result.get('all_facets_answered')}
                passed = (run.returncode == 0 and result.get('exact_quotes_verified') and result.get('visual_agreement',{}).get('matched')
                          and result.get('answer_status') == 'answered' and result.get('all_facets_answered') is True
                          and bool(result.get('answer',{}).get('citations')))
                summary['status'] = 'real_photo_smoke_passed' if passed else 'real_photo_needs_review'
    except KeyboardInterrupt: interrupted = True; summary.update(status='interrupted',error='Interrompu par l’utilisateur.')
    except Exception as error: summary.update(status='failed',error=str(error))
    finally:
        summary['total_seconds'] = time.perf_counter()-started; save(output/'summary.json',summary)
        with zipfile.ZipFile(output/'report.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in output.rglob('*'):
                if p.is_file() and p.name != 'report.zip': z.write(p,p.relative_to(output))
        print('Vérification : '+summary['status']+'\nRapport local : '+str(output/'report.zip'),flush=True)
    if args.open and not interrupted:
        app_local.main(['--port', str(args.port), '--open'])
    elif summary['status']=='failed': raise SystemExit(1)


if __name__=='__main__': main()
