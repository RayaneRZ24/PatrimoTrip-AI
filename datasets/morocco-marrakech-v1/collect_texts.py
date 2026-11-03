"""Collect the three French Wikipedia articles for the separate Marrakech pilot.

Standard Python, no model and no API key. Valid snapshots are reused unchanged.
Successful collection is source acquisition, not historical or visual validation.
"""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT=pathlib.Path(__file__).resolve().parent
API='https://fr.wikipedia.org/w/api.php'
UA='PatrimoTripAI/0.1 (academic heritage corpus acquisition)'


def digest(data):return hashlib.sha256(data).hexdigest()


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def validate_snapshot(snapshot,site):
    if snapshot.get('site_id')!=site['site_id']:raise ValueError('Monument différent de celui demandé.')
    text=snapshot.get('text')
    if not isinstance(text,str) or len(text.strip())<300:raise ValueError('Article vide ou trop court pour un corpus documentaire.')
    if digest(text.encode('utf-8'))!=snapshot.get('text_sha256'):raise ValueError('Texte modifié ou empreinte incorrecte.')
    source=snapshot.get('source',{})
    if source.get('license_url')!='https://creativecommons.org/licenses/by-sa/4.0/':raise ValueError('Licence de texte non reconnue.')
    if type(source.get('revision_id')) is not int or source['revision_id']<=0:raise ValueError('Révision source absente.')
    expected='https://fr.wikipedia.org/w/index.php?oldid='+str(source['revision_id'])
    if source.get('url')!=expected:raise ValueError('Lien source non figé à sa révision.')
    if source.get('wikibase_item')!=site['site_id']:raise ValueError('Correspondance article/monument non établie.')
    return snapshot


def from_response(data,site):
    if data.get('error'):raise ValueError('Erreur de l’API : '+str(data['error'].get('code')))
    query=data.get('query',{});pages=query.get('pages',{})
    if len(pages)!=1:raise ValueError('Une seule page attendue par monument.')
    page=next(iter(pages.values()))
    if 'missing' in page or page.get('pageprops',{}).get('disambiguation') is not None:raise ValueError('Article absent ou page d’homonymie.')
    rights=query.get('rightsinfo',{})
    notice_url=rights.get('url','')
    if not re.fullmatch(r'https://creativecommons\.org/licenses/by-sa/4\.0(?:/deed\.[A-Za-z_-]+)?/?',notice_url):raise ValueError('Licence à vérifier avant réutilisation : '+str(rights))
    license_url='https://creativecommons.org/licenses/by-sa/4.0/'
    revisions=page.get('revisions') or []
    if len(revisions)!=1:raise ValueError('Révision manquante ou ambiguë.')
    text=page.get('extract','');revision=revisions[0]
    snapshot={'site_id':site['site_id'],'text':text,'text_sha256':digest(text.encode('utf-8')),
              'acquired_at':dt.datetime.now(dt.timezone.utc).isoformat(),
              'source':{'type':'wikipedia','title':page['title'],'revision_id':revision['revid'],
                        'revision_timestamp':revision.get('timestamp'),'page_id':page['pageid'],
                        'url':'https://fr.wikipedia.org/w/index.php?oldid='+str(revision['revid']),
                        'wikibase_item':page.get('pageprops',{}).get('wikibase_item'),
                        'license':'CC BY-SA 4.0','license_url':license_url,'license_notice_url':notice_url,
                        'attribution':'Contributeurs de Wikipédia ; historique accessible depuis la révision liée.',
                        'review_status':'pending_historical_review'}}
    return validate_snapshot(snapshot,site)


def acquire(site):
    # One article per request: TextExtracts can otherwise silently omit pages.
    params={'action':'query','format':'json','prop':'extracts|revisions|pageprops','explaintext':1,
            'titles':site['wikipedia_title'],'redirects':1,'rvprop':'ids|timestamp',
            'meta':'siteinfo','siprop':'rightsinfo','maxlag':5}
    url=API+'?'+urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':UA}),timeout=45) as response:
                raw=response.read(3000001)
            if len(raw)>3000000:raise ValueError('Réponse trop volumineuse.')
            return from_response(json.loads(raw),site),raw
        except urllib.error.HTTPError as error:
            if error.code not in (429,503) or attempt==2:raise
            retry=error.headers.get('Retry-After','')
            delay=int(retry) if retry.isdigit() else 5*(attempt+1)
            time.sleep(min(30,max(1,delay)))
    raise RuntimeError('Acquisition interrompue.')


def collect(root=ROOT,offline=False):
    sites=json.loads((root/'sites.json').read_text(encoding='utf-8'))
    out=root/'text-snapshots';out.mkdir(exist_ok=True)
    rows=[]
    for site in sites:
        sid=site['site_id'];destination=out/(sid+'.json')
        try:
            if destination.exists():
                snapshot=validate_snapshot(json.loads(destination.read_text(encoding='utf-8')),site);reused=True
            elif offline:raise ValueError('Texte source non encore téléchargé.')
            else:
                print('Collecte : '+site['name'],flush=True)
                snapshot,raw=acquire(site)
                (out/(sid+'-response.json')).write_bytes(raw)
                save(destination,snapshot);reused=False
            rows.append({'site_id':sid,'status':'snapshot_verified','characters':len(snapshot['text']),
                         'revision_id':snapshot['source']['revision_id'],'reused':reused})
        except (ValueError,OSError,KeyError,TypeError) as error:
            rows.append({'site_id':sid,'status':'blocked_or_invalid','error':str(error)})
        if not offline:time.sleep(1)
    complete=all(r['status']=='snapshot_verified' for r in rows)
    summary={'status':'sources_acquired_pending_review' if complete else 'incomplete',
             'sites':len(sites),'verified_text_snapshots':sum(r['status']=='snapshot_verified' for r in rows),
             'historical_review_complete':False,'application_imported':False,'rows':rows}
    results=root/'results';results.mkdir(exist_ok=True)
    save(results/'text-acquisition.json',summary)
    files=sorted(out.glob('*.json'))+[results/'text-acquisition.json']
    checks={p.relative_to(root).as_posix():digest(p.read_bytes()) for p in files}
    save(results/'snapshot-checksums.json',checks)
    archive=results/'marrakech-textes.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[results/'snapshot-checksums.json']:z.write(path,path.relative_to(root))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Rapport et textes : '+str(archive),flush=True)
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--offline',action='store_true');args=parser.parse_args()
    raise SystemExit(0 if collect(offline=args.offline) else 1)


if __name__=='__main__':main()
