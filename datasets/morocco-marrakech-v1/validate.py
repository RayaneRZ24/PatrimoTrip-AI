"""Check the pilot's frozen files, provenance and image integrity without AI."""
import hashlib
import json
import pathlib

ROOT=pathlib.Path(__file__).resolve().parent


def validate(root=ROOT):
    checks=json.loads((root/'checksums.json').read_text(encoding='utf-8'))
    for name,expected in checks.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Fichier absent ou chemin invalide : '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Empreinte incorrecte : '+name)
    sites=json.loads((root/'sites.json').read_text(encoding='utf-8'))
    entities=json.loads((root/'sources/wikidata.json').read_text(encoding='utf-8'))['entities']
    pages=json.loads((root/'sources/commons.json').read_text(encoding='utf-8'))['query']['pages']
    if len(sites)!=3 or len({s['site_id'] for s in sites})!=3:raise ValueError('Trois monuments distincts attendus.')
    for site in sites:
        entity=entities[site['site_id']];image=site['image'];raw=(root/image['file']).read_bytes()
        if entity['lastrevid']!=site['wikidata_revision']:raise ValueError('Révision Wikidata incohérente.')
        countries=[c.get('mainsnak',{}).get('datavalue',{}).get('value',{}).get('id') for c in entity['claims'].get('P17',[])]
        if 'Q1028' not in countries:raise ValueError('Pays du monument non établi dans les métadonnées.')
        if not (31.5<site['coordinates']['latitude']<31.8 and -8.2<site['coordinates']['longitude']<-7.8):raise ValueError('Coordonnées hors de la zone pilote.')
        if not raw.startswith(b'\xff\xd8') or hashlib.sha256(raw).hexdigest()!=image['sha256'] or hashlib.sha1(raw).hexdigest()!=image['sha1']:raise ValueError('Photographie corrompue.')
        info=next(p['imageinfo'][0] for p in pages.values() if p['imageinfo'][0]['descriptionurl']==image['source_page'])
        if info['extmetadata']['LicenseShortName']['value']!=image['license']:raise ValueError('Licence photographique incohérente.')
        if info['sha1']!=image['sha1']:raise ValueError('Photographie différente de la source Commons.')
        if site['current_opening_hours'] is not None or site['current_ticket_prices'] is not None:raise ValueError('Informations actuelles non établies.')
    return {'frozen_files_verified':len(checks),'sites':len(sites),'photos':len(sites),'status':'seed_verified_texts_and_visual_evaluation_pending'}


if __name__=='__main__':print(json.dumps(validate(),ensure_ascii=False,indent=2))
