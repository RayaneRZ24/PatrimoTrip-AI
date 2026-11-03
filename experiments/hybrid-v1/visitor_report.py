"""Visitor-facing HTML and PDF reports, independent of private diagnostic archives.

PDF embeds the bundled licensed Unicode font; no runtime dependency.
Only the answer, cited public sources and an attributed reference image are exported.
"""
import datetime as dt
import html
import json
import re
import urllib.parse

from heritage_names import monument_name

import photo_local
import pdf_font


def safe_url(value):
    return value if isinstance(value, str) and urllib.parse.urlsplit(value).scheme in ('http', 'https') else ''


def document(result, root):
    answer = result.get('answer') or {}
    citations = answer.get('citations', [])
    evidence = result.get('evidence') or result.get('grounding', {}).get('evidence', [])
    sources = [e for e in evidence if e['citation_id'] in citations]
    site = result.get('identification', {}).get('site_id') or result.get('site_id')
    photo = bool(result.get('identification') or result.get('context_is_visual_proposal'))
    identity = ('Identité fournie par vous' if result.get('identity_origin') == 'user_provided' else
                'Monument proposé : l’identité de la photo reste à confirmer' if photo else 'Monument du contexte')
    image = None
    if re.fullmatch(r'Q\d+', site or ''):
        refs = json.loads((root/'visual-references-v1/references.json').read_text(encoding='utf-8'))['references']
        credit = next((r for r in refs if r['site_id'] == site), None)
        path = root/'visual-references-v1/images'/(site+'.jpg')
        if credit and path.exists(): image = {'path':path, 'url':'/media/'+site+'.jpg', 'credit':credit}
    return {'title':monument_name(result['site_name']) if result.get('site_name') else 'Votre découverte culturelle', 'identity':identity,
            'question':result.get('question', ''),
            'answer':answer.get('answer') or result.get('message') or 'Aucune réponse documentaire disponible.',
            'partial': result.get('all_facets_answered') is False or result.get('grounding', {}).get('all_facets_answered') is False,
            'sources':sources, 'image':image,
            'created':dt.datetime.now(dt.timezone.utc).strftime('%d/%m/%Y à %H:%M UTC')}


def sections(doc):
    yield 'Votre question', doc['question'] or 'Découvrir ce monument'
    yield 'Réponse documentaire', doc['answer']
    if doc['partial']: yield 'Réponse partielle', 'Certaines informations demandées ne sont pas établies dans les extraits disponibles.'
    yield 'À retenir', 'Les citations reproduisent des extraits du corpus. Leur exactitude historique dépend des sources. Les horaires et tarifs actuels ne sont pas vérifiés. Une proposition visuelle ne prouve pas l’identité de votre photo.'
    for e in doc['sources']:
        s=e['source']
        yield '['+e['citation_id']+'] '+s['title'], '\n'.join(filter(None,[
            'Section : '+' > '.join(e.get('section_path', [])), safe_url(s.get('url')),
            s.get('attribution') or s.get('publisher'), s.get('license'), safe_url(s.get('license_url')),
            'Date de la source : '+str(s['updated_at']) if s.get('updated_at') else '',
            'Extrait consulté : '+e['text']]))
    if doc['image']:
        c=doc['image']['credit']
        yield 'Crédit de la photographie de référence', '\n'.join(filter(None,[c.get('author'), c.get('license'),safe_url(c.get('source_page'))]))


def html_report(result, root, identifier):
    doc=document(result,root); esc=html.escape
    body = '<p class="eyebrow">PATRIMOTRIP AI · FICHE MONUMENT</p><h1>'+esc(doc['title'])+'</h1>'
    body += '<p class="badge">'+esc(doc['identity'])+'</p><p class="meta">Généré le '+esc(doc['created'])+'</p>'
    body += '<nav class="actions"><a href="/api/monument-report?id='+esc(identifier)+'&amp;format=pdf">Télécharger le PDF</a><button id="print-report">Imprimer / enregistrer en PDF</button></nav>'
    if doc['image']:
        body += '<figure><img src="'+esc(doc['image']['url'])+'" alt="Photographie de référence du monument"><figcaption>Photographie du catalogue, distincte de votre photo personnelle. Crédit indiqué en fin de fiche.</figcaption></figure>'
    for heading,text in sections(doc):
        body += '<section><h2>'+esc(heading)+'</h2>'
        for line in text.splitlines():
            url=safe_url(line)
            body += '<p>'+('<a rel="noopener noreferrer" target="_blank" href="'+esc(url,quote=True)+'">'+esc(line)+'</a>' if url else esc(line))+'</p>'
        body += '</section>'
    return ('<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(doc['title'])+' — PatrimoTrip AI</title><link rel="stylesheet" href="/report.css"><script src="/report.js" defer></script></head><body><main>'+body+'</main></body></html>').encode('utf-8')


def pdf_report(result, root):
    doc=document(result,root); objects=[]
    def add(value): objects.append(value); return len(objects)
    def stream(data, extra=b''): return b'<< /Length '+str(len(data)).encode()+b' '+extra+b' >>\nstream\n'+data+b'\nendstream'
    def literal(text):
        # Use hexadecimal strings to make PDF delimiters and user text inert.
        return b'<'+text.encode('utf-16-be',errors='replace').hex().encode()+b'>'
    catalog=add(b''); pages=add(b'')
    font=pdf_font.embed(add,stream); bold=font
    image_id=None; image_size=None
    if doc['image']:
        image_data, meta=photo_local.load_image(doc['image']['path'])
        # Catalogue JPEGs have three colour channels; images are never fetched remotely.
        image_id=add(stream(image_data, ('/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode' % (meta['width'],meta['height'])).encode()))
        scale=min(170/meta['width'],170/meta['height']);image_size=(meta['width']*scale,meta['height']*scale)
    page_ids=[]; commands=[]; annotations=[]; y=785
    def line(text, size=10, strong=False, url=''):
        nonlocal y, commands, annotations
        if y < 55: flush()
        commands.append(b'BT /'+(b'FB' if strong else b'F')+b' '+str(size).encode()+b' Tf 1 0 0 1 50 '+str(y).encode()+b' Tm '+literal(text)+b' Tj ET')
        if url:
            annotations.append(add(b'<< /Type /Annot /Subtype /Link /Rect [50 '+str(y-3).encode()+b' 545 '+str(y+11).encode()+b'] /Border [0 0 0] /A << /S /URI /URI '+b'<'+(b'\xfe\xff'+url.encode('utf-16-be')).hex().encode()+b'>'+b' >> >>'))
        y -= size+6
    def flush():
        nonlocal commands, annotations, y
        footer=b'BT /F 8 Tf 1 0 0 1 50 30 Tm '+literal('PatrimoTrip AI · '+str(len(page_ids)+1))+b' Tj ET'
        content=add(stream(b'\n'.join(commands+[footer])))
        resources=b'<< /Font << /F '+str(font).encode()+b' 0 R /FB '+str(bold).encode()+b' 0 R >>'
        if image_id: resources+=b' /XObject << /Photo '+str(image_id).encode()+b' 0 R >>'
        resources+=b' >>'
        page_ids.append(add(b'<< /Type /Page /Parent '+str(pages).encode()+b' 0 R /MediaBox [0 0 595 842] /Resources '+resources+b' /Contents '+str(content).encode()+b' 0 R /Annots ['+b' '.join(str(i).encode()+b' 0 R' for i in annotations)+b'] >>'))
        commands=[];annotations=[];y=785
    line('PATRIMOTRIP AI · FICHE MONUMENT',10,True)
    for text in pdf_font.wrap(doc['title'],20): line(text,20,True)
    for text in pdf_font.wrap(doc['identity'],10): line(text)
    line('Généré le '+doc['created'],9)
    if image_id:
        w,h=image_size;commands.append(('q %.2f 0 0 %.2f 50 %.2f cm /Photo Do Q' % (w,h,y-h)).encode());y-=h+18
        line('Photographie de référence du catalogue ; ce n’est pas votre photo personnelle.',8)
    for heading,text in sections(doc):
        if y < 100: flush()
        y-=8
        for title in pdf_font.wrap(heading,12):line(title,12,True)
        for paragraph in text.splitlines():
            url=safe_url(paragraph)
            for wrapped in pdf_font.wrap(paragraph,10):
                line(wrapped,10,url=url)
        y-=6
    flush()
    objects[catalog-1]=b'<< /Type /Catalog /Pages '+str(pages).encode()+b' 0 R >>'
    objects[pages-1]=b'<< /Type /Pages /Count '+str(len(page_ids)).encode()+b' /Kids ['+b' '.join(str(i).encode()+b' 0 R' for i in page_ids)+b'] >>'
    data=bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n');offsets=[0]
    for i,obj in enumerate(objects,1):
        offsets.append(len(data));data.extend(str(i).encode()+b' 0 obj\n'+obj+b'\nendobj\n')
    start=len(data);data.extend(b'xref\n0 '+str(len(offsets)).encode()+b'\n0000000000 65535 f \n')
    for offset in offsets[1:]:data.extend(('%010d 00000 n \n'%offset).encode())
    data.extend(b'trailer\n<< /Size '+str(len(offsets)).encode()+b' /Root '+str(catalog).encode()+b' 0 R >>\nstartxref\n'+str(start).encode()+b'\n%%EOF\n')
    return bytes(data)
