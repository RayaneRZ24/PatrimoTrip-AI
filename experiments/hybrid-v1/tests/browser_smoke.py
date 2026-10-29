# Developer browser test: UI inference is simulated; planning/save/export use real MCP.
import sys,pathlib,threading,json,re,argparse
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import app_local,hybrid,grounded_rag,facet_rag
from playwright.sync_api import sync_playwright
ROOT=app_local.ROOT; P=ROOT/'.app-state/ui-check'; P.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--browser-path');args=parser.parse_args()
server=app_local.create_server(0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url='http://127.0.0.1:'+str(server.server_port);saved_id=None
try:
 with sync_playwright() as p:
  browser=p.chromium.launch(executable_path=args.browser_path,headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
  page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(url);page.locator('.catalog-card').first.wait_for();assert page.locator('.catalog-card').count()==10
  page.screenshot(path=str(P/'desktop.png'),full_page=True)
  page.locator('[data-view="walk"]').click();page.locator('#walk-form button.primary').click();page.locator('#walk-result .route-step').first.wait_for()
  count=page.locator('#walk-result .route-step').count();assert count>=2
  with page.expect_response(lambda r:'/api/trips/save' in r.url) as response:page.get_by_role('button',name='Confirmer et enregistrer ce parcours',exact=True).click()
  saved=response.value.json();saved_id=saved['trip_id'];assert saved['status']=='saved'
  page.get_by_text('Parcours enregistré',exact=True).wait_for();page.locator('.saved-card').first.wait_for();assert page.locator('.saved-card').count()>=1
  page.screenshot(path=str(P/'itinerary.png'),full_page=True)
  downloaded=page.request.get(url+'/api/trip-export?id='+saved_id);assert downloaded.status==200 and downloaded.json()['trip_id']==saved_id
  corpus=hybrid.Corpus(ROOT);pid=next(pid for pid,e in corpus.passages.items() if e['site_id']=='Q243' and e['start']==0 and e['source_type']=='wikipedia');evidence=corpus.context([pid]);u=grounded_rag.sentence_units(evidence)[0]
  results={};n=[0]
  def fake_submit(route):
   body=route.request.post_data_json;n[0]+=1;key='synthetic-'+str(n[0]);assert body.get('image_base64')
   if n[0]==2: result={'status':'needs_clarification','message':'Précisez le lieu ou fournissez une autre photo.'}
   else:
    if n[0]==3:assert body.get('site_id')=='Q243'
    result={'status':'proposed_identification_with_answer','site_name':'tour Eiffel','identification':{'site_id':'Q243'},'identity_origin':'user_provided' if n[0]==3 else 'reference_and_qwen_proposal_not_verified','answer':{'status':'answered','answer':'« '+u['text']+' » [S1]','citations':['S1']},'evidence':evidence}
   results[key]=result;route.fulfill(status=202,content_type='application/json',body=json.dumps({'job_id':key}))
  page.route('**/api/photo',fake_submit)
  page.route('**/api/jobs/synthetic-*',lambda route:route.fulfill(content_type='application/json',body=json.dumps({'status':'done','result':results[route.request.url.rsplit('/',1)[-1]]})))
  page.locator('[data-view="explore"]').click();page.locator('#photo-file').set_input_files(str(ROOT/'evaluation-v1/images/a01.jpg'));page.locator('#photo-form button.primary').click();page.locator('#photo-result h2').filter(has_text='tour Eiffel').wait_for()
  assert page.locator('#photo-result .source').count()==1;page.locator('#photo-result summary').click();assert page.locator('#photo-result .source p').first.text_content()==evidence[0]['text']
  page.screenshot(path=str(P/'photo-result.png'),full_page=True)
  page.locator('#photo-form button.primary').click();page.locator('.confirmation').wait_for();page.locator('.confirmation select').select_option('Q243');page.locator('.confirmation button').click();page.get_by_text('Identité fournie par vous',exact=True).wait_for()
  page.set_viewport_size({'width':390,'height':844});page.locator('[data-view="walk"]').click();assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
  page.screenshot(path=str(P/'mobile.png'),full_page=True)
  assert not errors,errors
  print(json.dumps({'catalog_cards':10,'planned_stops':count,'save_and_export':'real MCP roundtrip passed','photo_ui':'accepted, abstained and explicit confirmation tested with simulated inference','mobile_overflow':False,'page_errors':errors},ensure_ascii=False))
  browser.close()
finally:
 server.application.close();server.shutdown();server.server_close();thread.join(2)
 if saved_id:(ROOT/'.app-state/trips'/(saved_id+'.json')).unlink(missing_ok=True)
