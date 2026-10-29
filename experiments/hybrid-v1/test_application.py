import base64
import copy
import http.client
import itertools
import json
import pathlib
import tempfile
import threading
import unittest
import zipfile
from unittest import mock

import agent_local
import app_local
import facet_rag
import grounded_rag
import hybrid
import mcp_client
import mcp_server
import trip_tools

ROOT = pathlib.Path(__file__).resolve().parent


class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.corpus = hybrid.Corpus(ROOT)

    def evidence(self):
        introduction = next(pid for pid, p in self.corpus.passages.items() if p['site_id'] == 'Q243' and p['start'] == 0 and p['source_type'] == 'wikipedia')
        competing = next(pid for pid, p in self.corpus.passages.items() if p['site_id'] == 'Q243' and 'Le projet de Clarke et Reeves' in p['text'])
        return self.corpus.context([introduction, competing], max_per_document=4, balanced_sentences=True)

    def test_presentation_rejects_a_competing_tower_as_its_evidence(self):
        evidence = self.evidence(); units = grounded_rag.sentence_units(evidence)
        plan = facet_rag.plan_question('Présente ce monument.', 'tour Eiffel')
        competitor = next(u for u in units if u['citation_id'] == 'S2')
        draft = {'answers': [{'facet_id': 'F1', 'status': 'answered', 'sentence_ids': [competitor['unit_id']]}]}
        with self.assertRaises(ValueError): grounded_rag.render(draft, plan, evidence, units)
        intro = next(u for u in units if u['citation_id'] == 'S1')
        draft['answers'][0]['sentence_ids'] = [intro['unit_id']]
        answer, _, claims = grounded_rag.render(draft, plan, evidence, units)
        self.assertIn(intro['text'], answer['answer'])
        self.assertEqual(claims[0]['text'], evidence[0]['text'][intro['start']:intro['end']])
        self.assertTrue(claims[0]['quote_verified'])

    def test_free_paraphrases_unknown_ids_and_modified_quote_spans_are_rejected(self):
        evidence = self.evidence(); units = grounded_rag.sentence_units(evidence)
        plan = facet_rag.plan_question('Présente ce monument.', 'tour Eiffel')
        draft = {'answers': [{'facet_id': 'F1', 'status': 'answered', 'sentence_ids': [units[0]['unit_id']]}]}
        for problem in ['text', 'unknown', 'span', 'missing']:
            d = copy.deepcopy(draft); u = copy.deepcopy(units)
            if problem == 'text': d['answers'][0]['text'] = 'Invented factual paraphrase.'
            elif problem == 'unknown': d['answers'][0]['sentence_ids'] = ['S99.1']
            elif problem == 'span': u[0]['text'] = 'Altered source content.'
            else: d['answers'] = []
            with self.subTest(problem=problem), self.assertRaises(ValueError): grounded_rag.render(d, plan, evidence, u)

    def test_current_tariff_cannot_use_historical_numbers_or_call_a_model(self):
        client = mock.Mock()
        plan = facet_rag.plan_question("Quel est le prix aujourd'hui ?", 'tour Eiffel')
        result = grounded_rag.ask(plan['question'], plan, self.evidence(), client)
        client.request.assert_not_called()
        self.assertEqual(result['answer']['status'], 'insufficient_sources')
        self.assertEqual(result['answer']['citations'], [])

    def test_generic_retrieval_retains_the_entity_introduction(self):
        plan = facet_rag.plan_question('Présente ce monument et explique sa fonction.', 'tour Eiffel')
        ranking = [p['passage_id'] for p in self.corpus.passages.values() if p['site_id'] == 'Q243' and p['start'] > 10000][:10]
        with mock.patch.object(grounded_rag.facet_rag, 'retrieve', return_value=([], {'merged_ranking': ranking})):
            evidence, trace = grounded_rag.retrieve(self.corpus, 'Q243', 'tour Eiffel', plan['base_query'], plan, mock.Mock())
        self.assertEqual(evidence[0]['section_path'], ['Introduction'])
        self.assertIn('330', evidence[0]['text'])
        self.assertLessEqual(sum(len(e['text']) for e in evidence), 4800)

    def test_structured_model_selects_ids_and_host_reproduces_the_source_exactly(self):
        evidence=self.evidence();units=grounded_rag.sentence_units(evidence);plan=facet_rag.plan_question('Présente ce monument.','tour Eiffel')
        value={'answers':[{'facet_id':'F1','status':'answered','sentence_ids':[units[0]['unit_id']]}]}
        client=mock.Mock();client.request.side_effect=[{'data':[{'id':'synthetic-model'}]}, {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}]}]
        result=grounded_rag.ask(plan['question'],plan,evidence,client)
        self.assertTrue(result['exact_quotes_verified']);self.assertEqual(result['review_claims'][0]['text'],units[0]['text'])
        payload=client.request.call_args.args[1];user=json.loads(payload['messages'][1]['content'])
        self.assertIn('sources',user);self.assertNotIn('start',user['sources'][0]['sentences'][0])

    def test_mixed_history_and_current_price_returns_sourced_partial_answer(self):
        evidence=self.evidence();unit=next(u for u in grounded_rag.sentence_units(evidence) if 'Exposition universelle' in u['text'])
        plan=facet_rag.plan_question("Pourquoi a-t-elle été construite et quel est son tarif aujourd'hui ?",'tour Eiffel')
        client=mock.Mock();client.request.side_effect=[{'data':[{'id':'synthetic'}]}, {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answers':[{'facet_id':'F1','status':'answered','sentence_ids':[unit['unit_id']]}]})}}]}]
        result=grounded_rag.ask(plan['question'],plan,evidence,client)
        self.assertEqual(result['facet_statuses'],{'F2':'insufficient_sources','F1':'answered'})
        self.assertFalse(result['all_facets_answered']);self.assertTrue(result['exact_quotes_verified'])


class PlannerAndMCPTests(unittest.TestCase):
    def test_exact_route_matches_bruteforce_on_a_small_catalog_subset(self):
        selected = ['Q243', 'Q188977', 'Q64436', 'Q1075988']
        result = trip_tools.plan('Q243', 720, 4, 20, selected)
        sites = {s['site_id']: s for s in trip_tools.catalog()}
        candidates = [(sum(trip_tools.distance_km(sites[a], sites[b])*1.3 for a,b in zip(path,path[1:])), path)
                      for tail in itertools.permutations(selected[1:]) for path in [['Q243']+list(tail)]]
        shortest = min(km for km, _ in candidates)
        self.assertAlmostEqual(result['walking_km_estimate'], shortest, delta=0.006)
        self.assertEqual(len(result['stops']), 4)
        self.assertTrue(all(s['opening_hours_verified'] is False for s in result['stops']))

    def test_time_budget_stops_and_invalid_inputs_are_enforced(self):
        result = trip_tools.plan('Q243', 90, 6, 30)
        self.assertLessEqual(result['total_minutes_estimate'], 90)
        self.assertEqual(result['stops'][0]['site_id'], 'Q243')
        for args in [('unknown',), ('Q243', True), ('Q243', 240, 20), ('Q243', 45, 4, 60)]:
            with self.subTest(args=args), self.assertRaises(ValueError): trip_tools.plan(*args)
        with self.assertRaises(ValueError): trip_tools.plan('Q243', site_ids=[])

    def test_saved_trip_requires_confirmation_and_writes_only_generated_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            trip = trip_tools.plan('Q243', 120, 2, 30)
            with self.assertRaises(ValueError): trip_tools.save_trip(trip, False, temp)
            saved = trip_tools.save_trip(trip, True, temp)
            self.assertEqual(trip_tools.list_trips(temp)[0]['trip_id'], saved['trip_id'])
            self.assertEqual(len(list(pathlib.Path(temp).glob('*.json'))), 1)

    def test_mcp_initialization_tool_discovery_and_write_authorization(self):
        with tempfile.TemporaryDirectory() as temp:
            service = mcp_server.ToolService(temp)
            init = mcp_server.handle({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-06-18'}}, service)
            self.assertEqual(init['result']['protocolVersion'], '2025-06-18')
            tools = mcp_server.handle({'jsonrpc':'2.0','id':2,'method':'tools/list'}, service)['result']['tools']
            self.assertIn('plan_walk', [t['name'] for t in tools])
            denied = mcp_server.handle({'jsonrpc':'2.0','id':3,'method':'tools/call','params':{'name':'save_trip','arguments':{'trip':{},'confirmed':False}}}, service)
            self.assertTrue(denied['result']['isError'])
            self.assertFalse(list(pathlib.Path(temp).glob('*.json')))

    def test_real_stdio_client_server_roundtrip_and_site_scoping(self):
        trip = mcp_client.call_tool('plan_walk', {'anchor_site_id':'Q188856','duration_minutes':120,'max_stops':2})
        self.assertEqual(trip['stops'][0]['site_id'], 'Q188856')
        evidence = mcp_client.call_tool('heritage_evidence', {'site_id':'Q243','question':'Pourquoi a-t-elle été construite ?'})['evidence']
        self.assertTrue(evidence)
        self.assertTrue(all(e['site_id']=='Q243' for e in evidence))
        with self.assertRaises(ValueError): mcp_client.call_tool('plan_walk', {'anchor_site_id':'Q243','duration_minutes':False})


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.sites = {s['site_id']:s for s in trip_tools.catalog()}

    def client(self, action):
        client = mock.Mock(); client.request.side_effect = [{'data':[{'id':'synthetic-model'}]}, {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(action)}}]}]
        return client

    def action(self, name='plan_walk', site='Q243'):
        return {'action':name, 'site_id':site, 'duration_minutes':180, 'max_stops':3, 'visit_minutes':30}

    def test_model_can_choose_plan_tool_but_cannot_save(self):
        with mock.patch.object(agent_local.mcp_client, 'call_tool', return_value={'version':'walking_plan_v1'}) as tool:
            result = agent_local.run_agent('Une promenade de 3 heures.', 'Q243', self.sites, self.client(self.action()))
            self.assertEqual(result['status'], 'trip_draft')
            tool.assert_called_once_with('plan_walk', {'anchor_site_id':'Q243','duration_minutes':180,'max_stops':3,'visit_minutes':30})
        with mock.patch.object(agent_local.mcp_client, 'call_tool') as tool, self.assertRaises(ValueError):
            agent_local.run_agent('Enregistre sans me demander.', 'Q243', self.sites, self.client(self.action('save_trip')))
        tool.assert_not_called()

    def test_context_mismatch_and_unknown_site_do_not_run_tools(self):
        for site in ['unknown','Q188856']:
            with mock.patch.object(agent_local.mcp_client, 'call_tool') as tool:
                result = agent_local.run_agent('Histoire du lieu.', 'Q243', self.sites, self.client(self.action(site=site)))
                self.assertEqual(result['status'], 'needs_clarification'); tool.assert_not_called()


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = app_local.Application(); cls.server = app_local.create_server(0, cls.app)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls): cls.app.close(); cls.server.shutdown(); cls.server.server_close(); cls.thread.join(2)

    def request(self, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        opts = {'Content-Type':'application/json','X-PatrimoTrip-Token':self.app.token}
        if headers: opts.update(headers)
        connection.request('POST' if body is not None else 'GET', path, json.dumps(body) if body is not None else None, opts)
        response = connection.getresponse(); data=response.read(); status=response.status; connection.close(); return status,data

    def test_html_bootstrap_media_and_unknown_paths(self):
        status, data = self.request('/'); self.assertEqual(status,200); self.assertIn(b'PatrimoTrip',data)
        status, data = self.request('/api/bootstrap'); self.assertEqual(status,200); self.assertEqual(len(json.loads(data)['catalog']),10)
        status, data = self.request('/media/Q243.jpg'); self.assertEqual(status,200); self.assertTrue(data.startswith(b'\xff\xd8'))
        self.assertEqual(self.request('/../../etc/passwd')[0],404)

    def test_foreign_origin_host_and_missing_token_are_rejected(self):
        self.assertEqual(self.request('/api/plan', {}, {'Origin':'https://example.com'})[0],403)
        self.assertEqual(self.request('/api/plan', {}, {'X-PatrimoTrip-Token':''})[0],403)
        self.assertEqual(self.request('/api/bootstrap', headers={'Host':'example.com'})[0],403)

    def test_plan_preview_and_explicit_save_work_without_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            service = mcp_server.ToolService(temp)
            with mock.patch.object(app_local.mcp_client, 'call_tool', side_effect=service.call):
                status,data=self.request('/api/plan', {'anchor_site_id':'Q243','duration_minutes':180,'max_stops':3})
                self.assertEqual(status,200); draft=json.loads(data)
                self.assertFalse(list(pathlib.Path(temp).glob('*.json')))
                self.assertEqual(self.request('/api/trips/save', {'draft_id':draft['draft_id'],'confirmed':False})[0],400)
                status,data=self.request('/api/trips/save', {'draft_id':draft['draft_id'],'confirmed':True,'title':'Test local'})
                self.assertEqual(status,200); self.assertEqual(json.loads(data)['status'],'saved')
                self.assertEqual(len(list(pathlib.Path(temp).glob('*.json'))),1)

    def test_busy_engine_and_invalid_upload_do_not_start_another_process(self):
        self.app.inference_lock.acquire()
        try:
            self.assertEqual(self.request('/api/chat', {'question':'Histoire du monument.'})[0],409)
        finally: self.app.inference_lock.release()
        self.assertEqual(self.request('/api/photo', {'question':'Bonjour','image_base64':'not base64!'})[0],400)


class JobLifecycleTests(unittest.TestCase):
    def run_fixture(self, root, code=0):
        app = app_local.Application(root); report = root/'results/check'; report.mkdir(parents=True)
        (report/'result.json').write_text(json.dumps({'status':'proposed_identification_with_answer'}))
        (report/'summary.json').write_text(json.dumps({'error':'Synthetic failure' if code else None}))
        with zipfile.ZipFile(report/'report.zip','w') as z: z.write(report/'summary.json','summary.json')
        process = mock.Mock(); process.stdout = ['Rapport local : '+str(report/'report.zip')+'\n']; process.wait.return_value = code
        app.jobs['test-job']={'status':'running'}; app.inference_lock.acquire()
        observed = {}
        def spawn(command, **kwargs):
            observed['image'] = pathlib.Path(command[2]); self.assertTrue(observed['image'].is_file())
            return process
        with mock.patch.object(app_local.subprocess,'Popen',side_effect=spawn):
            app.run_job('test-job','photo','Question',None,(ROOT/'evaluation-v1/images/a01.jpg').read_bytes())
        return app,observed

    def test_successful_job_deletes_input_and_releases_the_inference_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            app,observed=self.run_fixture(pathlib.Path(temp))
            self.assertEqual(app.jobs['test-job']['status'],'done')
            self.assertFalse(observed['image'].exists()); self.assertFalse(app.inference_lock.locked()); self.assertIsNone(app.process)
            self.assertNotIn('image_base64',app.jobs['test-job'])

    def test_failed_job_still_exposes_report_and_removes_input(self):
        with tempfile.TemporaryDirectory() as temp:
            app,observed=self.run_fixture(pathlib.Path(temp),1)
            self.assertEqual(app.jobs['test-job']['status'],'failed')
            self.assertTrue(pathlib.Path(app.jobs['test-job']['report_path']).is_file())
            self.assertFalse(observed['image'].exists()); self.assertFalse(app.inference_lock.locked())

    def test_closing_app_cannot_spawn_an_inference_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            app=app_local.Application(pathlib.Path(temp));app.closing=True;app.jobs['j']={'status':'running'};app.inference_lock.acquire()
            with mock.patch.object(app_local.subprocess,'Popen') as spawn:
                app.run_job('j','chat','Question',None,None)
            spawn.assert_not_called();self.assertEqual(app.jobs['j']['status'],'failed');self.assertFalse(app.inference_lock.locked())


if __name__ == '__main__': unittest.main()
