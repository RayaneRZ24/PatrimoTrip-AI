"""Regression cases from the user's screenshots: roles, context and visitor reports."""
import copy
import http.client
import json
import pathlib
import tempfile
import threading
import unittest
from unittest import mock

import agent_local
import app_local
import conversation
import facet_rag
import grounded_rag
import hybrid
import mcp_server
import trip_tools
import visitor_report

ROOT=pathlib.Path(__file__).resolve().parent


class RelevanceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.corpus=hybrid.Corpus(ROOT);cls.service=mcp_server.ToolService()

    def evidence_containing(self,site,terms):
        ids=[next(pid for pid,p in self.corpus.passages.items() if p['site_id']==site and term in p['text']) for term in terms]
        return self.corpus.context(ids,budget=8000,max_per_document=4)

    def test_pantheon_accepts_soufflot_and_rejects_lockroy_subproject(self):
        evidence=self.evidence_containing('Q188856',['Construit dans un style néo-classique','Le projet avait été conçu par Édouard Lockroy'])
        units=grounded_rag.sentence_units(evidence);plan=facet_rag.plan_question('Qui a conçu ce monument ?','Panthéon')
        good=next(u for u in units if 'sous la direction de Jacques-Germain Soufflot' in u['text'])
        bad=next(u for u in units if 'Le projet avait été conçu par Édouard Lockroy' in u['text'])
        self.assertIn(good,grounded_rag.eligible(plan['facets'][0],units))
        self.assertNotIn(bad,grounded_rag.eligible(plan['facets'][0],units))
        draft={'answers':[{'facet_id':'F1','status':'answered','sentence_ids':[bad['unit_id']]}]}
        with self.assertRaises(ValueError):grounded_rag.render(draft,plan,evidence,units)
        draft['answers'][0]['sentence_ids']=[good['unit_id']]
        answer,_,_=grounded_rag.render(draft,plan,evidence,units)
        self.assertIn('Soufflot',answer['answer']);self.assertNotIn('Lockroy',answer['answer'])

    def test_notre_dame_does_not_name_patron_or_restorer_as_original_designer(self):
        evidence=self.service.call('heritage_evidence',{'site_id':'Q2981','question':'Qui a conçu ce monument ?'})['evidence']
        plan=facet_rag.plan_question('Qui a conçu ce monument ?','Notre-Dame');units=grounded_rag.sentence_units(evidence)
        self.assertFalse(grounded_rag.eligible(plan['facets'][0],units))
        client=mock.Mock();answer=grounded_rag.ask(plan['question'],plan,evidence,client)
        client.request.assert_not_called();self.assertEqual(answer['answer']['status'],'insufficient_sources')
        self.assertIn('restaurateur',answer['answer']['answer']);self.assertNotIn('Maurice',answer['answer']['answer'])
        restoration=facet_rag.plan_question('Qui a restauré cette cathédrale ?','Notre-Dame')
        allowed=grounded_rag.eligible(restoration['facets'][0],units)
        self.assertTrue(any('Viollet-le-Duc' in u['text'] and 'restauration' in u['text'] for u in allowed))

    def test_eiffel_purpose_rejects_isolated_technical_sentence(self):
        evidence=self.evidence_containing('Q243',["Construite en deux ans",'La technique du fer permet ensuite de dépasser cette limite'])
        units=grounded_rag.sentence_units(evidence);plan=facet_rag.plan_question('Présente brièvement ce monument et explique pourquoi il a été construit.','tour Eiffel')
        bad=next(u for u in units if 'La technique du fer permet ensuite de dépasser cette limite' in u['text'])
        good=next(u for u in units if 'Construite en deux ans' in u['text'])
        for facet in plan['facets']:self.assertNotIn(bad,grounded_rag.eligible(facet,units))
        purpose=next(f for f in plan['facets'] if f['kind']=='purpose')
        self.assertIn(good,grounded_rag.eligible(purpose,units))
        draft={'answers':[{'facet_id':f['facet_id'],'status':'answered','sentence_ids':[good['unit_id']]} for f in plan['facets']]}
        answer,_,_=grounded_rag.render(draft,plan,evidence,units)
        self.assertEqual(answer['answer'].count('Construite en deux ans'),1)
        self.assertNotIn('cette limite',answer['answer'])

    def test_designer_evidence_keeps_construction_roles_available(self):
        evidence=self.service.call('heritage_evidence',{'site_id':'Q188856','question':'Qui a conçu le Panthéon ?'})['evidence']
        units=grounded_rag.sentence_units(evidence);plan=facet_rag.plan_question('Qui a conçu le Panthéon ?','Panthéon')
        self.assertTrue(any('Soufflot' in u['text'] for u in grounded_rag.eligible(plan['facets'][0],units)))
        self.assertTrue(any(u['text'].startswith('Auteur_de_l_edifice:') for u in units))


class ConversationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.sites={s['site_id']:s for s in trip_tools.catalog()}

    def test_pronoun_without_context_clarifies_without_model_or_tools(self):
        client=mock.Mock()
        result=agent_local.run_agent('Qui a conçu ce monument ?',None,self.sites,client)
        self.assertEqual(result['status'],'needs_clarification');self.assertIn('De quel monument',result['message'])
        client.request.assert_not_called();self.assertFalse(result['tool_trace'])

    def test_explicit_name_accents_and_hyphens_resolve_in_catalogue(self):
        for question,sid in [('Qui a conçu le Panthéon de Paris ?','Q188856'),('Histoire de Notre-Dame de Paris','Q2981'),('Pourquoi la tour Eiffel a-t-elle été construite ?','Q243'),('Explique les illuminations de la tour Eiffel de nuit','Q243')]:
            with self.subTest(question=question):self.assertEqual(conversation.resolve(question,None,self.sites),(sid,None))

    def test_explicit_context_is_reused_and_conflicts_are_not_silently_overridden(self):
        self.assertEqual(conversation.resolve('Qui a conçu ce monument ?','Q188856',self.sites),('Q188856',None))
        for question,selected in [('Qui a conçu le Panthéon de Rome ?',None),('Compare le Louvre et le Panthéon',None),('Qui a conçu le Panthéon ?','Q243')]:
            with self.subTest(question=question):
                sid,result=conversation.resolve(question,selected,self.sites);self.assertIsNone(sid);self.assertEqual(result['status'],'needs_clarification')

    def test_named_monument_is_passed_to_the_tool_without_manual_selection(self):
        choice={'action':'answer_heritage','site_id':'Q188856','duration_minutes':240,'max_stops':4,'visit_minutes':30}
        client=mock.Mock();client.request.side_effect=[{'data':[{'id':'test'}]}, {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(choice)}}]}]
        with mock.patch.object(agent_local.mcp_client,'call_tool',return_value={'evidence':[]}) as tool:
            result=agent_local.run_agent('Qui a conçu le Panthéon de Paris ?',None,self.sites,client)
        tool.assert_called_once_with('heritage_evidence',{'site_id':'Q188856','question':'Qui a conçu le Panthéon de Paris ?'})
        self.assertEqual(result['site_id'],'Q188856');self.assertEqual(result['context_origin'],'named_in_question')


class VisitorReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence=mcp_server.ToolService().call('heritage_evidence',{'site_id':'Q188856','question':'Qui a conçu ce monument ?'})['evidence']
        u=next(u for u in grounded_rag.sentence_units(evidence) if 'sous la direction de Jacques-Germain Soufflot' in u['text'])
        cls.result={'site_name':'Panthéon','site_id':'Q188856','question':'Qui a conçu ce monument ?','answer':{'status':'answered','answer':'« '+u['text']+' » ['+u['citation_id']+']','citations':[u['citation_id']]},'evidence':evidence,'private_debug':'SECRET_NOT_FOR_VISITORS'}

    def test_html_is_readable_escaped_and_excludes_diagnostics(self):
        result=copy.deepcopy(self.result);result['question']='<script>alert("test")</script>'
        page=visitor_report.html_report(result,ROOT,'test').decode()
        self.assertIn('&lt;script&gt;',page);self.assertNotIn('<script>alert',page)
        self.assertIn('Soufflot',page);self.assertIn('Télécharger le PDF',page);self.assertIn('Photographie du catalogue',page)
        self.assertNotIn('SECRET_NOT_FOR_VISITORS',page);self.assertNotIn('GPU',page)

    def test_pdf_embeds_unicode_font_images_and_real_page_tree(self):
        pdf=visitor_report.pdf_report(self.result,ROOT)
        self.assertTrue(pdf.startswith(b'%PDF-1.4'));self.assertTrue(pdf.endswith(b'%%EOF\n'))
        self.assertIn(b'/Subtype /Type0',pdf);self.assertIn(b'/ToUnicode',pdf);self.assertIn(b'/Subtype /Image',pdf)
        self.assertIn('Panthéon'.encode('utf-16-be').hex().encode(),pdf)
        self.assertNotIn(b'SECRET_NOT_FOR_VISITORS',pdf)

    def test_http_serves_fiche_and_pdf_separately_from_diagnostic_zip(self):
        app=app_local.Application();app.jobs['visitor-test']={'status':'done','result':self.result}
        server=app_local.create_server(0,app);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            for suffix,mime,magic in [('', 'text/html',b'<!doctype html>'),('&format=pdf','application/pdf',b'%PDF')]:
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
                conn.request('GET','/api/monument-report?id=visitor-test'+suffix)
                response=conn.getresponse();data=response.read()
                self.assertEqual(response.status,200);self.assertTrue(response.getheader('Content-Type').startswith(mime));self.assertTrue(data.startswith(magic))
                if suffix:self.assertIn('.pdf',response.getheader('Content-Disposition'))
                conn.close()
            conn=http.client.HTTPConnection('127.0.0.1',server.server_port);conn.request('GET','/api/monument-report?id=missing')
            r=conn.getresponse();r.read();self.assertEqual(r.status,404);conn.close()
        finally:app.close();server.shutdown();server.server_close();thread.join(2)


if __name__=='__main__':unittest.main()
