import copy
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import collect_texts
import validate

ROOT=pathlib.Path(__file__).resolve().parent


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.site={'site_id':'Q1137533','name':'Mosquée Koutoubia','wikipedia_title':'Mosquée Koutoubia'}
        self.response={'query':{'rightsinfo':{'url':'https://creativecommons.org/licenses/by-sa/4.0/deed.fr'},'pages':{'123':{
            'title':'Mosquée Koutoubia','pageid':123,'pageprops':{'wikibase_item':'Q1137533'},
            'extract':('Texte de source synthétique, sans assertion historique. '*10),
            'revisions':[{'revid':456,'timestamp':'2026-11-03T00:00:00Z'}]}}}}

    def test_article_identity_revision_and_license_are_required(self):
        snapshot=collect_texts.from_response(self.response,self.site)
        self.assertEqual(snapshot['source']['url'],'https://fr.wikipedia.org/w/index.php?oldid=456')
        self.assertEqual(snapshot['source']['license'],'CC BY-SA 4.0')
        for flaw in ['license','identity','missing','empty','revision','disambiguation']:
            value=copy.deepcopy(self.response);page=value['query']['pages']['123']
            if flaw=='license':value['query']['rightsinfo']={}
            elif flaw=='identity':page['pageprops']['wikibase_item']='Q243'
            elif flaw=='missing':page['missing']=''
            elif flaw=='empty':page['extract']=''
            elif flaw=='revision':page['revisions']=[]
            else:page['pageprops']['disambiguation']=''
            with self.subTest(flaw=flaw),self.assertRaises(ValueError):collect_texts.from_response(value,self.site)

    def test_corrupt_cached_text_is_not_silently_reused(self):
        snapshot=collect_texts.from_response(self.response,self.site);snapshot['text']+=' changed'
        with self.assertRaises(ValueError):collect_texts.validate_snapshot(snapshot,self.site)

    def test_partial_collection_is_not_reported_as_complete_and_cached_article_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);collect_texts.save(root/'sites.json',[self.site,{**self.site,'site_id':'Q2465115'}])
            (root/'text-snapshots').mkdir();collect_texts.save(root/'text-snapshots/Q1137533.json',collect_texts.from_response(self.response,self.site))
            with mock.patch.object(collect_texts,'acquire') as acquire,mock.patch('builtins.print'):
                complete=collect_texts.collect(root,offline=True)
            acquire.assert_not_called();self.assertFalse(complete)
            report=json.loads((root/'results/text-acquisition.json').read_text())
            self.assertEqual(report['verified_text_snapshots'],1);self.assertEqual(report['status'],'incomplete')
            self.assertFalse(report['application_imported']);self.assertFalse(report['historical_review_complete'])

    def test_frozen_public_seed_is_consistent(self):
        result=validate.validate(ROOT)
        self.assertEqual(result['sites'],3);self.assertEqual(result['photos'],3)


if __name__=='__main__':unittest.main()
