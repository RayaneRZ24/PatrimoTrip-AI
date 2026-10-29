"""MCP stdio server: JSON-RPC, initialize, tools/list, tools/call; no remote inference."""
import json
import sys

import hybrid
import trip_tools
from run_local import ROOT

VERSION = '2025-06-18'


def schema(properties, required):
    return {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}


TOOLS = [
    {'name': 'catalog_search', 'description': 'Chercher un monument culturel dans le catalogue local de Paris.',
     'inputSchema': schema({'query': {'type': 'string'}}, ['query'])},
    {'name': 'heritage_evidence', 'description': 'Lire les extraits documentaires locaux d’un site, avec leurs sources.',
     'inputSchema': schema({'site_id': {'type': 'string'}, 'question': {'type': 'string'}}, ['site_id', 'question'])},
    {'name': 'plan_walk', 'description': 'Construire un parcours culturel extérieur estimé à partir d’un monument.',
     'inputSchema': schema({'anchor_site_id': {'type': 'string'}, 'duration_minutes': {'type': 'integer'},
                           'max_stops': {'type': 'integer'}, 'visit_minutes': {'type': 'integer'},
                           'site_ids': {'type': 'array', 'items': {'type': 'string'}}}, ['anchor_site_id'])},
    {'name': 'save_trip', 'description': 'Enregistrer localement un parcours après confirmation explicite.',
     'inputSchema': schema({'trip': {'type': 'object'}, 'confirmed': {'type': 'boolean'}}, ['trip', 'confirmed'])},
    {'name': 'list_trips', 'description': 'Lister les parcours déjà enregistrés sur cet ordinateur.', 'inputSchema': schema({}, [])},
    {'name': 'tourism_reference_links', 'description': 'Liens officiels candidats pour vérifier horaires et tarifs ; ces informations ne sont pas établies localement.',
     'inputSchema': schema({'site_id': {'type': 'string'}}, ['site_id'])},
]


class ToolService:
    def __init__(self, storage=None):
        self.storage = storage; self.corpus = None

    def call(self, name, arguments):
        definition = next((t for t in TOOLS if t['name'] == name), None)
        if not definition or not isinstance(arguments, dict): raise ValueError('Outil ou arguments invalides.')
        spec = definition['inputSchema']
        if set(arguments)-set(spec['properties']) or set(spec['required'])-set(arguments): raise ValueError('Arguments inconnus ou manquants.')
        for key, value in arguments.items():
            expected = spec['properties'][key]['type']
            valid = {'string': isinstance(value, str), 'integer': isinstance(value, int) and not isinstance(value, bool),
                     'boolean': isinstance(value, bool), 'object': isinstance(value, dict), 'array': isinstance(value, list)}[expected]
            if not valid: raise ValueError('Type invalide pour '+key)
        sites = {s['site_id']: s for s in trip_tools.catalog()}
        if name == 'catalog_search':
            query = arguments['query'].casefold()[:500]
            return [s for s in sites.values() if query in s['name'].casefold() or query in s['site_id'].casefold() or not query]
        if name == 'plan_walk': return trip_tools.plan(**arguments)
        if name == 'save_trip': return trip_tools.save_trip(folder=self.storage, **arguments)
        if name == 'list_trips': return trip_tools.list_trips(self.storage)
        site_id = arguments['site_id']
        if site_id not in sites: raise ValueError('Site absent du catalogue.')
        if name == 'tourism_reference_links':
            return {'site_id': site_id, 'links': sites[site_id]['official_website_candidates'], 'current_hours_verified': False,
                    'current_prices_verified': False, 'message': 'Consultez les sites officiels pour les informations du jour.'}
        if self.corpus is None: self.corpus = hybrid.Corpus(ROOT)
        question = arguments['question']
        if not question.strip() or len(question) > 800: raise ValueError('Question vide ou trop longue.')
        ranking = self.corpus.bm25(question, site_id=site_id)
        intro = [pid for pid, p in self.corpus.passages.items() if p['site_id'] == site_id and p['start'] == 0 and p['source_type'] == 'wikipedia'][:1]
        return {'evidence': self.corpus.context(list(dict.fromkeys(intro+ranking)), max_per_document=3, balanced_sentences=True),
                'source_status': 'Frozen historical corpus; current practical information not verified.'}


def handle(request, service):
    if not isinstance(request, dict) or request.get('jsonrpc') != '2.0':
        return {'jsonrpc': '2.0', 'id': request.get('id') if isinstance(request, dict) else None, 'error': {'code': -32600, 'message': 'Invalid Request'}}
    method = request.get('method'); identifier = request.get('id')
    if 'id' not in request: return None
    try:
        if method == 'initialize':
            offered = request.get('params', {}).get('protocolVersion')
            result = {'protocolVersion': offered if offered in ['2024-11-05', '2025-03-26', VERSION] else VERSION,
                      'capabilities': {'tools': {}}, 'serverInfo': {'name': 'patrimotrip-local', 'version': '1.0.0'}}
        elif method == 'ping': result = {}
        elif method == 'tools/list': result = {'tools': TOOLS}
        elif method == 'tools/call':
            params = request.get('params', {})
            try:
                value = service.call(params.get('name'), params.get('arguments', {}))
                result = {'content': [{'type': 'text', 'text': json.dumps(value, ensure_ascii=False)}], 'isError': False}
            except (ValueError, TypeError) as error:
                result = {'content': [{'type': 'text', 'text': str(error)}], 'isError': True}
        else: return {'jsonrpc': '2.0', 'id': identifier, 'error': {'code': -32601, 'message': 'Method not found'}}
        return {'jsonrpc': '2.0', 'id': identifier, 'result': result}
    except Exception:
        return {'jsonrpc': '2.0', 'id': identifier, 'error': {'code': -32603, 'message': 'Internal tool error'}}


def main():
    service = ToolService()
    for line in sys.stdin:
        try:
            if len(line) > 200000: raise ValueError('Request too large')
            reply = handle(json.loads(line), service)
        except (ValueError, json.JSONDecodeError): reply = {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Parse error'}}
        if reply is not None: print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
