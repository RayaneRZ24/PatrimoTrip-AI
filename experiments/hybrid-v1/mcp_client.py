"""Short-lived MCP stdio client, one request batch per operation; traceable tool calls."""
import json
import os
import subprocess
import sys

from run_local import ROOT


def call_tool(name, arguments):
    requests = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                 'params': {'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 'patrimotrip-app', 'version': '1.0'}}},
                {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
                {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': name, 'arguments': arguments}}]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    result = subprocess.run([sys.executable, str(ROOT/'mcp_server.py')], input=''.join(json.dumps(r)+'\n' for r in requests),
                            text=True, encoding='utf-8', capture_output=True, timeout=45, env=env)
    if result.returncode: raise RuntimeError('Le serveur d’outils local n’a pas terminé correctement.')
    replies = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    reply = next((r for r in replies if r.get('id') == 2), {})
    response = reply.get('result', {})
    if reply.get('error') or response.get('isError') or not response.get('content'):
        raise ValueError(response.get('content', [{'text': 'Échec de l’outil MCP.'}])[0]['text'])
    return json.loads(response['content'][0]['text'])
