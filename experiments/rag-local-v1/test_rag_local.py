"""Offline corpus and transport checks; no model-quality score is produced."""
import http.server
import json
import pathlib
import tempfile
import threading
import unittest
from unittest import mock

import rag_local as rag

CORPUS = pathlib.Path(__file__).resolve().parent / "corpus"


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.database = pathlib.Path(cls.temp.name) / "index.sqlite3"
        cls.index = rag.build_index(CORPUS, cls.database)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_index_count_and_repeatability(self):
        self.assertEqual(self.index["passages"], 840)
        self.assertEqual(rag.build_index(CORPUS, self.database), {"passages": 840, "reused": True})

    def test_real_eiffel_evidence_and_traceability(self):
        evidence = rag.retrieve(self.database, "Pourquoi la tour Eiffel a-t-elle été construite ?")
        text = " ".join(e["text"] for e in evidence)
        self.assertIn("Exposition universelle", text)
        self.assertIn("1889", text)
        self.assertIn("Révolution française", text)
        self.assertLessEqual(sum(len(e["text"]) for e in evidence), 4800)
        for item in evidence:
            self.assertEqual(item["site_id"], "Q243")
            self.assertTrue(item["source"]["url"].startswith("https://"))
            with rag.connect(self.database) as connection:
                row = connection.execute("SELECT body,start_offset FROM passages WHERE passage_id=?", (item["passage_id"],)).fetchone()
            self.assertEqual(item["text"], row["body"][:item["end"]-item["start"]])
            self.assertEqual(item["start"], row["start_offset"])

    def test_location_and_author_sources_present(self):
        evidence = rag.retrieve(self.database, "Où se trouve le Panthéon de Paris ?")
        self.assertIn("5e arrondissement", " ".join(e["text"] for e in evidence))
        evidence = rag.retrieve(self.database, "Qui a construit le palais Garnier ?")
        self.assertIn("Charles Garnier", " ".join(e["text"] for e in evidence))
        self.assertFalse(any("bibliographiques" in " ".join(e["section_path"]) for e in evidence))

    def test_sql_like_input_is_data(self):
        rag.retrieve(self.database, 'Eiffel ") OR 1=1; DROP TABLE passages; --')
        with rag.connect(self.database) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM passages").fetchone()[0], 840)

    def test_no_match_does_not_call_model(self):
        client = mock.Mock()
        result = rag.ask(self.database, "zzzzunfindabletestword", client)
        self.assertFalse(result["model_called"])
        self.assertEqual(result["answer"]["status"], "insufficient_sources")
        client.request.assert_not_called()

    def test_reference_checks_reject_invented_or_missing_sources(self):
        evidence = [{"citation_id": "S1"}]
        invalid = [
            {"status": "answered", "answer": "Un fait sans référence", "citations": []},
            {"status": "answered", "answer": "Un fait [S999]", "citations": ["S999"]},
            {"status": "answered", "answer": "Un fait [S1]", "citations": []},
            {"status": "answered", "answer": "Un fait [S1]", "citations": ["S1", "S1"]},
        ]
        for answer in invalid:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                rag.validate_answer(answer, evidence)
        answer = {"status": "answered", "answer": "Un fait [S1]", "citations": ["S1"]}
        self.assertEqual(rag.validate_answer(answer, evidence), answer)

    def test_external_or_credentialled_endpoint_rejected(self):
        for url in ["http://example.com:8080", "https://127.0.0.1:8080", "http://user:secret@localhost:8080"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                rag.LocalModel(url)

    def test_http_flow_and_truncated_answer(self):
        payloads = []

        class Handler(http.server.BaseHTTPRequestHandler):
            truncated = False

            def log_message(self, *args):
                pass

            def send(self, value):
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send({"data": [{"id": "mock-model-not-real-inference"}]})

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payloads.append(data)
                answer = {"status": "answered", "answer": "Réponse simulée [S1]", "citations": ["S1"]}
                self.send({"choices": [{"message": {"content": json.dumps(answer)},
                                        "finish_reason": "length" if self.truncated else "stop"}], "usage": {"completion_tokens": 10}})

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = rag.LocalModel(f"http://127.0.0.1:{server.server_port}")
            result = rag.ask(self.database, "Pourquoi la tour Eiffel a-t-elle été construite ?", client)
            self.assertTrue(result["reference_checks_passed"])
            payload = payloads[0]
            schema = payload["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["properties"]["citations"]["items"]["enum"], [e["citation_id"] for e in result["evidence"]])
            self.assertEqual(payload["messages"][0]["content"], rag.SYSTEM)
            self.assertLess(len(payload["messages"][1]["content"]), 7500)
            self.assertIn("Exposition universelle", payload["messages"][1]["content"])
            Handler.truncated = True
            result = rag.ask(self.database, "Pourquoi la tour Eiffel a-t-elle été construite ?", client)
            self.assertFalse(result["reference_checks_passed"])
            self.assertIn("tronquée", result["validation_error"])
            self.assertNotIn("answer", result)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
