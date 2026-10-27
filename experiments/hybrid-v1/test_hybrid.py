import http.server
import io
import json
import pathlib
import tempfile
import threading
import unittest
from unittest import mock

import hybrid
import rag_local as rag
import run_local
import runtime
import compare_profiles

ROOT = pathlib.Path(__file__).resolve().parent


class HybridTests(unittest.TestCase):
    def test_vectors_normalize_and_reject_invalid_values(self):
        self.assertEqual(hybrid.unit_vector([3, 4, 0], 3), [0.6, 0.8, 0])
        for bad in [[0, 0, 0], [1, float("nan"), 2], [1, float("inf"), 2], [1, 2], [True, 2, 3]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                hybrid.unit_vector(bad, 3)

    def test_vector_cache_namespace_text_hash_and_completeness(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "vectors.sqlite3"
            cache = hybrid.VectorCache(path, "model-A", 3)
            encoder = mock.Mock()
            encoder.encode.return_value = [1, 0, 0]
            first = cache.encode(encoder, "document", "p1", "alpha")
            second = cache.encode(encoder, "document", "p1", "alpha")
            self.assertFalse(first[1]); self.assertTrue(second[1])
            encoder.encode.assert_called_once_with("alpha")
            self.assertIsNone(cache.get("document", "p1", "modified"))
            self.assertIsNone(hybrid.VectorCache(path, "model-B", 3).get("document", "p1", "alpha"))
            cache.put("document", "p2", "beta", [0, 1, 0], 0.1)
            self.assertEqual(cache.rank([1, 0, 0], {"p1", "p2"})[0][0], "p1")
            with self.assertRaises(ValueError):
                cache.rank([1, 0, 0], {"p1", "p2", "missing"})
            with rag.connect(path) as connection:
                connection.execute("UPDATE vectors SET embedding=? WHERE item_id='p1'", (b"broken",))
            with self.assertRaises(ValueError):
                cache.get("document", "p1", "alpha")

    def test_rank_fusion_combines_ranks_not_raw_scores(self):
        fused = hybrid.reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        self.assertEqual(fused[0], "b")
        self.assertEqual(set(fused), {"a", "b", "c"})

    def test_relevance_groups_and_negative_case(self):
        case = {"evidence_groups": [["a"], ["b", "c"]]}
        one = hybrid.evaluate_rank(["a", "x"], case)
        self.assertTrue(one["known_evidence_hit_at_5"])
        self.assertFalse(one["all_evidence_groups_at_5"])
        both = hybrid.evaluate_rank(["a", "b"], case)
        self.assertTrue(both["all_evidence_groups_at_5"])
        self.assertIsNone(hybrid.evaluate_rank(["a"], {"evidence_groups": []}))

    def test_model_card_and_file_validation(self):
        card = {"license": "apache-2.0", "base_model": runtime.BASE_MODEL}
        self.assertTrue(runtime.compatible_card(card))
        self.assertFalse(runtime.compatible_card(None))
        self.assertFalse(runtime.compatible_card({**card, "base_model": "unknown"}))
        self.assertFalse(runtime.compatible_card({**card, "license": "unknown"}))
        files = [{"rfilename": "model-Q4_K_M.gguf", "lfs": {"sha256": "a"*64, "size": 300}},
                 {"rfilename": "model-Q8_0.gguf", "lfs": {"sha256": "b"*64, "size": 600}}]
        self.assertEqual(runtime.choose_file(files)["rfilename"], "model-Q8_0.gguf")
        with self.assertRaises(ValueError):
            runtime.choose_file([{"rfilename": "model-Q8_0.gguf"}])

    def test_takeover_only_recognized_local_test_server(self):
        model = pathlib.Path("C:/Users/Test/PatrimoTripAI/models/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf")
        process = {"Name": "llama-server.exe", "CommandLine": f'"llama-server.exe" --model "{model}" --host 127.0.0.1 --port 8080'}
        self.assertTrue(runtime.is_expected_test_process(process, model))
        for changed in [{**process, "Name": "other.exe"},
                        {**process, "CommandLine": process["CommandLine"].replace("127.0.0.1", "0.0.0.0")},
                        {**process, "CommandLine": process["CommandLine"].replace("8080", "9090")},
                        {**process, "CommandLine": "llama-server.exe --model other.gguf --host 127.0.0.1 --port 8080"}]:
            with self.subTest(changed=changed):
                self.assertFalse(runtime.is_expected_test_process(changed, model))

    def test_real_annotations_and_lexical_rankings(self):
        corpus = hybrid.Corpus(ROOT)
        manifest = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        hybrid.validate_cases(manifest, corpus)
        self.assertEqual(len(manifest["cases"]), 32)
        self.assertEqual(len(corpus.passages), 840)
        ranking = corpus.bm25("Pourquoi la tour Eiffel a-t-elle été construite ?")
        context = corpus.context(ranking)
        self.assertLessEqual(sum(len(e["text"]) for e in context), 4800)
        self.assertEqual(len({e["citation_id"] for e in context}), len(context))
        self.assertIn("Q243", [corpus.passages[pid]["site_id"] for pid in ranking[:5]])

    def test_embedding_api_shape_and_query_instruction(self):
        requests = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def send(self, data):
                body = json.dumps(data).encode(); self.send_response(200)
                self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)
            def do_GET(self):
                self.send({"data": [{"id": "mock-embedding-model"}]})
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send({"data": [{"index": 0, "embedding": [3, 4, 0]}]})
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            encoder = hybrid.Embeddings(f"http://127.0.0.1:{server.server_port}", dimensions=3)
            self.assertEqual(encoder.encode(hybrid.query_input("Une question")), [0.6, 0.8, 0])
            self.assertTrue(requests[0]["input"].startswith("Instruct:"))
            self.assertIn("\nQuery:Une question", requests[0]["input"])
            self.assertEqual(requests[0]["encoding_format"], "float")
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_invalid_reranking_is_not_silently_repaired(self):
        corpus = hybrid.Corpus(ROOT)
        ranking = corpus.bm25("Panthéon")[:3]
        client = mock.Mock()
        client.request.side_effect = [{"data": [{"id": "mock"}]},
                                      {"choices": [{"message": {"content": '{"order":["C1","C1","C1"]}'}, "finish_reason": "stop"}]}]
        with self.assertRaises(ValueError):
            hybrid.rerank(client, "Panthéon", ranking, corpus)

    def test_model_switch_stops_owned_worker_before_starting_another(self):
        active, created = [], []
        class Process:
            def __init__(self):
                self.returncode = None
            def poll(self):
                return self.returncode
            def terminate(self):
                self.returncode = 0
            def wait(self, timeout=None):
                return self.returncode
        def spawn(args, **kwargs):
            self.assertFalse(any(p.poll() is None for p in active))
            process = Process(); active.append(process); created.append(args)
            return process
        opener = mock.Mock()
        opener.open.side_effect = lambda *a, **k: io.BytesIO(b'{"status":"ok"}')
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(runtime.shutil, "which", return_value="mock-server"), \
             mock.patch.object(runtime, "port_busy", return_value=False), mock.patch.object(runtime.subprocess, "Popen", side_effect=spawn), \
             mock.patch.object(runtime.urllib.request, "build_opener", return_value=opener):
            models = runtime.Models(pathlib.Path("generation.gguf"), pathlib.Path("projector.gguf"), pathlib.Path(temp))
            models.start("embedding", pathlib.Path("embedding.gguf"))
            models.start("generation")
            self.assertEqual(active[0].poll(), 0)
            self.assertIn("--pooling", created[0])
            self.assertIn("--mmproj-offload", created[1])
            models.start("generation", generation_vision=False)
            self.assertEqual(active[1].poll(), 0)
            self.assertNotIn("--mmproj", created[2])
            self.assertNotIn("--mmproj-offload", created[2])
            self.assertNotIn("projector.gguf", created[2])
            for flag in ["--n-gpu-layers", "--ctx-size", "--batch-size", "--ubatch-size", "--reasoning-budget"]:
                self.assertEqual(created[1][created[1].index(flag)+1], created[2][created[2].index(flag)+1])
            models.stop()
            self.assertFalse(any(p.poll() is None for p in active))

    def test_profile_resource_summary_handles_units_and_separate_gpus(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = pathlib.Path(temp)
            (folder / "memory.jsonl").write_text('{"available_ram_mib": 900}\n{"available_ram_mib": null}\n{"available_ram_mib": 600}\n')
            (folder / "gpu.csv").write_text('timestamp, index, name, memory.used [MiB], memory.free [MiB]\n'
                't, 0, NVIDIA, 2700 MiB, 1200 MiB\nt, 0, NVIDIA, 2800 MiB, 1100 MiB\nt, 1, Other, N/A, N/A\n')
            result = compare_profiles.resource_summary(folder)
            self.assertEqual(result["min_available_ram_mib"], 600)
            self.assertEqual(result["median_available_ram_mib"], 750)
            self.assertEqual(result["gpu"]["0"]["peak_used_mib"], 2800)
            self.assertEqual(result["gpu"]["0"]["min_free_mib"], 1100)
            self.assertIsNone(result["gpu"]["1"]["peak_used_mib"])

    def test_profile_comparison_uses_identical_contexts_and_flags_status_failures(self):
        corpus = hybrid.Corpus(ROOT)
        manifest = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        observed = []
        def answer(database, question, client, retrieval_fn, retrieval_name):
            observed.append((question, retrieval_fn(database, question)))
            return {"answer": {"status": "answered"}, "reference_checks_passed": True, "total_seconds": 1}
        manager = mock.Mock()
        manager.start.return_value = "http://127.0.0.1:8080"
        monitor = mock.Mock()
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(compare_profiles.rag, "ask", side_effect=answer):
            result = compare_profiles.compare(corpus, manifest, manager, pathlib.Path(temp), resource_factory=monitor)
            self.assertEqual(manager.start.call_args_list, [mock.call("generation", generation_vision=True),
                                                          mock.call("generation", generation_vision=False)])
            self.assertEqual(observed[:6], observed[6:])
            self.assertEqual(len(observed), 12)
            for profile in result.values():
                self.assertEqual(profile["reference_checks_passed"], 6)
                self.assertEqual(profile["expected_status_matches"], 5)  # Fake model incorrectly answers the tariff.
                self.assertEqual(profile["factual_review"], "pending")
            self.assertEqual(monitor.return_value.stop.call_count, 2)

    def test_failed_text_start_preserves_baseline_and_stops_monitor(self):
        corpus = hybrid.Corpus(ROOT)
        manifest = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        manager = mock.Mock()
        manager.start.side_effect = ["http://127.0.0.1:8080", runtime.ModelStartupError("mock failure")]
        monitor = mock.Mock()
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(compare_profiles.rag, "ask", return_value={
                "answer": {"status": "answered"}, "reference_checks_passed": True, "total_seconds": 1}):
            with self.assertRaises(runtime.ModelStartupError):
                compare_profiles.compare(corpus, manifest, manager, pathlib.Path(temp), resource_factory=monitor)
            saved = json.loads((pathlib.Path(temp) / "profiles.json").read_text())
            self.assertIn("vision_baseline", saved)
            monitor.return_value.stop.assert_called_once()

    def test_complete_evaluation_flow_with_synthetic_vectors_and_fake_model(self):
        corpus = hybrid.Corpus(ROOT)
        manifest = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp:
            cache = hybrid.VectorCache(pathlib.Path(temp) / "synthetic.sqlite3", "MOCK-NOT-A-REAL-MODEL", dimensions=3)
            for role, key, text in run_local.cache_items(corpus, manifest["cases"]):
                cache.put(role, key, text, [1, 0, 0], 0)
            client = mock.Mock()
            def request(path, payload=None):
                if path == "/v1/models":
                    return {"data": [{"id": "mock-not-real-generation"}]}
                props = payload["response_format"]["json_schema"]["schema"]["properties"]
                if "order" in props:
                    content = {"order": props["order"]["items"]["enum"]}
                else:
                    reference = props["claims"]["items"]["properties"]["citations"]["items"]["enum"][0]
                    content = {"status": "answered", "claims": [{"text": "Simulation technique uniquement", "citations": [reference]}]}
                return {"choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}]}
            client.request.side_effect = request
            output = pathlib.Path(temp) / "report"; output.mkdir()
            result = run_local.benchmark(corpus, manifest, cache, client, output)
            self.assertEqual(result["generation_examples"], 18)
            self.assertEqual(result["reference_checks_passed"], 18)
            self.assertEqual(result["rerank_failures"], 0)
            self.assertEqual(result["expected_status_failures"], 3)  # Fake model answers even when data is absent.
            self.assertEqual(result["retrieval_metrics"]["dense"]["cases"], 31)
            self.assertEqual(result["rerank_subset_metrics"]["hybrid_llm_rerank"]["cases"], 5)
            self.assertEqual(result["factual_review"], "pending")
            self.assertEqual(len(list((output / "answers").glob("*.json"))), 18)


if __name__ == "__main__":
    unittest.main()
