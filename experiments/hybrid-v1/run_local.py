"""One local run: verified model, embeddings, retrieval ablations, reranking, answers."""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

import hybrid
import rag_local as rag
import runtime

ROOT = pathlib.Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def query_id(question):
    return hashlib.sha256(hybrid.query_input(question).encode()).hexdigest()


def embedding_cache_key(metadata, corpus):
    return hashlib.sha256(json.dumps({"model_sha256": metadata["sha256"], "corpus": corpus.fingerprint,
                                     "input_format": hybrid.FORMAT_VERSION, "task": hybrid.QUERY_TASK,
                                     "dimensions": 1024, "pooling": "last"}, sort_keys=True).encode()).hexdigest()


def cache_items(corpus, cases):
    for passage in corpus.passages.values():
        yield "document", passage["passage_id"], hybrid.document_input(passage, corpus.sources[passage["document_id"]])
    for case in cases:
        yield "query", query_id(case["question"]), hybrid.query_input(case["question"])


def rank_methods(corpus, case, cache):
    question = case["question"]
    vector = cache.get("query", query_id(question), hybrid.query_input(question))
    if vector is None:
        raise ValueError("Question embedding missing.")
    bm25 = corpus.bm25(question)
    dense = [pid for pid, score in cache.rank(vector, set(corpus.passages))]
    return {"bm25": bm25, "routed_current": corpus.routed(question), "dense": dense,
            "hybrid_rrf": hybrid.reciprocal_rank_fusion([bm25, dense])}


def metrics(rows):
    groups = {}
    for row in rows:
        score = row.get("score")
        if score is not None:
            groups.setdefault(row["method"], []).append(score)
    return {name: {"cases": len(scores), **{key: statistics.mean(float(score[key]) for score in scores)
              for key in ["known_evidence_hit_at_5", "known_evidence_mrr_at_10", "all_evidence_groups_at_5"]}}
            for name, scores in groups.items()}


def markdown_report(output, report):
    lines = ["# PatrimoTrip AI — comparaison locale", "", "Statut : " + report["status"], "",
             "Jeu de développement : passages justificatifs partiels, pas un test final indépendant.", "",
             "Les valeurs mesurent les passages annotés retrouvés, pas la véracité des réponses.", ""]
    for title, data in [("Recherche sur tout le jeu annoté", report.get("retrieval_metrics", {})),
                        ("Comparaison du reranking sur le même sous-ensemble", report.get("rerank_subset_metrics", {}))]:
        lines += ["## " + title, "", "| Méthode | Cas | Passage connu dans top 5 | Rang réciproque top 10 | Tous les groupes dans top 5 |",
                  "|---|---:|---:|---:|---:|"]
        for name, row in data.items():
            lines.append(f"| {name} | {row['cases']} | {row['known_evidence_hit_at_5']:.3f} | {row['known_evidence_mrr_at_10']:.3f} | {row['all_evidence_groups_at_5']:.3f} |")
        lines.append("")
    if report.get("error"):
        lines += ["## Opération bloquée", "", report["error"], ""]
    lines += ["## Lecture des réponses", "", "Les réponses et extraits sont dans answers/. Leur soutien factuel et les contradictions restent à relire.",
              "Un format conforme et une référence existante ne prouvent pas qu'une réponse est correcte.", "",
              "Les timings de génération avec requêtes pré-encodées ne comprennent pas le coût d'un changement de modèle par question.",
              "Les chargements du lot figurent séparément dans model-events.json ; la durée totale inclut la préparation.", "",
              "GPU : mesure globale à une seconde ; RAM : mémoire physique disponible globale, pas celle du seul processus."]
    (output / "rapport.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def benchmark(corpus, manifest, cache, client, output):
    rankings, rows = {}, []
    for case in manifest["cases"]:
        methods = rank_methods(corpus, case, cache)
        rankings[case["case_id"]] = methods
        for method, ranking in methods.items():
            rows.append({"case_id": case["case_id"], "kind": case["kind"], "method": method,
                         "score": hybrid.evaluate_rank(ranking, case), "ranking": ranking[:10]})
    save(output / "retrieval.json", rows)
    subset, answers = [], []
    answers_dir = output / "answers"
    answers_dir.mkdir()
    for case in manifest["cases"]:
        if case["case_id"] not in manifest["generation_case_ids"]:
            continue
        methods = rankings[case["case_id"]]
        print("Reranking et réponses : " + case["case_id"], flush=True)
        rerank_error = None
        try:
            methods["hybrid_llm_rerank"] = hybrid.rerank(client, case["question"], methods["hybrid_rrf"], corpus)
        except (ValueError, RuntimeError, KeyError) as error:
            rerank_error = str(error)
            methods["hybrid_llm_rerank"] = []
        for method, ranking in methods.items():
            subset.append({"case_id": case["case_id"], "method": method, "score": hybrid.evaluate_rank(ranking, case),
                           "ranking": ranking[:10], "error": rerank_error if method == "hybrid_llm_rerank" else None})
        for method in ["routed_current", "hybrid_rrf", "hybrid_llm_rerank"]:
            if method == "hybrid_llm_rerank" and rerank_error:
                result = {"question": case["question"], "error": rerank_error, "generation_skipped": True,
                          "reference_checks_passed": False, "note": "Reranking failed; no silent fallback to another method."}
            else:
                evidence = corpus.context(methods[method])
                try:
                    result = rag.ask(corpus.database, case["question"], client,
                                     retrieval_fn=lambda database, question: evidence, retrieval_name=method)
                except (ValueError, RuntimeError, KeyError) as error:
                    result = {"question": case["question"], "error": str(error), "reference_checks_passed": False}
            result.update({"case_id": case["case_id"], "method": method, "expected_answer_status": case.get("expected_answer_status"),
                           "factual_review": "pending"})
            save(answers_dir / f"{case['case_id']}-{method}.json", result)
            answers.append({"case_id": case["case_id"], "method": method,
                            "reference_checks_passed": result.get("reference_checks_passed", False),
                            "status": result.get("answer", {}).get("status"), "seconds": result.get("total_seconds"),
                            "expected_status_matches": (result.get("answer", {}).get("status") == case["expected_answer_status"])
                                if case.get("expected_answer_status") else None,
                            "error": result.get("error") or result.get("validation_error"), "factual_review": "pending"})
    save(output / "rerank-subset.json", subset)
    save(output / "answers-summary.json", answers)
    return {"retrieval_metrics": metrics(rows), "rerank_subset_metrics": metrics(subset),
            "generation_examples": len(answers), "reference_checks_passed": sum(a["reference_checks_passed"] for a in answers),
            "expected_status_failures": sum(a["expected_status_matches"] is False for a in answers),
            "rerank_failures": sum(bool(row["error"]) for row in subset if row["method"] == "hybrid_llm_rerank"),
            "factual_review": "pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "ask", "lexical"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--take-over-test-server", action="store_true",
                        help="Allow stopping only the recognized local 4B test server on port 8080.")
    parser.add_argument("--text-only", action="store_true", help="Do not load the visual projector for generation.")
    default_models = pathlib.Path.home() / "PatrimoTripAI" / "models" / "qwen3.5-4b"
    parser.add_argument("--generation-model", type=pathlib.Path, default=default_models / "Qwen3.5-4B-Q4_K_M.gguf")
    parser.add_argument("--projector", type=pathlib.Path, default=default_models / "mmproj-F16.gguf")
    args = parser.parse_args()
    if args.command == "ask" and (not args.question or len(args.question) > 1000):
        parser.error("ask requires a question of at most 1000 characters.")
    output = ROOT / "results" / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + runtime.uuid_suffix())
    output.mkdir(parents=True)
    report = {"status": "started", "utc_started": dt.datetime.now(dt.timezone.utc).isoformat(),
              "command": args.command, "python": sys.version.split()[0], "factual_review": "pending",
              "sqlite": rag.sqlite3.sqlite_version,
              "annotation_limit": "Development set, partial known-evidence labels, not held-out accuracy.",
              "generation_server_left_running": False}
    manager, resources, success = None, None, False
    started = time.perf_counter()
    try:
        corpus = hybrid.Corpus(ROOT)
        manifest = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        hybrid.validate_cases(manifest, corpus)
        report["corpus_fingerprint"] = corpus.fingerprint
        report["cases_sha256"] = rag.sha_file(ROOT / "cases.json")
        report["script_sha256"] = {p.name: rag.sha_file(p) for p in ROOT.glob("*.py")}
        if args.command == "lexical":
            rows = []
            for case in manifest["cases"]:
                for method, ranking in [("bm25", corpus.bm25(case["question"])), ("routed_current", corpus.routed(case["question"]))]:
                    rows.append({"case_id": case["case_id"], "method": method, "score": hybrid.evaluate_rank(ranking, case)})
            report.update({"status": "lexical_only_complete", "retrieval_metrics": metrics(rows),
                           "dense": "not_run", "reranking": "not_run", "generation": "not_run"})
            success = True
            return
        for path in [args.generation_model] + ([] if args.text_only else [args.projector]):
            if not path.is_file():
                raise ValueError(f"Previously prepared 4B file not found: {path}")
            with path.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise ValueError("Generation model or projector is not a GGUF file.")
        manager = runtime.Models(args.generation_model.resolve(), args.projector.resolve(), output, args.take_over_test_server)
        version = subprocess.run([manager.executable, "--version"], capture_output=True, text=True, errors="replace", timeout=20, env=manager.environment)
        report["llama_version"] = (version.stdout + version.stderr)[:2000]
        resources = runtime.Resources(output, manager.environment)
        resources.start()
        embedding_path, metadata = runtime.download_embedding(ROOT / "models")
        report["embedding_model"] = metadata
        report["generation_model"] = {"name": args.generation_model.name, "sha256": rag.sha_file(args.generation_model),
                                      "projector_sha256": None if args.text_only else rag.sha_file(args.projector), "gpu_layers": 20, "ctx_size": 4096,
                                      "vision_offload": not args.text_only, "verification": "Files previously checked by user; local digests recorded, not re-downloaded."}
        model_key = embedding_cache_key(metadata, corpus)
        cache = hybrid.VectorCache(ROOT / "vectors.sqlite3", model_key)
        report.update({"embedding_cache_namespace": model_key, "embedding_input_format": hybrid.FORMAT_VERSION,
                       "query_instruction": hybrid.QUERY_TASK})
        cases = manifest["cases"] if args.command == "run" else [{"question": args.question}]
        missing = sum(cache.get(role, key, text) is None for role, key, text in cache_items(corpus, cases))
        report["embedding_items_missing_at_start"] = missing
        manager.prepare_ports()
        if missing:
            resources.phase = "embedding_start"
            try:
                url = manager.start("embedding", embedding_path)
            except runtime.ModelStartupError:
                manager.stop()
                print("Le démarrage GPU des embeddings a échoué ; essai CPU avec le 4B arrêté.", flush=True)
                url = manager.start("embedding", embedding_path, embedding_cpu=True)
            encoder = hybrid.Embeddings(url)
            resources.phase = "embedding_index_and_queries"
            embedding_rows = []
            for n, (role, key, text) in enumerate(cache_items(corpus, cases), 1):
                _, reused, seconds = cache.encode(encoder, role, key, text)
                embedding_rows.append({"role": role, "id": key, "cache_hit": reused, "seconds": seconds})
                if n % 25 == 0:
                    print(f"Embeddings : {n}/{len(corpus.passages)+len(cases)} (cache repris automatiquement)", flush=True)
            save(output / "embedding-work.json", embedding_rows)
            manager.stop()
        resources.phase = "generation_start"
        url = manager.start("generation", generation_vision=not args.text_only)
        client = rag.LocalModel(url)
        resources.phase = "evaluation" if args.command == "run" else "answer"
        if args.command == "run":
            report.update(benchmark(corpus, manifest, cache, client, output))
            report["status"] = "complete" if report["rerank_failures"] == 0 and report["expected_status_failures"] == 0 and report["reference_checks_passed"] == report["generation_examples"] else "complete_with_failures"
        else:
            case = {"question": args.question}
            rankings = rank_methods(corpus, case, cache)
            ranking = hybrid.rerank(client, args.question, rankings["hybrid_rrf"], corpus)
            evidence = corpus.context(ranking)
            answer = rag.ask(corpus.database, args.question, client, retrieval_fn=lambda database, question: evidence,
                             retrieval_name="hybrid_rrf_llm_rerank")
            save(output / "answer.json", answer)
            print(answer.get("answer", {}).get("answer", answer.get("validation_error", "No answer")))
            report["status"] = "complete" if answer.get("reference_checks_passed") else "complete_with_failures"
        success = True
        report["generation_server_left_running"] = True
        report["generation_server_pid"] = manager.process.pid
        manager.event("generation_left_running_for_user")
        manager.process = None  # Leave the known generation server usable; no embedding server remains.
    except (Exception, KeyboardInterrupt) as error:
        report.update({"status": "failed", "error": str(error) or "Interrupted by user"})
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("Opération arrêtée : " + report["error"], flush=True)
    finally:
        if manager:
            manager.stop()
            save(output / "model-events.json", manager.events)
        if resources:
            resources.stop()
        report["total_wall_seconds"] = time.perf_counter()-started
        save(output / "summary.json", report)
        markdown_report(output, report)
        archive = output / "report.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in output.rglob("*"):
                if path.is_file() and path != archive:
                    bundle.write(path, path.relative_to(output))
        print("Rapport local : " + str(archive), flush=True)
        if not success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
