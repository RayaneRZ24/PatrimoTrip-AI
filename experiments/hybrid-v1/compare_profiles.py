"""Compare the same local 4B with and without its visual projector; no download."""
import argparse
import csv
import datetime as dt
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

import hybrid
import rag_local as rag
import runtime
from run_local import save

ROOT = pathlib.Path(__file__).resolve().parent


def resource_summary(folder):
    result = {"scope": "Global available physical RAM and GPU use; one-second samples, not process allocations."}
    memory = folder / "memory.jsonl"
    if memory.exists():
        rows = [json.loads(line) for line in memory.read_text(encoding="utf-8").splitlines()]
        values = [r["available_ram_mib"] for r in rows if r["available_ram_mib"] is not None]
        if values:
            result.update(ram_samples=len(values), min_available_ram_mib=min(values),
                          median_available_ram_mib=statistics.median(values))
    gpu = folder / "gpu.csv"
    if gpu.exists():
        with gpu.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
            rows = list(csv.DictReader(stream))
        devices = {}
        for row in rows:
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            index = row.get("index", "unknown")
            values = devices.setdefault(index, {"name": row.get("name"), "used": [], "free": []})
            for key, target in [("memory.used [MiB]", "used"), ("memory.free [MiB]", "free")]:
                match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(?:MiB)?", row.get(key, ""))
                if match:
                    values[target].append(float(match[1]))
        result["gpu"] = {index: {"name": v["name"], "samples": len(v["used"]),
            "peak_used_mib": max(v["used"]) if v["used"] else None,
            "min_free_mib": min(v["free"]) if v["free"] else None} for index, v in devices.items()}
    return result


def compare(corpus, manifest, manager, output, resource_factory=runtime.Resources):
    selected = [c for c in manifest["cases"] if c["case_id"] in manifest["generation_case_ids"]]
    # Freeze identical passages before loading either profile; no new embeddings needed.
    inputs = [{"case": case, "evidence": corpus.context(corpus.routed(case["question"]))} for case in selected]
    save(output / "fixed-contexts.json", inputs)
    profiles = {}
    for name, vision in [("vision_baseline", True), ("text_only", False)]:
        folder = output / name
        folder.mkdir()
        manager.logs = folder
        monitor = None
        try:
            print("Profil : " + name, flush=True)
            started = time.perf_counter()
            url = manager.start("generation", generation_vision=vision)
            startup = time.perf_counter()-started
            monitor = resource_factory(folder, manager.environment)
            monitor.phase = "loaded_profile_questions"
            monitor.start()
            client = rag.LocalModel(url)
            answers = []
            for item in inputs:
                case, evidence = item["case"], item["evidence"]
                print(name + " : " + case["case_id"], flush=True)
                try:
                    answer = rag.ask(corpus.database, case["question"], client,
                                     retrieval_fn=lambda database, question: evidence,
                                     retrieval_name="fixed_routed_context_profile_comparison")
                except (ValueError, RuntimeError, KeyError, OSError) as error:
                    answer = {"error": str(error), "reference_checks_passed": False}
                expected = case.get("expected_answer_status", "answered")
                answer.update(case_id=case["case_id"], profile=name, expected_status=expected,
                              status_matches=answer.get("answer", {}).get("status") == expected,
                              factual_review="pending")
                save(folder / (case["case_id"] + ".json"), answer)
                answers.append({"case_id": case["case_id"], "seconds": answer.get("total_seconds"),
                                "reference_checks_passed": answer.get("reference_checks_passed", False),
                                "status_matches": answer["status_matches"], "error": answer.get("error") or answer.get("validation_error")})
            timings = [a["seconds"] for a in answers if a["seconds"] is not None]
            profiles[name] = {"vision_loaded": vision, "startup_seconds": startup, "answers": answers,
                              "reference_checks_passed": sum(a["reference_checks_passed"] for a in answers),
                              "expected_status_matches": sum(a["status_matches"] for a in answers),
                              "questions": len(inputs), "median_answer_seconds": statistics.median(timings) if timings else None,
                              "factual_review": "pending"}
        finally:
            if monitor:
                monitor.stop()
        profiles[name]["resources"] = resource_summary(folder)
        save(output / "profiles.json", profiles)  # Preserve the baseline if the next startup fails.
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take-over-test-server", action="store_true")
    models = pathlib.Path.home() / "PatrimoTripAI" / "models" / "qwen3.5-4b"
    parser.add_argument("--generation-model", type=pathlib.Path, default=models / "Qwen3.5-4B-Q4_K_M.gguf")
    parser.add_argument("--projector", type=pathlib.Path, default=models / "mmproj-F16.gguf")
    args = parser.parse_args()
    output = ROOT / "results" / ("profiles-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + runtime.uuid_suffix())
    output.mkdir(parents=True)
    report = {"status": "started", "python": sys.version.split()[0], "factual_review": "pending",
              "utc_started": dt.datetime.now(dt.timezone.utc).isoformat(), "server_left_running": False,
              "limitation": "One sequential pass per profile, global resource samples, six development questions. Reference/status checks do not prove factual quality."}
    manager, success = None, False
    started = time.perf_counter()
    try:
        corpus = hybrid.Corpus(ROOT)
        manifest = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
        hybrid.validate_cases(manifest, corpus)
        for path in [args.generation_model, args.projector]:
            with path.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise ValueError("Not a GGUF model: " + str(path))
        report.update(corpus_fingerprint=corpus.fingerprint, cases_sha256=rag.sha_file(ROOT / "cases.json"),
                      generation_sha256=rag.sha_file(args.generation_model), projector_sha256=rag.sha_file(args.projector),
                      script_sha256={p.name: rag.sha_file(p) for p in ROOT.glob("*.py")},
                      settings={"gpu_layers": 20, "ctx_size": 4096, "batch_size": 256, "ubatch_size": 128, "parallel": 1})
        manager = runtime.Models(args.generation_model.resolve(), args.projector.resolve(), output, args.take_over_test_server)
        version = subprocess.run([manager.executable, "--version"], capture_output=True, text=True, errors="replace", timeout=20, env=manager.environment)
        report["llama_version"] = (version.stdout + version.stderr)[:2000]
        manager.prepare_ports()
        profiles = compare(corpus, manifest, manager, output)
        report["profiles"] = profiles
        passed = all(p["reference_checks_passed"] == p["questions"] == p["expected_status_matches"] for p in profiles.values())
        report["status"] = "complete" if passed else "complete_with_failures"
        report.update(server_left_running=True, server_profile="text_only", server_pid=manager.process.pid)
        manager.event("text_server_left_running_for_user")
        manager.process = None
        success = True
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error) or "Interrupted")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("Comparaison arrêtée : " + report["error"], flush=True)
    finally:
        if manager:
            manager.stop()
            save(output / "model-events.json", manager.events)
        report["total_wall_seconds"] = time.perf_counter()-started
        save(output / "summary.json", report)
        with zipfile.ZipFile(output / "report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.rglob("*"):
                if path.is_file() and path.name != "report.zip":
                    archive.write(path, path.relative_to(output))
        print("Rapport local : " + str(output / "report.zip"), flush=True)
        if success:
            print("Le serveur texte reste prêt sur http://127.0.0.1:8080 ; les photos nécessitent le profil vision.", flush=True)
        else:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
