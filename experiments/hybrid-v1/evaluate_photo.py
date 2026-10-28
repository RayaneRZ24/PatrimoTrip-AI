"""Frozen additional evaluation of the local photo pipeline, without gold-label prompts."""
import argparse
import datetime as dt
import json
import pathlib
import statistics
import subprocess
import time
import traceback
import zipfile

import hybrid
import photo_local as photo
import rag_local as rag
import runtime
from compare_profiles import resource_summary
from run_local import ROOT, save

DATASET = ROOT / "evaluation-v1"
IDENTITY_REGRESSION_CASES = {"q01", "q03", "q04", "q05", "q11", "q12"}


def select_cases(manifest, identity_regression=False):
    if not identity_regression:
        return manifest
    selected = [c for c in manifest["cases"] if c["case_id"] in IDENTITY_REGRESSION_CASES]
    if {c["case_id"] for c in selected} != IDENTITY_REGRESSION_CASES:
        raise ValueError("Cas manquants dans le sous-ensemble de régression.")
    return {**manifest, "cases": selected, "selection": "previously_seen_identity_regression_v1"}


def previous_comparison(results_root, output, rows, manifest_sha, assets_sha):
    for path in sorted(results_root.glob("evaluation-*/summary.json"), reverse=True):
        if path.parent == output:
            continue
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if (summary.get("status") != "complete" or summary.get("cases_planned") != 12 or
                    summary.get("manifest_sha256") != manifest_sha or summary.get("assets_sha256") != assets_sha):
                continue
            before = {r["case_id"]: r for r in json.loads((path.parent / "cases-summary.json").read_text(encoding="utf-8"))}
            if len(before) != 12 or any(r["case_id"] not in before or before[r["case_id"]]["asset_id"] != r["asset_id"] for r in rows):
                continue
            fields = ["predicted_site_id", "visual_status", "identity_exact", "safe_rejection_without_history", "answer_status"]
            comparison = {"before_run": path.parent.name, "scope": "Same selected cases before/after a development correction, not an independent accuracy estimate.",
                          "cases": [{"case_id": r["case_id"], "asset_id": r["asset_id"],
                                     "before": {k: before[r["case_id"]].get(k) for k in fields},
                                     "after": {k: r.get(k) for k in fields}} for r in rows]}
            save(output / "identity-before-after.json", comparison)
            return path.parent.name
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def verify_dataset(dataset, corpus):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assets_list = json.loads((dataset / "assets.json").read_text(encoding="utf-8"))
    assets = {a["asset_id"]: a for a in assets_list}
    if len(assets) != len(assets_list):
        raise ValueError("Duplicate image identifier.")
    previous = set(manifest["previous_image_sha256"])
    hashes = set()
    for a in assets.values():
        path = (dataset / a["image"]).resolve()
        if not path.is_relative_to(dataset.resolve()):
            raise ValueError("Image outside dataset.")
        _, metadata = photo.load_image(path)
        if metadata["sha256"] != a["sha256"] or a["sha256"] in previous or a["sha256"] in hashes:
            raise ValueError("Image checksum mismatch or repeated reference image.")
        if a["license"] not in {"CC BY-SA 3.0", "CC BY-SA 4.0"} or not a["artist"] or not a["source_page"]:
            raise ValueError("Incomplete image attribution.")
        hashes.add(a["sha256"])
    ids = set()
    for c in manifest["cases"]:
        if c["case_id"] in ids or not c["case_id"].isalnum() or c["asset_id"] not in assets:
            raise ValueError("Invalid or duplicate case identifier.")
        ids.add(c["case_id"])
        if not isinstance(c["question"], str) or not c["question"].strip() or len(c["question"]) > 800:
            raise ValueError("Invalid question.")
        for annotation in c["known_evidence"]:
            p = corpus.passages[annotation["passage_id"]]
            if not annotation["quote"] or annotation["quote"] not in p["text"] or p["site_id"] != assets[c["asset_id"]]["expected_site_id"]:
                raise ValueError("Incorrect known-evidence annotation.")
    return manifest, assets


def score_case(case, asset, result, evidence):
    prediction = result.get("identification", {})
    identity_check = prediction.get("identity_check", {})
    predicted = prediction.get("site_id")
    status = prediction.get("status")
    expected = asset["expected_site_id"]
    exact = status == "identified" and predicted == expected if expected != "unknown" else None
    rejected = status in {"uncertain", "out_of_scope"} and predicted == "unknown" and result.get("historical_answer_generated") is False
    known_hit = any(e.get("passage_id") == a["passage_id"] and a["quote"] in e.get("text", "")
                    for a in case["known_evidence"] for e in evidence) if case["known_evidence"] else None
    expected_status = case["expected_answer_status"]
    return {"predicted_site_id": predicted, "visual_status": status, "identity_exact": exact,
            "identity_check_version": identity_check.get("version"),
            "model_canonical_name": identity_check.get("canonical_name"),
            "model_city": identity_check.get("city"), "model_country": identity_check.get("country"),
            "identity_conflicts": identity_check.get("conflicts", []),
            "safe_rejection_without_history": rejected,
            "outside_classified": status == "out_of_scope" and predicted == "unknown" if asset["kind"] == "outside" else None,
            "detail_correct_or_abstained": bool(exact or rejected) if asset["kind"] == "detail" else None,
            "ambiguous_clarification": status == "uncertain" and rejected if asset["kind"] == "ambiguous" else None,
            "answer_status": result.get("answer_status"), "known_evidence_in_context": known_hit,
            "facet_statuses": result.get("facet_statuses"), "all_facets_answered": result.get("all_facets_answered"),
            "reference_checks_passed": result.get("reference_checks_passed", False),
            "format_and_expected_status_on_correct_identity": bool(exact and result.get("reference_checks_passed") and
                result.get("answer_status") == expected_status) if expected_status else None,
            "factual_review": "pending"}


def review_entries(case_id, folder):
    path = folder / "answer.json"
    if not path.exists():
        return []
    answer = json.loads(path.read_text(encoding="utf-8"))
    if not answer.get("reference_checks_passed"):
        return [{"case_id": case_id, "kind": "rejected_answer", "review": "not_reviewed",
                 "validation_error": answer.get("validation_error")}]
    sources = {e["citation_id"]: e for e in answer["evidence"]}
    claims = answer.get("review_claims", answer.get("claims_response", {}).get("claims", []))
    return [{"case_id": case_id, "claim_index": n, "answer_status": answer["answer"]["status"],
             "facet_id": claim.get("facet_id"), "facet_status": claim.get("facet_status"),
             "claim": claim["text"], "citations": claim["citations"],
             "cited_excerpts": [{"reference": ref, "text": sources[ref]["text"], "title": sources[ref]["source"]["title"],
                                "url": sources[ref]["source"]["url"]} for ref in claim["citations"]],
             "all_supplied_excerpts": [{"reference": ref, "text": e["text"]} for ref, e in sources.items()],
             "support": "not_reviewed", "answers_question": "not_reviewed", "notes": ""}
            for n, claim in enumerate(claims, 1)]


def summarize(rows):
    unique = {}
    disagreements = 0
    for row in rows:
        first = unique.setdefault(row["asset_id"], row)
        if first is not row and (row.get("predicted_site_id"), row.get("visual_status")) != (first.get("predicted_site_id"), first.get("visual_status")):
            disagreements += 1
    standard = [r for r in unique.values() if r["asset_kind"] == "standard"]
    outside = [r for r in unique.values() if r["asset_kind"] == "outside"]
    detail = [r for r in unique.values() if r["asset_kind"] == "detail"]
    ambiguous = [r for r in unique.values() if r["asset_kind"] == "ambiguous"]
    documented = [r for r in rows if r["case_kind"] == "documented"]
    missing = [r for r in rows if r["case_kind"] == "missing_current"]
    measured = [r["seconds"] for r in rows if not r.get("error")]
    return {"cases_attempted": len(rows), "case_errors": sum(bool(r.get("error")) for r in rows),
            "unique_images_attempted": len(unique), "repeated_image_prediction_disagreements": disagreements,
            "standard_identity_exact": sum(bool(r.get("identity_exact")) for r in standard), "standard_images": len(standard),
            "outside_safe_rejections": sum(bool(r.get("safe_rejection_without_history")) for r in outside),
            "outside_classified": sum(bool(r.get("outside_classified")) for r in outside), "outside_images": len(outside),
            "detail_correct_or_abstained": sum(bool(r.get("detail_correct_or_abstained")) for r in detail),
            "detail_identity_exact": sum(bool(r.get("identity_exact")) for r in detail), "detail_images": len(detail),
            "ambiguous_clarifications": sum(bool(r.get("ambiguous_clarification")) for r in ambiguous), "ambiguous_images": len(ambiguous),
            "documented_known_evidence_in_context": sum(bool(r.get("known_evidence_in_context")) for r in documented),
            "documented_format_and_expected_status": sum(bool(r.get("format_and_expected_status_on_correct_identity")) for r in documented),
            "documented_cases": len(documented),
            "missing_current_abstentions_on_correct_identity": sum(bool(r.get("format_and_expected_status_on_correct_identity")) for r in missing),
            "missing_current_cases": len(missing), "median_completed_case_seconds": statistics.median(measured) if measured else None,
            "factual_review": "pending", "automatic_factual_accuracy": "not_measured"}


def run_suite(corpus, sites, manifest, assets, dataset, manager, output, resource_factory=runtime.Resources):
    rows, review = [], []
    for n, case in enumerate(manifest["cases"], 1):
        asset = assets[case["asset_id"]]
        folder = output / "cases" / case["case_id"]
        folder.mkdir(parents=True)
        manager.logs = folder
        monitor = resource_factory(folder, manager.environment)
        row = {"case_id": case["case_id"], "asset_id": case["asset_id"], "case_kind": case["kind"], "asset_kind": asset["kind"]}
        started = time.perf_counter()
        print(f"Évaluation : {n}/{len(manifest['cases'])} — {case['case_id']}", flush=True)
        try:
            data, metadata = photo.load_image(dataset / asset["image"])
            monitor.start()
            # Never pass expected identity, expected status or annotated evidence to inference.
            result = photo.pipeline(corpus, sites, data, metadata, case["question"], manager, folder, monitor)
            save(folder / "result.json", result)
            retrieval = folder / "retrieval.json"
            evidence = json.loads(retrieval.read_text(encoding="utf-8"))["evidence"] if retrieval.exists() else []
            row.update(score_case(case, asset, result, evidence))
            row["pipeline_status"] = result["status"]
            review.extend(review_entries(case["case_id"], folder))
        except Exception as error:
            manager.stop()
            row["error"] = str(error)
            (folder / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            monitor.stop()
        row["seconds"] = time.perf_counter()-started
        row["resources"] = resource_summary(folder)
        rows.append(row)
        save(output / "cases-summary.json", rows)
        save(output / "review.json", review)
    return summarize(rows)


def write_report(output, report):
    lines = ["# Évaluation supplémentaire du parcours photo", "", "Statut d'exécution : " + report["status"], "",
             "Les scores techniques et de statut ne prouvent pas la véracité des réponses.",
             "Petit jeu documentaire et visuel ; pas une estimation d'exactitude générale.", ""]
    if report.get("evaluation_kind") == "identity_development_regression":
        lines += ["Régression de développement : six cas déjà examinés, après correction de l'identité. Ces résultats ne constituent pas un nouveau test indépendant et ne se comparent pas directement aux proportions du jeu complet.", ""]
    metrics = report.get("metrics", {})
    for good, total, label in [("standard_identity_exact", "standard_images", "Identités exactes, vues générales"),
        ("detail_correct_or_abstained", "detail_images", "Détails : identité correcte ou abstention prudente"),
        ("outside_safe_rejections", "outside_images", "Hors corpus : rejet sans histoire parisienne"),
        ("ambiguous_clarifications", "ambiguous_images", "Image sans contexte : clarification"),
        ("documented_known_evidence_in_context", "documented_cases", "Passage annoté présent dans le contexte"),
        ("documented_format_and_expected_status", "documented_cases", "Identité, format et statut attendus ; faits à relire"),
        ("missing_current_abstentions_on_correct_identity", "missing_current_cases", "Tarif/horaire : abstention après bonne identification")]:
        lines.append(f"- {label} : {metrics.get(good, 0)}/{metrics.get(total, 0)}")
    lines += ["", "Les répétitions d'une image ne gonflent pas le score visuel : seule sa première tentative compte.",
              "Les preuves annotées sont partielles : un autre passage peut être pertinent.", "",
              "## Relecture", "", "Dans review.json, examiner chaque affirmation et les extraits cités. Renseigner support avec supported, partial, unsupported ou contradicted ; answers_question avec yes, partial ou no. Laisser not_reviewed tant que la lecture n'est pas faite.",
              "Vérifier les dates, les superlatifs, les détails ajoutés et les abstentions injustifiées. Les observations visuelles restent aussi à comparer aux images du jeu.", "",
              "Les durées par cas incluent les changements de modèle et les calculs nécessaires, avec réutilisation du cache. Les mesures de mémoire sont globales et échantillonnées à une seconde."]
    if report.get("error"):
        lines += ["", "Erreur : " + report["error"]]
    (output / "rapport.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take-over-test-server", action="store_true")
    parser.add_argument("--identity-regression", action="store_true", help="Run six previously seen cases to check the identity correction; not an independent evaluation.")
    models = pathlib.Path.home() / "PatrimoTripAI" / "models" / "qwen3.5-4b"
    parser.add_argument("--generation-model", type=pathlib.Path, default=models / "Qwen3.5-4B-Q4_K_M.gguf")
    parser.add_argument("--projector", type=pathlib.Path, default=models / "mmproj-F16.gguf")
    args = parser.parse_args()
    prefix = "evaluation-identity-" if args.identity_regression else "evaluation-"
    output = ROOT / "results" / (prefix+dt.datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+runtime.uuid_suffix())
    output.mkdir(parents=True)
    manager, started, success = None, time.perf_counter(), False
    report = {"status": "started", "utc_started": dt.datetime.now(dt.timezone.utc).isoformat(), "factual_review": "pending",
              "identity_version": photo.IDENTITY_VERSION,
              "retrieval_version": photo.facet_rag.VERSION,
              "evaluation_kind": "identity_development_regression" if args.identity_regression else "full_additional_evaluation",
              "scope": "Previously examined cases after identity correction; development regression, not an independent test." if args.identity_regression else
                       "Additional small frozen evaluation; no claim of general accuracy or independence from model training."}
    try:
        corpus = hybrid.Corpus(ROOT)
        sites = {s["site_id"]: s for s in json.loads((ROOT / "corpus/sites.json").read_text(encoding="utf-8"))}
        manifest, assets = verify_dataset(DATASET, corpus)
        manifest = select_cases(manifest, args.identity_regression)
        for path in [args.generation_model, args.projector]:
            with path.open("rb") as stream:
                if stream.read(4) != b"GGUF":
                    raise ValueError("Invalid GGUF file: " + path.name)
        report.update(cases_planned=len(manifest["cases"]), corpus_fingerprint=corpus.fingerprint,
                      manifest_sha256=rag.sha_file(DATASET / "manifest.json"), assets_sha256=rag.sha_file(DATASET / "assets.json"),
                      generation_sha256=rag.sha_file(args.generation_model), projector_sha256=rag.sha_file(args.projector),
                      script_sha256={p.name: rag.sha_file(p) for p in ROOT.glob("*.py")},
                      generation_settings={"gpu_layers": 20, "context": 4096, "batch": 256, "ubatch": 128,
                                           "parallel": 1, "reasoning_budget": 0,
                                           "single_prompt_version": "photo_partial_answers_v2", "compound_prompt_version": "photo_facets_v1"})
        report["selected_case_ids"] = [c["case_id"] for c in manifest["cases"]]
        save(output / "evaluation-manifest.json", manifest)
        save(output / "evaluation-assets.json", list(assets.values()))
        manager = runtime.Models(args.generation_model.resolve(), args.projector.resolve(), output, args.take_over_test_server)
        version = subprocess.run([manager.executable, "--version"], capture_output=True, text=True, errors="replace", timeout=20, env=manager.environment)
        report["llama_version"] = (version.stdout+version.stderr)[:2000]
        manager.prepare_ports()
        report["metrics"] = run_suite(corpus, sites, manifest, assets, DATASET, manager, output)
        if args.identity_regression:
            rows = json.loads((output / "cases-summary.json").read_text(encoding="utf-8"))
            report["comparison_before_run"] = previous_comparison(ROOT / "results", output, rows,
                                                                 report["manifest_sha256"], report["assets_sha256"])
        report["status"] = "complete_with_case_errors" if report["metrics"]["case_errors"] else "complete"
        success = report["metrics"]["case_errors"] == 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error) or "Interrupted")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("Évaluation arrêtée : " + report["error"], flush=True)
    finally:
        if manager:
            manager.stop()
            save(output / "model-events.json", manager.events)
        report["server_left_running"] = False
        report["total_wall_seconds"] = time.perf_counter()-started
        save(output / "summary.json", report)
        write_report(output, report)
        with zipfile.ZipFile(output / "report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output.rglob("*"):
                if path.is_file() and path.name != "report.zip":
                    archive.write(path, path.relative_to(output))
        print("Rapport local : " + str(output / "report.zip"), flush=True)
        if not success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
