"""Bounded French question splitting, per-facet retrieval and individually validated answers."""
import json
import re
import time

import hybrid
import rag_local as rag
from run_local import query_id

VERSION = "facet_coverage_v2"
# Query expansion is retrieval-only: no monument, date, price or expected answer is inserted.
EXPANSIONS = {
    "commemoration": "commémoration anniversaire célébration célébrer célébrant",
    "event": "événement manifestation exposition inauguration",
    "designer": "architecte conception conçu auteur",
    "date": "année date construction début inauguration",
    "purpose": "fonction origine usage destination construit",
    "price": "tarif prix billet adulte actuel",
    "hours": "horaire fermeture ouverture actuel",
    "generic": "",
}


def normalized(text):
    return " ".join(rag.normalized_words(text))


def facet_kind(question):
    text = normalized(question)
    for kind, stems in [("price", ["tarif", "prix", "cout", "combien"]),
                        ("hours", ["horaire", "heure", "ferme", "ouvre"]),
                        ("commemoration", ["commemor", "celebr", "annivers"]),
                        ("designer", ["architect", "concu", "conception"]),
                        ("event", ["evenement", "occasion"]),
                        ("date", ["quand", "annee", "date"]),
                        ("purpose", ["fonction", "usage", "destin", "pourquoi"])]:
        if any(stem in text for stem in stems):
            return kind
    return "generic"


def plan_question(question, site_name):
    if not isinstance(question, str) or not question.strip() or len(question) > 800:
        raise ValueError("La question doit contenir entre 1 et 800 caractères.")
    # Conservative syntax rules, not a general semantic parser. Preserve the full query on overflow.
    lead = r"(?:(?:à|en|pour)\s+)?(?:qui\b|que\b|quel\w*\b|quand\b|comment\b|pourquoi\b|où\b|explique\w*\b|présente\w*\b|(?:sa|son|ses|la|le)\s+(?:fonction|usage|architecte|date|histoire)\b)"
    parts = re.split(r"\s+(?:et|ainsi que|puis)\s+(?="+lead+r")|\?\s+(?=\S)|;\s*", question, flags=re.I)
    parts = [p.strip(' ,;?') for p in parts if p.strip(' ,;?')]
    compound = 1 < len(parts) <= 3
    if not compound:
        parts = [question.strip().rstrip('?')]
    base_query = f"À propos du monument « {site_name} » : {question}"
    facets = []
    for n, part in enumerate(parts, 1):
        part = part.rstrip('?')+' ?'
        kind = facet_kind(part)
        query = f"À propos du monument « {site_name} » : {part}"
        if compound and EXPANSIONS[kind]:
            query += '\nTermes de recherche : '+EXPANSIONS[kind]
        facets.append({"facet_id": f"F{n}", "question": part, "kind": kind, "query": query})
    # Preserve the event antecedent of a commemoration question for dense retrieval.
    # Sparse terms stay facet-specific, avoiding repetition of the other clause's generic words.
    event_parts = [f['question'] for f in facets if f['kind'] == 'event']
    if compound and event_parts:
        for facet in facets:
            if facet['kind'] == 'commemoration':
                facet['lexical_query'] = facet['query']
                facet['query'] += '\nContexte de l’événement demandé : '+' '.join(event_parts)
    return {"version": VERSION, "question": question, "base_query": base_query, "facets": facets,
            "compound": compound, "split_method": "bounded_french_syntax_rules" if compound else "single_query_fallback",
            "limitation": "A candidate retrieved for a facet is not proof that the facet is answered."}


def extra_queries(plan):
    return [f["query"] for f in plan["facets"]] if plan["compound"] else []


def lexical_query(query, site_name):
    ignored = set(rag.normalized_words(site_name)) | {"propos", "monument", "batiment", "edifice", "termes", "recherche"}
    return " ".join(word for word in rag.keywords(query) if word not in ignored)


def retrieve(corpus, site_id, site_name, plan, cache):
    allowed = {pid for pid, passage in corpus.passages.items() if passage["site_id"] == site_id}
    rankings = []
    for facet in plan["facets"]:
        vector = cache.get("query", query_id(facet["query"]), hybrid.query_input(facet["query"]))
        if vector is None:
            raise ValueError("Encodage d'une sous-question absent.")
        lexical = corpus.bm25(lexical_query(facet.get("lexical_query", facet["query"]), site_name), site_id=site_id)
        dense = [pid for pid, _ in cache.rank(vector, allowed)]
        fused = hybrid.reciprocal_rank_fusion([lexical, dense])
        rankings.append({"facet_id": facet["facet_id"], "question": facet["question"], "query": facet["query"],
                         "bm25": lexical, "dense": dense, "hybrid_rrf": fused})
    # Round-robin candidates let each aspect contribute before one ranking consumes the context.
    merged = []
    for position in range(50):
        for ranking in rankings:
            candidates = ranking["hybrid_rrf"]
            if position < len(candidates) and candidates[position] not in merged:
                merged.append(candidates[position])
    evidence = corpus.context(merged, max_per_document=4, balanced_sentences=True)
    if any(e["site_id"] != site_id for e in evidence):
        raise ValueError("Une preuve appartient à un autre monument.")
    return evidence, {"version": VERSION, "context_policy": "balanced_sentence_prefix_v1",
                      "context_limit_chars": 4800, "context_limit_passages": 4,
                      "plan": plan, "per_facet": rankings, "merged_ranking": merged[:20],
                      "candidate_presence_not_semantic_support": {f["facet_id"]: [e["passage_id"] for e in evidence
                          if e["passage_id"] in f["hybrid_rrf"][:5]] for f in rankings}}


def render_answers(draft, plan, evidence):
    if not isinstance(draft, dict) or set(draft) != {"answers"} or not isinstance(draft["answers"], list):
        raise ValueError("Réponse par sous-question invalide.")
    expected = [f["facet_id"] for f in plan["facets"]]
    parts = {}
    for part in draft["answers"]:
        if not isinstance(part, dict) or set(part) != {"facet_id", "status", "claims"}:
            raise ValueError("Champs d'une sous-réponse invalides.")
        key = part["facet_id"]
        if not isinstance(key, str) or key not in expected or key in parts or not isinstance(part["claims"], list) or len(part["claims"]) != 1:
            raise ValueError("Sous-question absente, dupliquée ou réponse trop longue.")
        rendered = rag.render_claims({"status": part["status"], "claims": part["claims"]}, evidence)
        parts[key] = (part, rendered)
    if set(parts) != set(expected):
        raise ValueError("Toutes les sous-questions doivent avoir un statut explicite.")
    statuses = [parts[k][1]["status"] for k in expected]
    status = "answered" if "answered" in statuses else "needs_clarification" if "needs_clarification" in statuses else "insufficient_sources"
    text, refs, review = [], [], []
    for key in expected:
        part, rendered = parts[key]
        text.append(rendered["answer"])
        refs.extend(ref for ref in rendered["citations"] if ref not in refs)
        review.extend({**claim, "facet_id": key, "facet_status": part["status"]} for claim in part["claims"])
    answer = rag.validate_answer({"status": status, "answer": '\n'.join(text), "citations": refs}, evidence)
    return answer, {key: parts[key][1]["status"] for key in expected}, review


def ask(question, plan, evidence, client, system_prompt):
    started = time.perf_counter()
    result = {"question": question, "retrieval": VERSION, "evidence": evidence, "plan": plan,
              "prompt_version": "photo_facets_v1",
              "generation_format": "facet_claims_v1", "semantic_support_review": "pending"}
    if not evidence:
        result.update(answer={"status": "insufficient_sources", "answer": "Aucun extrait correspondant n'a été retrouvé.", "citations": []},
                      model_called=False, reference_checks_passed=True, facet_statuses={f['facet_id']: 'insufficient_sources' for f in plan['facets']},
                      review_claims=[], total_seconds=time.perf_counter()-started)
        return result
    ids = [f["facet_id"] for f in plan["facets"]]
    refs = [e["citation_id"] for e in evidence]
    claim = {"type": "object", "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 400},
             "citations": {"type": "array", "items": {"type": "string", "enum": refs}, "maxItems": len(refs)}},
             "required": ["text", "citations"], "additionalProperties": False}
    part = {"type": "object", "properties": {"facet_id": {"type": "string", "enum": ids},
            "status": {"type": "string", "enum": ["answered", "insufficient_sources", "needs_clarification"]},
            "claims": {"type": "array", "minItems": 1, "maxItems": 1, "items": claim}},
            "required": ["facet_id", "status", "claims"], "additionalProperties": False}
    schema = {"type": "object", "properties": {"answers": {"type": "array", "minItems": len(ids), "maxItems": len(ids), "items": part}},
              "required": ["answers"], "additionalProperties": False}
    sources = [{"reference": e["citation_id"], "title": e["source"]["title"], "section": e["section_path"],
                "updated_at": e["source"].get("updated_at"), "text": e["text"]} for e in evidence]
    prompt = system_prompt + "\nProduis answers avec une entrée pour chaque facet_id demandé. Chaque entrée possède son propre status et une phrase dans claims. Traite exactement l'aspect demandé. Si cet aspect manque dans les extraits, marque cette entrée insufficient_sources sans compléter avec tes connaissances. Une autre sous-question peut néanmoins être answered. Ne confonds pas les objets des sous-questions."
    model = client.request('/v1/models')['data'][0]['id']
    payload = {"model": model, "messages": [{"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"question": question, "aspects": [{"facet_id": f['facet_id'], "question": f['question']} for f in plan['facets']],
                                                "extraits_documentaires_donnees": sources}, ensure_ascii=False)}],
        "temperature": 0, "seed": 42, "max_tokens": 768, "stream": False, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": "facet_answers", "strict": True, "schema": schema}}}
    reply = client.request('/v1/chat/completions', payload)
    choice = reply['choices'][0]
    result.update(model_called=True, model_name=str(model).replace('\\', '/').rsplit('/', 1)[-1], raw_response=choice['message']['content'],
                  finish_reason=choice.get('finish_reason'), timings=reply.get('timings'), usage=reply.get('usage'))
    try:
        if result['finish_reason'] == 'length':
            raise ValueError("Réponse par aspects tronquée.")
        draft = json.loads(result['raw_response'])
        result['facet_response'] = draft
        answer, statuses, review = render_answers(draft, plan, evidence)
        result.update(answer=answer, facet_statuses=statuses, review_claims=review, reference_checks_passed=True,
                      all_facets_answered=all(s == 'answered' for s in statuses.values()))
    except (ValueError, TypeError, KeyError) as error:
        result.update(reference_checks_passed=False, validation_error=str(error))
    result['total_seconds'] = time.perf_counter()-started
    return result
