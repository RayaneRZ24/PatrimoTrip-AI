"use strict";
const $ = (id) => document.getElementById(id);
let token = "", catalog = [], credits = {}, selectedFile = null, previewURL = null;
function element(tag, text, className) { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (className) el.className = className; return el; }
function notify(message) { $("notice").textContent = message; $("notice").classList.toggle("hidden", !message); }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-PatrimoTrip-Token": token}, body: JSON.stringify(body)});
  const value = await response.json(); if (!response.ok) throw new Error(value.error || "La demande n’a pas abouti."); return value;
}
function show(view) {
  document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === view));
  document.querySelectorAll(".tab").forEach(el => { const active = el.dataset.view === view; el.classList.toggle("active", active); el.setAttribute("aria-selected", String(active)); el.tabIndex = active ? 0 : -1; });
}
document.querySelector(".tabs").setAttribute("role", "tablist");
document.querySelectorAll(".tab").forEach((button, index, buttons) => {
  button.setAttribute("role", "tab"); button.id = `tab-${button.dataset.view}`; button.setAttribute("aria-controls", button.dataset.view); button.tabIndex = index === 0 ? 0 : -1;
  $(button.dataset.view).setAttribute("role", "tabpanel"); $(button.dataset.view).setAttribute("aria-labelledby", button.id);
  button.addEventListener("click", () => show(button.dataset.view));
  button.addEventListener("keydown", event => { let next = index; if (event.key === "ArrowRight") next = (index+1)%buttons.length; else if (event.key === "ArrowLeft") next = (index+buttons.length-1)%buttons.length; else if (event.key === "Home") next = 0; else if (event.key === "End") next = buttons.length-1; else return; event.preventDefault(); buttons[next].click(); buttons[next].focus(); });
});
function fillSelect(select, blank = false, placeholder = "Choisir une identité connue…") {
  select.replaceChildren(); if (blank) { const option = element("option", placeholder); option.value = ""; select.append(option); }
  for (const site of catalog) { const option = element("option", `${site.name} · ${site.city}`); option.value = site.site_id; select.append(option); }
}
function link(url, label) {
  try { const parsed = new URL(url); if (!["https:", "http:"].includes(parsed.protocol)) return element("span", label); }
  catch { return element("span", label); }
  const a = element("a", label); a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; return a;
}
function sources(panel, evidence, citations) {
  for (const item of evidence || []) {
    if (citations && !citations.includes(item.citation_id)) continue;
    const details = element("details", undefined, "source"); details.append(element("summary", `[${item.citation_id}] ${item.source.title}`));
    details.append(element("p", item.text)); details.append(link(item.source.url, "Ouvrir la source originale →"));
    const rights = element("p", `${item.source.attribution || item.source.publisher || "Source documentaire"} · ${item.source.license || "Licence indiquée sur la source"}`, "subtle");
    if (item.source.license_url) rights.append(document.createTextNode(" · "), link(item.source.license_url, "Licence")); details.append(rights);
    if (item.source.updated_at) details.append(element("p", `Source datée du ${item.source.updated_at}. Les informations du jour ne sont pas confirmées.`, "subtle"));
    panel.append(details);
  }
}
function chooseWalk(site) { $("walk-anchor").value = site; show("walk"); $("walk-anchor").focus(); }
function renderResult(panel, result, jobId) {
  panel.replaceChildren();
  if (result.trip) { renderTrip(panel, result.trip); return; }
  if (result.site_name) {
    panel.append(element("span", result.identity_origin === "user_provided" ? "Identité fournie par vous" : result.identification ? "Monument proposé" : "Monument du contexte", "badge"));
    panel.append(element("h2", result.site_name));
  }
  if (result.answer) {
    if (result.answer.status !== "answered") panel.append(element("p", "Information non établie dans le corpus", "badge caution"));
    if (result.all_facets_answered === false) panel.append(element("p", "Réponse partielle : certaines informations manquent dans les sources.", "badge caution"));
    panel.append(element("p", result.answer.answer, "answer"));
    sources(panel, result.evidence, result.answer.citations);
    if (result.identification) panel.append(element("p", "Ces extraits concernent le monument proposé ; ils ne prouvent pas l’identité de la photo.", "subtle"));
  } else {
    panel.append(element("span", "Une précision peut aider", "badge caution"));
    panel.append(element("h2", result.status === "current_information_unverified" ? "Vérifier les informations du jour" : "Poursuivons avec un peu de contexte."));
    panel.append(element("p", result.message || "Le corpus ne permet pas de répondre à cette demande."));
  }
  if (result.practical) for (const url of result.practical.links || []) { const p = element("p"); p.append(link(url, new URL(url).hostname + " →")); panel.append(p); }
  const site = result.identification?.site_id || result.site_id;
  const actions = element("div", undefined, "actions");
  if (site && site !== "unknown") { const button = element("button", "Composer un parcours à partir de ce lieu"); button.type = "button"; button.addEventListener("click", () => chooseWalk(site)); actions.append(button); }
  if (jobId) { const a = element("a", "Télécharger le rapport"); a.href = `/api/report?id=${encodeURIComponent(jobId)}`; actions.append(a); }
  panel.append(actions);
  if (result.status === "needs_clarification" && selectedFile && panel.id === "photo-result") {
    const confirmation = element("div", undefined, "confirmation"); const select = element("select"); select.setAttribute("aria-label", "Identité du monument connue par vous"); fillSelect(select, true);
    confirmation.append(element("p", "Vous connaissez ce monument ? Vous pouvez fournir son identité explicitement.", "subtle"), select);
    const button = element("button", "Utiliser l’identité que j’ai choisie", "primary"); button.type = "button";
    button.addEventListener("click", () => { if (!select.value) return notify("Choisissez vous-même le monument connu."); submitPhoto(select.value).catch(error => notify(error.message)); });
    confirmation.append(button); panel.append(confirmation);
  }
}
async function pollJob(id, panel) {
  let reportAvailable = false;
  const busy = element("div", undefined, "busy"); const message = element("p", "Préparation de l’analyse…");
  busy.append(element("div", undefined, "spinner"), element("h2", "Un instant de découverte."), message,
    element("p", "L’analyse s’exécute sur votre ordinateur et peut prendre plus d’une minute. Gardez la fenêtre de lancement ouverte.", "subtle"));
  panel.replaceChildren(busy);
  try {
    while (true) {
      const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
      reportAvailable = Boolean(job.report_available);
      if (job.status === "done") { renderResult(panel, job.result, id); return; }
      if (job.status === "failed") throw new Error(job.error || "L’analyse a été interrompue.");
      if (message.textContent !== job.phase) message.textContent = job.phase; await new Promise(resolve => setTimeout(resolve, 1000));
    }
  } catch (error) { panel.replaceChildren(element("h2", "L’analyse n’a pas pu terminer."), element("p", error.message, "error")); if (reportAvailable) { const a = element("a", "Télécharger le rapport de diagnostic", "report-link"); a.href = `/api/report?id=${encodeURIComponent(id)}`; panel.append(a); } throw error; }
}
$("photo-file").addEventListener("change", () => {
  selectedFile = $("photo-file").files[0] || null; if (previewURL) URL.revokeObjectURL(previewURL);
  $("photo-preview").classList.toggle("hidden", !selectedFile);
  if (selectedFile) { previewURL = URL.createObjectURL(selectedFile); $("photo-preview").src = previewURL; }
});
function readBase64(file) { return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(",")[1]); reader.onerror = () => reject(new Error("Photo illisible.")); reader.readAsDataURL(file); }); }
async function submitPhoto(site = "") {
  if (!selectedFile) throw new Error("Choisissez une photo."); if (selectedFile.size > 8*1024*1024) throw new Error("La photo doit faire au maximum 8 Mio.");
  notify(""); const body = {image_base64: await readBase64(selectedFile), question: $("photo-question").value, site_id: site};
  const {job_id} = await api("/api/photo", body); await pollJob(job_id, $("photo-result"));
}
$("photo-form").addEventListener("submit", async event => { event.preventDefault(); const button = event.submitter; button.disabled = true; try { await submitPhoto(); } catch (error) { notify(error.message); } finally { button.disabled = false; } });
$("chat-form").addEventListener("submit", async event => {
  event.preventDefault(); const button = event.submitter; button.disabled = true; notify("");
  try { const {job_id} = await api("/api/chat", {question: $("chat-question").value, site_id: $("chat-site").value}); await pollJob(job_id, $("chat-result")); }
  catch (error) { notify(error.message); } finally { button.disabled = false; }
});
function routeMap(stops) {
  const ns = "http://www.w3.org/2000/svg"; const svg = document.createElementNS(ns, "svg"); svg.classList.add("route-map"); svg.setAttribute("viewBox", "0 0 520 245"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Schéma des positions géographiques des étapes, sans réseau de rues");
  const lats = stops.map(s => s.coordinates.latitude), lons = stops.map(s => s.coordinates.longitude);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats), minLon = Math.min(...lons), maxLon = Math.max(...lons);
  const points = stops.map(s => ({x: 55+(s.coordinates.longitude-minLon)/(maxLon-minLon || 1)*410, y: 200-(s.coordinates.latitude-minLat)/(maxLat-minLat || 1)*155}));
  const line = document.createElementNS(ns, "polyline"); line.setAttribute("points", points.map(p => `${p.x},${p.y}`).join(" ")); line.setAttribute("fill", "none"); line.setAttribute("stroke", "#98a775"); line.setAttribute("stroke-width", "2"); line.setAttribute("stroke-dasharray", "5 5"); svg.append(line);
  points.forEach((p, i) => { const circle = document.createElementNS(ns, "circle"); circle.setAttribute("cx", p.x); circle.setAttribute("cy", p.y); circle.setAttribute("r", "13"); circle.setAttribute("fill", "#dfe9ab"); circle.setAttribute("stroke", "#55734d"); const text = document.createElementNS(ns, "text"); text.setAttribute("x", p.x); text.setAttribute("y", p.y+4); text.setAttribute("text-anchor", "middle"); text.textContent = String(i+1); const title = document.createElementNS(ns, "title"); title.textContent = stops[i].name; circle.append(title); svg.append(circle, text); });
  return svg;
}
function renderTrip(panel, trip) {
  panel.replaceChildren(element("span", trip.status === "saved" ? "Parcours enregistré" : "Parcours à relire", "badge"), element("h2", trip.title || "Votre promenade culturelle"));
  panel.append(element("p", `${trip.stops.length} étapes · ${Math.round(trip.total_minutes_estimate)} min estimées · ${trip.walking_km_estimate} km de marche estimée`, "subtle"), routeMap(trip.stops), element("p", "Positions schématiques ; les pointillés ne représentent pas les rues.", "subtle"));
  trip.stops.forEach((stop, i) => { const row = element("div", undefined, "route-step"); const content = element("div"); content.append(element("h3", stop.name), element("p", `${stop.walking_minutes_from_previous_estimate ? Math.round(stop.walking_minutes_from_previous_estimate)+" min de marche estimée · " : ""}${stop.visit_minutes_assumption} min sur place prévues`)); if (stop.official_links?.length) content.append(link(stop.official_links[0], "Consulter les informations du site →")); row.append(element("span", String(i+1), "step-number"), content); panel.append(row); });
  const assumptions = element("ul", undefined, "assumptions"); trip.assumptions.forEach(text => assumptions.append(element("li", text))); panel.append(assumptions);
  if (trip.draft_id) {
    const input = element("input"); input.type = "text"; input.maxLength = 120; input.value = "Promenade culturelle à Paris"; input.setAttribute("aria-label", "Nom du parcours"); panel.append(input);
    const button = element("button", "Confirmer et enregistrer ce parcours", "primary"); button.type = "button";
    button.addEventListener("click", async () => { button.disabled = true; try { const saved = await api("/api/trips/save", {draft_id: trip.draft_id, confirmed: true, title: input.value}); renderTrip(panel, saved); await refreshTrips(); } catch (error) { notify(error.message); button.disabled = false; } }); panel.append(button);
  } else if (trip.trip_id) {
    const actions = element("div", undefined, "actions"); const a = element("a", "Exporter le parcours en JSON"); a.href = `/api/trip-export?id=${encodeURIComponent(trip.trip_id)}`; const print = element("button", "Imprimer"); print.type = "button"; print.addEventListener("click", () => window.print()); actions.append(a, print); panel.append(actions);
  }
}
$("walk-form").addEventListener("submit", async event => {
  event.preventDefault(); const button = event.submitter; button.disabled = true; notify("");
  try { const trip = await api("/api/plan", {anchor_site_id: $("walk-anchor").value, duration_minutes: Number($("walk-duration").value), max_stops: Number($("walk-stops").value), visit_minutes: Number($("walk-visit").value), site_ids: [...document.querySelectorAll("#site-checks input:checked")].map(el => el.value)}); renderTrip($("walk-result"), trip); }
  catch (error) { notify(error.message); } finally { button.disabled = false; }
});
async function refreshTrips() {
  const {trips} = await api("/api/trips"); const panel = $("saved-trips"); panel.replaceChildren();
  if (!trips.length) panel.append(element("p", "Vos parcours apparaîtront ici après confirmation de leur enregistrement.", "subtle"));
  for (const trip of trips) { const card = element("article", undefined, "saved-card"); card.append(element("h3", trip.title || "Promenade culturelle"), element("p", `${trip.stops.length} étapes · ${Math.round(trip.total_minutes_estimate)} min estimées`, "subtle")); const button = element("button", "Voir le parcours →", "quiet"); button.type = "button"; button.addEventListener("click", () => { show("walk"); renderTrip($("walk-result"), trip); $("walk-result").scrollIntoView({behavior: "smooth", block: "start"}); }); card.append(button); panel.append(card); }
}
$("refresh-trips").addEventListener("click", () => refreshTrips().catch(error => notify(error.message)));
$("status-button").addEventListener("click", async () => {
  try { const {status} = await api("/api/bootstrap"); const panel = $("status-content"); panel.replaceChildren(); for (const [key, label] of [["llama_found", "Moteur de génération"], ["generation_model_found", "Modèle de texte"], ["projector_found", "Composant photo Qwen"], ["visual_environment_found", "Environnement de comparaison visuelle"]]) { const row = element("div", label, "state-line"); row.append(element("b", status[key] ? "Repéré" : "Absent")); panel.append(row); } panel.append(element("p", "Ces contrôles repèrent les fichiers et programmes. Leur chargement est vérifié lors d’une analyse.", "subtle")); $("status-dialog").showModal(); } catch (error) { notify(error.message); }
});
async function initialize() {
  const data = await api("/api/bootstrap"); token = data.token; catalog = data.catalog; credits = data.image_credits || {};
  document.querySelectorAll(".site-select").forEach(select => fillSelect(select));
  fillSelect($("chat-site"), true, "Déduire le monument de ma demande");
  for (const site of catalog) {
    const check = element("input"); check.type = "checkbox"; check.value = site.site_id; check.checked = true; const label = element("label"); label.append(check, document.createTextNode(site.name)); $("site-checks").append(label);
    const card = element("article", undefined, "catalog-card"); const button = element("button", undefined, "catalog-open"); button.type = "button"; const image = element("img"); image.src = `/media/${site.site_id}.jpg`; image.alt = site.name; image.loading = "lazy";
    const title = element("div"); title.append(element("strong", site.name), element("small", site.city)); button.append(image, title); button.addEventListener("click", () => { $("chat-site").value = site.site_id; $("walk-anchor").value = site.site_id; show("assistant"); $("chat-question").focus(); $("assistant").scrollIntoView({behavior: "smooth", block: "start"}); }); card.append(button);
    const credit = credits[site.site_id]; if (credit) { const note = element("div", undefined, "credit"); const author = credit.author.includes("William Crochot") ? "William Crochot (Medium69)" : credit.author; note.append(link(credit.source_page, `${author} · ${credit.license}`)); card.append(note); } $("catalog-grid").append(card);
  }
  if (!data.status.llama_found || !data.status.generation_model_found) notify("Le catalogue et les parcours sont disponibles. Le moteur local et ses modèles doivent être installés pour analyser les photos et répondre avec l’IA.");
  await refreshTrips();
}
initialize().catch(error => notify(error.message));
