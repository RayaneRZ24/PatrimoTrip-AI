"""Local cultural walking plans with exact subset routing, explicitly estimated travel."""
import json
import hashlib
import math
import pathlib
import uuid
import datetime as dt

from run_local import ROOT


def catalog():
    data = (ROOT/'corpus/sites.json').read_bytes()
    checksums = json.loads((ROOT/'corpus/checksums.json').read_text(encoding='utf-8'))
    if hashlib.sha256(data).hexdigest() != checksums['sites.json']: raise ValueError('Catalogue modifié ou corrompu.')
    sites = json.loads(data)
    for site in sites:
        coordinates = site['coordinates']
        if not all(isinstance(coordinates[k], (float, int)) and math.isfinite(coordinates[k]) and abs(coordinates[k]) <= limit
                   for k, limit in [('latitude',90),('longitude',180)]): raise ValueError('Coordonnées invalides.')
    return sites


def distance_km(a, b):
    p, q = a['coordinates'], b['coordinates']
    x, y = math.radians(p['latitude']), math.radians(q['latitude'])
    dlat = y-x; dlon = math.radians(q['longitude']-p['longitude'])
    h = math.sin(dlat/2)**2+math.cos(x)*math.cos(y)*math.sin(dlon/2)**2
    return 6371.0088*2*math.asin(min(1, math.sqrt(h)))


def plan(anchor_site_id, duration_minutes=240, max_stops=4, visit_minutes=30, site_ids=None):
    sites = {s['site_id']: s for s in catalog()}
    if anchor_site_id not in sites: raise ValueError('Choisissez un monument du catalogue comme point de départ.')
    for value, low, high, name in [(duration_minutes, 45, 720, 'Durée'), (max_stops, 1, 8, 'Nombre d’étapes'), (visit_minutes, 10, 120, 'Temps sur place')]:
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high: raise ValueError(name+' hors limites.')
    selected = list(dict.fromkeys(list(sites) if site_ids is None else site_ids))
    if not selected or len(selected) > 10 or any(s not in sites for s in selected): raise ValueError('Sélection de monuments invalide.')
    if anchor_site_id not in selected: selected.insert(0, anchor_site_id)
    candidates = [s for s in selected if s != anchor_site_id]
    # Held–Karp dynamic programming over subsets; shortest open walking route
    # for each subset and end point. Visit times are user assumptions, not data.
    states = {(0, anchor_site_id): (0.0, [anchor_site_id])}
    best = (1, 0.0, [anchor_site_id])
    if visit_minutes > duration_minutes: raise ValueError('Le temps sur place dépasse la durée disponible.')
    for mask in range(1 << len(candidates)):
        for end in [anchor_site_id]+candidates:
            state = states.get((mask, end))
            if state is None: continue
            walked, route = state
            if len(route) >= max_stops: continue
            for index, target in enumerate(candidates):
                bit = 1 << index
                if mask & bit: continue
                km = walked+distance_km(sites[end], sites[target])*1.3
                minutes = km/4.5*60+(len(route)+1)*visit_minutes
                if minutes > duration_minutes: continue
                key = (mask | bit, target); next_route = route+[target]
                if key not in states or km < states[key][0]: states[key] = (km, next_route)
                objective = (len(next_route), -km)
                if objective > (best[0], best[1]): best = (len(next_route), -km, next_route)
    route = best[2]; elapsed = 0.0; stops = []
    for i, site in enumerate(route):
        walk = distance_km(sites[route[i-1]], sites[site])*1.3/4.5*60 if i else 0
        elapsed += walk
        s = sites[site]
        stops.append({'site_id': site, 'name': s['name'], 'coordinates': s['coordinates'],
                      'walking_minutes_from_previous_estimate': round(walk, 1), 'arrival_after_minutes_estimate': round(elapsed, 1),
                      'visit_minutes_assumption': visit_minutes, 'opening_hours_verified': False,
                      'official_links': [u for u in s['official_website_candidates'] if u.startswith('https://')]})
        elapsed += visit_minutes
    return {'version': 'walking_plan_v1', 'anchor_site_id': anchor_site_id, 'stops': stops,
            'total_minutes_estimate': round(elapsed, 1), 'walking_km_estimate': round(-best[1], 2),
            'available_minutes': duration_minutes, 'status': 'draft',
            'assumptions': ['Parcours extérieur indicatif, sans vérification des horaires ni accès intérieurs.',
                            'Marche estimée à 4,5 km/h ; distance à vol d’oiseau majorée de 30 %, sans calcul de rues.',
                            'Temps sur place choisi par l’utilisateur ; tarifs et réservations non établis.'],
            'optimizer': 'Exact subset dynamic programming: maximize stops then minimize estimated walking distance; open route.'}


def save_trip(trip, confirmed, folder=None):
    if confirmed is not True: raise ValueError('Confirmez le parcours avant son enregistrement.')
    if not isinstance(trip, dict) or trip.get('version') != 'walking_plan_v1' or not trip.get('stops'): raise ValueError('Parcours invalide.')
    sites = {s['site_id'] for s in catalog()}
    if len(trip['stops']) > 8 or any(s.get('site_id') not in sites for s in trip['stops']): raise ValueError('Étapes invalides.')
    # Persist only a bounded, finite JSON object, never a caller-selected path.
    raw = json.dumps(trip, ensure_ascii=False, allow_nan=False)
    if len(raw) > 60000: raise ValueError('Parcours trop volumineux.')
    folder = pathlib.Path(folder or ROOT/'.app-state/trips'); folder.mkdir(parents=True, exist_ok=True)
    trip_id = uuid.uuid4().hex
    value = {**trip, 'status': 'saved', 'trip_id': trip_id, 'saved_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    (folder/(trip_id+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return value


def list_trips(folder=None):
    folder = pathlib.Path(folder or ROOT/'.app-state/trips')
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('*.json'), reverse=True)[:50]]
