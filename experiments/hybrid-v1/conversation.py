"""Explicit single-monument context resolution within the finite local catalogue."""
import re
import unicodedata

import facet_rag
import rag_local


def resolve(question, selected_site, sites):
    text = facet_rag.normalized(question)
    cased = re.sub(r'[^A-Za-z0-9]+', ' ', ''.join(c for c in unicodedata.normalize('NFKD', question) if not unicodedata.combining(c)))
    named = set()
    foreign = False
    for sid, site in sites.items():
        aliases = list(rag_local.SITE_ALIASES.get(sid, []))+[site['name']]
        for alias in aliases:
            alias = facet_rag.normalized(alias)
            if not re.search(r'\b'+re.escape(alias)+r'\b', text): continue
            named.add(sid)
            # An explicit different city must not be silently mapped to Paris.
            location = re.search(r'(?i:\b'+re.escape(alias)+r'\s+(?:de|a)\s+)([A-Z][A-Za-z]+)', cased)
            if location and location.group(1).lower() not in facet_rag.normalized(site['city']).split(): foreign = True
    reason = None
    if foreign:
        reason = 'Ce lieu semble désigner un monument hors du catalogue parisien. Précisez la ville et choisissez un monument pris en charge.'
    elif len(named) > 1:
        reason = 'Votre demande mentionne plusieurs monuments. Choisissez celui sur lequel porte cette question.'
    elif selected_site and named and selected_site not in named:
        reason = 'Le monument nommé dans votre question diffère du contexte sélectionné. Changez le contexte ou précisez votre demande.'
    elif selected_site:
        return selected_site, None
    elif named:
        return next(iter(named)), None
    else:
        reason = 'De quel monument parlez-vous ? Écrivez son nom dans la question (par exemple : « Qui a conçu le Panthéon de Paris ? »), choisissez un monument ou reprenez explicitement le dernier lieu exploré.'
    return None, {'question': question, 'status': 'needs_clarification', 'message': reason, 'tool_trace': [], 'model_called': False}
