"""Display capitalization for heritage names; source quotations stay verbatim."""
import re


def monument_name(name):
    return re.sub(r'\b[^\W\d_]', lambda match: match[0].upper(), name)
