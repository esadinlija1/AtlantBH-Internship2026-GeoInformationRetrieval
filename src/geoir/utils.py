import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

SKIP = {"id", "tags", "visible", "version", "timestamp", "changeset",
        "geometry", "osm_type", "lon", "lat"}


def parse_tags(t):
    """Pretvara sadržaj kolone `tags` (JSON string ili dict) u dict."""
    if isinstance(t, dict):
        return t
    if not isinstance(t, str) or not t:
        return {}
    try:
        return json.loads(t)
    except Exception:
        return {}


def _is_empty(v):
    if v is None:
        return True
    if isinstance(v, (list, tuple, np.ndarray)):
        return len(v) == 0
    try:
        return bool(pd.isna(v)) or v == ""
    except Exception:
        return False


def build_attrs(row):
    """Spaja `tags` i popunjene kolone reda u jedan dict; kolona pobjeđuje tag."""
    d = dict(parse_tags(row.get("tags")))
    for c, v in row.items():
        if c in SKIP or _is_empty(v):
            continue
        d[c] = v
    return d