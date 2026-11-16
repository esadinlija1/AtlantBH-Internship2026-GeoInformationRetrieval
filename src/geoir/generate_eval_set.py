"""Build a 100-query evaluation set (dev 50 / test 50) from data/processed/pois_clean.parquet.

Leakage rules
-------------
* `nlp_text` is never loaded: the parquet is read with an explicit column list that excludes it.
* Query wording is generated from templates and from `primary_name` only, never from tags or `nlp_text`.
* Ground truth (`expected_ids`) is computed from structured columns:
  `id`, `primary_name`, `primary_category`, `lat`, `lon`, `city`; additionally `category_label`
  (only to tell mosques from churches) and `filters` / `opening_hours` (only for descriptive queries).

Query types (per split)
-----------------------
* entity           (15) paraphrased search for one named POI; expected = POIs with the same normalized
                        name within 1 km of it.
* spatial_category (15) category intent + city-centre coordinates + radius; expected = every POI of that
                        category inside the radius (haversine).
* descriptive      (12) category + 1-2 features described in plain Bosnian ("kafić sa baštom u Mostaru");
                        expected = every POI of that category with all features within CITY_RADIUS_KM of the
                        city centre. lat/lon are null: the city must be resolved from the query text.
* cross_lingual     (8) English category queries ("museums in Sarajevo") and Cyrillic queries
                        (entity names and category + city), expected computed as above.

Usage:  python -m geoir.generate_eval_set   (or python src/geoir/generate_eval_set.py)
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

def fold(s: str) -> str:
    s = s.casefold().replace("đ", "dj")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "processed" / "pois_clean.parquet"
OUT_DIR = ROOT / "data" / "eval"
COLUMNS = ["id", "primary_name", "primary_category", "category_label", "lat", "lon", "city", "filters", "opening_hours"]

N_PER_SPLIT = {"entity": 15, "spatial_category": 15, "descriptive": 12, "cross_lingual": 8}
CITY_RADIUS_KM = 3.0      # "in <city>" for urban categories
NATURE_RADIUS_KM = 15.0   # "near <place>" for nature categories
ENTITY_SAME_RADIUS_KM = 1.0

# city centres: (lat, lon, Latin locative, Cyrillic nominative, Cyrillic locative)
CITIES = {
    "Sarajevo": (43.8590, 18.4250, "u Sarajevu", "Сарајево", "у Сарајеву"),
    "Banja Luka": (44.7722, 17.1910, "u Banjoj Luci", "Бања Лука", "у Бањој Луци"),
    "Mostar": (43.3438, 17.8078, "u Mostaru", "Мостар", "у Мостару"),
    "Tuzla": (44.5384, 18.6763, "u Tuzli", "Тузла", "у Тузли"),
    "Zenica": (44.2034, 17.9077, "u Zenici", "Зеница", "у Зеници"),
}
MAIN_CITIES = list(CITIES)
# other towns: used to name the location of entity queries and nature queries
TOWNS = {
    **{k: v[:3] for k, v in CITIES.items()},
    "Bihać": (44.8169, 15.8708, "u Bihaću"), "Bijeljina": (44.7569, 19.2150, "u Bijeljini"),
    "Doboj": (44.7349, 18.0843, "u Doboju"), "Prijedor": (44.9799, 16.7134, "u Prijedoru"),
    "Trebinje": (42.7119, 18.3442, "u Trebinju"), "Travnik": (44.2264, 17.6658, "u Travniku"),
    "Brčko": (44.8726, 18.8098, "u Brčkom"), "Jajce": (44.3420, 17.2700, "u Jajcu"),
    "Konjic": (43.6519, 17.9611, "u Konjicu"), "Visoko": (43.9889, 18.1781, "u Visokom"),
    "Goražde": (43.6680, 18.9760, "u Goraždu"), "Livno": (43.8269, 17.0075, "u Livnu"),
    "Foča": (43.5064, 18.7747, "u Foči"), "Cazin": (44.9672, 15.9431, "u Cazinu"),
    "Gradiška": (45.1450, 17.2540, "u Gradišci"), "Bugojno": (44.0572, 17.4508, "u Bugojnu"),
}

# Bosnian category words used in queries
CAT_BS = {
    "cafe": "kafić", "restaurant": "restoran", "bar": "bar", "pub": "pivnica", "fast_food": "brza hrana",
    "hotel": "hotel", "motel": "motel", "guest_house": "pansion", "hostel": "hostel", "museum": "muzej",
    "castle": "tvrđava", "fort": "utvrda", "monument": "spomenik", "memorial": "spomenik", "attraction": "znamenitost",
    "archaeological_site": "arheološko nalazište", "peak": "vrh", "waterfall": "vodopad", "spring": "izvor",
    "cave_entrance": "pećina", "park": "park", "pharmacy": "apoteka", "bank": "banka", "school": "škola",
    "hospital": "bolnica", "clinic": "ambulanta", "post_office": "pošta", "university": "fakultet",
    "supermarket": "supermarket", "clothes": "butik", "mall": "tržni centar", "bakery": "pekara",
    "fuel": "benzinska pumpa", "hairdresser": "frizer", "place_of_worship": "bogomolja",
}

# entity sampling strata
ENTITY_GROUPS = {
    "food": ["cafe", "restaurant", "bar", "pub", "fast_food"],
    "lodging": ["hotel", "motel", "guest_house", "hostel"],
    "heritage": ["museum", "castle", "fort", "monument", "memorial", "attraction", "archaeological_site", "place_of_worship"],
    "nature": ["peak", "waterfall", "spring", "cave_entrance", "park"],
    "services": ["pharmacy", "bank", "school", "hospital", "clinic", "post_office", "university"],
    "shops": ["supermarket", "clothes", "mall", "bakery", "hairdresser"],
}
GENERIC_WORDS = {
    "restoran", "restaurant", "caffe", "cafe", "kafe", "kafic", "bar", "apoteka", "pekara", "hotel", "motel", "market",
    "prodavnica", "trgovina", "doo", "d.o.o", "d.o.o.", "frizerski", "salon", "caffe-bar", "pizzeria", "picerija",
    "aparthotel", "pansion", "hostel", "the", "and", "i",
}

STOP_WORDS = {"sa", "s", "u", "na", "i", "od", "kod", "za", "do", "pod", "the", "of", "and", "at", "-"}
# words that describe a type of place rather than name one place
GENERIC_DESC = ({fold(w) for phrase in CAT_BS.values() for w in phrase.split()} | GENERIC_WORDS | STOP_WORDS | {
    "orthodox", "catholic", "cemetery", "church", "mosque", "groblje", "crkva", "dzamija", "mesdzid", "skola", "podrucna",
    "osnovna", "srednja", "gradski", "gradska", "spomenik", "spomen", "obiljezje", "kafana", "bife", "kiosk", "trafika",
    "parking", "stadion", "igraliste", "dom", "zdravlja", "ju", "jzu", "os", "zavod", "centar", "center", "nekropola",
    "stecci", "kapela", "dzamija", "turbe", "vrelo", "izvor", "vidikovac", "most", "kula", "grad", "stari", "novi",
    "veliki", "mali", "gornji", "donji", "brdo", "vrh", "glavica", "gradina", "ambulanta", "poliklinika", "trznica",
    "pijaca", "pumpa", "benzinska", "mjesna", "zajednica", "biblioteka", "galerija",
})
ENGLISH_WORDS = {"at", "the", "of", "and", "drink", "party", "caffe", "cafe", "pub", "bar", "shop", "house", "city", "club",
                 "lounge", "food", "pizza", "coffee", "burger", "steak", "grill", "fast", "beauty", "hair", "studio",
                 "fitness", "gym", "store", "street", "hotel", "hostel", "apartment", "apartments", "rooms", "bistro",
                 "good", "deluxe", "garden", "inn", "point", "corner", "time", "fashion", "style"}

# spatial intents: query word -> predicate on the dataframe
SPATIAL_INTENTS = {
    "apoteka": lambda d: d.primary_category == "pharmacy",
    "bankomat": lambda d: (d.primary_category == "atm") | (d.filters.fillna("").str.contains('"atm": "yes"')),
    "kafić": lambda d: d.primary_category == "cafe",
    "park": lambda d: d.primary_category == "park",
    "restoran": lambda d: d.primary_category == "restaurant",
    "pekara": lambda d: d.primary_category == "bakery",
    "benzinska pumpa": lambda d: d.primary_category == "fuel",
    "supermarket": lambda d: d.primary_category == "supermarket",
    "pošta": lambda d: d.primary_category == "post_office",
    "banka": lambda d: d.primary_category == "bank",
    "džamija": lambda d: d.category_label.fillna("").str.startswith("mosque"),
    "crkva": lambda d: d.category_label.fillna("").str.contains("church"),
    "hotel": lambda d: d.primary_category == "hotel",
    "frizer": lambda d: d.primary_category == "hairdresser",
    "dječije igralište": lambda d: d.primary_category == "playground",
    "brza hrana": lambda d: d.primary_category == "fast_food",
}
SPATIAL_TEMPLATES = ["{w} u blizini", "najbliža {w}", "gdje je {w} u blizini", "{w} blizu mene", "{w} u krugu od {r} km"]
FEMININE = {"apoteka", "benzinska pumpa", "pošta", "banka", "džamija", "crkva", "pekara", "brza hrana"}

# descriptive features: name -> (predicate on parsed filters + opening hours, Bosnian phrase)
FEATURES = {
    "outdoor": (lambda f, oh: f.get("outdoor_seating") == "yes", "sa baštom"),
    "free_wifi": (lambda f, oh: f.get("internet_access") in ("wlan", "yes") and f.get("internet_access:fee") == "no",
                  "sa besplatnim internetom"),
    "wifi": (lambda f, oh: f.get("internet_access") == "wlan", "sa bežičnim internetom"),
    "wheelchair": (lambda f, oh: f.get("wheelchair") == "yes", "s pristupom za invalidska kolica"),
    "no_smoking": (lambda f, oh: f.get("smoking") == "no", "bez pušenja"),
    "smoking": (lambda f, oh: f.get("smoking") == "yes", "gdje se smije pušiti"),
    "aircon": (lambda f, oh: f.get("air_conditioning") == "yes", "sa klimom"),
    "takeaway": (lambda f, oh: f.get("takeaway") in ("yes", "only"), "sa hranom za ponijeti"),
    "cards": (lambda f, oh: any(f.get(k) == "yes" for k in ("payment:credit_cards", "payment:debit_cards", "payment:cards",
                                                            "payment:visa", "payment:mastercard", "payment:maestro")),
              "gdje mogu platiti karticom"),
    "lpg": (lambda f, oh: f.get("fuel:lpg") == "yes", "sa autoplinom"),
    "diesel": (lambda f, oh: f.get("fuel:diesel") == "yes", "sa dizelom"),
    "nonstop": (lambda f, oh: oh == "24/7", "koja radi non-stop"),
    "atm": (lambda f, oh: f.get("atm") == "yes", "sa bankomatom"),
}
DESCRIPTIVE_CATS = ["cafe", "restaurant", "bar", "fast_food", "hotel", "fuel", "pharmacy", "bank", "supermarket", "bakery"]
DESCRIPTIVE_TEMPLATES = ["{c} {f} {city}", "tražim {acc} {f} {city}", "gdje mogu naći {acc} {f} {city}", "{c} {city} {f}"]

# cross-lingual English intents: (English plural, predicate, urban?)
EN_INTENTS = {
    "museums": (lambda d: d.primary_category == "museum", True),
    "pharmacies": (lambda d: d.primary_category == "pharmacy", True),
    "mosques": (lambda d: d.category_label.fillna("").str.startswith("mosque"), True),
    "Orthodox churches": (lambda d: d.category_label.fillna("").str.contains("Orthodox church"), True),
    "Catholic churches": (lambda d: d.category_label.fillna("").str.contains("Catholic church"), True),
    "hotels": (lambda d: d.primary_category == "hotel", True),
    "bakeries": (lambda d: d.primary_category == "bakery", True),
    "waterfalls": (lambda d: d.primary_category == "waterfall", False),
    "fortresses": (lambda d: d.primary_category.isin(["castle", "fort"]), False),
    "caves": (lambda d: d.primary_category == "cave_entrance", False),
    "mountain huts": (lambda d: d.primary_category.isin(["alpine_hut", "wilderness_hut"]), False),
    "viewpoints": (lambda d: d.primary_category == "viewpoint", False),
}
EN_TEMPLATES_URBAN = ["{x} in {p}", "{x} in {p} city centre", "where can I find {x} in {p}"]
EN_TEMPLATES_NATURE = ["{x} near {p}", "{x} around {p}"]
CYR_CATEGORY = {"апотека": "pharmacy", "пекара": "bakery", "хотел": "hotel", "музеј": "museum", "банка": "bank",
                "пошта": "post_office", "ресторан": "restaurant", "кафић": "cafe"}

LAT2CYR = {"lj": "љ", "nj": "њ", "dž": "џ", "Lj": "Љ", "Nj": "Њ", "Dž": "Џ", "LJ": "Љ", "NJ": "Њ", "DŽ": "Џ"}
LAT2CYR_1 = dict(zip("abcčćdđefghijklmnoprsštuvzž", "абцчћдђефгхијклмнопрсштувзж"))
LAT2CYR_1.update({k.upper(): v.upper() for k, v in LAT2CYR_1.items()})


# --------------------------------------------------------------------------------------------- helpers
def haversine_km(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def name_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", fold(s)).strip()


def accusative(phrase: str) -> str:
    """Accusative of the category phrases used here: feminine -a nouns/adjectives take -u, the rest are unchanged."""
    return " ".join(w[:-1] + "u" if phrase in FEMININE and w.endswith("a") else w for w in phrase.split())


def to_cyrillic(s: str) -> str:
    for lat, cyr in LAT2CYR.items():
        s = s.replace(lat, cyr)
    return "".join(LAT2CYR_1.get(ch, ch) for ch in s)


def nearest_town(lat, lon, max_km=8.0):
    best = min(TOWNS.items(), key=lambda kv: haversine_km(lat, lon, kv[1][0], kv[1][1]))
    return best if haversine_km(lat, lon, best[1][0], best[1][1]) <= max_km else None


def entry(qtype, query, expected, lat=None, lon=None, radius_km=None):
    return {"query_id": None, "type": qtype, "query": query, "lat": lat, "lon": lon,
            "radius_km": radius_km, "expected_ids": sorted(int(i) for i in expected)}


# --------------------------------------------------------------------------------------------- generators
def entity_core(name: str, rng: random.Random, allow_fold: bool = True) -> str | None:
    """Natural rendering of a name: no quotes/legal form, generic words dropped, sometimes partial or unaccented."""
    words = [w for w in re.split(r"[\s\"'„“”,&]+", name) if w]
    core = [w for w in words if fold(w).strip(".") not in GENERIC_WORDS] or words
    if len(core) >= 3 and rng.random() < 0.5:
        part = core[:2]                                   # partial name ...
        while part and fold(part[-1]) in STOP_WORDS:
            part = part[:-1]                              # ... not ending in 'sa', 'u', 'na', ...
        if part and not is_generic(part):                 # ... and still distinctive
            core = part
    if is_generic(core):
        return None                                       # 'Supermarket', 'Orthodox cemetery', 'Područna škola'
    text = " ".join(core).lower()
    if allow_fold and rng.random() < 0.5:
        text = fold(text)                                 # typed without diacritics
    return text if len(re.sub(r"\W", "", text)) >= 4 and not text.isdigit() else None


def is_generic(words) -> bool:
    """True when every word is a category/descriptive word, i.e. the name does not identify one place."""
    return all(fold(w).strip(".-") in GENERIC_DESC for w in words)


def entity_candidates(df: pd.DataFrame) -> pd.DataFrame:
    named = df[df.primary_name.notna()].copy()
    named["nkey"] = named.primary_name.map(name_key)
    freq = named.nkey.value_counts()
    named = named[named.nkey.map(freq) <= 2]              # no chains (Konzum, UniCredit, ...)
    town_keys = {name_key(t) for t in TOWNS} | {name_key(c) for c in df.city.dropna().unique()}
    named = named[~named.nkey.isin(town_keys)]            # 'Posušje' names a town, not one POI
    return named[~named.primary_name.str.fullmatch(r"[\d\W]+")]


def looks_local(name: str) -> bool:
    """For Cyrillic queries: skip names with English words, which nobody would type in Cyrillic."""
    return not any(fold(w) in ENGLISH_WORDS for w in re.findall(r"\w+", name))


def entity_expected(df_named: pd.DataFrame, row) -> list[int]:
    same = df_named[df_named.nkey == row.nkey]
    d = haversine_km(same.lat.to_numpy(), same.lon.to_numpy(), row.lat, row.lon)
    return same.id[d <= ENTITY_SAME_RADIUS_KM].tolist()


def gen_entity(df, rng, n_total, used_ids):
    cand = entity_candidates(df)
    per_group = n_total // len(ENTITY_GROUPS)
    out = []
    for group, cats in ENTITY_GROUPS.items():
        pool = cand[cand.primary_category.isin(cats)].sample(frac=1, random_state=rng.randrange(10**6))
        got = 0
        for row in pool.itertuples():
            if got == per_group:
                break
            core = entity_core(row.primary_name, rng)
            if not core or row.id in used_ids:
                continue
            town = nearest_town(row.lat, row.lon)
            cat = CAT_BS.get(row.primary_category)
            templates = ["{core}", "gdje je {core}", "{core} adresa"]
            if town:
                templates += ["{core} {town}", "{core} {loc}", "gdje je {core} {loc}"]
            if cat and fold(cat) not in fold(core):
                templates += ["{cat} {core}"] + (["{cat} {core} {loc}"] if town else [])
            q = rng.choice(templates).format(core=core, cat=cat, town=town[0] if town else "",
                                             loc=town[1][2] if town else "")
            out.append((group, entry("entity", q, entity_expected(cand, row))))
            used_ids.add(row.id)
            got += 1
    return out


def gen_spatial(df, rng, n_total):
    combos = []
    for city in MAIN_CITIES:
        lat0, lon0 = CITIES[city][:2]
        dist = haversine_km(df.lat.to_numpy(), df.lon.to_numpy(), lat0, lon0)
        for word, pred in SPATIAL_INTENTS.items():
            r = rng.choice([1.0, 1.5, 2.0])
            ids = df.id[pred(df).to_numpy() & (dist <= r)].tolist()
            if 1 <= len(ids) <= 40:
                combos.append((city, word, r, ids))
    rng.shuffle(combos)
    per_city = n_total // len(MAIN_CITIES)
    out = []
    for city in MAIN_CITIES:
        for _, word, r, ids in [c for c in combos if c[0] == city][:per_city]:
            tpl = rng.choice(SPATIAL_TEMPLATES)
            if tpl.startswith("najbliža") and word not in FEMININE:
                tpl = "najbliži {w}"
            q = tpl.format(w=word, r=str(r).rstrip("0").rstrip("."))
            lat0, lon0 = CITIES[city][:2]
            out.append((city, entry("spatial_category", q, ids, lat0, lon0, r)))
    return out


def gen_descriptive(df, rng, n_total):
    parsed = df.filters.map(lambda s: json.loads(s) if isinstance(s, str) else {})
    feat_mask = {name: np.array([pred(f, oh) for f, oh in zip(parsed, df.opening_hours)])
                 for name, (pred, _) in FEATURES.items()}
    combos = []
    for city in MAIN_CITIES:
        in_city = haversine_km(df.lat.to_numpy(), df.lon.to_numpy(), *CITIES[city][:2]) <= CITY_RADIUS_KM
        for cat in DESCRIPTIVE_CATS:
            base = in_city & (df.primary_category == cat).to_numpy()
            names = list(FEATURES)
            sets = [(a,) for a in names] + [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
            for fs in sets:
                if {"wifi", "free_wifi"} <= set(fs) or {"smoking", "no_smoking"} <= set(fs):
                    continue
                m = base.copy()
                for f in fs:
                    m &= feat_mask[f]
                ids = df.id[m].tolist()
                if 1 <= len(ids) <= 25:
                    combos.append((city, cat, fs, ids))
    rng.shuffle(combos)
    out, seen_cat_feat, used_cities = [], set(), {}
    for city, cat, fs, ids in sorted(combos, key=lambda c: -len(c[2]) + rng.random()):   # prefer 2-feature combos
        key = (cat, fs)
        if key in seen_cat_feat or used_cities.get(city, 0) >= -(-n_total // len(MAIN_CITIES)):
            continue
        seen_cat_feat.add(key)
        used_cities[city] = used_cities.get(city, 0) + 1
        feats = " i ".join(FEATURES[f][1] for f in fs)
        q = rng.choice(DESCRIPTIVE_TEMPLATES).format(c=CAT_BS[cat], acc=accusative(CAT_BS[cat]), f=feats, city=CITIES[city][2])
        out.append((cat, entry("descriptive", q, ids)))
        if len(out) == n_total:
            break
    return out


def gen_cross_lingual(df, rng, n_total, used_ids):
    n_en, n_cyr_ent = n_total // 2, n_total // 4
    n_cyr_cat = n_total - n_en - n_cyr_ent
    out = []
    # English category + place
    combos = []
    for label, (pred, urban) in EN_INTENTS.items():
        places = MAIN_CITIES if urban else ["Jajce", "Konjic", "Trebinje", "Bihać", "Foča", "Mostar", "Sarajevo", "Livno"]
        radius = CITY_RADIUS_KM if urban else NATURE_RADIUS_KM
        for p in places:
            dist = haversine_km(df.lat.to_numpy(), df.lon.to_numpy(), *TOWNS[p][:2])
            ids = df.id[pred(df).to_numpy() & (dist <= radius)].tolist()
            if 1 <= len(ids) <= 40:
                combos.append((label, p, urban, ids))
    rng.shuffle(combos)
    seen = set()
    for label, p, urban, ids in combos:
        if label in seen:
            continue
        seen.add(label)
        tpl = rng.choice(EN_TEMPLATES_URBAN if urban else EN_TEMPLATES_NATURE)
        out.append(("en", entry("cross_lingual", tpl.format(x=label, p=p), ids)))
        if len(seen) == n_en:
            break
    # Cyrillic entity names (Latin names written in Cyrillic, no foreign letters)
    cand = entity_candidates(df)
    cand = cand[~cand.primary_name.str.contains(r"[qwxyQWXY]") & cand.primary_name.map(looks_local)
                & cand.primary_category.isin(sum(ENTITY_GROUPS.values(), []))]
    got = 0
    for row in cand.sample(frac=1, random_state=rng.randrange(10**6)).itertuples():
        if got == n_cyr_ent:
            break
        core = entity_core(row.primary_name, random.Random(rng.random()), allow_fold=False)
        if not core or row.id in used_ids or any(ord(c) > 127 and c not in "čćšžđ" for c in core):
            continue
        town = nearest_town(row.lat, row.lon)
        q = to_cyrillic(core) + (f" {to_cyrillic(town[0])}" if town else "")
        out.append(("cyr_entity", entry("cross_lingual", q, entity_expected(cand, row))))
        used_ids.add(row.id)
        got += 1
    # Cyrillic category + city
    combos = []
    for word, cat in CYR_CATEGORY.items():
        for city in MAIN_CITIES:
            dist = haversine_km(df.lat.to_numpy(), df.lon.to_numpy(), *CITIES[city][:2])
            ids = df.id[(df.primary_category == cat).to_numpy() & (dist <= CITY_RADIUS_KM)].tolist()
            if 1 <= len(ids) <= 40:
                combos.append((word, city, ids))
    rng.shuffle(combos)
    seen = set()
    for word, city, ids in combos:
        if word in seen:
            continue
        seen.add(word)
        out.append(("cyr_category", entry("cross_lingual", f"{word} {CITIES[city][4]}", ids)))
        if len(seen) == n_cyr_cat:
            break
    return out


# --------------------------------------------------------------------------------------------- split & main
def stratified_split(items, rng):
    """items: [(stratum, entry)]; alternate dev/test inside each stratum so both splits get the same mix."""
    dev, test = [], []
    by_stratum = {}
    for stratum, e in items:
        by_stratum.setdefault(stratum, []).append(e)
    turn = 0
    for stratum in sorted(by_stratum):
        group = by_stratum[stratum]
        rng.shuffle(group)
        group.sort(key=lambda e: len(e["expected_ids"]))   # alternate in size order -> similar ground-truth sizes
        for e in group:
            (dev if turn % 2 == 0 else test).append(e)
            turn += 1
    return dev, test


def main():
    rng = random.Random(SEED)
    df = pd.read_parquet(IN_PATH, columns=COLUMNS)          # nlp_text is deliberately not loaded
    assert "nlp_text" not in df.columns

    used_ids: set[int] = set()
    generated = {
        "entity": gen_entity(df, rng, 2 * N_PER_SPLIT["entity"], used_ids),
        "spatial_category": gen_spatial(df, rng, 2 * N_PER_SPLIT["spatial_category"]),
        "descriptive": gen_descriptive(df, rng, 2 * N_PER_SPLIT["descriptive"]),
        "cross_lingual": gen_cross_lingual(df, rng, 2 * N_PER_SPLIT["cross_lingual"], used_ids),
    }

    splits = {"dev": [], "test": []}
    for qtype, items in generated.items():
        assert len(items) == 2 * N_PER_SPLIT[qtype], f"{qtype}: generated {len(items)}"
        dev, test = stratified_split(items, rng)
        splits["dev"] += dev
        splits["test"] += test

    abbr = {"entity": "ent", "spatial_category": "spa", "descriptive": "des", "cross_lingual": "xl"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, entries in splits.items():
        counters = {}
        for e in entries:
            counters[e["type"]] = counters.get(e["type"], 0) + 1
            e["query_id"] = f"{split}-{abbr[e['type']]}-{counters[e['type']]:02d}"
        path = OUT_DIR / f"{split}_queries.json"
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    # sanity checks
    all_q = splits["dev"] + splits["test"]
    # the same text may repeat for spatial queries at different coordinates, but never with the same location
    keys = [(e["query"], e["lat"], e["lon"]) for e in all_q]
    assert len(set(keys)) == len(keys), f"duplicate queries: {[k for k in keys if keys.count(k) > 1]}"
    assert all(e["expected_ids"] for e in all_q), "query without ground truth"
    valid = set(df.id)
    assert all(i in valid for e in all_q for i in e["expected_ids"])

    # summary
    print(f"Wrote {OUT_DIR / 'dev_queries.json'} and {OUT_DIR / 'test_queries.json'}\n")
    rows = [{"split": s, "type": e["type"], "n_expected": len(e["expected_ids"])} for s, es in splits.items() for e in es]
    summary = (pd.DataFrame(rows).groupby(["type", "split"]).n_expected
               .agg(queries="size", avg_expected="mean", min_expected="min", max_expected="max").round(2)
               .unstack("split"))
    print(summary.to_string())
    totals = pd.DataFrame(rows).groupby("split").n_expected.agg(queries="size", avg_expected="mean").round(2)
    print("\nTotals:\n" + totals.to_string())
    print("\nExamples:")
    for qtype in N_PER_SPLIT:
        for e in [e for e in splits["dev"] if e["type"] == qtype][:3]:
            loc = f"  @({e['lat']}, {e['lon']}, r={e['radius_km']} km)" if e["lat"] is not None else ""
            print(f"  {e['query_id']:12s} {e['query']!r}{loc}  -> {len(e['expected_ids'])} expected")


if __name__ == "__main__":
    main()
