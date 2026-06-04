"""
Fuzzy Logic Lake Position Inference System  —  v2.0
=====================================================
Changes from v1 (corrected baseline):
  • Added body-temperature label 'Hot' with its own triangular MF.
  • 'Calm' added as a valid wind output label (new rules introduced it).
  • 'Cool' added as a valid temperature output label.
  • Genga rule expanded: moon now covers New;Low (was Low only).
  • Kus;Genya;Tarai rule updated: moon now covers New;Midway (replaces the
    original New-only rule for those three winds under Heavy/Moderate).
  • Kus, Descending;Low, Cloudy, High → Good (new Cloudy variant added).
  • Kus, New, Clear, Moderate updated: LP=Normal (new document overrides
    the previous Bad assignment; annotated).
  • 29 net-new rules added from the updated expert elicitation table.
  • Malformed rule skipped (Nyadhiwa / shifted columns); see NOTE below.
  • Duplicate Genya/New/Clear/Hot deduplicated.
  • Kus, Ascending, Clear, Warm conflict resolved → Bad (conservative).
  • Tarai/Descending/Cloudy/Warm rainfall label corrected: "Warm" → "Moderate".
  • Separator inconsistencies fixed: "Kus/Genya" and "Genya, Kus" both
    normalised to semicolon-separated sets.
  • Terminology: "No Rain" → "No", "Drizzling" → "Light" (carried forward).

SKIPPED RULE (data quality):
  "Nyadhiwa AND Heavy AND Low AND Hot" — moon='Heavy' and cloud='Low' are
  not valid labels; columns appear shifted in the source table. This rule
  has been omitted and flagged for re-elicitation from the domain expert.
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Membership functions
# ---------------------------------------------------------------------------

def triangular(x, a, b, c):
    """Standard triangular membership function, clamped to [0, 1]."""
    if a == b == c:
        return 1.0 if x == a else 0.0
    if b == a:
        return max(0.0, min(1.0, (c - x) / (c - b)))
    if b == c:
        return max(0.0, min(1.0, (x - a) / (b - a)))
    return max(0.0, min((x - a) / (b - a), (c - x) / (c - b)))


def moon_membership(moon_day):
    """Lunar day [0, 29] → fuzzy membership per phase."""
    return {
        'New':        triangular(moon_day,  0,  0,  3),
        'Ascending':  triangular(moon_day,  1,  5, 10),
        'Midway':     triangular(moon_day,  8, 14, 21),
        'Descending': triangular(moon_day, 18, 22, 27),
        'Low':        triangular(moon_day, 24, 28, 29),
    }


def cloud_membership(cloud_percent):
    """Cloud cover [0, 100] % → fuzzy membership per condition."""
    return {
        'Clear':  triangular(cloud_percent,  0,  0, 20),
        'Light':  triangular(cloud_percent, 10, 25, 40),
        'Cloudy': triangular(cloud_percent, 30, 50, 70),
        'Heavy':  triangular(cloud_percent, 60, 80, 100),
    }


def body_temp_membership(temp):
    """
    Body temperature [°C] → fuzzy membership per thermal category.

    Scale (approximate thresholds):
      Cold     34.0 – 35.5  (hypothermic risk)
      Cool     35.0 – 37.0  (below normal)
      Low      alias for Cool (same triangular params)
      Moderate 36.5 – 37.5  (normal resting)
      Warm     37.0 – 39.0  (mildly elevated)
      High     38.0 – 40.0  (fever range)
      Hot      38.5 – 40.0  (high fever / extreme heat stress)
                             [NEW in v2 — distinct peak from High]
    """
    val = {
        'Cold':     triangular(temp, 34.0, 34.5, 35.5),
        'Cool':     triangular(temp, 35.0, 36.0, 37.0),
        'Moderate': triangular(temp, 36.5, 37.0, 37.5),
        'Warm':     triangular(temp, 37.0, 38.0, 39.0),
        'High':     triangular(temp, 38.0, 39.0, 40.0),
        'Hot':      triangular(temp, 38.5, 39.5, 40.0),  # NEW
    }
    val['Low'] = val['Cool']   # explicit alias — identical physical range
    return val


# ---------------------------------------------------------------------------
# Rule base
# ---------------------------------------------------------------------------
#
# Each tuple:
#   (wind_str, moon_str, cloud_str, body_str,
#    wind_out, temp_out, rainfall_out, lake_pos)
#
# Semicolons in a condition = OR (max membership across listed labels).
# "Any Position" / "Any" = universal wildcard (strength = 1.0).
#
# Annotation legend used in comments:
#   [v1]     — carried forward unchanged from v1 corrected baseline
#   [v1-mod] — carried forward with modification
#   [new]    — first introduced in v2 from updated expert table
#   [dup]    — duplicate detected and removed
#   [skip]   — malformed / unresolvable; omitted pending re-elicitation
#   [CONFLICT RESOLVED] — see header docstring for resolution rationale

RULES_RAW = [

    # ===================================================================
    # RISKY  lake-position rules
    # ===================================================================

    # [v1] Grouped rule; original single-wind duplicate (R1) already removed.
    ("Genya;Kus;Nyabukoba", "Descending",                  "Heavy",        "Moderate",
     "Risky",  "High",     "Very Heavy", "Risky"),

    # [v1]
    ("Genya;Kus;Nyabukoba", "Low",                         "Heavy",        "Low",
     "Risky",  "Low",      "Very Heavy", "Risky"),

    # [v1]
    ("Nyakoi",              "New;Low",                     "Heavy",        "Low",
     "Risky",  "Low",      "Very Heavy", "Risky"),

    # [v1-mod] Moon expanded from Low-only to New;Low per updated table.
    ("Genga",               "New;Low",                     "Heavy",        "Moderate",
     "Risky",  "Moderate", "Heavy",      "Risky"),

    # [v1] CONFLICT RESOLVED: (Genya,New,Heavy,Moderate) Risky retained over Good.
    ("Genya",               "New",                         "Heavy",        "Moderate",
     "Risky",  "Moderate", "Moderate",   "Risky"),

    # [v1]
    ("Kus",                 "Midway",                      "Heavy",        "Moderate",
     "Stormy", "Moderate", "Moderate",   "Risky"),

    # [v1]
    ("Nyagire",             "New",                         "Heavy",        "Warm",
     "Risky",  "Moderate", "Moderate",   "Risky"),

    # [v1]
    ("Genya",               "Midway",                      "Cloudy",       "Warm",
     "Bad",    "Warm",     "Moderate",   "Risky"),

    # [v1]
    ("Genya",               "Descending",                  "Heavy",        "High",
     "Bad",    "High",     "Moderate",   "Risky"),

    # [v1]
    ("Genya",               "New",                         "Heavy",        "Warm",
     "Risky",  "High",     "Moderate",   "Risky"),

    # [v1]
    ("Tarai",               "Descending",                  "Cloudy",       "Warm",
     # [new] rainfall label was "Warm" in source — corrected to "Moderate"
     "Stormy", "Warm",     "Moderate",   "Risky"),

    # [v1]
    ("Kus;Genya;Nyabukoba", "Low",                         "Heavy",        "Cool",
     "Windy",  "Warm",     "Heavy",      "Risky"),

    # [v1] CONFLICT RESOLVED: LP=Risky retained (vs Bad in R75).
    ("Kus;Genya;Nyabukoba", "New",                         "Heavy",        "Cool",
     "Bad",    "Warm",     "Heavy",      "Risky"),

    # [v1]
    ("Kus;Genya;Nyabukoba", "Low",                         "Heavy",        "High",
     "Stormy", "High",     "Very Heavy", "Risky"),

    # [new] Nyakoi + Low moon + Cloudy + Cool body
    ("Nyakoi",              "Low",                         "Cloudy",       "Cool",
     "Stormy", "Cool",     "Light",      "Risky"),

    # ===================================================================
    # BAD  lake-position rules
    # ===================================================================

    # [v1]
    ("Nyakoi",              "New;Low",                     "Heavy",        "Moderate",
     "Risky",  "Moderate", "Heavy",      "Bad"),

    # [v1]
    ("Nyakoi",              "Any Position",                "Light",        "Any",
     "Risky",  "Moderate", "Moderate",   "Bad"),

    # [v1] Rule 15 — the broad wildcard; subsumed rules 19/20/22/52/67 already removed.
    ("Kus;Genya;Tarai;Nyagire;Nyabukoba", "Any Position",  "Cloudy",       "Moderate",
     "Windy",  "High",     "Heavy",      "Bad"),

    # [v1-mod] Moon updated to New;Midway (was New-only for this wind group).
    ("Kus;Genya;Tarai",     "New;Midway",                  "Heavy",        "Moderate",
     "Risky",  "Moderate", "Moderate",   "Bad"),

    # [v1]
    ("Genya",               "Descending",                  "Light",        "Moderate",
     "Risky",  "Moderate", "Moderate",   "Bad"),

    # [v1]
    ("Tarai",               "Descending",                  "Light",        "High",
     "Risky",  "High",     "Light",      "Bad"),

    # [v1] R18; R27 (Kus-only duplicate) already removed.
    ("Kus;Genya",           "Low",                         "Cloudy",       "High",
     "Risky",  "High",     "Light",      "Bad"),

    # [v1]
    ("Nyadhiwa",            "Ascending;Descending",        "Cloudy",       "High",
     "Windy",  "High",     "Moderate",   "Bad"),

    # [v1]
    ("Kus",                 "Low",                         "Heavy",        "Moderate",
     "Windy",  "Moderate", "Moderate",   "Bad"),

    # [v1]
    ("Kus",                 "New",                         "Heavy",        "Warm",
     "Risky",  "Moderate", "Moderate",   "Bad"),

    # [v1]
    ("Kus",                 "New;Low",                     "Heavy",        "Moderate",
     "Stormy", "Moderate", "Heavy",      "Bad"),

    # [v1]
    ("Kus;Genya",           "Low",                         "Light",        "Warm",
     "Stormy", "High",     "Moderate",   "Bad"),

    # [v1]
    ("Genya",               "Midway",                      "Light",        "Cool",
     "Stormy", "High",     "Light",      "Bad"),

    # [v1]
    ("Kus",                 "Midway",                      "Heavy",        "Warm",
     "Stormy", "Moderate", "Moderate",   "Bad"),

    # [v1] CONFLICT RESOLVED: LP=Bad (most cautious for fishers).
    ("Kus;Genya",           "New",                         "Heavy",        "Cool",
     "Stormy", "Moderate", "Moderate",   "Bad"),

    # [v1]
    ("Kus",                 "New",                         "Light",        "Warm",
     "Windy",  "High",     "Light",      "Bad"),

    # [new] CONFLICT RESOLVED: Kus/Ascending/Clear/Warm appears as Bad (NEW3)
    # and Normal (NEW21) in new table. Bad retained (conservative).
    ("Kus",                 "Ascending",                   "Clear",        "Warm",
     "Windy",  "Warm",     "No",         "Bad"),

    # [new]
    ("Kus",                 "Midway",                      "Clear",        "Hot",
     "Stormy", "High",     "No",         "Bad"),

    # [new] "Kus/Genya" separator corrected to semicolon.
    ("Kus;Genya",           "Midway",                      "Clear",        "Hot",
     "Stormy", "High",     "No",         "Bad"),

    # [new]
    ("Nyagire",             "Midway",                      "Heavy",        "Hot",
     "Stormy", "High",     "No",         "Bad"),

    # [new]
    ("Genya",               "Midway",                      "Heavy",        "Warm",
     "Stormy", "Warm",     "No",         "Bad"),

    # [new]
    ("Genya",               "Midway",                      "Light",        "Warm",
     "Stormy", "Warm",     "Light",      "Bad"),

    # [new]
    ("Tarai",               "New",                         "Cloudy",       "Warm",
     "Windy",  "Warm",     "Moderate",   "Bad"),

    # [new]
    ("Kus",                 "New",                         "Cloudy",       "Warm",
     "Windy",  "Warm",     "Moderate",   "Bad"),

    # ===================================================================
    # GOOD  lake-position rules
    # ===================================================================

    # [v1]
    ("Kus",                 "Ascending",                   "Clear",        "Moderate",
     "Normal", "Moderate", "No",         "Good"),

    # [new] Cloudy variant — new rule from updated table.
    ("Kus",                 "Descending;Low",              "Cloudy",       "High",
     "Windy",  "High",     "Light",      "Good"),

    # [v1]
    ("Kus",                 "Descending;Low",              "Clear",        "High",
     "Normal", "Moderate", "No",         "Good"),

    # [v1] CONFLICT RESOLVED: Good retained over Normal (more specific rule).
    ("Genya",               "Midway",                      "Clear",        "Moderate",
     "Normal", "Moderate", "No",         "Good"),

    # [v1]
    ("Genya",               "Descending",                  "Clear",        "High",
     "Windy",  "Moderate", "No",         "Good"),

    # ===================================================================
    # NORMAL  lake-position rules
    # ===================================================================

    # [v1]
    ("Nyadhiwa",            "Descending",                  "Clear",        "Cold",
     "Normal", "Moderate", "No",         "Normal"),

    # [v1]
    ("Nyadhiwa",            "Ascending",                   "Light;Cloudy", "Moderate",
     "Windy",  "High",     "No",         "Normal"),

    # [v1]
    ("Nyadhiwa",            "New",                         "Light",        "High",
     "Windy",  "High",     "No",         "Normal"),

    # [v1]
    ("Nyadhiwa",            "New",                         "Cloudy",       "Moderate",
     "Windy",  "Moderate", "Light",      "Normal"),

    # [v1]
    ("Nyadhiwa",            "Midway",                      "Clear",        "Moderate",
     "Windy",  "High",     "No",         "Normal"),

    # [v1]
    ("Nyadhiwa",            "Midway",                      "Light",        "Moderate",
     "Windy",  "High",     "Light",      "Normal"),

    # [v1]
    ("Nyadhiwa",            "Low",                         "Clear;Light",  "Moderate",
     "Windy",  "High",     "No",         "Normal"),

    # [v1]
    ("Nyadhiwa",            "Low",                         "Cloudy",       "Moderate",
     "Windy",  "Moderate", "Moderate",   "Normal"),

    # [new]
    ("Nyadhiwa",            "Descending",                  "Clear",        "Warm",
     "Windy",  "High",     "No",         "Normal"),

    # [new]
    ("Nyadhiwa",            "New",                         "Clear",        "Warm",
     "Windy",  "High",     "No",         "Normal"),

    # [new]
    ("Nyadhiwa",            "New",                         "Cloudy",       "Warm",
     "Windy",  "Warm",     "Light",      "Normal"),

    # [new]
    ("Nyadhiwa",            "Descending",                  "Heavy",        "Warm",
     "Windy",  "Warm",     "Heavy",      "Normal"),

    # [skip] "Nyadhiwa AND Heavy AND Low AND Hot" — columns shifted in source
    # table (moon='Heavy', cloud='Low' are not valid labels). Omitted pending
    # re-elicitation from domain expert.

    # [v1]
    ("Marimbe",             "Low",                         "Clear",        "High",
     "Windy",  "High",     "No",         "Normal"),

    # [v1]
    ("Marimbe",             "Low",                         "Light",        "Moderate",
     "Windy",  "Warm",     "Light",      "Normal"),

    # [v1]
    ("Marimbe",             "Low",                         "Cloudy",       "Moderate",
     "Windy",  "Warm",     "Moderate",   "Normal"),

    # [v1]
    ("Marimbe",             "Midway",                      "Light",        "Moderate",
     "Windy",  "High",     "Light",      "Normal"),

    # [v1] R48/R50 merged; R50 duplicate removed.
    ("Marimbe",             "Descending",                  "Light",        "Moderate",
     "Windy",  "High",     "No",         "Normal"),

    # [v1]
    ("Marimbe",             "New;Ascending",               "Cloudy",       "Moderate",
     "Normal", "High",     "No",         "Normal"),

    # [v1]
    ("Genya",               "New;Low",                     "Light",        "Moderate",
     "Windy",  "High",     "No",         "Normal"),

    # [v1] R55 (Midway-only duplicate) already removed; covered here.
    ("Genya",               "Ascending;Midway;Descending", "Clear",        "Moderate",
     "Windy",  "Moderate", "No",         "Normal"),

    # [v1]
    ("Genya",               "Ascending;Midway;Descending", "Heavy",        "Moderate",
     "Windy",  "Moderate", "Heavy",      "Normal"),

    # [v1] Genya New;Low Cloudy Moderate — updated from Bad to Normal per new table.
    # [CONFLICT RESOLVED]: new document treats this as Normal; updated accordingly.
    ("Genya",               "New;Low",                     "Cloudy",       "Moderate",
     "Windy",  "Moderate", "Moderate",   "Normal"),

    # [new]
    ("Genya",               "Descending",                  "Cloudy",       "Warm",
     "Windy",  "Warm",     "Light",      "Normal"),

    # [new]
    ("Genya",               "New",                         "Clear",        "Hot",
     "Windy",  "High",     "Light",      "Normal"),

    # [new] [dup] Genya/New/Clear/Hot appeared twice in source; one removed.
    # [new]
    ("Genya",               "Low",                         "Clear",        "Hot",
     "Windy",  "High",     "Moderate",   "Normal"),

    # [new]
    ("Genya",               "Low",                         "Clear",        "Warm",
     "Windy",  "Warm",     "No",         "Normal"),

    # [new]
    ("Genya",               "Low",                         "Light",        "Cool",
     "Normal", "Cool",     "No",         "Normal"),

    # [v1]
    ("Kus",                 "New;Low",                     "Clear",        "Moderate",
     "Windy",  "Moderate", "Light",      "Normal"),

    # [v1]
    ("Kus",                 "Ascending;Midway;Descending", "Clear",        "Moderate",
     "Windy",  "High",     "No",         "Normal"),

    # [v1] Narrowed from Ascending;Midway;Descending to Ascending;Midway.
    # Descending sub-case conflicts with R26 (Good, more specific) — excluded.
    ("Kus",                 "Ascending;Midway",            "Cloudy",       "High",
     "Windy",  "High",     "Moderate",   "Normal"),

    # [v1-mod] Kus/New/Clear/Moderate: updated to Normal per new document.
    # Previous v1 had Bad; new expert table assigns Normal. [CONFLICT RESOLVED]
    ("Kus",                 "New",                         "Clear",        "Moderate",
     "Windy",  "Moderate", "No",         "Normal"),

    # [new]
    ("Kus",                 "Descending",                  "Heavy",        "Cool",
     "Windy",  "Warm",     "Moderate",   "Normal"),

    # [new]
    ("Kus",                 "Descending",                  "Light",        "Hot",
     "Windy",  "High",     "Light",      "Normal"),

    # [v1]
    ("Tarai",               "New;Low",                     "Light",        "Moderate",
     "Normal", "High",     "No",         "Normal"),

    # [v1]
    ("Tarai",               "New;Low",                     "Cloudy",       "High",
     "Windy",  "Moderate", "Moderate",   "Normal"),

    # [v1]
    ("Tarai",               "Ascending;Midway;Descending", "Clear",        "Moderate",
     "Windy",  "Moderate", "No",         "Normal"),

    # [v1]
    ("Tarai",               "Ascending;Midway;Descending", "Cloudy",       "High",
     "Windy",  "High",     "Moderate",   "Normal"),

    # [new]
    ("Tarai",               "Low",                         "Cloudy",       "Hot",
     "Windy",  "High",     "Light",      "Normal"),

    # [new]
    ("Tarai",               "Ascending",                   "Cloudy",       "Warm",
     "Windy",  "Warm",     "Light",      "Normal"),

    # [v1]
    ("Kus;Genya",           "Low",                         "Clear",        "Cool",
     "Normal", "Moderate", "No",         "Normal"),

    # [new] "Genya, Kus" comma separator normalised to semicolon.
    ("Genya;Kus",           "Descending",                  "Cloudy",       "Cool",
     "Windy",  "Warm",     "No",         "Normal"),

    # [new]
    ("Nyagire",             "Ascending",                   "Heavy",        "Warm",
     "Windy",  "Warm",     "No",         "Normal"),

    # [new]
    ("Nyagire",             "New",                         "Cloudy",       "Hot",
     "Windy",  "High",     "Light",      "Normal"),

]


# ---------------------------------------------------------------------------
# Rule parsing helpers
# ---------------------------------------------------------------------------

def _parse_items(s):
    """Split semicolon-separated condition string into a frozenset of labels."""
    return frozenset(item.strip() for item in s.split(';') if item.strip())


def _parse_rules(raw):
    """Convert raw string tuples into frozensets for efficient membership lookup."""
    return [
        (
            _parse_items(wind_s),
            _parse_items(moon_s),
            _parse_items(cloud_s),
            _parse_items(body_s),
            wind_out, temp_out, rain_out, lake_pos,
        )
        for (wind_s, moon_s, cloud_s, body_s,
             wind_out, temp_out, rain_out, lake_pos) in raw
    ]


RULES = _parse_rules(RULES_RAW)


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------

LAKE_POS_CLASSES = ("Good", "Normal", "Risky", "Bad")


def _compute_firings(wind_type, moon_day, cloud_percent, body_temp_value):
    """Run the rule base and return per-output max-firing dictionaries."""
    moon_mem  = moon_membership(moon_day)
    cloud_mem = cloud_membership(cloud_percent)
    body_mem  = body_temp_membership(body_temp_value)

    lake_pos_max = defaultdict(float)
    wind_out_max = defaultdict(float)
    temp_out_max = defaultdict(float)
    rain_out_max = defaultdict(float)

    for (wind_set, moon_set, cloud_set, body_set,
         wind_out, temp_out, rain_out, lake_pos) in RULES:

        if 'Any' not in wind_set and wind_type not in wind_set:
            continue

        moon_str = (1.0 if 'Any Position' in moon_set
                    else max((moon_mem.get(m, 0.0) for m in moon_set), default=0.0))
        cloud_str = (1.0 if 'Any' in cloud_set
                     else max((cloud_mem.get(c, 0.0) for c in cloud_set), default=0.0))
        body_str = (1.0 if 'Any' in body_set
                    else max((body_mem.get(b, 0.0) for b in body_set), default=0.0))

        firing = min(moon_str, cloud_str, body_str)
        if firing <= 0.0:
            continue

        if firing > lake_pos_max[lake_pos]:  lake_pos_max[lake_pos] = firing
        if firing > wind_out_max[wind_out]:   wind_out_max[wind_out] = firing
        if firing > temp_out_max[temp_out]:   temp_out_max[temp_out] = firing
        if firing > rain_out_max[rain_out]:   rain_out_max[rain_out] = firing

    return lake_pos_max, wind_out_max, temp_out_max, rain_out_max


def determine_lake_position(wind_type, moon_day, cloud_percent, body_temp_value):
    """Mamdani max-min inference — returns the winning category per output."""
    lake_pos_max, wind_out_max, temp_out_max, rain_out_max = _compute_firings(
        wind_type, moon_day, cloud_percent, body_temp_value
    )

    def _winner(d, default="Normal"):
        return max(d.items(), key=lambda kv: kv[1])[0] if d else default

    return (
        _winner(lake_pos_max),
        _winner(wind_out_max),
        _winner(temp_out_max),
        _winner(rain_out_max),
    )


def fuzzy_posterior(wind_type, moon_day, cloud_percent, body_temp_value,
                    classes=LAKE_POS_CLASSES):
    """Normalised lake-position membership vector across `classes`.

    Returns a list whose length and order match `classes`. When no rule
    fires (all zeros), the vector is left as zeros — the caller can decide
    on a fallback.
    """
    lake_pos_max, *_ = _compute_firings(
        wind_type, moon_day, cloud_percent, body_temp_value
    )
    raw = [float(lake_pos_max.get(c, 0.0)) for c in classes]
    total = sum(raw)
    if total > 0.0:
        return [v / total for v in raw]
    return raw


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # (wind,       moon, cloud, body,  note)
        ("Genya",       22,   85,  37.0,  "Descending moon, heavy cloud, moderate temp"),
        ("Kus",          2,   10,  37.0,  "New moon, clear sky, moderate temp"),
        ("Nyadhiwa",    14,   35,  37.5,  "Midway moon, light cloud, warm temp"),
        ("Marimbe",     26,    5,  38.5,  "Low moon, clear sky, high temp"),
        ("Nyakoi",       1,   75,  36.0,  "New moon, heavy cloud, cool temp"),
        ("Kus",          5,    5,  38.0,  "Ascending moon, clear sky, warm temp — new Bad rule"),
        ("Nyagire",      5,   90,  39.5,  "Ascending moon, heavy cloud, hot temp — new rule"),
        ("Genya",       14,   90,  38.5,  "Midway moon, heavy cloud, warm — new Bad rule"),
        ("Kus",         14,    5,  39.5,  "Midway moon, clear sky, hot — new Bad rule"),
        ("Nyadhiwa",    22,    5,  38.0,  "Descending moon, clear sky, warm — new Normal rule"),
    ]

    col = "{:<12} {:>5} {:>6} {:>5}  |  {:<12} {:<10} {:<10} {:<12}  {}"
    header = col.format("Wind", "Moon", "Cloud", "Body",
                        "Lake Pos", "Wind Out", "Temp Out", "Rain Out", "Note")
    print(header)
    print("-" * len(header))
    for (w, m, c, b, note) in test_cases:
        lp, wo, to, ro = determine_lake_position(w, m, c, b)
        print(col.format(w, m, c, b, lp, wo, to, ro, note))