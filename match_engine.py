"""
match_engine.py
================
Tiered medication-identity matching.

Tier A — normalized exact match (free, instant)
Tier B — RxNorm API lookup, resolved to ingredient-level RxCUI so that
         brand names (Atarax) and generics (hydroxyzine) resolve to the
         same concept (free, no API key, no patient data sent — only the
         bare drug name string)
Tier C — fuzzy string match via RapidFuzz, last resort for anything
         RxNorm can't resolve

Anything that still doesn't match after all three tiers is a genuine
candidate omission — not a name-formatting artifact.
"""

import copy
import re
from collections import defaultdict

import requests
from rapidfuzz import fuzz

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
FUZZY_THRESHOLD = 85  # 0-100, RapidFuzz token_sort_ratio

# Frequency text -> doses per day. The frequency string is lowercased and
# stripped before lookup (case-insensitive). Anything not listed makes a pair's
# dose comparison unparseable, so it is skipped and logged rather than guessed.
# Extend this table as new frequency phrasings appear in real extractions.
FREQUENCY_TO_DOSES_PER_DAY = {
    "daily": 1, "qd": 1, "q24h": 1, "qhs": 1, "qam": 1,
    "bid": 2, "q12h": 2,
    "tid": 3, "q8h": 3,
    "qid": 4, "q6h": 4,
    "q4h": 6,
    "q3h": 8,
}

# Dose discrepancies compare total daily dose (per-dose mg x doses/day). Flag
# only when the relative TDD difference exceeds this fraction, so schedule-only
# differences that net the same daily dose (100 mg BID vs 50 mg q6h) don't flag.
DOSE_TDD_THRESHOLD = 0.10

# Route comparison uses the extraction route enum, normalized (lowercase,
# hyphens/spaces removed). PO and oral are the same route. A route mismatch is
# *flagged* ONLY when the discharge AVS says by-mouth (PO) but the inpatient MAR
# used an enteral feeding tube (G-tube/J-tube/NG) -- a real "can this patient
# actually swallow at home?" concern, attributed to the AVS. Every other
# mismatch is logged for visibility but not reported.
_PO_ROUTES = {"po", "oral"}
_ENTERAL_TUBE_ROUTES = {"gtube", "jtube", "ng"}

# Names (brand + generic) that identify the acetaminophen ingredient. Stored
# already-normalized (lowercase, no punctuation) so they compare directly to
# normalize(med["name"]). Acetaminophen is near-universal and rarely a genuine
# reconciliation discrepancy, so its presence on one list but not the other is
# noise; clean_lists drops it when ignore_acetaminophen is on.
_ACETAMINOPHEN_NAMES = {"acetaminophen", "paracetamol", "tylenol", "ofirmev"}

# Default cleaning configuration. clean_lists layers any caller overrides on top,
# so a partial config dict only changes the toggles it names.
CLEAN_DEFAULTS = {
    "ignore_acetaminophen": True,
    "ignore_prn": "all",        # "all" | "mar_only" | "none"
    "ignore_one_time": True,
    "ignore_cancelled": True,
}


def normalize(name: str) -> str:
    """Tier A: lowercase, strip punctuation/whitespace."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _flag_true(value) -> bool:
    """Interpret an extracted boolean-ish flag. Only a genuine True (or the
    literal string "true") counts; missing/None/False/anything else is treated
    as False, so cleaning never drops a med on an ambiguous flag — keeping a med
    for reconciliation is the safe failure mode."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return value is True


def _removal_reason(med: dict, is_mar: bool, cfg: dict) -> str | None:
    """Return the name of the first filter that drops `med`, or None to keep it.
    Filters are checked in a fixed precedence so each removed med is attributed
    to exactly one reason and per-filter counts sum to the total removed."""
    name_norm = normalize(med.get("name") or "")

    if cfg["ignore_acetaminophen"] and name_norm in _ACETAMINOPHEN_NAMES:
        return "acetaminophen"

    if is_mar and cfg["ignore_cancelled"] and _flag_true(med.get("is_cancelled")):
        return "cancelled"

    if cfg["ignore_one_time"] and med.get("schedule_type") == "one_time":
        return "one_time"

    prn_mode = cfg["ignore_prn"]
    prn_applies = prn_mode == "all" or (prn_mode == "mar_only" and is_mar)
    if prn_applies and (_flag_true(med.get("is_prn")) or med.get("schedule_type") == "prn"):
        return "prn"

    return None


def clean_lists(avs_data: dict, mar_data: dict, config: dict | None = None):
    """
    Filter both medication lists BEFORE the four detectors run, returning
    filtered COPIES of each list plus a log of everything removed and why.
    The inputs are NEVER mutated — raw extraction stays intact for
    scoring/debugging.

    config toggles (defaults in CLEAN_DEFAULTS; a partial dict overrides only
    the keys it names):
      ignore_acetaminophen (bool): drop acetaminophen / Tylenol / Ofirmev /
                                   paracetamol from BOTH lists.
      ignore_prn ("all"|"mar_only"|"none"): drop meds where is_prn is true OR
                                   schedule_type == "prn". "all" = both lists,
                                   "mar_only" = MAR only, "none" = keep.
      ignore_one_time (bool): drop schedule_type == "one_time" from BOTH lists.
      ignore_cancelled (bool): drop is_cancelled MAR meds (AVS has no such field).

    Returns (avs_clean, mar_clean, removed) where removed is:
      {"acetaminophen": [...], "prn": [...], "one_time": [...], "cancelled": [...]}
    and each entry is {"list": "avs"|"mar", "med": <deep copy of the dropped med>}.
    """
    cfg = {**CLEAN_DEFAULTS, **(config or {})}

    avs_meds = avs_data.get("medications") or []
    mar_meds = mar_data.get("medications") or []

    removed: dict[str, list] = {
        "acetaminophen": [], "prn": [], "one_time": [], "cancelled": [],
    }

    def _filter(meds: list, is_mar: bool) -> list:
        which = "mar" if is_mar else "avs"
        kept = []
        for med in meds:
            if not isinstance(med, dict):
                kept.append(copy.deepcopy(med))  # defensive: pass non-dicts through
                continue
            reason = _removal_reason(med, is_mar, cfg)
            if reason is None:
                kept.append(copy.deepcopy(med))
            else:
                removed[reason].append({"list": which, "med": copy.deepcopy(med)})
        return kept

    avs_clean = _filter(avs_meds, is_mar=False)
    mar_clean = _filter(mar_meds, is_mar=True)
    return avs_clean, mar_clean, removed


def rxnorm_ingredient_rxcui(name: str, session: requests.Session, cache: dict) -> str | None:
    """
    Tier B: resolve a drug name to its ingredient-level RxCUI via RxNorm.
    Only the bare name string is sent -- no dose, no patient/case data.
    Returns None if RxNorm has no match (falls through to Tier C).
    """
    if name in cache:
        return cache[name]

    try:
        # Step 1: approximate match to find the closest RxNorm concept
        r = session.get(f"{RXNORM_BASE}/approximateTerm.json",
                         params={"term": name, "maxEntries": 1}, timeout=5)
        r.raise_for_status()
        candidates = r.json().get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            cache[name] = None
            return None
        rxcui = candidates[0]["rxcui"]

        # Step 2: walk up to the ingredient (TTY=IN) level so brand/generic
        # and different strengths/forms of the same drug collapse together
        r2 = session.get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json",
                          params={"tty": "IN"}, timeout=5)
        r2.raise_for_status()
        groups = r2.json().get("relatedGroup", {}).get("conceptGroup", [])
        for g in groups:
            if g.get("tty") == "IN" and g.get("conceptProperties"):
                ing_rxcui = g["conceptProperties"][0]["rxcui"]
                cache[name] = ing_rxcui
                return ing_rxcui

        # No ingredient-level concept found; fall back to the raw rxcui
        cache[name] = rxcui
        return rxcui

    except requests.RequestException:
        cache[name] = "__UNREACHABLE__"
        return "__UNREACHABLE__"


def match_medications(avs_meds: list, mar_meds: list, offline_cache: dict | None = None,
                      session: requests.Session | None = None,
                      rxcui_cache: dict | None = None):
    """
    Returns (matched_pairs, unmatched_avs, unmatched_mar, stats).

    unmatched_avs are AVS meds with no MAR match after all three tiers.
    unmatched_mar are MAR meds no AVS med claimed (the leftover MAR pool) — the
    set Stage C now reports as omissions (a med given inpatient on the MAR but
    not carried onto the discharge AVS). The tier logic is identical either way;
    only which leftover set the caller reports differs. stats attributes each
    matched pair to exactly one tier:
        {"tier_a": int, "tier_b": int, "tier_c": int, "unmatched": int,
         "unmatched_mar": int, "avs_total": int, "mar_total": int}
    where "unmatched" is the AVS-side count (kept for debugging).

    Pass session/rxcui_cache to share one HTTP session and one RxCUI cache
    with other lookups in the same pipeline run; each defaults to a fresh one.
    """
    if session is None:
        session = requests.Session()
    if rxcui_cache is None:
        rxcui_cache = {}
    matched_pairs = []
    unmatched_avs = []
    stats = {"tier_a": 0, "tier_b": 0, "tier_c": 0, "unmatched": 0,
             "avs_total": len(avs_meds), "mar_total": len(mar_meds)}
    mar_pool = list(mar_meds)  # entries get removed as they're claimed

    for avs_med in avs_meds:
        match = None
        matched_tier = None

        # --- Tier A: normalized exact match ---
        avs_norm = normalize(avs_med["name"])
        for mar_med in mar_pool:
            if normalize(mar_med["name"]) == avs_norm:
                match = mar_med
                matched_tier = "tier_a"
                break

        # --- Tier B: RxNorm ingredient-level match ---
        if match is None:
            avs_rxcui = (offline_cache.get(avs_med["name"]) if offline_cache
                         else rxnorm_ingredient_rxcui(avs_med["name"], session, rxcui_cache))
            if avs_rxcui and avs_rxcui != "__UNREACHABLE__":
                for mar_med in mar_pool:
                    mar_rxcui = (offline_cache.get(mar_med["name"]) if offline_cache
                                 else rxnorm_ingredient_rxcui(mar_med["name"], session, rxcui_cache))
                    if mar_rxcui and mar_rxcui == avs_rxcui:
                        match = mar_med
                        matched_tier = "tier_b"
                        break

        # --- Tier C: fuzzy fallback ---
        if match is None:
            best_score, best_med = 0, None
            for mar_med in mar_pool:
                score = fuzz.token_sort_ratio(avs_med["name"], mar_med["name"])
                if score > best_score:
                    best_score, best_med = score, mar_med
            if best_score >= FUZZY_THRESHOLD:
                match = best_med
                matched_tier = "tier_c"

        if match is not None:
            stats[matched_tier] += 1
            matched_pairs.append((avs_med, match))
            mar_pool.remove(match)
        else:
            stats["unmatched"] += 1
            unmatched_avs.append(avs_med)

    # Whatever remains in mar_pool was never claimed by any AVS med -> the
    # MAR-side unmatched set (Stage C reports these as omissions).
    unmatched_mar = mar_pool
    stats["unmatched_mar"] = len(unmatched_mar)

    return matched_pairs, unmatched_avs, unmatched_mar, stats


def _doses_per_day_from_frequency(frequency) -> int | None:
    """Map a frequency string to doses/day via FREQUENCY_TO_DOSES_PER_DAY.
    Returns None if the text isn't a recognized token (caller skips + logs)."""
    if not isinstance(frequency, str):
        return None
    return FREQUENCY_TO_DOSES_PER_DAY.get(frequency.strip().lower())


def _mar_doses_per_day(mar_med: dict) -> int | None:
    """MAR doses/day: the count of scheduled administration times if any are
    present, otherwise fall back to mapping the frequency text."""
    scheduled = mar_med.get("scheduled_times")
    if isinstance(scheduled, list) and scheduled:
        return len(scheduled)
    return _doses_per_day_from_frequency(mar_med.get("frequency"))


def _norm_route(route) -> str:
    """Normalize a route enum value for comparison: lowercase, no hyphens/spaces
    (so 'G-tube' -> 'gtube', 'PO' -> 'po')."""
    return (route or "").lower().replace("-", "").replace(" ", "")


def compare_matched_pairs(matched_pairs: list):
    """Deterministic field diff on already-matched medication pairs.

    Dose discrepancies compare TOTAL DAILY DOSE (per-dose mg x doses/day) with a
    DOSE_TDD_THRESHOLD relative tolerance, so a schedule difference that nets the
    same daily dose (e.g. 100 mg BID vs 50 mg q6h, both 200 mg/day) is not
    flagged. If per-dose mg or doses/day is missing/unparseable on either side,
    the pair's dose comparison is SKIPPED and logged rather than flagged.
    """
    dosage_discrepancies = []
    route_discrepancies = []

    for avs_med, mar_med in matched_pairs:
        # --- Dose discrepancy via total daily dose (TDD) ---
        avs_per_dose = avs_med.get("dose_mg")
        mar_per_dose = mar_med.get("dose_mg")
        avs_dpd = _doses_per_day_from_frequency(avs_med.get("frequency"))
        mar_dpd = _mar_doses_per_day(mar_med)

        missing = []
        if avs_per_dose is None:
            missing.append("avs dose_mg")
        if mar_per_dose is None:
            missing.append("mar dose_mg")
        if avs_dpd is None:
            missing.append(f"avs doses/day (frequency={avs_med.get('frequency')!r})")
        if mar_dpd is None:
            missing.append(
                f"mar doses/day (frequency={mar_med.get('frequency')!r}, "
                f"scheduled_times={mar_med.get('scheduled_times')!r})")

        if missing:
            print(f"[DOSE-SKIP] {avs_med.get('name')}: cannot compute TDD — "
                  f"missing {', '.join(missing)}")
        else:
            tdd_avs = avs_per_dose * avs_dpd
            tdd_mar = mar_per_dose * mar_dpd
            denom = max(tdd_avs, tdd_mar)
            pct = 0.0 if denom == 0 else abs(tdd_avs - tdd_mar) / denom
            if pct > DOSE_TDD_THRESHOLD:
                dosage_discrepancies.append({
                    "name": avs_med["name"],
                    "tdd_avs": tdd_avs,
                    "tdd_mar": tdd_mar,
                    "pct_diff": round(pct * 100, 1),
                })

        # --- Route discrepancy ---
        avs_route, mar_route = avs_med.get("route"), mar_med.get("route")
        if not (avs_route and mar_route):
            continue  # incomplete route info on one side -> nothing to compare

        norm_avs_route = _norm_route(avs_route)
        norm_mar_route = _norm_route(mar_route)
        if norm_avs_route == norm_mar_route:
            continue  # same route
        if norm_avs_route in _PO_ROUTES and norm_mar_route in _PO_ROUTES:
            continue  # PO and oral are the same route

        # Genuine mismatch. Flag ONLY the actionable case (AVS PO vs MAR enteral
        # tube), attributed to the AVS; log every other mismatch for visibility.
        if norm_avs_route in _PO_ROUTES and norm_mar_route in _ENTERAL_TUBE_ROUTES:
            route_discrepancies.append({
                "name": avs_med["name"],
                "avs_route": avs_route,
                "mar_route": mar_route,
                "attributed_to": "avs",
            })
        else:
            print(f"[ROUTE-INFO] {avs_med.get('name')}: unflagged route mismatch "
                  f"AVS={avs_route!r} vs MAR={mar_route!r}")

    return dosage_discrepancies, route_discrepancies


def find_avs_duplications(avs_meds: list, session: requests.Session | None = None,
                          rxcui_cache: dict | None = None,
                          offline_cache: dict | None = None) -> list[dict]:
    """
    Detect Duplications on the AVS list ONLY: two or more entries sharing the
    SAME ingredient AND the SAME formulation.

    Ingredient identity reuses Tier B RxNorm ingredient resolution
    (rxnorm_ingredient_rxcui), so brand and generic collapse to one identity
    (e.g. Keppra == levetiracetam). When RxNorm can't give a usable RxCUI the
    normalized name is used as a fallback identity, so same-named entries still
    group. Formulation is the coarse dose-form enum extracted on the AVS side.

    Grouping is by (ingredient identity, formulation), so same ingredient in a
    DIFFERENT formulation is NOT a duplication:
        Keppra liquid + levetiracetam liquid              -> flagged
        ondansetron liquid + ondansetron tablet           -> NOT flagged
        albuterol inhaler + albuterol nebulizer_solution  -> NOT flagged
    (Acetaminophen is not a useful example here: clean_lists drops it from both
    lists before this check, so a Tylenol x2 pair never reaches duplication
    detection at all.)

    Entries whose formulation is unknown (null/empty) are skipped: without a
    formulation we cannot assert "same formulation", and grouping on unknown
    form would resurrect false positives. This replaces the former RxClass ATC
    grouping, which flagged unrelated drugs sharing a broad class (e.g. the
    "Other antiepileptics" catch-all).

    Only bare drug-name strings are ever sent to RxNorm -- no dose, route, or
    case data. Returns a list of {"ingredient", "formulation", "drugs"} groups,
    where "drugs" is the display names of the colliding entries.
    """
    if session is None:
        session = requests.Session()
    if rxcui_cache is None:
        rxcui_cache = {}

    by_group: dict[tuple, list] = defaultdict(list)  # (identity, form) -> [names]
    for med in avs_meds:
        if not isinstance(med, dict):
            continue
        name = med.get("name")
        if not name:
            continue
        form = (med.get("formulation") or "").strip().lower()
        if not form:
            continue  # unknown formulation -> can't confirm a same-form duplication

        rxcui = (offline_cache.get(name) if offline_cache is not None
                 else rxnorm_ingredient_rxcui(name, session, rxcui_cache))
        identity = rxcui if (rxcui and rxcui != "__UNREACHABLE__") else normalize(name)

        by_group[(identity, form)].append(name)

    duplications = []
    for (identity, form), names in by_group.items():
        if len(names) >= 2:
            duplications.append({
                "ingredient": names[0],   # display name of the first occurrence
                "formulation": form,
                "drugs": names,
            })
    return duplications
