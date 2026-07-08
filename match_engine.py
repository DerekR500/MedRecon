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

import re
import requests
from rapidfuzz import fuzz

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
FUZZY_THRESHOLD = 85  # 0-100, RapidFuzz token_sort_ratio


def normalize(name: str) -> str:
    """Tier A: lowercase, strip punctuation/whitespace."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


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
    Returns (matched_pairs, unmatched_avs) where unmatched_avs are
    candidate omissions after exhausting all three tiers.

    Pass session/rxcui_cache to share one HTTP session and one RxCUI cache
    with other lookups in the same pipeline run; each defaults to a fresh one.
    """
    if session is None:
        session = requests.Session()
    if rxcui_cache is None:
        rxcui_cache = {}
    matched_pairs = []
    unmatched_avs = []
    mar_pool = list(mar_meds)  # entries get removed as they're claimed

    for avs_med in avs_meds:
        match = None

        # --- Tier A: normalized exact match ---
        avs_norm = normalize(avs_med["name"])
        for mar_med in mar_pool:
            if normalize(mar_med["name"]) == avs_norm:
                match = mar_med
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

        if match is not None:
            matched_pairs.append((avs_med, match))
            mar_pool.remove(match)
        else:
            unmatched_avs.append(avs_med)

    return matched_pairs, unmatched_avs


def compare_matched_pairs(matched_pairs: list):
    """Deterministic field diff on already-matched medication pairs."""
    dosage_discrepancies = []
    route_discrepancies = []

    for avs_med, mar_med in matched_pairs:
        avs_dose, mar_dose = avs_med.get("dose_mg"), mar_med.get("dose_mg")
        if avs_dose is not None and mar_dose is not None and avs_dose != mar_dose:
            dosage_discrepancies.append({
                "name": avs_med["name"], "avs_dose_mg": avs_dose, "mar_dose_mg": mar_dose,
            })

        avs_route, mar_route = avs_med.get("route"), mar_med.get("route")
        norm_avs_route = (avs_route or "").lower().replace("-", "").replace(" ", "")
        norm_mar_route = (mar_route or "").lower().replace("-", "").replace(" ", "")
        # Treat PO/oral as equivalent; j-tube vs g-tube are genuinely different devices
        po_equivalents = {"po", "oral"}
        if norm_avs_route in po_equivalents and norm_mar_route in po_equivalents:
            continue
        if avs_route and mar_route and norm_avs_route != norm_mar_route:
            route_discrepancies.append({
                "name": avs_med["name"], "avs_route": avs_route, "mar_route": mar_route,
            })

    return dosage_discrepancies, route_discrepancies
