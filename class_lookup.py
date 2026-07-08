"""
class_lookup.py
===============
Deterministic Duplications detection via RxClass, RxNorm's companion API
for pharmacological drug classes.

The current LLM Stage C flags Patient 1's split-dose entries -- the two
clonidine entries and the linked hydroxyzine 7 mg / 9 mg group -- as
duplications, even though the source documents annotate them as intentional
split dosing. A genuine duplication is two *different* drugs from the same
pharmacological class prescribed concurrently, not one drug given in
staggered doses.

This module makes that distinction deterministic, the same way
match_engine.py made Omissions/Dosage/Routes deterministic:

  Step 1 -- resolve each MAR drug to its pharmacological class(es) via RxClass
  Step 2 -- group MAR meds by class
  Step 3 -- within a class, collapse entries that share an ingredient-level
            RxCUI (same drug, split dosing); only 2+ *distinct* ingredients
            in one class is a real duplication candidate

Only the bare drug-name string is ever sent to RxClass -- no dose, route,
case ID, or any other patient data.
"""

from collections import defaultdict

import requests

from match_engine import normalize, rxnorm_ingredient_rxcui

RXCLASS_BASE = "https://rxnav.nlm.nih.gov/REST/rxclass"

# relaSource for the class lookup. We use ATC (Anatomical Therapeutic
# Chemical) rather than MED-RT: ATC is a single, stable hierarchy and its
# byDrugName response resolves to the ATC-4 pharmacological subgroup
# (e.g. "Benzodiazepine derivatives", "Sulfonamides, plain") -- exactly the
# granularity that makes two benzodiazepines or two loop diuretics collide
# while keeping unrelated drugs apart. MED-RT instead returns several
# overlapping relations per drug (may_treat, mechanism of action, physiologic
# effect, established pharmacologic class), which would make deterministic
# single-class grouping ambiguous.
RELA_SOURCE = "ATC"


def get_drug_class(name: str, session: requests.Session, cache: dict) -> list[str] | None:
    """
    Resolve a drug name to its pharmacological class name(s) via RxClass.
    Only the bare name string is sent -- no dose, no patient/case data.

    Returns a list of ATC-4 class names (a drug can belong to more than one),
    an empty list if RxClass has no class for the drug, or None if the API is
    unreachable -- callers treat None the same graceful way match_engine treats
    "__UNREACHABLE__". Results are cached in-memory by normalized drug name,
    mirroring rxnorm_ingredient_rxcui().
    """
    key = normalize(name)
    if key in cache:
        return cache[key]

    try:
        r = session.get(f"{RXCLASS_BASE}/class/byDrugName.json",
                        params={"drugName": name, "relaSource": RELA_SOURCE}, timeout=5)
        r.raise_for_status()
        infos = r.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
        # De-duplicate class names (some sources list a drug under a class more
        # than once) and keep a stable, sorted order for reproducible grouping.
        classes = sorted({info["rxclassMinConceptItem"]["className"]
                          for info in infos if info.get("rxclassMinConceptItem")})
        cache[key] = classes
        return classes

    except requests.RequestException:
        cache[key] = None
        return None


def find_duplications(mar_meds: list, session: requests.Session, cache: dict,
                      offline_class_cache: dict | None = None,
                      offline_rxcui_cache: dict | None = None,
                      rxcui_cache: dict | None = None) -> list[dict]:
    """
    Detect genuine Duplications: two *different* drugs from the same
    pharmacological class prescribed concurrently.

    Split/staggered dosing of a single drug (Patient 1's clonidine and the
    linked hydroxyzine 7 mg / 9 mg group) is NOT a duplication -- those entries
    share an ingredient-level RxCUI and are collapsed to one identity here.

    `cache` is the in-memory RxClass class cache (keyed by normalized name).
    The ingredient-level identity check reuses rxnorm_ingredient_rxcui() from
    match_engine rather than re-implementing it; pass rxcui_cache to share one
    RxCUI cache with other lookups in the same pipeline run (defaults to a
    fresh internal one).

    In --offline-demo mode the caller passes offline_class_cache /
    offline_rxcui_cache (name -> value maps) which are consulted instead of the
    live APIs, following match_medications()'s offline_cache pattern.

    Returns a list of {"class", "drugs"} candidate duplications.
    """
    if rxcui_cache is None:
        rxcui_cache = {}
    by_class = defaultdict(list)  # class name -> list of (display_name, identity)

    for med in mar_meds:
        name = med["name"]

        # --- Step 1: pharmacological class(es) ---
        classes = (offline_class_cache.get(name) if offline_class_cache is not None
                   else get_drug_class(name, session, cache))
        if not classes:  # None (unreachable) or [] (no ATC class) -> can't group
            continue

        # --- ingredient-level identity, to collapse split-dose duplicates ---
        rxcui = (offline_rxcui_cache.get(name) if offline_rxcui_cache is not None
                 else rxnorm_ingredient_rxcui(name, session, rxcui_cache))
        # Fall back to the normalized name when RxNorm can't give a usable
        # RxCUI, so same-named split doses still collapse to one identity.
        identity = rxcui if (rxcui and rxcui != "__UNREACHABLE__") else normalize(name)

        for cls in classes:
            by_class[cls].append((name, identity))

    # --- Step 3: any class with 2+ distinct ingredients is a candidate ---
    duplications = []
    for cls, entries in by_class.items():
        distinct = {}  # identity -> display name (first one wins)
        for disp, identity in entries:
            distinct.setdefault(identity, disp)
        if len(distinct) >= 2:
            duplications.append({"class": cls, "drugs": sorted(distinct.values())})

    return duplications
