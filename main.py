# pip install langchain-ollama langchain-core ollama pillow
"""
medgemma_reconciliation.py
==========================
Medication Reconciliation Pipeline using MedGemma 1.5
Strategy: Decompose into stages + force structured JSON output at each step
to minimize hallucinations.

Pipeline Stages:
  Stage A — Extract home medications (preadmission form) → structured JSON
  Stage B — Extract MAR medications → structured JSON
  Stage C — Deterministic comparison via RxNorm/RxClass (no LLM call)

Usage:
  python main.py

Requirements:
  pip install langchain-ollama langchain-core ollama pillow
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import re
import threading
from collections import Counter

import requests
from PIL import Image

from model_adapter import call_model
from match_engine import (
    match_medications,
    compare_matched_pairs,
    clean_lists,
    find_avs_duplications,
)


# ─────────────────────────────────────────────
# HELPER — Parse JSON from model output safely
# ─────────────────────────────────────────────

def parse_json_response(raw_text: str) -> dict | list | None:
    """
    Strip markdown fences and parse JSON from the model's raw output.
    Returns None if parsing fails.
    """
    # Remove ```json ... ``` or ``` ... ``` wrappers if present
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

    # Find the first '{' or '[' to skip any leading prose
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = cleaned.find(start_char)
        if start != -1:
            end = cleaned.rfind(end_char)
            if end != -1:
                json_str = cleaned[start:end + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass  # fall through to next attempt

    print("[WARN] Could not extract valid JSON from model response.")
    print("[RAW OUTPUT]:", raw_text[:500])
    return None

# ─────────────────────────────────────────────
# Extraction observability — log-only, NEVER mutates extracted data
# ─────────────────────────────────────────────

REPETITION_RUN_THRESHOLD = 3        # identical consecutive records before warning
PAGE_MED_VOLUME_CEILING = 20        # medications per page before volume warning
OMISSION_RATIO_WARN_THRESHOLD = 0.5  # omissions / AVS meds fraction before warning

# Cumulative warning tallies for the [RUN-SUMMARY] line. Snapshot-and-cleared
# by stage_c_compare. Single-case runs are exact; concurrent batch cases share
# the counter, so a summary may attribute another in-flight case's extraction
# warnings to itself.
_WARNING_COUNTS: Counter = Counter()
_WARNING_LOCK = threading.Lock()


def _warn(tag: str, msg: str) -> None:
    with _WARNING_LOCK:
        _WARNING_COUNTS[tag] += 1
    print(f"[WARN][{tag}] {msg}")


def _check_extraction_quality(meds: list) -> None:
    """Repetition + volume warnings on one page's raw medication list.
    Observer only: logs and returns, never mutates/filters/reorders `meds`.

    scheduled_times is part of the identity tuple deliberately — legitimate
    split dosing differs only in that field (e.g. Norditropin 0.5 mg SubQ
    @ 0502 and @ 2100), and collapsing on name alone would flag it as a loop.
    """
    def key(m):
        if not isinstance(m, dict):
            return ("<non-dict>", repr(m))
        st = m.get("scheduled_times")
        if isinstance(st, list):
            st = tuple(st)
        return (m.get("name"), m.get("dose_raw"), m.get("route"),
                m.get("frequency"), st)

    i, n = 0, len(meds)
    while i < n:
        j = i + 1
        while j < n and key(meds[j]) == key(meds[i]):
            j += 1
        run_length = j - i
        if run_length >= REPETITION_RUN_THRESHOLD:
            name = meds[i].get("name") if isinstance(meds[i], dict) else meds[i]
            _warn("REPETITION",
                  f"'{name}' repeated {run_length}x consecutively - possible generation loop")
        i = j

    if n > PAGE_MED_VOLUME_CEILING:
        _warn("VOLUME",
              f"page emitted {n} medications (ceiling {PAGE_MED_VOLUME_CEILING}) - "
              "verify against source image")


# ─────────────────────────────────────────────
# Shared extraction filtering
# ─────────────────────────────────────────────

PHANTOM_MED_BLOCKLIST = {
    "signed", "signature", "sign", "provider", "physician",
    "nurse", "attestation", "attested", "verified", "approved"
}

def _is_valid_med(entry: dict) -> bool:
    name = (entry.get("name") or "").strip().lower()
    return bool(name) and name not in PHANTOM_MED_BLOCKLIST

# ─────────────────────────────────────────────
# STAGE A (image variant) — AVS is an image
# ─────────────────────────────────────────────

STAGE_A_IMAGE_PROMPT = """You are a clinical data extraction assistant.
Extract all information from this After Visit Summary (AVS) image.

IMPORTANT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Produce two top-level keys:
    "allergies": array of strings (each allergy as listed)
    "medications": array of medication objects with fields:
        "name"       : first word only of the generic or brand name (e.g., "Furosemide" not "Furosemide 40 mg") (string)
        "dose_mg"    : numeric dose in mg of the ACTUAL ADMINISTERED dose, or null if not a mg dose.
                       If the document shows both a formulation/tablet strength AND a parenthetical
                       actual administered dose, ALWAYS extract the parenthetical actual dose —
                       never the leading formulation strength.
                       Examples:
                         "amiodarone 100 mg tablet ... Take 0.5 tablets (50 mg) by mouth" -> dose_mg: 50
                         "gabapentin 250 mg/5 mL solution ... Give 2.5 mL (125 mg)"       -> dose_mg: 125
        "dose_raw"   : the actual administered dose exactly as written (string) — the same
                       administered dose used for dose_mg, not the formulation strength
                       (e.g., "0.5 tablets (50 mg)" not "100 mg tablet")
        "route"      : exactly one of: "PO", "IV", "IM", "SubQ", "G-tube", "J-tube", "NG",
                       "Nasal", "Inhaled", "Topical", "PR". Map any other phrasing in the
                       document to the closest value in this list, e.g.:
                         "by mouth" / "oral" / "mouth"     -> "PO"
                         "under the skin" / "subcutaneous" -> "SubQ"
                         "tube" (type unspecified)         -> the specific tube type ("G-tube",
                           "J-tube", or "NG") stated elsewhere in the document for that same
                           medication or access route, if determinable from context
                       If the route genuinely maps to none of these values, use null.
                       NEVER output a free-text route outside this list.
        "frequency"  : e.g. "Once daily", "BID", "PRN" (string)
        "indication" : what it is for, or null if not listed (string)
        "is_prn"     : boolean — true if the order is PRN / "as needed" / "as directed";
                       otherwise false. Default false.
        "schedule_type": exactly one of: "scheduled", "prn", "one_time".
                       "prn" for as-needed orders, "one_time" for a single/once-only order,
                       "scheduled" for standing recurring orders. Default "scheduled".
        "formulation": coarse dose form, exactly one of: "tablet", "capsule", "liquid",
                       "suppository", "inhaler", "nebulizer_solution", "cream_ointment",
                       "patch", "drops", "spray", "injection", "other", or null.
                       Map "suspension" / "solution" / "syrup" -> "liquid". Use null if the
                       form is not determinable from the image. Do NOT guess.
- Do NOT invent or infer any information not explicitly visible in the image.
- If the entire page contains no medications (e.g. it is a signature page, attestation page,
  provider sign-off, or contains only administrative text with no drug names), return an
  empty array for "medications": []. NEVER output null for "medications" or "allergies" —
  always use an empty array [] when there is nothing to list.
- Ignore signature lines, provider names, attestation text, and all non-medication content.
  Do NOT extract the word "Signed", "Signature", a provider's name, or any administrative
  label as a medication entry.
- If a field is not visible, use null.

EXAMPLES (showing the new fields populated):
  "amiodarone 100 mg tablet ... Take 0.5 tablets (50 mg) by mouth twice a day"
    -> {"name":"amiodarone","dose_mg":50,"dose_raw":"0.5 tablets (50 mg)","route":"PO",
        "frequency":"BID","indication":null,"is_prn":false,
        "schedule_type":"scheduled","formulation":"tablet"}
  "senna 8.8 mg/5 mL syrup ... Take 5 mL (8.8 mg) by mouth as needed for constipation"
    -> {"name":"senna","dose_mg":8.8,"dose_raw":"5 mL (8.8 mg)","route":"PO",
        "frequency":"PRN","indication":"constipation","is_prn":true,
        "schedule_type":"prn","formulation":"liquid"}

Respond with ONLY the JSON object.
"""

def stage_a_extract_from_image(model_name: str, image) -> dict | None:
    """
    Stage A (image variant): Extract medications + allergies from an AVS image.
    Input: PIL Image object
    Returns: dict with 'allergies' and 'medications' keys, or None on failure
    """
    print("\n── Stage A (image): Extracting from AVS image ──")

    raw = call_model(STAGE_A_IMAGE_PROMPT, model_name, image=image)

    result = parse_json_response(raw)
    if result is not None:
        _check_extraction_quality(result.get("medications") or [])
        result["medications"] = [
            m for m in (result.get("medications") or []) if _is_valid_med(m)
        ]
        result["allergies"] = result.get("allergies") or []
        print(f"  ✓ Extracted {len(result['medications'])} medications, "
              f"{len(result['allergies'])} allergies from image")
        print(json.dumps(result, indent=2))
    return result


# ─────────────────────────────────────────────
# STAGE B (image variant) — MAR is an image
# ─────────────────────────────────────────────

STAGE_B_IMAGE_PROMPT = """You are a clinical data extraction assistant.
Extract all information from this hospital MAR (Medication Administration Record) image.

IMPORTANT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Produce two top-level keys:
    "allergies_on_mar": array of strings (copy allergy section exactly, or ["NKDA"] if stated)
    "medications": array of medication objects with fields:
        "name"       : first word only of the generic or brand name (e.g., "Furosemide" not "Furosemide 40 mg") (string)
        "dose_mg"    : numeric dose in mg, or null if not a mg dose
        "dose_raw"   : dose exactly as written (string)
        "route"      : exactly one of: "PO", "IV", "IM", "SubQ", "G-tube", "J-tube", "NG",
                       "Nasal", "Inhaled", "Topical", "PR". Map any other phrasing in the
                       document to the closest value in this list, e.g.:
                         "by mouth" / "oral" / "mouth"     -> "PO"
                         "under the skin" / "subcutaneous" -> "SubQ"
                         "tube" (type unspecified)         -> the specific tube type ("G-tube",
                           "J-tube", or "NG") stated elsewhere in the document for that same
                           medication or access route, if determinable from context
                       If the route genuinely maps to none of these values, use null.
                       NEVER output a free-text route outside this list.
        "frequency"  : e.g. "Once daily", "BID", "PRN" (string)
        "scheduled_times": ALWAYS a JSON array of "HHMM" strings, e.g. ["0800", "2000"].
                       Use an empty array [] when no administration times are shown — never
                       null and never a bare string, so downstream can count doses/day.
        "is_prn"     : boolean — true if the order is PRN / "as needed" / "as directed";
                       otherwise false. Default false.
        "schedule_type": exactly one of: "scheduled", "prn", "one_time".
                       "prn" for as-needed orders, "one_time" for a single/once-only order,
                       "scheduled" for standing recurring orders. Default "scheduled".
        "is_cancelled": boolean — best-effort true ONLY if the row is struck through /
                       crossed out / marked discontinued. Default false. If unclear, false.
                       NEVER fabricate a cancellation.
- Do NOT invent or infer any information not explicitly visible in the image.
- Ignore signature lines, provider names, and non-medication text.
- If a field is not visible, use null (except scheduled_times, which uses [] — see above).

EXAMPLES (showing the new fields populated):
  "amiodarone (Pacerone) tablet 50 mg ... g-tube, Daily ... Given: 0852, 0856, 0824"
    -> {"name":"amiodarone","dose_mg":50,"dose_raw":"50 mg","route":"G-tube",
        "frequency":"Daily","scheduled_times":["0852","0856","0824"],
        "is_prn":false,"schedule_type":"scheduled","is_cancelled":false}
  A struck-through / discontinued row for "ranitidine 15 mg ... j-tube, PRN"
    -> {"name":"ranitidine","dose_mg":15,"dose_raw":"15 mg","route":"J-tube",
        "frequency":"PRN","scheduled_times":[],
        "is_prn":true,"schedule_type":"prn","is_cancelled":true}

Respond with ONLY the JSON object.
"""

def stage_b_extract_mar_from_image(model_name: str, image) -> dict | None:
    """
    Stage B (image variant): Extract MAR medications and allergy listing from a MAR image.
    Input: PIL Image object
    Returns: dict with 'allergies_on_mar' and 'medications' keys, or None on failure
    """
    print("\n── Stage B (image): Extracting from MAR image ──")

    raw = call_model(STAGE_B_IMAGE_PROMPT, model_name, image=image)

    result = parse_json_response(raw)
    if result is not None:
        _check_extraction_quality(result.get("medications") or [])
        meds = result.get("medications", [])
        print(f"  ✓ Extracted {len(meds)} MAR medications from image")
        print(json.dumps(result, indent=2))
    return result


# ─────────────────────────────────────────────
# STAGE C — Deterministic Medication Comparison
# ─────────────────────────────────────────────

def _first_word(name: str) -> str:
    """Reduce a medication name to its first word — the shape app.py and the
    frontend already expect (e.g. "Furosemide" not "Furosemide 40 mg")."""
    words = (name or "").split()
    return words[0] if words else ""


def stage_c_compare(avs_data: dict, mar_data: dict) -> dict | None:
    """
    Stage C: deterministic comparison of AVS and MAR medication lists via
    RxNorm (match_engine) — no LLM call.

    Only bare drug-name strings are ever sent to RxNorm — no dose, route, case
    ID, or file name. If RxNorm is unreachable, identity matching falls back to
    Tier C fuzzy matching and duplication grouping falls back to normalized
    names (both work offline), with a logged warning — the reconciliation never
    crashes on network failure.

    Returns dict with omissions, duplications, dosage_discrepancies,
    incorrect_routes — each a list of medication name strings (first word
    only, the shape the rest of the app already depends on).
    """
    print("\n── Stage C: Deterministic matching (RxNorm/RxClass) ──")

    # Pre-detection cleaning: drop acetaminophen / PRN / one-time / cancelled
    # meds so the four detectors only see rows worth reconciling. Returns COPIES
    # — avs_data/mar_data (raw extraction) are left intact for scoring/debugging.
    raw_avs = avs_data.get("medications") or []
    raw_mar = mar_data.get("medications") or []
    avs_meds, mar_meds, removed = clean_lists(avs_data, mar_data)
    n_removed = sum(len(v) for v in removed.values())
    print(f"[CLEAN] removed {n_removed} "
          f"(acetaminophen={len(removed['acetaminophen'])}, prn={len(removed['prn'])}, "
          f"one_time={len(removed['one_time'])}, cancelled={len(removed['cancelled'])}) | "
          f"AVS {len(raw_avs)}->{len(avs_meds)} MAR {len(raw_mar)}->{len(mar_meds)}")

    # One shared HTTP session + caches for this pipeline run, so repeated drug
    # names within this case's AVS+MAR don't trigger duplicate API calls.
    # Caches are deliberately per-case: batch cases run concurrently, and a
    # fresh cache per case is correct and safe (no cross-case data mixing).
    # A global cross-request cache is a possible future optimization only.
    session = requests.Session()
    rxcui_cache: dict = {}   # raw drug name -> ingredient-level RxCUI (shared by
                             # matching and the AVS duplication check)

    matched_pairs, unmatched_avs, unmatched_mar, match_stats = match_medications(
        avs_meds, mar_meds, session=session, rxcui_cache=rxcui_cache)
    print(f"[STAGE-C] AVS={match_stats['avs_total']} MAR={match_stats['mar_total']} | "
          f"matched: A={match_stats['tier_a']} B={match_stats['tier_b']} "
          f"C={match_stats['tier_c']} | unmatched_avs={match_stats['unmatched']} "
          f"unmatched_mar={match_stats['unmatched_mar']}")

    # Omissions are the MAR-side unmatched set: a med given inpatient (on the
    # MAR) but not carried onto the discharge AVS. The AVS-side unmatched set is
    # still computed and logged below for debugging, but is no longer reported.
    #
    # TODO(Faith): restrict the omission reference to the LAST 3 DAYS of the
    # discharge MAR (a med stopped early in the stay shouldn't count as forgotten
    # at discharge). For now we compare against the full scheduled (cleaned) MAR
    # and treat the missing 3-day window as a known limitation.
    print(f"[STAGE-C][debug] AVS-side unmatched (NOT reported as omissions): "
          f"{[_first_word(m['name']) for m in unmatched_avs]}")

    dosage_discrepancies, route_discrepancies = compare_matched_pairs(matched_pairs)

    # Duplications: on the cleaned AVS list ONLY, flag two or more entries that
    # share the same ingredient (RxNorm ingredient-level, so brand==generic) AND
    # the same formulation. Different formulations of one ingredient are not a
    # duplication. Replaces the old RxClass/ATC grouping (which false-flagged
    # unrelated drugs under broad classes like "Other antiepileptics").
    try:
        dup_groups = find_avs_duplications(avs_meds, session=session, rxcui_cache=rxcui_cache)
    except Exception as e:
        print(f"[WARN] AVS duplication check failed ({e}) — reporting no duplications.")
        dup_groups = []

    if "__UNREACHABLE__" in rxcui_cache.values():
        print("[WARN] RxNorm unreachable — identity matching and duplication grouping "
              "fell back to normalized names.")

    # Flatten duplication groups to a deduped list of drug name strings.
    dup_names: list[str] = []
    for group in dup_groups:
        for drug in group["drugs"]:
            name = _first_word(drug)
            if name not in dup_names:
                dup_names.append(name)

    findings = {
        "omissions":            [_first_word(m["name"]) for m in unmatched_mar],
        "duplications":         dup_names,
        "dosage_discrepancies": [_first_word(d["name"]) for d in dosage_discrepancies],
        "incorrect_routes":     [_first_word(r["name"]) for r in route_discrepancies],
    }

    # Omissions are MAR-side now, so the ratio is measured against the MAR total:
    # a large fraction of inpatient meds missing from discharge points at AVS
    # extraction failure (or a genuinely large omission set).
    n_mar = match_stats["mar_total"]
    n_omissions = len(findings["omissions"])
    if n_mar and n_omissions > OMISSION_RATIO_WARN_THRESHOLD * n_mar:
        _warn("OMISSION-RATIO",
              f"{n_omissions}/{n_mar} ({round(100 * n_omissions / n_mar)}%) of MAR meds "
              "not carried to discharge AVS - check AVS extraction before trusting this result")

    print(
        f"  ✓ {len(findings['omissions'])} omission(s), "
        f"{len(findings['duplications'])} duplication(s), "
        f"{len(findings['dosage_discrepancies'])} dose discrepancy(ies), "
        f"{len(findings['incorrect_routes'])} route issue(s)"
    )

    # Machine-readable one-line summary — greppable across batch logs.
    # model_name is not in scope here (stage_c_compare receives only the two
    # extracted med dicts), so it is omitted rather than threading a new
    # parameter through the call stack.
    with _WARNING_LOCK:
        warning_counts = {k: v for k, v in _WARNING_COUNTS.items()}
        _WARNING_COUNTS.clear()
    run_summary = {
        "avs_total": match_stats["avs_total"],
        "mar_total": n_mar,
        "tier_a": match_stats["tier_a"],
        "tier_b": match_stats["tier_b"],
        "tier_c": match_stats["tier_c"],
        "unmatched_avs": match_stats["unmatched"],
        "unmatched_mar": match_stats["unmatched_mar"],
        "omissions": n_omissions,
        "duplications": len(findings["duplications"]),
        "dose_discrepancies": len(findings["dosage_discrepancies"]),
        "route_issues": len(findings["incorrect_routes"]),
        "warnings": warning_counts,
    }
    print("[RUN-SUMMARY] " + json.dumps(run_summary, separators=(",", ":")))

    return findings

