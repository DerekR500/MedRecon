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

import requests
from PIL import Image

from model_adapter import call_model
from match_engine import match_medications, compare_matched_pairs
from class_lookup import find_duplications


MODEL_NAME = "gemma-3-27b-it (UF Navigator)"


# ─────────────────────────────────────────────
# 0. MODEL LOAD
# ─────────────────────────────────────────────

def load_model(model_name: str = MODEL_NAME) -> str:
    """Return the model name for use throughout the pipeline."""
    print(f"✓ Model selected: {model_name}")
    return model_name


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
# STAGE A — Extract Home Medications
# ─────────────────────────────────────────────

STAGE_A_PROMPT_TEMPLATE = """You are a clinical data extraction assistant.
Extract all medications from the following After Visit Summary (AVS) into a JSON array.

IMPORTANT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Each item must have exactly these fields:
    "name"       : first word only of the generic or brand name (e.g., "Furosemide" not "Furosemide 40 mg") (string)
    "dose_mg"    : numeric dose in mg, or null if not a mg dose
    "dose_raw"   : dose exactly as written (string)
    "route"      : e.g. "PO", "IV" (string)
    "frequency"  : e.g. "Once daily", "BID", "PRN" (string)
    "indication" : what it is for (string)
- If a field is not present in the document, use null.
- Do NOT invent or infer any information not explicitly in the text.

AFTER VISIT SUMMARY (AVS) TEXT:
{avs_text}

Respond with ONLY the JSON array.
"""

def stage_a_extract_home_meds(model_name: str, avs_text: str) -> list[dict] | None:
    """
    Stage A: Extract medications from the After Visit Summary (AVS) into JSON.
    Input: raw text of the AVS
    Returns: list of medication dicts, or None on failure
    """
    print("\n── Stage A: Extracting medications from AVS ──")

    prompt = STAGE_A_PROMPT_TEMPLATE.replace("{avs_text}", avs_text)
    raw = call_model(prompt, model_name)

    result = parse_json_response(raw)
    if result is not None:
        print(f"  ✓ Extracted {len(result)} home medications")
    return result


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
        "dose_mg"    : numeric dose in mg, or null if not a mg dose
        "dose_raw"   : dose exactly as written (string)
        "route"      : e.g. "PO", "IV" (string)
        "frequency"  : e.g. "Once daily", "BID", "PRN" (string)
        "indication" : what it is for, or null if not listed (string)
- Do NOT invent or infer any information not explicitly visible in the image.
- If a field is not visible, use null.

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
        meds = result.get("medications", [])
        allergies = result.get("allergies", [])
        print(f"  ✓ Extracted {len(meds)} medications, {len(allergies)} allergies from image")
        print(json.dumps(result, indent=2))
    return result


# ─────────────────────────────────────────────
# STAGE B — Extract MAR Medications
# ─────────────────────────────────────────────

STAGE_B_PROMPT_TEMPLATE = """You are a clinical data extraction assistant.
Extract all medications from the following hospital MAR (Medication Administration Record)
into a JSON object.

IMPORTANT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Produce two top-level keys:
    "allergies_on_mar": array of strings (copy allergy section exactly, or ["NKDA"] if stated)
    "medications": array of objects with fields:
        "name"       : first word only of the generic or brand name (e.g., "Furosemide" not "Furosemide 40 mg") (string)
        "dose_mg"    : numeric dose in mg, or null if not a mg dose
        "dose_raw"   : dose exactly as written (string)
        "route"      : e.g. "PO", "IV" (string)
        "frequency"  : e.g. "Once daily", "BID", "PRN" (string)
        "scheduled_times": array of strings e.g. ["0800", "2000"], or null
- Do NOT invent or infer any information not explicitly in the text.
- If a field is not present, use null.

MAR TEXT:
{mar_text}

Respond with ONLY the JSON object.
"""

def stage_b_extract_mar(model_name: str, mar_text: str) -> dict | None:
    """
    Stage B: Extract MAR medications and allergy listing into JSON.
    Input: raw text of the MAR
    Returns: dict with 'allergies_on_mar' and 'medications' keys, or None on failure
    """
    print("\n── Stage B: Extracting MAR medications ──")

    prompt = STAGE_B_PROMPT_TEMPLATE.replace("{mar_text}", mar_text)
    raw = call_model(prompt, model_name)

    result = parse_json_response(raw)
    if result is not None:
        meds = result.get("medications", [])
        print(f"  ✓ Extracted {len(meds)} MAR medications")
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
        "route"      : e.g. "PO", "IV" (string)
        "frequency"  : e.g. "Once daily", "BID", "PRN" (string)
        "scheduled_times": array of strings e.g. ["0800", "2000"], or null
- Do NOT invent or infer any information not explicitly visible in the image.
- If a field is not visible, use null.

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
    RxNorm/RxClass (match_engine + class_lookup) — no LLM call.

    Only bare drug-name strings are ever sent to RxNorm/RxClass — no dose,
    route, case ID, or file name. If the APIs are unreachable, identity
    matching falls back to Tier C fuzzy matching (works offline) and the
    duplications check simply returns fewer/no results, with a logged
    warning — the reconciliation never crashes on network failure.

    Returns dict with omissions, duplications, dosage_discrepancies,
    incorrect_routes — each a list of medication name strings (first word
    only, the shape the rest of the app already depends on).
    """
    print("\n── Stage C: Deterministic matching (RxNorm/RxClass) ──")

    avs_meds = avs_data.get("medications") or []
    mar_meds = mar_data.get("medications") or []

    # One shared HTTP session + caches for this pipeline run, so repeated drug
    # names within this case's AVS+MAR don't trigger duplicate API calls.
    # Caches are deliberately per-case: batch cases run concurrently, and a
    # fresh cache per case is correct and safe (no cross-case data mixing).
    # A global cross-request cache is a possible future optimization only.
    session = requests.Session()
    rxcui_cache: dict = {}   # raw drug name        -> ingredient-level RxCUI
    class_cache: dict = {}   # normalized drug name -> ATC class name list

    matched_pairs, unmatched_avs = match_medications(
        avs_meds, mar_meds, session=session, rxcui_cache=rxcui_cache)
    dosage_discrepancies, route_discrepancies = compare_matched_pairs(matched_pairs)

    try:
        dup_groups = find_duplications(mar_meds, session, class_cache,
                                       rxcui_cache=rxcui_cache)
    except Exception as e:
        print(f"[WARN] RxClass duplication check failed ({e}) — reporting no duplications.")
        dup_groups = []

    if "__UNREACHABLE__" in rxcui_cache.values():
        print("[WARN] RxNorm unreachable — identity matching fell back to fuzzy (Tier C).")
    if any(v is None for v in class_cache.values()):
        print("[WARN] RxClass unreachable for some drugs — duplication results may be incomplete.")

    # Flatten duplication groups to a deduped list of drug name strings.
    dup_names: list[str] = []
    for group in dup_groups:
        for drug in group["drugs"]:
            name = _first_word(drug)
            if name not in dup_names:
                dup_names.append(name)

    findings = {
        "omissions":            [_first_word(m["name"]) for m in unmatched_avs],
        "duplications":         dup_names,
        "dosage_discrepancies": [_first_word(d["name"]) for d in dosage_discrepancies],
        "incorrect_routes":     [_first_word(r["name"]) for r in route_discrepancies],
    }

    print(
        f"  ✓ {len(findings['omissions'])} omission(s), "
        f"{len(findings['duplications'])} duplication(s), "
        f"{len(findings['dosage_discrepancies'])} dose discrepancy(ies), "
        f"{len(findings['incorrect_routes'])} route issue(s)"
    )
    return findings


# ─────────────────────────────────────────────
# MAIN — Wire all stages together
# ─────────────────────────────────────────────

def run_reconciliation(
    model_name: str,
    avs_text: str,
    mar_text: str,
    output_path: str = "reconciliation_report.json",
) -> dict:
    """
    Run the full 3-stage reconciliation pipeline (text inputs) and save a JSON report.

    Args:
        model_name:  Model display name or ID (routed via model_adapter)
        avs_text:    Raw text of the After Visit Summary (AVS)
        mar_text:    Raw text of the hospital MAR
        output_path: Where to save the final JSON report

    Returns:
        The final report as a dict
    """
    print("\n" + "=" * 55)
    print("  MEDICATION RECONCILIATION PIPELINE")
    print("=" * 55)

    # Stage A — extract AVS medications from text
    home_meds = stage_a_extract_home_meds(model_name, avs_text)
    if not home_meds:
        print("[ERROR] Stage A failed — could not extract AVS medications.")
        return {}

    # Stage B — extract MAR meds from text
    mar_data = stage_b_extract_mar(model_name, mar_text)
    if not mar_data:
        print("[ERROR] Stage B failed — could not extract MAR medications.")
        return {}

    # Stage C — deterministic comparison (no LLM call)
    avs_data = {"medications": home_meds}
    findings = stage_c_compare(avs_data, mar_data)
    if not findings:
        print("[ERROR] Stage C failed — could not compare documents.")
        return {}

    # Assemble final report
    final_report = {
        "pipeline_stages": {
            "stage_a_home_meds": home_meds,
            "stage_b_mar_data":  mar_data,
            "stage_c_findings":  findings,
        }
    }

    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n✓ Full report saved to: {output_path}")
    print("=" * 55)

    return final_report


if __name__ == "__main__":
    model = load_model()

    print("\n" + "=" * 55)
    print("  MEDICATION RECONCILIATION PIPELINE")
    print("=" * 55)

