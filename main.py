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
  Stage C — Compare medications and extract 4 values of discrepancy

Usage:
  python main.py

Requirements:
  pip install langchain-ollama langchain-core ollama pillow
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import re
from PIL import Image
from model_adapter import call_model


MODEL_NAME = "medgemma1.5:latest"


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
    return result


# ─────────────────────────────────────────────
# STAGE C — LLM Medication Comparison
# ─────────────────────────────────────────────

STAGE_C_PROMPT_TEMPLATE = """You are a clinical data extraction assistant comparing two medication lists.

AFTER VISIT SUMMARY (AVS) MEDICATIONS:
{avs_medications_json}

MEDICATION ADMINISTRATION RECORD (MAR) MEDICATIONS:
{mar_medications_json}

Identify issues in exactly these four categories and return ONLY valid JSON.

DEFINITIONS:
- "omissions": medications present in one document but absent from the other — check both directions (AVS→MAR and MAR→AVS). For each medication, look up its "name" in the other list. If it is not found there, it is an omission.
- "duplications": medications listed MORE THAN ONCE in the SAME document (AVS or MAR)
- "dosage_discrepancies": medications appearing in BOTH documents but with DIFFERENT doses
- "incorrect_routes": medications appearing in BOTH documents but with DIFFERENT administration routes (e.g., PO vs IV)

RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Each value must be a JSON array of medication name strings.
- Use ONLY the first word of the medication name (e.g., "Furosemide" not "Furosemide 40 mg PO").
- Do NOT invent issues not clearly supported by the data above.

Respond with ONLY a JSON object with keys: "omissions", "duplications", "dosage_discrepancies", "incorrect_routes".
"""


def stage_c_compare(model_name: str, avs_data: dict, mar_data: dict) -> dict | None:
    """
    Stage C: LLM-based comparison of AVS and MAR medication lists.
    Returns dict with omissions, duplications, dosage_discrepancies, incorrect_routes.
    """
    print("\n── Stage C: LLM comparison of AVS vs MAR ──")

    avs_meds_json = json.dumps(avs_data.get("medications") or [], indent=2)
    mar_meds_json = json.dumps(mar_data.get("medications") or [], indent=2)

    prompt = (STAGE_C_PROMPT_TEMPLATE
              .replace("{avs_medications_json}", avs_meds_json)
              .replace("{mar_medications_json}", mar_meds_json))

    raw = call_model(prompt, model_name)
    result = parse_json_response(raw)

    if result is not None:
        print(
            f"  ✓ {len(result.get('omissions', []))} omission(s), "
            f"{len(result.get('duplications', []))} duplication(s), "
            f"{len(result.get('dosage_discrepancies', []))} dose discrepancy(ies), "
            f"{len(result.get('incorrect_routes', []))} route issue(s)"
        )
    return result


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

    # Stage C — LLM comparison
    avs_data = {"medications": home_meds}
    findings = stage_c_compare(model_name, avs_data, mar_data)
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

