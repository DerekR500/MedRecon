# pip install google-generativeai pillow
"""
medgemma_reconciliation.py
==========================
Medication Reconciliation Pipeline using Google Gemini 1.5 Flash
Strategy: Decompose into stages + force structured JSON output at each step
to minimize hallucinations.

Pipeline Stages:
  Stage A — Extract home medications (preadmission form) → structured JSON
  Stage B — Extract MAR medications → structured JSON
  Stage C — Deterministic comparison using Python (no LLM involved)
  Stage D — Send only the flagged discrepancies back to Gemini for
             clinical interpretation

Usage:
  python medgemma_reconciliation.py

Requirements:
  pip install google-generativeai pillow
"""

import json
import re
import os
import google.genai as genai
from google.genai import types
from PIL import Image

MODEL_NAME = "gemini-2.5-flash"


# ─────────────────────────────────────────────
# 0. AUTH & MODEL LOAD
# ─────────────────────────────────────────────

def load_model() -> object:
    """Load Gemini 1.5 Flash once and reuse across all stages."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "  Windows:   set GEMINI_API_KEY=your_key_here\n"
            "  Linux/Mac: export GEMINI_API_KEY=your_key_here"
        )
    client = genai.Client(api_key=api_key)
    print("✓ Gemini 2.5 Flash client ready")
    return client


# ─────────────────────────────────────────────
# STAGE A — Extract Home Medications
# ─────────────────────────────────────────────

STAGE_A_PROMPT_TEMPLATE = """You are a clinical data extraction assistant.
Extract all medications from the following preadmission form into a JSON array.

IMPORTANT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Each item must have exactly these fields:
    "name"       : generic or brand name as written (string)
    "dose_mg"    : numeric dose in mg, or null if not a mg dose
    "dose_raw"   : dose exactly as written (string)
    "route"      : e.g. "PO", "IV" (string)
    "frequency"  : e.g. "Once daily", "BID", "PRN" (string)
    "indication" : what it is for (string)
- If a field is not present in the document, use null.
- Do NOT invent or infer any information not explicitly in the text.

PREADMISSION FORM TEXT:
{preadmission_text}

Respond with ONLY the JSON array.
"""

def stage_a_extract_home_meds(model, preadmission_text: str) -> list[dict] | None:
    """
    Stage A: Extract home medications from the preadmission form into JSON.
    Input: raw text of the preadmission form
    Returns: list of medication dicts, or None on failure
    """
    print("\n── Stage A: Extracting home medications ──")

    prompt = STAGE_A_PROMPT_TEMPLATE.format(preadmission_text=preadmission_text)

    response = model.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    result = json.loads(response.text)

    if result is not None:
        print(f"  ✓ Extracted {len(result)} home medications")
    return result


# ─────────────────────────────────────────────
# STAGE A (image variant) — MAR is an image
# ─────────────────────────────────────────────

STAGE_A_IMAGE_PROMPT = """You are a clinical data extraction assistant.
Extract all information from this preadmission/MAR document image.

IMPORTANT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Produce two top-level keys:
    "allergies": array of strings (each allergy as listed)
    "medications": array of medication objects with fields:
        "name"       : generic or brand name as written (string)
        "dose_mg"    : numeric dose in mg, or null if not a mg dose
        "dose_raw"   : dose exactly as written (string)
        "route"      : e.g. "PO", "IV" (string)
        "frequency"  : e.g. "Once daily", "BID", "PRN" (string)
        "indication" : what it is for, or null if not listed (string)
- Do NOT invent or infer any information not explicitly visible in the image.
- If a field is not visible, use null.

Respond with ONLY the JSON object.
"""

def stage_a_extract_from_image(model, image) -> dict | None:
    """
    Stage A (image variant): Extract medications + allergies from a MAR image.
    Input: PIL Image object
    Returns: dict with 'allergies' and 'medications' keys, or None on failure
    """
    print("\n── Stage A (image): Extracting from MAR image ──")

    response = model.models.generate_content(
        model=MODEL_NAME,
        contents=[image, STAGE_A_IMAGE_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    result = json.loads(response.text)

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
        "name"       : generic or brand name as written (string)
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

def stage_b_extract_mar(model, mar_text: str) -> dict | None:
    """
    Stage B: Extract MAR medications and allergy listing into JSON.
    Input: raw text of the MAR
    Returns: dict with 'allergies_on_mar' and 'medications' keys, or None on failure
    """
    print("\n── Stage B: Extracting MAR medications ──")

    prompt = STAGE_B_PROMPT_TEMPLATE.format(mar_text=mar_text)

    response = model.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    result = json.loads(response.text)

    if result is not None:
        meds = result.get("medications", [])
        print(f"  ✓ Extracted {len(meds)} MAR medications")
    return result


# ─────────────────────────────────────────────
# STAGE B (image variant) — MAR is an image
# ─────────────────────────────────────────────

STAGE_B_IMAGE_PROMPT = """You are a clinical data extraction assistant.
Extract all medication orders and allergy information from this hospital MAR
(Medication Administration Record) image.

IMPORTANT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Produce two top-level keys:
    "allergies_on_mar": array of strings (copy allergy section exactly, or ["NKDA"] if stated)
    "medications": array of objects with fields:
        "name"            : generic or brand name as written (string)
        "dose_mg"         : numeric dose in mg, or null if not a mg dose
        "dose_raw"        : dose exactly as written (string)
        "route"           : e.g. "PO", "IV" (string)
        "frequency"       : e.g. "Once daily", "BID", "PRN" (string)
        "scheduled_times" : array of strings e.g. ["0800", "2000"], or null
- Each drug must appear ONCE. Consolidate multiple scheduled times for the same
  drug into a single entry with all times in the "scheduled_times" array.
- Do NOT invent or infer any information not explicitly visible in the image.
- If a field is not visible, use null.

Respond with ONLY the JSON object.
"""

def stage_b_extract_mar_from_image(model, image: Image.Image) -> dict | None:
    """
    Stage B (image variant): Extract MAR medications and allergies from an image.
    Input: PIL Image object of the MAR
    Returns: dict with 'allergies_on_mar' and 'medications' keys, or None on failure
    """
    print("\n── Stage B (image): Extracting MAR from image ──")

    response = model.models.generate_content(
        model=MODEL_NAME,
        contents=[image, STAGE_B_IMAGE_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    result = json.loads(response.text)

    if result is not None:
        meds = result.get("medications", [])
        print(f"  ✓ Extracted {len(meds)} MAR medications from image")
    return result


# ─────────────────────────────────────────────
# STAGE C — Deterministic Python Comparison
# (No LLM — pure logic, no hallucinations possible)
# ─────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Lowercase and strip punctuation for fuzzy name matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Map of brand → generic and common aliases for matching
MED_ALIASES = {
    "lasix":        "furosemide",
    "coreg":        "carvedilol",
    "lipitor":      "atorvastatin",
    "xanax":        "alprazolam",
    "tylenol":      "acetaminophen",
    "bactrim":      "sulfamethoxazoletrimethoprim",
    "bactrimds":    "sulfamethoxazoletrimethoprim",
}

# Map of drug → pharmacological class for therapeutic duplication check
DRUG_CLASSES = {
    "carvedilol":   "beta_blocker",
    "metoprolol":   "beta_blocker",
    "metoprololsuccinate": "beta_blocker",
    "atenolol":     "beta_blocker",
    "furosemide":   "loop_diuretic",
    "bumetanide":   "loop_diuretic",
    "torsemide":    "loop_diuretic",
    "atorvastatin": "statin",
    "rosuvastatin": "statin",
    "simvastatin":  "statin",
    "lisinopril":   "ace_inhibitor",
    "enalapril":    "ace_inhibitor",
    "metformin":    "biguanide",
    "sulfamethoxazoletrimethoprim": "sulfonamide_antibiotic",
}

SULFA_DRUGS = {"sulfamethoxazoletrimethoprim", "sulfadiazine", "dapsone"}


def canonical(name: str) -> str:
    # Strip parenthetical content e.g. "Lasix (Furosemide)" → "Lasix"
    name = re.sub(r"\(.*?\)", "", name).strip()
    n = normalize_name(name)
    return MED_ALIASES.get(n, n)


def parse_dose_mg(med: dict) -> float | None:
    """Extract a float mg dose from a medication dict."""
    if med.get("dose_mg") is not None:
        try:
            return float(med["dose_mg"])
        except (ValueError, TypeError):
            pass
    # Try parsing from dose_raw as fallback
    raw = med.get("dose_raw") or ""
    match = re.search(r"([\d.]+)\s*mg", raw, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def stage_c_compare(
    home_meds: list[dict],
    home_allergies: list[str],
    mar_data: dict,
) -> dict:
    """
    Stage C: Deterministic comparison — no LLM involved.

    Returns a structured findings dict with:
      - allergy_conflict
      - dose_mismatches
      - frequency_mismatches
      - omissions        (home meds missing from MAR)
      - new_hospital_meds
      - therapeutic_duplications
    """
    print("\n── Stage C: Deterministic comparison ──")

    mar_meds      = mar_data.get("medications", [])
    mar_allergies = mar_data.get("allergies_on_mar", [])

    # Build lookup: canonical name → med dict for each list
    home_lookup = {canonical(m["name"]): m for m in home_meds}
    mar_lookup  = {canonical(m["name"]): m for m in mar_meds}

    findings = {
        "allergy_conflict":        None,
        "dose_mismatches":         [],
        "frequency_mismatches":    [],
        "omissions":               [],
        "new_hospital_meds":       [],
        "therapeutic_duplications":[],
    }

    # ── Allergy check ──────────────────────────────────────────────────────────
    home_allergy_str = " ".join(home_allergies).lower()
    mar_allergy_str  = " ".join(mar_allergies).lower()

    home_has_allergy = bool(home_allergies) and "nkda" not in home_allergy_str
    mar_has_nkda     = "nkda" in mar_allergy_str or not mar_allergies

    if home_has_allergy and mar_has_nkda:
        findings["allergy_conflict"] = {
            "home_list_allergies": home_allergies,
            "mar_allergy_field":   mar_allergies,
            "flag": "CRITICAL: Home list records allergies but MAR shows NKDA.",
        }

    # Check if any MAR drug belongs to an allergic class
    for cname, med in mar_lookup.items():
        drug_class = DRUG_CLASSES.get(cname, "")
        # Crude sulfa check
        if cname in SULFA_DRUGS and "sulfa" in home_allergy_str:
            findings["allergy_conflict"] = findings["allergy_conflict"] or {}
            findings["allergy_conflict"]["dangerous_order"] = (
                f"CRITICAL: {med['name']} is a sulfonamide and patient has a sulfa allergy."
            )

    # ── Dose & frequency mismatches for shared medications ────────────────────
    for cname, home_med in home_lookup.items():
        if cname not in mar_lookup:
            continue  # handled in omissions below
        mar_med = mar_lookup[cname]

        home_dose = parse_dose_mg(home_med)
        mar_dose  = parse_dose_mg(mar_med)

        if home_dose is not None and mar_dose is not None:
            if abs(home_dose - mar_dose) > 0.01:
                findings["dose_mismatches"].append({
                    "medication":  home_med["name"],
                    "home_dose":   home_med.get("dose_raw"),
                    "mar_dose":    mar_med.get("dose_raw"),
                    "flag": f"Dose changed: {home_dose}mg (home) → {mar_dose}mg (MAR)",
                })

        home_freq = (home_med.get("frequency") or "").strip().lower()
        mar_freq  = (mar_med.get("frequency") or "").strip().lower()
        if home_freq and mar_freq and home_freq != mar_freq:
            findings["frequency_mismatches"].append({
                "medication":       home_med["name"],
                "home_frequency":   home_med.get("frequency"),
                "mar_frequency":    mar_med.get("frequency"),
                "flag": f"Frequency changed: '{home_med.get('frequency')}' (home) → '{mar_med.get('frequency')}' (MAR)",
            })

    # ── Omissions: home meds not on MAR ──────────────────────────────────────
    for cname, home_med in home_lookup.items():
        if cname not in mar_lookup:
            findings["omissions"].append({
                "medication": home_med["name"],
                "home_dose":  home_med.get("dose_raw"),
                "flag": f"'{home_med['name']}' is on the home list but ABSENT from the MAR.",
            })

    # ── New hospital medications: on MAR but not at home ─────────────────────
    for cname, mar_med in mar_lookup.items():
        if cname not in home_lookup:
            findings["new_hospital_meds"].append({
                "medication": mar_med["name"],
                "mar_dose":   mar_med.get("dose_raw"),
                "frequency":  mar_med.get("frequency"),
                "flag": f"'{mar_med['name']}' is a new hospital order (not on home list).",
            })

    # ── Therapeutic duplications: same drug class appearing twice on MAR ──────
    mar_classes: dict[str, list[str]] = {}
    for cname, mar_med in mar_lookup.items():
        drug_class = DRUG_CLASSES.get(cname)
        if drug_class:
            mar_classes.setdefault(drug_class, []).append(mar_med["name"])

    for drug_class, meds_in_class in mar_classes.items():
        if len(meds_in_class) > 1:
            findings["therapeutic_duplications"].append({
                "drug_class": drug_class,
                "medications": meds_in_class,
                "flag": (
                    f"THERAPEUTIC DUPLICATION: {' and '.join(meds_in_class)} "
                    f"are both {drug_class.replace('_', ' ')}s ordered concurrently."
                ),
            })

    # Summary counts
    total_flags = (
        (1 if findings["allergy_conflict"] else 0)
        + len(findings["dose_mismatches"])
        + len(findings["frequency_mismatches"])
        + len(findings["omissions"])
        + len(findings["new_hospital_meds"])
        + len(findings["therapeutic_duplications"])
    )
    print(f"  ✓ Comparison complete — {total_flags} issue(s) flagged")
    return findings


# ─────────────────────────────────────────────
# STAGE D — Clinical Interpretation (LLM)
# Only receives the flagged discrepancies, not raw docs
# ─────────────────────────────────────────────

STAGE_D_PROMPT_TEMPLATE = """You are an expert clinical AI assistant specializing in
Medication Reconciliation and Patient Safety.

A deterministic algorithm has already compared the two documents and identified the
following structured discrepancies. Your job is to interpret these findings clinically
and produce a final reconciliation report in JSON format.

IDENTIFIED DISCREPANCIES (Python-generated, not inferred):
{discrepancies_json}

IMPORTANT RULES:
- Output ONLY valid JSON. No prose, no markdown, no explanation outside the JSON.
- Base your clinical commentary ONLY on the discrepancies provided above.
- Do NOT reference medications or findings not listed above.
- Produce a JSON object with these top-level keys:
    "critical_safety_alerts":       array of string descriptions
    "therapeutic_duplications":     array of string descriptions
    "dose_or_frequency_mismatches": array of string descriptions
    "omissions":                    array of string descriptions
    "new_hospital_medications":     array of string descriptions
    "overall_risk_level":           one of "LOW", "MODERATE", "HIGH", "CRITICAL"
    "recommended_actions":          array of string action items for the clinical team

Respond with ONLY the JSON object.
"""

def stage_d_clinical_interpretation(model, findings: dict) -> dict | None:
    """
    Stage D: Send only the flagged discrepancies to Gemini for clinical commentary.
    The model cannot hallucinate medications it was never told about.
    """
    print("\n── Stage D: Clinical interpretation ──")

    discrepancies_json = json.dumps(findings, indent=2)
    prompt = STAGE_D_PROMPT_TEMPLATE.format(discrepancies_json=discrepancies_json)

    response = model.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    result = json.loads(response.text)

    if result is not None:
        print(f"  ✓ Clinical report generated (risk level: {result.get('overall_risk_level', 'N/A')})")
    return result


# ─────────────────────────────────────────────
# MAIN — Wire all stages together
# ─────────────────────────────────────────────

def run_reconciliation(
    model,
    preadmission_text: str,
    mar_text: str,
    preadmission_allergies: list[str],
    output_path: str = "reconciliation_report.json",
) -> dict:
    """
    Run the full 4-stage reconciliation pipeline and save a JSON report.

    Args:
        model:                  Loaded Gemini GenerativeModel
        preadmission_text:      Raw text of the preadmission/home medication form
        mar_text:               Raw text of the hospital MAR
        preadmission_allergies: List of allergy strings from preadmission form
        output_path:            Where to save the final JSON report

    Returns:
        The final report as a dict
    """
    print("\n" + "=" * 55)
    print("  MEDICATION RECONCILIATION PIPELINE")
    print("=" * 55)

    # Stage A — extract home meds from text
    home_meds = stage_a_extract_home_meds(model, preadmission_text)
    if not home_meds:
        print("[ERROR] Stage A failed — could not extract home medications.")
        return {}

    # Stage B — extract MAR meds from text
    mar_data = stage_b_extract_mar(model, mar_text)
    if not mar_data:
        print("[ERROR] Stage B failed — could not extract MAR medications.")
        return {}

    # Stage C — deterministic Python comparison (no LLM)
    findings = stage_c_compare(home_meds, preadmission_allergies, mar_data)

    # Stage D — LLM interprets only the flagged findings
    clinical_report = stage_d_clinical_interpretation(model, findings)

    # Assemble final report
    final_report = {
        "pipeline_stages": {
            "stage_a_home_meds":    home_meds,
            "stage_b_mar_data":     mar_data,
            "stage_c_findings":     findings,
            "stage_d_clinical":     clinical_report,
        }
    }

    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n✓ Full report saved to: {output_path}")
    print("=" * 55)

    return final_report


# ─────────────────────────────────────────────
# EXAMPLE — Test 2 from your notebook
# ─────────────────────────────────────────────

PREADMISSION_TEXT1 = """
Patient Name: Pendelton, Arthur
DOB: 11/14/1951 | Age: 74 | Gender: M
Admission Date: 05/20/2026
Reason for Admission: Acute Decompensated Heart Failure exacerbation

KNOWN ALLERGIES:
- Sulfa Drugs (Reaction: Anaphylaxis / Severe Swelling)

Home Medications (Confirmed by Patient & Outpatient Pharmacy):
1. Lasix (Furosemide) | 40 mg | PO | Once daily in the morning | Heart Failure
2. Coreg (Carvedilol) | 12.5 mg | PO | Twice daily (BID) | Heart Failure
3. Metformin | 500 mg | PO | Twice daily with meals | Type 2 Diabetes
4. Atorvastatin | 20 mg | PO | Once daily at bedtime | Hyperlipidemia
5. Aspirin | 81 mg | PO | Once daily | Cardioprotection
"""

PREADMISSION_ALLERGIES1 = ["Sulfa Drugs (Reaction: Anaphylaxis / Severe Swelling)"]

PREADMISSION_ALLERGIES2 = ["Penicillin (Reaction: Hives/Rash)"]

PREADMISSION_TEXT2 = """
            Facility: Gulf Coast State College Clinic
            Date of Assessment: 1/18/2016 (Prior to Hospital Admission)
            Patient Name: Davis, Celia  DOB: 5/25/1955  Age: 61 | 
            Gender: FWeight: 175 lbs
            ALLERGIES: Penicillin (Reaction: Hives/Rash)
            Home Medications (Reported by Patient/Pharmacy):
            - Lipitor | 40 mg | PO | Once daily at bedtime | High Cholesterol
            - Lisinopril | 10 mg | PO | Once daily in the morning | Hypertension
            - Xanax | 0.5 mg | PO | Twice daily (BID) | Anxiety
            - Ibuprofen | 400 mg | PO | Every 6 hours PRN | Joint Pain / Knee
            - Centrum Silver | 1 tablet | PO | Once daily | Supplement
"""

MAR_TEXT = """
Patient Name: Pendelton, Arthur
Room: 412-B | Attending: Dr. Vance
Allergies Listed on MAR: No Known Drug Allergies (NKDA)

Active Orders & Administration Grid:
1. Furosemide 40 mg | PO | Twice Daily (BID)
   - Scheduled Times: 0800, 1600
2. Carvedilol 6.25 mg | PO | Twice Daily (BID)
   - Scheduled Times: 0800, 2000
3. Metoprolol Succinate 50 mg | PO | Once Daily
   - Scheduled Time: 0800
4. Atorvastatin 20 mg | PO | Once Daily
   - Scheduled Time: 2100
5. Bactrim DS (Sulfamethoxazole-Trimethoprim) 1 tablet | PO | Once Daily
   - Scheduled Time: 0900
"""


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Set it before running:")
        print("  Windows:   set GEMINI_API_KEY=your_key_here")
        print("  Linux/Mac: export GEMINI_API_KEY=your_key_here")
        raise SystemExit(1)

    model = load_model()

    # ── Load MAR image ──────────────────────────────────────────────────────
    mar_image = Image.open("exampleMAR.png")

    # ── Stage A: extract home meds from preadmission text ───────────────────
    home_meds = stage_a_extract_home_meds(model, PREADMISSION_TEXT2)
    if not home_meds:
        print("[ERROR] Stage A failed — could not extract home medications.")
        raise SystemExit(1)

    # ── Stage B (image): extract MAR meds from image ─────────────────────────
    mar_data = stage_b_extract_mar_from_image(model, mar_image)
    if not mar_data:
        print("[ERROR] Stage B failed — could not extract MAR medications.")
        raise SystemExit(1)

    # ── Stage C: deterministic comparison ────────────────────────────────────
    findings = stage_c_compare(home_meds, PREADMISSION_ALLERGIES2, mar_data)

    # ── Stage D: clinical interpretation ─────────────────────────────────────
    clinical_report = stage_d_clinical_interpretation(model, findings)

    # ── Assemble and save final report ───────────────────────────────────────
    final_report = {
        "pipeline_stages": {
            "stage_a_home_meds": home_meds,
            "stage_b_mar_data":  mar_data,
            "stage_c_findings":  findings,
            "stage_d_clinical":  clinical_report,
        }
    }
    with open("reconciliation_report1.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print("\n✓ Full report saved to: reconciliation_report.json")

    print("\n── FINAL CLINICAL REPORT ──")
    print(json.dumps(clinical_report, indent=2))