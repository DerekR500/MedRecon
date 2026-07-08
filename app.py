"""
app.py
======
Gradio frontend for the Medication Reconciliation Pipeline.

Usage:
  python app.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pyperclip
import gradio as gr
from PIL import Image

from main import (
    stage_a_extract_from_image,
    stage_b_extract_mar_from_image,
    stage_c_compare,
)

_MAX_IMAGE_WIDTH = 1400  # px — keeps text readable without overwhelming the model

def _combine_images(file_list) -> Image.Image | None:
    """Load one or more uploaded image files and concatenate them vertically.
    Scales down to _MAX_IMAGE_WIDTH if needed so multi-page stacks stay readable."""
    if not file_list:
        return None
    pages = []
    for f in file_list:
        path = f.name if hasattr(f, "name") else f
        pages.append(Image.open(path).convert("RGB"))
    if len(pages) == 1:
        img = pages[0]
    else:
        total_height = sum(p.height for p in pages)
        max_width    = max(p.width  for p in pages)
        combined = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))
        y = 0
        for p in pages:
            combined.paste(p, (0, y))
            y += p.height
        img = combined
    if img.width > _MAX_IMAGE_WIDTH:
        scale = _MAX_IMAGE_WIDTH / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


MODEL_CHOICES = [
    "medgemma-27b-it (UF Navigator)",
    "gemma-3-27b-it (UF Navigator)",
    "mistral-small-3.1 (UF Navigator)",
    "granite-3.3-8b-instruct (UF Navigator)",
]

_STAGE_C_COLUMNS = ["Omissions", "Duplications", "Dosage Discrepancies", "Incorrect Routes"]


def _make_cell(meds: list) -> str:
    if not meds:
        return "0"
    return f"{len(meds)}: {', '.join(meds)}"


def _stage_c_to_table(findings: dict) -> tuple[pd.DataFrame, list]:
    omissions = findings.get("omissions", []) or []
    dups      = findings.get("duplications", []) or []
    dose      = findings.get("dosage_discrepancies", []) or []
    routes    = findings.get("incorrect_routes", []) or []
    row = [_make_cell(omissions), _make_cell(dups), _make_cell(dose), _make_cell(routes)]
    df  = pd.DataFrame([row], columns=_STAGE_C_COLUMNS)
    return df, row


def copy_row(row_data: list) -> str:
    if not row_data:
        return "Nothing to copy — run the pipeline first."
    try:
        pyperclip.copy("\t".join(str(x) for x in row_data))
        return "Copied to clipboard."
    except Exception as e:
        return f"Copy failed: {e}"


def run_pipeline(avs_files, mar_files, model_name):
    """Generator — yields (stage_c_df, row_data, log_str) after each stage."""
    logs: list[str] = []
    empty_df = pd.DataFrame(columns=_STAGE_C_COLUMNS)

    def log(msg: str):
        logs.append(msg)

    avs_image = _combine_images(avs_files)
    mar_image = _combine_images(mar_files)

    if avs_image is None or mar_image is None:
        yield empty_df, [], "Please upload both AVS and MAR image(s) before submitting."
        return

    avs_pages = len(avs_files) if avs_files else 0
    mar_pages = len(mar_files) if mar_files else 0
    log(f"Loaded {avs_pages} AVS page(s), {mar_pages} MAR page(s)")

    # ── Stage A ──────────────────────────────────────────────────────────────
    log("── Stage A: Extracting from AVS image ──")
    yield empty_df, [], "\n".join(logs)

    home_data = stage_a_extract_from_image(model_name, avs_image)
    if not home_data:
        log("[ERROR] Stage A failed — could not extract from AVS image.")
        yield empty_df, [], "\n".join(logs)
        return

    home_meds      = home_data.get("medications") or []
    home_allergies = home_data.get("allergies")   or []
    log(f"  ✓ {len(home_meds)} home medication(s) | {len(home_allergies)} allergy entry(ies)")
    yield empty_df, [], "\n".join(logs)

    # ── Stage B ──────────────────────────────────────────────────────────────
    log("── Stage B: Extracting from MAR image ──")
    yield empty_df, [], "\n".join(logs)

    mar_data = stage_b_extract_mar_from_image(model_name, mar_image)
    if not mar_data:
        log("[ERROR] Stage B failed — could not extract from MAR image.")
        yield empty_df, [], "\n".join(logs)
        return

    mar_meds = mar_data.get("medications") or []
    log(f"  ✓ {len(mar_meds)} MAR medication(s) extracted")
    yield empty_df, [], "\n".join(logs)

    # ── Stage C ──────────────────────────────────────────────────────────────
    log("── Stage C: Deterministic matching (RxNorm/RxClass) ──")
    yield empty_df, [], "\n".join(logs)

    findings = stage_c_compare(home_data, mar_data)
    if not findings:
        log("[ERROR] Stage C failed — could not compare documents.")
        yield empty_df, [], "\n".join(logs)
        return

    stage_c_df, row_data = _stage_c_to_table(findings)
    log("  ✓ Comparison complete")
    log("\nPipeline complete.")
    yield stage_c_df, row_data, "\n".join(logs)


def analyze_avs(avs_files, model_name):
    """Run Stage A only and return the extracted AVS medication data."""
    avs_image = _combine_images(avs_files)
    if avs_image is None:
        return {}, "Please upload AVS image(s) first."
    result = stage_a_extract_from_image(model_name, avs_image)
    if not result:
        return {}, "Stage A failed — could not extract from AVS image."
    meds      = result.get("medications") or []
    allergies = result.get("allergies")   or []
    log = (
        f"AVS Analysis complete\n"
        f"  {len(meds)} medication(s) found\n"
        f"  {len(allergies)} allergy entry(ies) found"
    )
    return result, log


def analyze_mar(mar_files, model_name):
    """Run Stage B only and return the extracted MAR medication data."""
    mar_image = _combine_images(mar_files)
    if mar_image is None:
        return {}, "Please upload MAR image(s) first."
    result = stage_b_extract_mar_from_image(model_name, mar_image)
    if not result:
        return {}, "Stage B failed — could not extract from MAR image."
    meds = result.get("medications") or []
    log = (
        f"MAR Analysis complete\n"
        f"  {len(meds)} medication(s) found"
    )
    return result, log


# ── Gradio layout ─────────────────────────────────────────────────────────────

with gr.Blocks(title="MedRecon — Medication Reconciliation") as demo:
    gr.Markdown("## Medication Reconciliation Pipeline")

    row_state = gr.State([])

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            avs_input = gr.File(
                label="AVS — upload one or more pages",
                file_count="multiple",
                file_types=["image"],
            )
            mar_input = gr.File(
                label="MAR — upload one or more pages",
                file_count="multiple",
                file_types=["image"],
            )
            model_dropdown = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES[0],
                label="Model",
            )
            submit_btn = gr.Button("Run Reconciliation", variant="primary")
            with gr.Row():
                avs_btn = gr.Button("Analyze AVS", variant="secondary")
                mar_btn = gr.Button("Analyze MAR", variant="secondary")

        # ── Right column: outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            stage_c_table = gr.Dataframe(
                headers=_STAGE_C_COLUMNS,
                label="Reconciliation Results",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                copy_btn    = gr.Button("Copy Row", variant="secondary", scale=1)
                copy_status = gr.Textbox(label="", show_label=False, interactive=False, scale=2)
            output_json = gr.JSON(label="Extraction Result")
            log_box = gr.Textbox(
                label="Pipeline Log",
                lines=12,
                interactive=False,
                placeholder="Pipeline output will appear here...",
            )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[avs_input, mar_input, model_dropdown],
        outputs=[stage_c_table, row_state, log_box],
    )
    copy_btn.click(
        fn=copy_row,
        inputs=[row_state],
        outputs=[copy_status],
    )
    avs_btn.click(
        fn=analyze_avs,
        inputs=[avs_input, model_dropdown],
        outputs=[output_json, log_box],
    )
    mar_btn.click(
        fn=analyze_mar,
        inputs=[mar_input, model_dropdown],
        outputs=[output_json, log_box],
    )

demo.queue()
demo.launch()
