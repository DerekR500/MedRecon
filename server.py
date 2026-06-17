"""
server.py
=========
FastAPI backend for the Medication Reconciliation Pipeline.

Usage:
  uvicorn server:app --reload --port 8000

Requirements (beyond existing project deps):
  pip install "fastapi[standard]" python-multipart pymupdf
"""

import asyncio
import io
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from main import (
    stage_a_extract_from_image,
    stage_b_extract_mar_from_image,
    stage_c_compare,
)

_MAX_IMAGE_WIDTH = 1400

app = FastAPI(title="MedRecon API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── File loading ──────────────────────────────────────────────────────────────

def _pdf_to_pil(data: bytes) -> Image.Image:
    import fitz
    doc    = fitz.open(stream=data, filetype="pdf")
    pages  = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    doc.close()
    if len(pages) == 1:
        return pages[0]
    total_h = sum(p.height for p in pages)
    max_w   = max(p.width  for p in pages)
    canvas  = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for p in pages:
        canvas.paste(p, (0, y))
        y += p.height
    return canvas


async def _upload_to_image(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    name = (upload.filename or "").lower()
    if name.endswith(".pdf"):
        img = await asyncio.to_thread(_pdf_to_pil, data)
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.width > _MAX_IMAGE_WIDTH:
        scale = _MAX_IMAGE_WIDTH / img.width
        img   = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


# ── Reconciliation endpoint ───────────────────────────────────────────────────

@app.post("/reconcile")
async def reconcile(
    avs_file: list[UploadFile] = File(...),
    mar_file: list[UploadFile] = File(...),
    model:    str              = Form(...),
):
    async def stream():
        def evt(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        # Stage A — one API call per AVS page, results merged
        yield evt("log", {"msg": f"── Stage A: Extracting from AVS ({len(avs_file)} page(s)) ──"})
        avs_meds, avs_allergies = [], []
        for i, upload in enumerate(avs_file):
            try:
                img       = await _upload_to_image(upload)
                page_data = await asyncio.to_thread(stage_a_extract_from_image, model, img)
            except Exception as e:
                yield evt("error", {"msg": f"Stage A error (page {i + 1}): {e}"})
                return
            if not page_data:
                yield evt("error", {"msg": f"Stage A failed on page {i + 1} — could not extract from AVS."})
                return
            p_meds = page_data.get("medications") or []
            p_alrg = page_data.get("allergies")   or []
            avs_meds.extend(p_meds)
            avs_allergies.extend(p_alrg)
            yield evt("log", {"msg": f"  ✓ Page {i + 1}: {len(p_meds)} medication(s) | {len(p_alrg)} allergy entry(ies)"})

        home_data = {"medications": avs_meds, "allergies": avs_allergies}
        yield evt("log", {"msg": f"  Total: {len(avs_meds)} medication(s) | {len(avs_allergies)} allergy entry(ies)"})

        # Stage B — one API call per MAR page, results merged
        yield evt("log", {"msg": f"── Stage B: Extracting from MAR ({len(mar_file)} page(s)) ──"})
        mar_meds, mar_allergies = [], []
        for i, upload in enumerate(mar_file):
            try:
                img       = await _upload_to_image(upload)
                page_data = await asyncio.to_thread(stage_b_extract_mar_from_image, model, img)
            except Exception as e:
                yield evt("error", {"msg": f"Stage B error (page {i + 1}): {e}"})
                return
            if not page_data:
                yield evt("error", {"msg": f"Stage B failed on page {i + 1} — could not extract from MAR."})
                return
            p_meds = page_data.get("medications")    or []
            p_alrg = page_data.get("allergies_on_mar") or []
            mar_meds.extend(p_meds)
            mar_allergies.extend(p_alrg)
            yield evt("log", {"msg": f"  ✓ Page {i + 1}: {len(p_meds)} MAR medication(s)"})

        mar_data = {"medications": mar_meds, "allergies_on_mar": mar_allergies}
        yield evt("log", {"msg": f"  Total: {len(mar_meds)} MAR medication(s)"})

        # Stage C
        yield evt("log", {"msg": "── Stage C: LLM comparison ──"})
        try:
            findings = await asyncio.to_thread(stage_c_compare, model, home_data, mar_data)
        except Exception as e:
            yield evt("error", {"msg": f"Stage C error: {e}"})
            return
        if not findings:
            yield evt("error", {"msg": "Stage C failed — could not compare documents."})
            return

        yield evt("log", {"msg": "  ✓ Comparison complete\n\nPipeline complete."})
        yield evt("result", {"findings": findings})

    return StreamingResponse(stream(), media_type="text/event-stream")


# Static frontend — mount last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
