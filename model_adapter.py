"""
model_adapter.py
================
Unified LLM call interface for the Medication Reconciliation Pipeline.

Supported backends:
  - MedGemma 1.5 via local Ollama  (http://localhost:11434)
  - UF Navigator models via OpenAI-compatible API (https://api.ai.it.ufl.edu/v1)
"""

import io
import os
import base64
from dotenv import load_dotenv

load_dotenv()

LOCAL_MODEL_ID = "medgemma1.5:latest"
_UF_BASE_URL = "https://api.ai.it.ufl.edu/v1"

# Maps Gradio dropdown display names → internal API model IDs
_DISPLAY_TO_ID: dict[str, str] = {
    "MedGemma (Local)":                      LOCAL_MODEL_ID,
    "medgemma-27b-it (UF Navigator)":        "medgemma-27b-it",
    "gemma-3-27b-it (UF Navigator)":         "gemma-3-27b-it",
    "mistral-small-3.1 (UF Navigator)":      "mistral-small-3.1",
    "granite-3.3-8b-instruct (UF Navigator)":"granite-3.3-8b-instruct",
}


def call_model(prompt: str, model_name: str, image=None) -> str:
    """
    Call the selected model and return the raw text response.

    Args:
        prompt:     Full prompt string to send.
        model_name: Display name (from dropdown) or raw model ID string.
        image:      Optional PIL Image for vision/multimodal calls.

    Returns:
        Raw response string from the model.
    """
    model_id = _DISPLAY_TO_ID.get(model_name, model_name)

    if model_id == LOCAL_MODEL_ID:
        return _call_ollama(prompt, model_id, image)
    else:
        return _call_uf_navigator(prompt, model_id, image)


# ── Backend: local Ollama ────────────────────────────────────────────────────

def _call_ollama(prompt: str, model_id: str, image=None) -> str:
    import ollama as _ollama
    kwargs: dict = {"model": model_id, "prompt": prompt, "format": "json"}
    if image is not None:
        kwargs["images"] = [_to_png_bytes(image)]
    response = _ollama.generate(**kwargs)
    return response["response"]


# ── Backend: UF Navigator (OpenAI-compatible REST API) ───────────────────────

def _call_uf_navigator(prompt: str, model_id: str, image=None) -> str:
    from openai import OpenAI

    api_key = os.getenv("KEY_NAME")
    if not api_key:
        raise ValueError(
            "KEY_NAME is not set. Add it to your .env file."
        )

    client = OpenAI(base_url=_UF_BASE_URL, api_key=api_key)

    if image is not None:
        b64 = base64.b64encode(_to_png_bytes(image)).decode("utf-8")
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
    else:
        content = prompt

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content


# ── Helper ───────────────────────────────────────────────────────────────────

def _to_png_bytes(image) -> bytes:
    """Convert a PIL Image (or raw bytes-like object) to PNG bytes."""
    from PIL import Image as PILImage

    buf = io.BytesIO()
    if isinstance(image, PILImage.Image):
        image.save(buf, format="PNG")
    else:
        buf.write(bytes(image))
    return buf.getvalue()
