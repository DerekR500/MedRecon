"""
model_adapter.py
================
Unified LLM call interface for the Medication Reconciliation Pipeline.

Supported backend:
  - UF Navigator models via OpenAI-compatible API (https://api.ai.it.ufl.edu/v1)
"""

import io
import os
import base64
import threading
from dotenv import load_dotenv

load_dotenv()

_UF_BASE_URL = "https://api.ai.it.ufl.edu/v1"

# Maps dropdown display names → internal API model IDs
_DISPLAY_TO_ID: dict[str, str] = {
    "medgemma-27b-it (UF Navigator)":        "medgemma-27b-it",
    "gemma-3-27b-it (UF Navigator)":         "gemma-3-27b-it",
    "mistral-small-3.1 (UF Navigator)":      "mistral-small-3.1",
    "granite-3.3-8b-instruct (UF Navigator)":"granite-3.3-8b-instruct",
}


class EmptyModelResponseError(RuntimeError):
    """Raised when the UF Navigator API returns a response with no content
    (empty completion, truncated stream, or malformed body). Retryable."""


def call_model(prompt: str, model_name: str, image=None) -> str:
    """
    Call the selected model and return the raw text response.

    Args:
        prompt:     Full prompt string to send.
        model_name: Display name (from dropdown) or raw model ID string.
        image:      Optional PIL Image for vision/multimodal calls.

    Returns:
        Raw response string from the model. Guaranteed non-None.
    """
    model_id = _DISPLAY_TO_ID.get(model_name, model_name)
    return _call_uf_navigator(prompt, model_id, image)


# ── Backend: UF Navigator (OpenAI-compatible REST API) ───────────────────────

# Single shared client for the whole process. _call_uf_navigator runs up to
# 8x concurrently (one thread per AVS/MAR page); constructing a new OpenAI
# client per call leaked an HTTP connection pool each time, which is never
# closed in a long-lived server process. The OpenAI client is thread-safe,
# so one instance serves all threads.
_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from openai import OpenAI

                api_key = os.getenv("KEY_NAME")
                if not api_key:
                    raise ValueError(
                        "KEY_NAME is not set. Add it to your .env file."
                    )
                _client = OpenAI(base_url=_UF_BASE_URL, api_key=api_key)
    return _client


def _extract_content(response) -> str:
    """Pull the message text out of a chat completion, raising a clear error
    instead of letting a None slip out and crash downstream with
    "object of type 'NoneType' has no len()"."""
    if not getattr(response, "choices", None):
        raise EmptyModelResponseError(
            "UF Navigator returned a response with no choices (malformed body)."
        )
    content = response.choices[0].message.content
    if content is None or not content.strip():
        raise EmptyModelResponseError(
            "UF Navigator returned an empty completion for this page."
        )
    return content


def _call_uf_navigator(prompt: str, model_id: str, image=None) -> str:
    import time
    from openai import APIConnectionError, APITimeoutError

    client = _get_client()

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

    # Retry once on transient failures -- connection/timeout errors AND empty
    # completions -- so a single blip on one page doesn't kill a whole
    # multi-page reconciliation run.
    # Auth/invalid-request errors are NOT retried -- they would fail identically
    # again. If the retry also fails, the exception propagates up so server.py's
    # evt("error", ...) handling still sees it.
    max_attempts = 2
    backoff_seconds = 2

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
            )
            return _extract_content(response)
        except (APIConnectionError, APITimeoutError, EmptyModelResponseError) as e:
            if attempt == max_attempts:
                raise
            print(f"[WARN] UF Navigator call failed ({e}), retrying once in {backoff_seconds}s...")
            time.sleep(backoff_seconds)


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
