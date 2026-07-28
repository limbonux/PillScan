"""
Pill Identification Service (Vision LLM fallback)
=================================================

Identifies a medication from a photo using the Gemini vision model
and returns structured candidates. Used by the scan
router as a fallback when the local CV model (YOLOv8 + EfficientNet) is
unavailable, disabled, or produces no usable/mappable prediction.

The model is asked to reply in strict JSON so results can be mapped to the
drug database. Returns ``None`` when no provider key is configured (so the
caller can degrade to an honest "not identified" instead of a fake match).
"""

import base64
import json
import re
from typing import Optional

from app.config import get_settings
from app.services import llm_keys
from app.services import provider_keys

settings = get_settings()


class PillIdError(Exception):
    """Raised when the vision LLM call fails or returns an unusable response."""


IDENTIFY_PROMPT = (
    "You are a pharmacist assistant. Look at the photo of a medication "
    "(pill, tablet, capsule, blister strip, or box) and identify the medicine.\n\n"
    "Respond with ONLY a JSON object — no markdown, no explanation — using "
    "exactly this shape:\n"
    "{\n"
    '  "identified": true or false,\n'
    '  "candidates": [\n'
    '    {"name_en": "brand name", "name_ar": "الاسم بالعربية", '
    '"generic_en": "active ingredient", "strength": "e.g. 500mg", '
    '"dosage_form": "e.g. tablet", "confidence": 0.0}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Provide up to 3 candidates, most likely first.\n"
    "- confidence is your certainty from 0 to 1.\n"
    "- If you cannot identify any medicine, return "
    '{"identified": false, "candidates": []}.\n'
    "- Fill name_ar in Arabic (transliterate if there is no common Arabic name).\n"
    "- Use empty strings for fields you cannot determine.\n"
    "- Output valid JSON only."
)


def is_configured(user=None) -> bool:
    """True when any vision-capable key (Gemini or a vision provider) is set."""
    if llm_keys.resolve_gemini_keys(user):
        return True
    return any(
        provider_keys.user_provider_keys(user, provider)
        for provider in provider_keys.vision_providers()
    )


async def _call_gemini(image_b64: str, mime_type: str, api_key: str) -> str:
    """
    Call Gemini for pill identification and return the raw JSON text.

    Model fallback, error classification, timeouts and network errors are
    handled centrally in ``llm_keys.gemini_generate``.
    """
    parts = [
        {"text": IDENTIFY_PROMPT},
        {"inline_data": {"mime_type": mime_type, "data": image_b64}},
    ]
    try:
        return await llm_keys.gemini_generate(parts, api_key, temperature=0.1)
    except llm_keys.GeminiError as e:
        raise PillIdError(str(e))


def _extract_json(text: str) -> dict:
    """Parse the model's reply into a dict, tolerating code fences / extra prose."""
    cleaned = text.strip()
    # Strip ```json ... ``` fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block in the text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise PillIdError(f"Could not parse JSON from model reply: {text[:200]}")


def _normalize_candidates(parsed: dict) -> list[dict]:
    """Clean and clamp the candidate list from the parsed model reply."""
    raw = parsed.get("candidates") or []
    candidates = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name_en = str(item.get("name_en", "") or "").strip()
        name_ar = str(item.get("name_ar", "") or "").strip()
        if not name_en and not name_ar:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        candidates.append({
            "name_en": name_en,
            "name_ar": name_ar,
            "generic_en": str(item.get("generic_en", "") or "").strip(),
            "strength": str(item.get("strength", "") or "").strip(),
            "dosage_form": str(item.get("dosage_form", "") or "").strip(),
            "confidence": confidence,
        })
    return candidates


async def identify_pill(
    image_bytes: bytes,
    content_type: Optional[str],
    user=None,
    db=None,
) -> Optional[dict]:
    """
    Identify a medication from an image using Gemini.

    Tries the user's Gemini keys (settings page) first, then any server env
    keys, moving to the next key automatically when one fails or is exhausted.

    Returns:
        {"provider": "gemini", "model": str, "candidates": [ {name_en, name_ar,
         generic_en, strength, dosage_form, confidence}, ... ]}
        or None if no Gemini key is configured.

    Raises:
        PillIdError when every configured key fails or the reply is unparseable.
    """
    gemini_keys = await llm_keys.resolve_gemini_keys_async(user, db)

    # Vision failover providers (Pixtral / OpenRouter vision) — user, then
    # admin-shared, then env keys.
    vision_extra: list[tuple[str, str]] = []
    for provider in provider_keys.vision_providers():
        for key in await provider_keys.resolve_keys_async(provider, user, db):
            vision_extra.append((provider, key))

    if not gemini_keys and not vision_extra:
        return None

    model = settings.GEMINI_MODEL
    mime_type = content_type if (content_type or "").startswith("image/") else "image/jpeg"
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    last_error: Optional[Exception] = None

    # 1) Gemini vision keys.
    for key in gemini_keys:
        try:
            text = await _call_gemini(image_b64, mime_type, key)
            return {"provider": "gemini", "model": model,
                    "candidates": _normalize_candidates(_extract_json(text))}
        except Exception as e:  # noqa: BLE001 - try the next key/provider
            last_error = e

    # 2) Extra vision providers (OpenAI-compatible).
    for provider, key in vision_extra:
        try:
            text = await provider_keys.openai_compatible_vision(
                provider, IDENTIFY_PROMPT, image_b64, mime_type, key
            )
            return {"provider": provider,
                    "model": provider_keys.provider_vision_model(provider),
                    "candidates": _normalize_candidates(_extract_json(text))}
        except Exception as e:  # noqa: BLE001 - try the next key/provider
            last_error = e

    # Keys existed but all failed.
    raise PillIdError(str(last_error) if last_error else "All vision keys failed.")
