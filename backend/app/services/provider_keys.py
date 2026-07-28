"""
Multi-Provider API Key Store & Failover (text models)
=====================================================

Beyond Gemini (which powers vision — pill scan + leaflet — and text), the app
lets users add keys for extra **free, strong** text providers that serve as
failover for the drug assistant when Gemini quota is exhausted or a key fails:

    • Mistral     (mistral-large-latest)
    • Groq        (llama-3.3-70b-versatile)
    • OpenRouter  (meta-llama/llama-3.3-70b-instruct:free)

All three expose an **OpenAI-compatible** ``/chat/completions`` endpoint, so a
single client handles them. Each provider gets up to ``SLOTS_PER_PROVIDER`` key
slots, stored **encrypted** in the ``users.provider_api_keys`` JSON column as
``{provider: [enc_key, ...]}``. Admin-stored keys are shared with every user,
exactly like Gemini.
"""

from typing import Any, Dict, List, Optional

import httpx

from app.config import get_settings
from app.utils.crypto import decrypt_secret, encrypt_secret

settings = get_settings()

# How many key slots each extra provider exposes on the AI-settings page.
SLOTS_PER_PROVIDER = 10

# Provider registry. ``model`` is a sensible free/strong default; ``base_url``
# is the OpenAI-compatible chat-completions endpoint.
PROVIDERS: Dict[str, Dict[str, str]] = {
    "mistral": {
        "label": "Mistral",
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-large-latest",
        "env_prefix": "MISTRAL_API_KEY",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "env_prefix": "GROQ_API_KEY",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "env_prefix": "OPENROUTER_API_KEY",
    },
}


def ordered_providers() -> List[str]:
    """Provider keys in the order they should be tried as failover."""
    return list(PROVIDERS.keys())


def is_valid_provider(provider: str) -> bool:
    return provider in PROVIDERS


def provider_label(provider: str) -> str:
    return PROVIDERS.get(provider, {}).get("label", provider)


def provider_model(provider: str) -> str:
    return PROVIDERS.get(provider, {}).get("model", "")


# ── Storage (users.provider_api_keys JSON column) ──────────────────────────

def _raw_store(user: Optional[Any]) -> Dict[str, list]:
    """The user's raw {provider: [encrypted_key, ...]} map (never None)."""
    if user is None:
        return {}
    store = getattr(user, "provider_api_keys", None)
    return dict(store) if isinstance(store, dict) else {}


def user_provider_keys(user: Optional[Any], provider: str) -> List[str]:
    """Decrypted, non-empty keys the user stored for ``provider``, in order."""
    if not is_valid_provider(provider):
        return []
    encrypted = _raw_store(user).get(provider) or []
    keys: List[str] = []
    for enc in encrypted:
        val = decrypt_secret(enc)
        if val and val.strip():
            keys.append(val.strip())
    return keys


def user_provider_slots(user: Optional[Any], provider: str) -> List[Optional[str]]:
    """
    The stored (still-encrypted) value for each of the provider's slots, padded
    to ``SLOTS_PER_PROVIDER`` with None — used to render slot status.
    """
    encrypted = list(_raw_store(user).get(provider) or [])
    encrypted = encrypted[:SLOTS_PER_PROVIDER]
    encrypted += [None] * (SLOTS_PER_PROVIDER - len(encrypted))
    return encrypted


def set_user_provider_slot(user: Any, provider: str, slot: int, plaintext: Optional[str]) -> None:
    """
    Set (or clear, when ``plaintext`` is empty) one 1-based slot for a provider.
    Reassigns the JSON column so SQLAlchemy detects the change.
    """
    if not is_valid_provider(provider) or not (1 <= slot <= SLOTS_PER_PROVIDER):
        return
    store = _raw_store(user)
    slots = list(store.get(provider) or [])
    slots = slots[:SLOTS_PER_PROVIDER]
    slots += [None] * (SLOTS_PER_PROVIDER - len(slots))
    slots[slot - 1] = encrypt_secret((plaintext or "").strip())
    # Drop trailing Nones so the stored list stays compact.
    while slots and slots[-1] is None:
        slots.pop()
    store = dict(store)
    if slots:
        store[provider] = slots
    else:
        store.pop(provider, None)
    user.provider_api_keys = store


def _env_keys(provider: str) -> List[str]:
    """Server-wide keys for a provider from the environment (PREFIX, PREFIX_2..)."""
    prefix = PROVIDERS[provider]["env_prefix"]
    keys: List[str] = []
    for slot in range(1, SLOTS_PER_PROVIDER + 1):
        attr = prefix if slot == 1 else f"{prefix}_{slot}"
        val = getattr(settings, attr, None)
        if val and str(val).strip():
            keys.append(str(val).strip())
    return keys


def _dedupe(keys: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


async def _admin_shared_keys(provider: str, db: Any) -> List[str]:
    """Keys stored by ADMIN accounts for this provider — shared with all users."""
    if db is None:
        return []
    from sqlalchemy import select, or_
    from app.models.user import User

    result = await db.execute(
        select(User).where(or_(User.role == "ADMIN", User.is_admin == True))  # noqa: E712
    )
    keys: List[str] = []
    for admin in result.scalars().all():
        keys.extend(user_provider_keys(admin, provider))
    return keys


async def resolve_keys_async(provider: str, user: Optional[Any], db: Any) -> List[str]:
    """Ordered, de-duplicated keys for a provider: user → admin-shared → env."""
    if not is_valid_provider(provider):
        return []
    ordered = (
        user_provider_keys(user, provider)
        + await _admin_shared_keys(provider, db)
        + _env_keys(provider)
    )
    return _dedupe(ordered)


# ── OpenAI-compatible chat call ────────────────────────────────────────────

async def openai_compatible_generate(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Call a provider's OpenAI-compatible chat-completions endpoint and return the
    assistant message text. Raises on any HTTP/network error so the caller can
    fail over to the next key/provider.
    """
    cfg = PROVIDERS[provider]
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    if provider == "openrouter":
        # OpenRouter asks callers to identify themselves (optional but polite).
        headers["HTTP-Referer"] = "https://pillscan-web.onrender.com"
        headers["X-Title"] = "PillScan"

    timeout = settings.LLM_TIMEOUT_SECONDS
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(cfg["base_url"], json=payload, headers=headers)

    if resp.status_code != 200:
        raise RuntimeError(f"{provider} HTTP {resp.status_code}: {(resp.text or '')[:200]}")

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"{provider}: empty choices")
    content = (choices[0].get("message") or {}).get("content") or ""
    if not content.strip():
        raise RuntimeError(f"{provider}: empty content")
    return content
