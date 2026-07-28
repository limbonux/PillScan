"""
Tests for the extra text providers (Mistral / Groq / OpenRouter):
key storage/masking endpoints and assistant failover onto them.
"""

import json
import pytest
from httpx import AsyncClient

from app.services import provider_keys, assistant_service


class TestProviderKeyStorage:
    def test_set_and_read_roundtrip(self):
        class U:
            provider_api_keys = None
        u = U()
        provider_keys.set_user_provider_slot(u, "mistral", 1, "SECRET-1")
        provider_keys.set_user_provider_slot(u, "mistral", 3, "SECRET-3")
        keys = provider_keys.user_provider_keys(u, "mistral")
        assert keys == ["SECRET-1", "SECRET-3"]
        # Stored values are encrypted, not plaintext.
        raw = u.provider_api_keys["mistral"]
        assert "SECRET-1" not in json.dumps(raw)

    def test_clear_slot(self):
        class U:
            provider_api_keys = None
        u = U()
        provider_keys.set_user_provider_slot(u, "groq", 1, "K")
        assert provider_keys.user_provider_keys(u, "groq") == ["K"]
        provider_keys.set_user_provider_slot(u, "groq", 1, "")
        assert provider_keys.user_provider_keys(u, "groq") == []
        # Provider key removed entirely when empty.
        assert "groq" not in (u.provider_api_keys or {})

    def test_invalid_provider_ignored(self):
        class U:
            provider_api_keys = None
        u = U()
        provider_keys.set_user_provider_slot(u, "not-a-provider", 1, "K")
        assert (u.provider_api_keys or {}) == {}


class TestProviderKeyEndpoints:
    @pytest.mark.asyncio
    async def test_get_lists_all_providers(self, client: AsyncClient, test_user: dict):
        r = await client.get("/api/v1/users/me/ai-keys", headers=test_user["auth_header"])
        assert r.status_code == 200
        data = r.json()
        assert data["slots_per_provider"] == 10
        provs = {p["provider"] for p in data["providers"]}
        assert provs == {"mistral", "groq", "openrouter"}
        for p in data["providers"]:
            assert len(p["keys"]) == 10
            assert p["configured_count"] == 0

    @pytest.mark.asyncio
    async def test_set_and_mask_key(self, client: AsyncClient, test_user: dict):
        r = await client.put(
            "/api/v1/users/me/ai-keys",
            headers=test_user["auth_header"],
            json={"provider": "mistral", "slot": 2, "key": "sk-mistral-abcd1234"},
        )
        assert r.status_code == 200
        mistral = next(p for p in r.json()["providers"] if p["provider"] == "mistral")
        assert mistral["configured_count"] == 1
        slot2 = next(k for k in mistral["keys"] if k["slot"] == 2)
        assert slot2["configured"] is True
        # Raw key never returned; only a masked hint ending in the last 4 chars.
        assert slot2["hint"].endswith("1234")
        assert "sk-mistral-abcd1234" not in json.dumps(r.json())

    @pytest.mark.asyncio
    async def test_clear_key(self, client: AsyncClient, test_user: dict):
        await client.put(
            "/api/v1/users/me/ai-keys",
            headers=test_user["auth_header"],
            json={"provider": "groq", "slot": 1, "key": "GKEY"},
        )
        r = await client.put(
            "/api/v1/users/me/ai-keys",
            headers=test_user["auth_header"],
            json={"provider": "groq", "slot": 1, "key": ""},
        )
        groq = next(p for p in r.json()["providers"] if p["provider"] == "groq")
        assert groq["configured_count"] == 0

    @pytest.mark.asyncio
    async def test_invalid_provider_rejected(self, client: AsyncClient, test_user: dict):
        r = await client.put(
            "/api/v1/users/me/ai-keys",
            headers=test_user["auth_header"],
            json={"provider": "openai", "slot": 1, "key": "x"},
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_slot_out_of_range_rejected(self, client: AsyncClient, test_user: dict):
        r = await client.put(
            "/api/v1/users/me/ai-keys",
            headers=test_user["auth_header"],
            json={"provider": "mistral", "slot": 11, "key": "x"},
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_keys_require_auth(self, client: AsyncClient):
        assert (await client.get("/api/v1/users/me/ai-keys")).status_code in (401, 403)


class TestAssistantProviderFailover:
    @pytest.mark.asyncio
    async def test_assistant_uses_provider_when_no_gemini(
        self, client: AsyncClient, test_user: dict, monkeypatch
    ):
        """With only a Mistral key (no Gemini), the assistant answers via Mistral."""
        # Store a provider key (no Gemini key configured for this user).
        await client.put(
            "/api/v1/users/me/ai-keys",
            headers=test_user["auth_header"],
            json={"provider": "mistral", "slot": 1, "key": "MISTRAL-KEY"},
        )

        called = {}

        async def fake_openai(provider, system_prompt, user_prompt, api_key, **kwargs):
            called["provider"] = provider
            called["api_key"] = api_key
            return json.dumps({
                "name": "بروفنيد",
                "activeIngredient": "بروميد الإبراتروبيوم",
                "uses": ["توسيع الشعب الهوائية"],
                "sideEffects": ["جفاف الفم"],
                "recognized": True,
            })

        monkeypatch.setattr(provider_keys, "openai_compatible_generate", fake_openai)

        r = await client.post(
            "/api/v1/assistant/drug-info",
            headers=test_user["auth_header"],
            json={"name": "Brofnid"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["recognized"] is True
        assert data["name"] == "بروفنيد"
        assert data["uses"] == ["توسيع الشعب الهوائية"]
        # Proves the Mistral provider path (not Gemini) was taken.
        assert called["provider"] == "mistral"
        assert called["api_key"] == "MISTRAL-KEY"
        assert "mistral" in data["model"]

    @pytest.mark.asyncio
    async def test_assistant_not_configured_without_any_key(
        self, client: AsyncClient, test_user: dict
    ):
        """No Gemini and no provider keys → graceful not-configured response."""
        r = await client.post(
            "/api/v1/assistant/drug-info",
            headers=test_user["auth_header"],
            json={"name": "Panadol"},
        )
        assert r.status_code == 200
        assert r.json()["is_configured"] is False
