"""
llm_provider.py
===============
Shared LLM caller with automatic provider and key fallback.

Priority order: cerebras → groq → gemini → ollama

Multiple keys per provider are supported via numbered env vars:
    CEREBRAS_API_KEY_1=...
    CEREBRAS_API_KEY_2=...
    CEREBRAS_API_KEY_3=...
    GEMINI_API_KEY_1=...
    ...

When one key hits its daily quota, the next key for the same provider is
tried before moving on to the next provider entirely.

Fallback logic:
  - Per-minute rate limit  → retry up to 3× with backoff, same key
  - Daily quota exhausted  → try next key for same provider, then next provider
  - All keys/providers done → raise RuntimeError

Usage:
    from src.llm_provider import call_llm, available_providers

    text = call_llm(system_prompt, user_message)   # auto-detects all keys
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Key collection
# ---------------------------------------------------------------------------

def _get_keys(env_prefix: str) -> list[str]:
    """Collect all keys for a provider from env vars.

    Reads PROVIDER_KEY, PROVIDER_KEY_1, PROVIDER_KEY_2, ... in that order.
    Deduplicates while preserving order.
    """
    keys: list[str] = []
    base = os.environ.get(env_prefix, "").strip()
    if base:
        keys.append(base)
    for i in range(1, 20):
        k = os.environ.get(f"{env_prefix}_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def available_providers() -> list[str]:
    """Return providers that have at least one key configured, in priority order."""
    result: list[str] = []
    for provider, env_prefix in [
        ("cerebras",  "CEREBRAS_API_KEY"),
        ("sambanova", "SAMBANOVA_API_KEY"),
        ("groq",      "GROQ_API_KEY"),
        ("gemini",    "GEMINI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("openai",    "OPENAI_API_KEY"),
    ]:
        if _get_keys(env_prefix):
            result.append(provider)

    try:
        import urllib.request
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        result.append("ollama")
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Per-provider callers (accept explicit api_key)
# ---------------------------------------------------------------------------

def _call_one(system_prompt: str, user_message: str, provider: str,
              api_key: str) -> str:
    """Single attempt at one provider with a specific key."""

    if provider == "cerebras":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
        resp = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_completion_tokens=512,
        )
        return resp.choices[0].message.content

    elif provider == "sambanova":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.sambanova.ai/v1")
        resp = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content

    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=512,
            ),
        )
        return resp.text

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content

    elif provider == "ollama":
        import urllib.request
        model   = os.environ.get("OLLAMA_MODEL", "llama3.2")
        host    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{host}/api/chat", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return data["message"]["content"]

    else:
        raise ValueError(f"Unknown provider '{provider}'.")


# ---------------------------------------------------------------------------
# Quota detection and retry
# ---------------------------------------------------------------------------

def _is_quota_exhausted(exc: Exception, provider: str) -> bool:
    msg = str(exc).lower()
    if provider == "groq":
        return "tokens per day" in msg or "tpd" in msg
    return "quota" in msg or "exceeded your" in msg or "insufficient_quota" in msg


def _call_provider(system_prompt: str, user_message: str,
                   provider: str, keys: list[str]) -> str:
    """Try all keys for one provider, with per-minute retry per key.

    Raises when all keys for this provider are exhausted or fail.
    """
    last_exc: Exception | None = None

    for ki, key in enumerate(keys, 1):
        key_label = f"key {ki}/{len(keys)}" if len(keys) > 1 else "key"
        for attempt in range(3):
            try:
                return _call_one(system_prompt, user_message, provider, key)
            except Exception as exc:
                if _is_quota_exhausted(exc, provider):
                    last_exc = exc
                    print(
                        f"[WARN] {provider} {key_label} quota exhausted, "
                        f"{'trying next key' if ki < len(keys) else 'no more keys'}.",
                        file=sys.stderr,
                    )
                    break  # move to next key
                msg = str(exc)
                is_rate_limit = "429" in msg or "rate limit" in msg.lower()
                if is_rate_limit and attempt < 2:
                    wait = 60 * (attempt + 1)
                    print(
                        f"[WARN] {provider} {key_label}: per-minute limit, "
                        f"waiting {wait}s ...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                else:
                    last_exc = exc
                    break  # move to next key

    raise last_exc or RuntimeError(f"{provider}: all keys exhausted.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ENV_PREFIXES: dict[str, str] = {
    "cerebras":  "CEREBRAS_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "groq":      "GROQ_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "ollama":    "",
}


def call_llm(
    system_prompt: str,
    user_message: str,
    providers: list[str] | None = None,
) -> str:
    """Call the LLM with automatic key and provider fallback.

    Tries each key for each provider before moving to the next provider.
    Raises RuntimeError if all providers and keys are exhausted.
    """
    if providers is None:
        providers = available_providers()

    if not providers:
        raise RuntimeError(
            "No LLM providers available. Set CEREBRAS_API_KEY (or _1/_2/_3), "
            "GROQ_API_KEY, GEMINI_API_KEY, or run Ollama locally."
        )

    last_exc: Exception | None = None

    for i, provider in enumerate(providers):
        keys = _get_keys(_ENV_PREFIXES.get(provider, "")) or [""]  # ollama needs no key
        n_keys = len([k for k in keys if k])
        label  = f"{provider} ({n_keys} key{'s' if n_keys > 1 else ''})"

        try:
            return _call_provider(system_prompt, user_message, provider, keys)
        except Exception as exc:
            last_exc = exc
            remaining = providers[i + 1:]
            if remaining:
                print(
                    f"[WARN] {label} exhausted. Switching to {remaining[0]} ...",
                    file=sys.stderr,
                )
            else:
                print(f"[ERROR] {label} failed and no providers remain.", file=sys.stderr)

    raise RuntimeError(f"All LLM providers exhausted. Last error: {last_exc}")
