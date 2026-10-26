"""Isolated Anthropic wrapper — the ONLY module that talks to the LLM.

Keeping the provider in one file is deliberate: the LLM is confined to language
tasks (understand a question, phrase an answer over numbers that code already
found and verified). Swapping models or providers is a one-line change here, and
nothing downstream trusts the model with a number.

The model is the cheap one (config.ANTHROPIC_MODEL = claude-haiku-4-5): the
architecture offloads all hard numeric work to code, so a light model suffices.
These calls are short, so we skip streaming / extended thinking.
"""
from __future__ import annotations

import json
import os
import re

import config

try:  # load a local .env if present (never commit the key)
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_client = None


import threading

_local = threading.local()


def force_offline(flag: bool) -> None:
    """Per-thread switch (one Streamlit session = one thread): when a spending
    cap is hit, that session drops to the deterministic voice — same numbers,
    template wording, no API calls — instead of being blocked."""
    _local.offline = bool(flag)


def available() -> bool:
    """True if an API key is configured. The deterministic spine runs without
    one; only the language steps need it."""
    if getattr(_local, "offline", False):
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        # accept-encoding: identity -> request uncompressed responses. The SDK's
        # bundled HTTP stack otherwise tries to brotli/zstd-decompress the reply
        # and, with some env package versions, hits a decoder API mismatch
        # ("process() takes no keyword arguments"). Uncompressed skips that path.
        _client = anthropic.Anthropic(  # reads ANTHROPIC_API_KEY
            default_headers={"accept-encoding": "identity"})
    return _client


def complete(system: str, user: str, max_tokens: int = 512,
             model: str | None = None) -> str:
    """Single-shot completion, returns the text. Raises if no key."""
    if not available():
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    resp = _get_client().messages.create(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def complete_json(system: str, user: str, max_tokens: int = 512,
                  model: str | None = None) -> dict:
    """Completion constrained to return one JSON object. Robust to stray prose
    around the object."""
    text = complete(system + "\nReturn ONLY one JSON object, no prose.",
                    user, max_tokens, model)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in model output: {text!r}")
    return json.loads(m.group(0))
