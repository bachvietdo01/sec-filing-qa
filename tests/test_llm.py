"""Unit tests for llm.py — the isolated Anthropic wrapper. No real client is
ever constructed: _get_client is monkeypatched and the key is toggled via env."""
from __future__ import annotations

import pytest

import llm


# --------------------------------------------------------------------------- #
# Fakes for the Anthropic client surface used by llm.complete()
# --------------------------------------------------------------------------- #
class _Block:
    def __init__(self, type, text):
        self.type = type
        self.text = text


class _Resp:
    def __init__(self, content):
        self.content = content


class _Messages:
    def __init__(self, resp, captured):
        self._resp = resp
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return self._resp


class _Client:
    def __init__(self, resp, captured):
        self.messages = _Messages(resp, captured)


def _install_client(monkeypatch, blocks):
    captured = {}
    client = _Client(_Resp(blocks), captured)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    return captured


# --------------------------------------------------------------------------- #
# available()
# --------------------------------------------------------------------------- #
def test_available_true_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abc")
    assert llm.available() is True


def test_available_false_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.available() is False


# --------------------------------------------------------------------------- #
# complete()
# --------------------------------------------------------------------------- #
def test_complete_raises_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        llm.complete("sys", "user")


def test_complete_concatenates_text_blocks(monkeypatch):
    _install_client(monkeypatch, [_Block("text", "Hello "), _Block("text", "world")])
    assert llm.complete("sys", "user") == "Hello world"


def test_complete_skips_non_text_blocks(monkeypatch):
    blocks = [_Block("text", "keep"), _Block("tool_use", "drop"),
              _Block("text", " me")]
    _install_client(monkeypatch, blocks)
    assert llm.complete("sys", "user") == "keep me"


def test_complete_strips_surrounding_whitespace(monkeypatch):
    _install_client(monkeypatch, [_Block("text", "  padded  ")])
    assert llm.complete("sys", "user") == "padded"


def test_complete_passes_default_model(monkeypatch):
    import config
    captured = _install_client(monkeypatch, [_Block("text", "ok")])
    llm.complete("sys", "user")
    assert captured["model"] == config.ANTHROPIC_MODEL
    assert captured["system"] == "sys"
    assert captured["messages"] == [{"role": "user", "content": "user"}]
    assert captured["max_tokens"] == 512


def test_complete_model_override(monkeypatch):
    captured = _install_client(monkeypatch, [_Block("text", "ok")])
    llm.complete("sys", "user", max_tokens=99, model="my-model")
    assert captured["model"] == "my-model"
    assert captured["max_tokens"] == 99


# --------------------------------------------------------------------------- #
# complete_json() — robust extraction of one JSON object
# --------------------------------------------------------------------------- #
def test_complete_json_clean_object(monkeypatch):
    _install_client(monkeypatch, [_Block("text", '{"company": "TSLA", "n": 3}')])
    assert llm.complete_json("sys", "user") == {"company": "TSLA", "n": 3}


def test_complete_json_embedded_in_prose(monkeypatch):
    text = 'Sure! Here is the result:\n{"ok": true, "value": 42}\nHope that helps.'
    _install_client(monkeypatch, [_Block("text", text)])
    assert llm.complete_json("sys", "user") == {"ok": True, "value": 42}


def test_complete_json_nested_object(monkeypatch):
    text = 'prefix {"a": {"b": 1}, "c": [1, 2]} suffix'
    _install_client(monkeypatch, [_Block("text", text)])
    assert llm.complete_json("sys", "user") == {"a": {"b": 1}, "c": [1, 2]}


def test_complete_json_raises_when_no_object(monkeypatch):
    _install_client(monkeypatch, [_Block("text", "there is no json here")])
    with pytest.raises(ValueError):
        llm.complete_json("sys", "user")


def test_complete_json_appends_instruction_to_system(monkeypatch):
    captured = _install_client(monkeypatch, [_Block("text", "{}")])
    llm.complete_json("base-system", "user")
    # complete_json augments the system prompt to force JSON-only output
    assert "base-system" in captured["system"]
    assert "JSON" in captured["system"]
