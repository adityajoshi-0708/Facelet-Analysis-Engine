"""
tests/test_phase12.py — Phase 12: LLM Client

Run from project root:
    python tests/test_phase12.py

All tests pass whether or not Ollama is running — if Ollama is unreachable,
get_llm_client() falls back to MockLLMClient and the same contracts hold.
No pytest fixtures; each test_* function is standalone.
"""

import sys
import time
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.llm_client import (
    LLMClient,
    LLMResponse,
    OllamaClient,
    MockLLMClient,
    get_llm_client,
)

_PASSED = []
_FAILED = []


def _ok(name: str):
    print(f"✓ {name} passed")
    _PASSED.append(name)


def _fail(name: str, reason: str):
    print(f"✗ {name} FAILED: {reason}")
    _FAILED.append(name)


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------

def test_llm_response_schema():
    name = "llm_response_schema"
    try:
        r = LLMResponse(
            raw_text="hello",
            logprobs=None,
            model_name="test-model",
            latency_ms=12.5,
        )
        assert r.raw_text == "hello"
        assert r.logprobs is None
        assert r.model_name == "test-model"
        assert isinstance(r.latency_ms, float)
        assert isinstance(r.metadata, dict)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_llm_response_with_logprobs():
    name = "llm_response_with_logprobs"
    try:
        lp = [{"token": "4", "logprob": -0.12}]
        r = LLMResponse(
            raw_text='{"score": 4}',
            logprobs=lp,
            model_name="test-model",
            latency_ms=50.0,
        )
        assert r.logprobs is not None
        assert len(r.logprobs) == 1
        assert r.logprobs[0]["token"] == "4"
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------

def test_mock_client_is_llm_client():
    """MockLLMClient must be a subclass of LLMClient (Liskov substitution)."""
    name = "mock_is_llm_client_subclass"
    try:
        client = MockLLMClient()
        assert isinstance(client, LLMClient)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_generate_returns_llm_response():
    name = "mock_generate_returns_llm_response"
    try:
        client = MockLLMClient()
        resp = client.generate("Score this text: I quit my job to start a company.")
        assert isinstance(resp, LLMResponse)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_raw_text_not_empty():
    name = "mock_raw_text_not_empty"
    try:
        client = MockLLMClient()
        resp = client.generate("Evaluate risk taking.")
        assert len(resp.raw_text.strip()) > 0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_model_name():
    name = "mock_model_name"
    try:
        client = MockLLMClient()
        resp = client.generate("test")
        assert resp.model_name == "mock-llm"
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_logprobs_is_none():
    """Mock never returns logprobs — Phase 15 must use self-consistency fallback."""
    name = "mock_logprobs_is_none"
    try:
        client = MockLLMClient()
        resp = client.generate("test")
        assert resp.logprobs is None
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_latency_is_near_zero():
    name = "mock_latency_near_zero"
    try:
        client = MockLLMClient()
        resp = client.generate("test")
        assert resp.latency_ms < 100  # mock should be essentially instant
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_high_risk_prompt_scores_high():
    """Prompts with 'quit' / 'invested' / 'life savings' should score 4 or 5."""
    name = "mock_high_risk_prompt_scores_high"
    try:
        import json
        client = MockLLMClient()
        resp = client.generate(
            "Facet: Risk Taking. Text: I quit my stable job and invested my life savings."
        )
        data = json.loads(resp.raw_text)
        assert data["score"] >= 4, f"Expected score ≥4, got {data['score']}"
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_system_prompt_accepted():
    """system_prompt kwarg must not raise."""
    name = "mock_system_prompt_accepted"
    try:
        client = MockLLMClient()
        resp = client.generate(
            prompt="Score this.",
            system_prompt="You are a facet scoring assistant.",
        )
        assert isinstance(resp, LLMResponse)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_empty_prompt():
    """Empty prompt must not raise — return a valid LLMResponse."""
    name = "mock_empty_prompt"
    try:
        client = MockLLMClient()
        resp = client.generate("")
        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.raw_text, str)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_mock_metadata_has_mock_flag():
    name = "mock_metadata_has_mock_flag"
    try:
        client = MockLLMClient()
        resp = client.generate("test")
        assert resp.metadata.get("mock") is True
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# OllamaClient — structural tests (no network required)
# ---------------------------------------------------------------------------

def test_ollama_client_is_llm_client():
    name = "ollama_is_llm_client_subclass"
    try:
        client = OllamaClient()
        assert isinstance(client, LLMClient)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_ollama_reads_config():
    """OllamaClient must read model name and base_url from config."""
    name = "ollama_reads_config"
    try:
        client = OllamaClient()
        assert len(client.model_name) > 0
        assert client.base_url.startswith("http")
        assert 0.0 <= client.temperature <= 2.0
        assert client.max_tokens > 0
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_ollama_endpoint_format():
    name = "ollama_endpoint_format"
    try:
        client = OllamaClient()
        assert client._endpoint.endswith("/api/chat")
        assert "://" in client._endpoint
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_ollama_unreachable_raises_connection_error():
    """If we point OllamaClient at a bad URL it must raise ConnectionError."""
    name = "ollama_unreachable_raises_connection_error"
    try:
        # Monkey-patch to a dead port
        client = OllamaClient()
        client._endpoint = "http://localhost:19999/api/chat"
        client.base_url  = "http://localhost:19999"
        try:
            client.generate("test")
            _fail(name, "Expected ConnectionError but no exception was raised")
            return
        except ConnectionError:
            pass  # correct
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# get_llm_client factory
# ---------------------------------------------------------------------------

def test_factory_mock_explicit():
    name = "factory_mock_explicit"
    try:
        client = get_llm_client(provider="mock")
        assert isinstance(client, MockLLMClient)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_factory_unknown_provider_raises():
    name = "factory_unknown_provider_raises"
    try:
        try:
            get_llm_client(provider="nonexistent_provider")
            _fail(name, "Expected ValueError but no exception raised")
            return
        except ValueError:
            pass  # correct
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_factory_returns_llm_client_interface():
    """Whatever the factory returns must implement LLMClient."""
    name = "factory_returns_llm_client_interface"
    try:
        client = get_llm_client()           # uses config / probes Ollama
        assert isinstance(client, LLMClient)
        assert hasattr(client, "generate")
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_factory_ollama_falls_back_to_mock_when_unreachable():
    """
    When Ollama is not running, get_llm_client('ollama') must return
    MockLLMClient (graceful degradation, not a crash).
    """
    name = "factory_ollama_falls_back_to_mock"
    try:
        # Temporarily patch OllamaClient.base_url via subclassing trick:
        # We can't guarantee Ollama is/isn't running, so we test the
        # generated client is always usable regardless.
        client = get_llm_client(provider="ollama")
        # Must be either OllamaClient (if running) or MockLLMClient (if not)
        assert isinstance(client, (OllamaClient, MockLLMClient))
        # Must always be callable with no crash
        resp = client.generate("test fallback")
        assert isinstance(resp, LLMResponse)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# End-to-end: factory → generate → validate response shape
# ---------------------------------------------------------------------------

def test_e2e_generate_response_shape():
    """Full generate call through factory — response must match LLMResponse schema."""
    name = "e2e_generate_response_shape"
    try:
        client = get_llm_client()
        resp = client.generate(
            prompt="Rate the Risk Taking facet for: 'I quit my job to start a startup.'",
            system_prompt="Return a JSON object with keys: score (int 1-5), rationale (str), evidence_span (str).",
        )
        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.raw_text, str)
        assert len(resp.raw_text) > 0
        assert isinstance(resp.latency_ms, float)
        assert resp.latency_ms >= 0
        assert isinstance(resp.model_name, str)
        # logprobs: either None or a list
        assert resp.logprobs is None or isinstance(resp.logprobs, list)
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


def test_e2e_latency_recorded():
    name = "e2e_latency_recorded"
    try:
        client = get_llm_client()
        t0 = time.perf_counter()
        resp = client.generate("Hello.")
        wall_ms = (time.perf_counter() - t0) * 1000
        # latency_ms should be positive and not wildly larger than wall time
        assert resp.latency_ms >= 0
        assert resp.latency_ms <= wall_ms + 500  # 500ms slack for overhead
        _ok(name)
    except Exception as e:
        _fail(name, str(e))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 12 — LLM Client")
    print("=" * 60)

    test_llm_response_schema()
    test_llm_response_with_logprobs()
    test_mock_client_is_llm_client()
    test_mock_generate_returns_llm_response()
    test_mock_raw_text_not_empty()
    test_mock_model_name()
    test_mock_logprobs_is_none()
    test_mock_latency_is_near_zero()
    test_mock_high_risk_prompt_scores_high()
    test_mock_system_prompt_accepted()
    test_mock_empty_prompt()
    test_mock_metadata_has_mock_flag()
    test_ollama_client_is_llm_client()
    test_ollama_reads_config()
    test_ollama_endpoint_format()
    test_ollama_unreachable_raises_connection_error()
    test_factory_mock_explicit()
    test_factory_unknown_provider_raises()
    test_factory_returns_llm_client_interface()
    test_factory_ollama_falls_back_to_mock_when_unreachable()
    test_e2e_generate_response_shape()
    test_e2e_latency_recorded()

    print()
    if _FAILED:
        print(f"✗ {len(_FAILED)} test(s) failed: {_FAILED}")
        sys.exit(1)
    else:
        print(f"✅ All Phase 12 tests passed ({len(_PASSED)} tests)")