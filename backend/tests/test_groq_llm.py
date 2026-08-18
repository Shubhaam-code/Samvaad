"""Unit tests for the Groq Cloud LLM provider adapter (Phase 5.4).

All tests use injected OpenAI client stubs — zero real network requests are made.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from openai import OpenAIError

from app.llm.base import LLMError
from app.llm.groq_llm import (
    DEFAULT_GROQ_MODEL,
    GroqLLM,
    _redact_key,
    is_groq_configured,
)
from app.llm.models import FinishReason, LLMRequest


def _make_mock_choice(text: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=text),
        finish_reason=finish_reason,
    )


def _make_mock_response(text: str, model: str = DEFAULT_GROQ_MODEL, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[_make_mock_choice(text, finish_reason=finish_reason)],
        model=model,
        usage=SimpleNamespace(prompt_tokens=15, completion_tokens=25),
    )


def test_is_groq_configured():
    assert is_groq_configured(None) is False
    assert is_groq_configured("") is False
    assert is_groq_configured("   ") is False
    assert is_groq_configured("gsk_test_api_key_123") is True


def test_key_redaction():
    key = "gsk_secret_groq_key_999"
    msg = f"Failed with {key} at endpoint"
    redacted = _redact_key(msg, key)
    assert key not in redacted
    assert "[REDACTED]" in redacted


def test_groq_llm_initialization():
    with pytest.raises(ValueError, match="api_key is required"):
        GroqLLM(api_key=None)

    llm = GroqLLM(api_key="test-key", model_name="llama-3.1-8b-instant")
    assert llm.model_name == "llama-3.1-8b-instant"
    assert llm.provider == "groq"


def test_groq_llm_successful_generation():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(
        "नई दिल्ली भारत की राजधानी है।"
    )

    llm = GroqLLM(api_key="test-key", client=mock_client)
    req = LLMRequest(
        prompt="भारत की राजधानी क्या है?",
        system_prompt="You are a grounded assistant.",
        temperature=0.2,
        max_tokens=100,
    )

    response = llm.generate(req)

    assert response.text == "नई दिल्ली भारत की राजधानी है।"
    assert response.finish_reason == FinishReason.STOP
    assert response.provider == "groq"
    assert response.model == "llama-3.1-8b-instant"
    assert response.latency_ms is not None and response.latency_ms >= 0.0
    assert response.usage is not None
    assert response.usage.prompt_tokens == 15
    assert response.usage.completion_tokens == 25
    assert response.usage.total_tokens == 40

    mock_client.chat.completions.create.assert_called_once_with(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a grounded assistant."},
            {"role": "user", "content": "भारत की राजधानी क्या है?"},
        ],
        max_tokens=100,
        temperature=0.2,
    )


def test_groq_llm_length_finish_reason():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(
        "Truncated output...", finish_reason="length"
    )

    llm = GroqLLM(api_key="test-key", client=mock_client)
    req = LLMRequest(prompt="Tell me a very long story")
    response = llm.generate(req)

    assert response.finish_reason == FinishReason.LENGTH


def test_groq_llm_error_wrapping_and_redaction():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = OpenAIError(
        "Authentication failed with key gsk_secret_key_12345"
    )

    llm = GroqLLM(api_key="gsk_secret_key_12345", client=mock_client)
    req = LLMRequest(prompt="Hello")

    with pytest.raises(LLMError) as exc_info:
        llm.generate(req)

    assert "gsk_secret_key_12345" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_groq_llm_batch_generation():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response("Answer")

    llm = GroqLLM(api_key="test-key", client=mock_client)
    reqs = [
        LLMRequest(prompt="Question 1"),
        LLMRequest(prompt="Question 2"),
    ]

    responses = llm.generate_batch(reqs)
    assert len(responses) == 2
    assert responses[0].text == "Answer"
    assert responses[1].text == "Answer"
    assert mock_client.chat.completions.create.call_count == 2
