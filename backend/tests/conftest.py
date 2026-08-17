"""Shared pytest fixtures for the backend test suite.

Enforces deterministic behavior regardless of the developer machine:

- The LLM provider configuration is pinned to "unconfigured" for every
  test, so the 501 availability gate (POST /api/chat without a real
  provider) behaves identically in CI and on machines that have a
  populated .env file. Tests that need a configured LLM provider
  re-set the settings attributes inside the test body.

- The STT provider configuration is also pinned to "unconfigured" for
  every test. FakeSTT is injected explicitly in STT tests via
  create_fake_stt(); get_stt() will never accidentally return a real
  provider during test execution.

- The module-level provider caches (``_llm_cache`` and ``_stt_cache`` in
  ``app.api.dependencies``) are cleared between tests. Without this,
  a previous test that configured a real provider would leak that
  cached instance into later tests even after monkeypatch resets the
  settings, leading to phantom SDK clients sitting in the test process.
"""

import pytest

from app.api.dependencies import (
    _llm_cache,
    _orchestrator_cache,
    _stt_cache,
    _tts_cache,
)
from app.settings import settings


@pytest.fixture(autouse=True)
def unconfigured_llm_provider(monkeypatch):
    """Force the LLM provider to its unconfigured state for every test."""
    monkeypatch.setattr(settings, "llm_provider", "openai_compatible")
    monkeypatch.setattr(settings, "llm_api_key", None)
    monkeypatch.setattr(settings, "llm_base_url", None)
    yield


@pytest.fixture(autouse=True)
def unconfigured_stt_provider(monkeypatch):
    """Force the STT provider to its unconfigured state for every test.

    This prevents get_stt() from accidentally constructing a real OpenAI
    client (and failing) when LLM_API_KEY or STT_API_KEY happens to be
    set in the developer's local .env file. STT tests inject FakeSTT or
    a stub OpenAIWhisperSTT directly via create_fake_stt() /
    create_openai_whisper_stt().
    """
    monkeypatch.setattr(settings, "stt_provider", "openai_whisper")
    monkeypatch.setattr(settings, "stt_api_key", None)
    monkeypatch.setattr(settings, "stt_base_url", None)
    monkeypatch.setattr(settings, "stt_model", "whisper-1")
    monkeypatch.setattr(settings, "stt_language", None)
    monkeypatch.setattr(settings, "stt_timeout_seconds", 30.0)
    monkeypatch.setattr(settings, "stt_max_audio_size_mb", 10.0)
    yield


@pytest.fixture(autouse=True)
def unconfigured_tts_provider(monkeypatch):
    """Force the TTS provider to its unconfigured state for every test.

    This prevents get_tts() from accidentally constructing a real OpenAI
    client (and failing) when LLM_API_KEY or TTS_API_KEY happens to be
    set in the developer's local .env file. TTS tests inject FakeTTS or
    a stub OpenAITTS directly via create_fake_tts() / create_openai_tts().
    """
    monkeypatch.setattr(settings, "tts_provider", "openai_tts")
    monkeypatch.setattr(settings, "tts_api_key", None)
    monkeypatch.setattr(settings, "tts_base_url", None)
    monkeypatch.setattr(settings, "tts_model", "tts-1")
    monkeypatch.setattr(settings, "tts_voice", "alloy")
    monkeypatch.setattr(settings, "tts_output_format", "mp3")
    monkeypatch.setattr(settings, "tts_speed", 1.0)
    monkeypatch.setattr(settings, "tts_timeout_seconds", 30.0)
    monkeypatch.setattr(settings, "tts_max_text_length", 4096)
    monkeypatch.setattr(settings, "tts_max_audio_size_mb", 10.0)
    yield


@pytest.fixture(autouse=True)
def _clear_provider_caches():
    """Drop any cached provider instances between tests.

    The provider caches are module-level and live for the entire
    pytest process. Without clearing, a test that calls ``get_tts()``
    while ``tts_api_key`` is set would leave a real
    ``OpenAITTS`` cached under that key. Subsequent tests that
    reset settings would still get that cached instance back if the
    cache key matched — silently bypassing the autouse isolation.
    """
    _llm_cache.clear()
    _stt_cache.clear()
    _tts_cache.clear()
    _orchestrator_cache.clear()
    yield
    _llm_cache.clear()
    _stt_cache.clear()
    _tts_cache.clear()
    _orchestrator_cache.clear()
