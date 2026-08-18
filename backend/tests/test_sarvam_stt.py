"""Unit tests for the Sarvam AI STT provider adapter (Phase 5.3).

All tests use injected HTTP client stubs — zero real network requests are made.
"""

from unittest.mock import MagicMock
import httpx
import pytest

from app.stt.base import STTError
from app.stt.models import STTRequest
from app.stt.sarvam_stt import (
    SarvamSTT,
    _map_language_to_sarvam,
    _redact_key,
    is_sarvam_stt_configured,
)

# Minimal 44-byte RIFF WAV header fixture
VALID_WAV_BYTES = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00"
    b"\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


def test_is_sarvam_stt_configured():
    assert is_sarvam_stt_configured(None) is False
    assert is_sarvam_stt_configured("") is False
    assert is_sarvam_stt_configured("   ") is False
    assert is_sarvam_stt_configured("sk_sarvam_test_123") is True


def test_language_mapping():
    assert _map_language_to_sarvam("hi") == "hi-IN"
    assert _map_language_to_sarvam("HI-IN") == "hi-IN"
    assert _map_language_to_sarvam("en") == "en-IN"
    assert _map_language_to_sarvam("ta") == "ta-IN"
    assert _map_language_to_sarvam(None) == "unknown"
    assert _map_language_to_sarvam("unknown") == "unknown"


def test_key_redaction():
    key = "secret_sarvam_key_999"
    msg = f"Failed to authenticate with key {key} at endpoint"
    redacted = _redact_key(msg, key)
    assert key not in redacted
    assert "[REDACTED]" in redacted


def test_sarvam_stt_initialization():
    with pytest.raises(ValueError, match="api_key is required"):
        SarvamSTT(api_key=None)

    stt = SarvamSTT(api_key="test-key", model="saaras:v2")
    assert stt.model_name == "saaras:v2"
    assert stt.provider == "sarvam"


def test_sarvam_stt_successful_transcription():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transcript": "भारत की राजधानी नई दिल्ली है।",
        "language_code": "hi-IN",
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    stt = SarvamSTT(api_key="test-key", http_client=mock_client)
    req = STTRequest(
        audio=VALID_WAV_BYTES,
        filename="query.wav",
        language="hi",
    )

    response = stt.transcribe(req)

    assert response.text == "भारत की राजधानी नई दिल्ली है।"
    assert response.language == "hi-IN"
    assert response.model == "saaras:v2"
    assert response.provider == "sarvam"
    assert response.latency_ms is not None and response.latency_ms > 0.0

    # Verify headers and files were passed to POST
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["headers"]["api-subscription-key"] == "test-key"
    assert call_kwargs["data"]["language_code"] == "hi-IN"


def test_sarvam_stt_authentication_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Invalid subscription key: test-secret-key"

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    stt = SarvamSTT(api_key="test-secret-key", http_client=mock_client)
    req = STTRequest(audio=VALID_WAV_BYTES, filename="query.wav")

    with pytest.raises(STTError) as exc_info:
        stt.transcribe(req)

    assert "authentication failed" in str(exc_info.value).lower()
    # Ensure raw secret key is never present in exception string
    assert "test-secret-key" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_sarvam_stt_timeout_error():
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.TimeoutException("Read timed out")

    stt = SarvamSTT(api_key="test-key", timeout_seconds=5.0, http_client=mock_client)
    req = STTRequest(audio=VALID_WAV_BYTES, filename="query.wav")

    with pytest.raises(STTError, match="timed out"):
        stt.transcribe(req)
