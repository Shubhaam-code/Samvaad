"""Unit tests for the Sarvam AI TTS provider adapter (Phase 5.3).

All tests use injected HTTP client stubs — zero real network requests are made.
"""

import base64
from unittest.mock import MagicMock
import httpx
import pytest

from app.tts.base import TTSError
from app.tts.models import TTSRequest
from app.tts.sarvam_tts import (
    SarvamTTS,
    _map_language_to_sarvam,
    _redact_key,
    is_sarvam_tts_configured,
)

# Minimal 44-byte RIFF WAV header fixture
VALID_WAV_BYTES = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00"
    b"\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)
VALID_B64_AUDIO = base64.b64encode(VALID_WAV_BYTES).decode("ascii")


def test_is_sarvam_tts_configured():
    assert is_sarvam_tts_configured(None) is False
    assert is_sarvam_tts_configured("") is False
    assert is_sarvam_tts_configured("sk_sarvam_test_123") is True


def test_language_mapping_tts():
    assert _map_language_to_sarvam("hi") == "hi-IN"
    assert _map_language_to_sarvam("en") == "en-IN"
    assert _map_language_to_sarvam(None) == "hi-IN"


def test_sarvam_tts_initialization():
    with pytest.raises(ValueError, match="api_key is required"):
        SarvamTTS(api_key=None)

    tts = SarvamTTS(api_key="test-key", model="bulbul:v2", speaker="arvind")
    assert tts.model_name == "bulbul:v2"
    assert tts.provider == "sarvam"


def test_sarvam_tts_successful_synthesis():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "audios": [VALID_B64_AUDIO],
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    tts = SarvamTTS(api_key="test-key", speaker="meera", http_client=mock_client)
    req = TTSRequest(
        text="भारत की राजधानी नई दिल्ली है।",
        language="hi",
        speed=1.2,
    )

    response = tts.synthesize(req)

    assert response.audio == VALID_WAV_BYTES
    assert response.format == "wav"
    assert response.content_type == "audio/wav"
    assert response.model == "bulbul:v2"
    assert response.provider == "sarvam"
    assert response.latency_ms is not None and response.latency_ms > 0.0

    # Verify headers and payload
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["headers"]["api-subscription-key"] == "test-key"
    assert call_kwargs["json"]["speaker"] == "meera"
    assert call_kwargs["json"]["target_language_code"] == "hi-IN"
    assert call_kwargs["json"]["pace"] == 1.2


def test_sarvam_tts_empty_audio_list_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"audios": []}

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    tts = SarvamTTS(api_key="test-key", http_client=mock_client)
    req = TTSRequest(text="Hello world")

    with pytest.raises(TTSError, match="empty audio payload"):
        tts.synthesize(req)


def test_sarvam_tts_authentication_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized with key: my-secret-sarvam-key"

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    tts = SarvamTTS(api_key="my-secret-sarvam-key", http_client=mock_client)
    req = TTSRequest(text="Hello world")

    with pytest.raises(TTSError) as exc_info:
        tts.synthesize(req)

    assert "authentication failed" in str(exc_info.value).lower()
    assert "my-secret-sarvam-key" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_sarvam_tts_timeout_error():
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.TimeoutException("Read timed out")

    tts = SarvamTTS(api_key="test-key", http_client=mock_client)
    req = TTSRequest(text="Hello world")

    with pytest.raises(TTSError, match="timed out"):
        tts.synthesize(req)
