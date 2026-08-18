"""
Tests for the LLM harness (Phase 6.1).

Covers the provider-agnostic interface, the deterministic FakeLLM,
configuration model, request/response models, and all validation rules.

All tests use tiny synthetic strings only.
No real MSMARCO-XI data. No network access. No model downloads.
No external LLM SDKs required.
"""

import socket

import pytest

from app.llm import (
    BaseLLM,
    FinishReason,
    LLMConfig,
    LLMError,
    LLMPrompt,
    LLMProtocol,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMText,
    LLMUsage,
    FakeLLM,
    create_fake_llm,
    validate_batch,
    validate_generated_text,
    validate_max_tokens,
    validate_prompt,
    validate_system_prompt,
    validate_temperature,
    validate_top_p,
)


# ---------------------------------------------------------------------------
# Interface importability
# ---------------------------------------------------------------------------


def test_llm_interface_can_be_imported():
    """Test that the LLM interface can be imported."""
    assert BaseLLM is not None
    assert hasattr(BaseLLM, "generate")
    assert hasattr(BaseLLM, "generate_batch")
    assert hasattr(BaseLLM, "model_name")
    assert hasattr(BaseLLM, "provider")


def test_llm_protocol_can_be_imported():
    """Test that LLMProtocol can be imported."""
    assert LLMProtocol is not None


def test_type_aliases_exist():
    """Test that predictable LLM types are defined."""
    assert LLMPrompt == str
    assert LLMText == str


def test_llm_error_is_exception():
    """Test that LLMError is an Exception subclass."""
    assert issubclass(LLMError, Exception)


def test_base_llm_is_abstract():
    """Test that BaseLLM cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseLLM()


def test_base_llm_requires_generate():
    """Test that BaseLLM subclass must implement generate()."""
    class IncompleteLLM(BaseLLM):
        @property
        def model_name(self):
            return None

        @property
        def provider(self):
            return None

        def generate_batch(self, requests):
            return []

    with pytest.raises(TypeError):
        IncompleteLLM()


def test_base_llm_requires_generate_batch():
    """Test that BaseLLM subclass must implement generate_batch()."""
    class IncompleteLLM(BaseLLM):
        @property
        def model_name(self):
            return None

        @property
        def provider(self):
            return None

        def generate(self, request):
            return None

    with pytest.raises(TypeError):
        IncompleteLLM()


def test_base_llm_requires_model_name():
    """Test that BaseLLM subclass must implement model_name."""
    class IncompleteLLM(BaseLLM):
        @property
        def provider(self):
            return None

        def generate(self, request):
            return None

        def generate_batch(self, requests):
            return []

    with pytest.raises(TypeError):
        IncompleteLLM()


def test_base_llm_requires_provider():
    """Test that BaseLLM subclass must implement provider."""
    class IncompleteLLM(BaseLLM):
        @property
        def model_name(self):
            return None

        def generate(self, request):
            return None

        def generate_batch(self, requests):
            return []

    with pytest.raises(TypeError):
        IncompleteLLM()


def test_llm_protocol_duck_typing():
    """Test that any class with the right methods satisfies LLMProtocol."""
    class DuckTypedLLM:
        """Not inheriting from BaseLLM, but has the right methods."""

        @property
        def model_name(self):
            return "duck-model"

        @property
        def provider(self):
            return "duck"

        def generate(self, request):
            return LLMResponse(text="duck answer")

        def generate_batch(self, requests):
            return [self.generate(r) for r in requests]

    llm = DuckTypedLLM()
    assert hasattr(llm, "generate")
    assert hasattr(llm, "generate_batch")
    assert hasattr(llm, "model_name")
    assert hasattr(llm, "provider")
    assert callable(llm.generate)
    assert callable(llm.generate_batch)


# ---------------------------------------------------------------------------
# Fake LLM basics
# ---------------------------------------------------------------------------


def test_fake_llm_works():
    """Test that FakeLLM produces non-empty responses."""
    llm = FakeLLM(model_name="fake-model", max_tokens=64)
    assert llm.model_name == "fake-model"
    assert llm.max_tokens == 64
    assert llm.provider == "fake"

    response = llm.generate(LLMRequest(prompt="what is goa?"))
    assert isinstance(response, LLMResponse)
    assert isinstance(response.text, str)
    assert response.text
    assert response.text.strip()
    assert response.finish_reason == FinishReason.STOP
    assert response.provider == "fake"
    assert response.model == "fake-model"


def test_fake_llm_defaults():
    """Test that FakeLLM has sensible defaults."""
    llm = FakeLLM()
    assert llm.model_name == "fake-llm"
    assert llm.max_tokens == 256
    assert llm.provider == "fake"


def test_fake_llm_create_factory():
    """Test the create_fake_llm factory."""
    llm = create_fake_llm(model_name="test-model", max_tokens=32)
    assert isinstance(llm, FakeLLM)
    assert llm.model_name == "test-model"
    assert llm.max_tokens == 32


def test_fake_llm_invalid_model_name_raises():
    """Test that FakeLLM rejects invalid model names."""
    with pytest.raises(ValueError):
        FakeLLM(model_name="")
    with pytest.raises(ValueError):
        FakeLLM(model_name="   ")


def test_fake_llm_invalid_max_tokens_raises():
    """Test that FakeLLM rejects invalid max_tokens."""
    with pytest.raises(ValueError):
        FakeLLM(max_tokens=0)
    with pytest.raises(ValueError):
        FakeLLM(max_tokens=-5)


def test_fake_llm_latency_reported():
    """Test that FakeLLM reports configured latency."""
    llm = FakeLLM(latency_ms=12.5)
    response = llm.generate(LLMRequest(prompt="hi"))
    assert response.latency_ms == 12.5


# ---------------------------------------------------------------------------
# Deterministic output
# ---------------------------------------------------------------------------


def test_deterministic_output_same_instance():
    """Test that the same prompt produces identical text on one instance."""
    llm = FakeLLM()
    request = LLMRequest(prompt="capital of goa")
    assert llm.generate(request).text == llm.generate(request).text


def test_deterministic_output_across_instances():
    """Test that the same prompt produces identical text across instances."""
    llm1 = FakeLLM()
    llm2 = FakeLLM()
    request = LLMRequest(prompt="best beaches")
    assert llm1.generate(request).text == llm2.generate(request).text


def test_deterministic_batch_output():
    """Test that batches are deterministic across instances."""
    requests = [LLMRequest(prompt="q1"), LLMRequest(prompt="q2")]
    llm1 = FakeLLM()
    llm2 = FakeLLM()
    assert [r.text for r in llm1.generate_batch(requests)] == [
        r.text for r in llm2.generate_batch(requests)
    ]


def test_different_prompts_differ():
    """Test that different prompts produce different text."""
    llm = FakeLLM()
    t1 = llm.generate(LLMRequest(prompt="tourism in goa")).text
    t2 = llm.generate(LLMRequest(prompt="food in goa")).text
    assert t1 != t2


def test_usage_counts_deterministic():
    """Test that token usage is deterministic and consistent."""
    llm = FakeLLM()
    prompt = "tell me about goa tourism"
    r1 = llm.generate(LLMRequest(prompt=prompt))
    r2 = llm.generate(LLMRequest(prompt=prompt))
    assert r1.usage.prompt_tokens == r2.usage.prompt_tokens
    assert r1.usage.completion_tokens == r2.usage.completion_tokens
    assert r1.usage.prompt_tokens == len(prompt.split())


# ---------------------------------------------------------------------------
# Single generation
# ---------------------------------------------------------------------------


def test_single_generation_returns_response():
    """Test generating a single response."""
    llm = FakeLLM()
    response = llm.generate(LLMRequest(prompt="hello"))
    assert isinstance(response, LLMResponse)
    assert response.text


def test_generate_accepts_duck_typed_request():
    """Test that generate() accepts duck-typed request-like objects."""
    llm = FakeLLM()

    class SimpleRequest:
        def __init__(self, prompt):
            self.prompt = prompt

    response = llm.generate(SimpleRequest("goa weather"))
    assert response.text


def test_generate_rejects_missing_request():
    """Test that generate() rejects missing requests."""
    llm = FakeLLM()
    with pytest.raises(ValueError, match="prompt attribute"):
        llm.generate(None)


def test_generate_rejects_missing_prompt_attribute():
    """Test that generate() rejects objects without a prompt attribute."""
    llm = FakeLLM()

    class NoPrompt:
        pass

    with pytest.raises(ValueError, match="prompt attribute"):
        llm.generate(NoPrompt())


def test_generate_rejects_empty_prompt():
    """Test that generate() rejects empty prompts."""
    llm = FakeLLM()
    with pytest.raises(ValueError):
        llm.generate(LLMRequest(prompt="   "))


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def test_batch_generation():
    """Test generating a batch of responses."""
    llm = FakeLLM()
    requests = [LLMRequest(prompt="one"), LLMRequest(prompt="two"), LLMRequest(prompt="three")]
    responses = llm.generate_batch(requests)
    assert len(responses) == 3
    assert all(isinstance(r, LLMResponse) for r in responses)


def test_batch_generation_preserves_ordering():
    """Test that generate_batch() preserves input ordering exactly."""
    llm = FakeLLM()
    requests = [LLMRequest(prompt="A"), LLMRequest(prompt="B"), LLMRequest(prompt="C")]

    responses = llm.generate_batch(requests)
    expected = [llm.generate(r).text for r in requests]
    assert [r.text for r in responses] == expected

    reversed_texts = [r.text for r in llm.generate_batch(list(reversed(requests)))]
    assert reversed_texts == list(reversed(expected))


def test_batch_generation_empty_raises():
    """Test that generating an empty batch raises ValueError."""
    llm = FakeLLM()
    with pytest.raises(ValueError):
        llm.generate_batch([])


# ---------------------------------------------------------------------------
# Validation: prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\t\n", " \u00a0 "])
def test_generate_empty_or_whitespace_raises(bad_prompt):
    """Test that empty/whitespace-only prompts raise ValueError."""
    llm = FakeLLM()
    with pytest.raises(ValueError):
        llm.generate(LLMRequest(prompt=bad_prompt))


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\t\n"])
def test_validate_prompt_rejects_empty_or_whitespace(bad_prompt):
    """Test the shared validate_prompt() rule directly."""
    with pytest.raises(ValueError):
        validate_prompt(bad_prompt)


def test_validate_prompt_accepts_valid_text():
    """Test that validate_prompt() accepts non-empty text."""
    assert validate_prompt("  hello  ") == "  hello  "
    assert validate_prompt("नमस्ते") == "नमस्ते"


def test_validate_prompt_rejects_non_string():
    """Test that validate_prompt() rejects non-string inputs."""
    with pytest.raises(ValueError):
        validate_prompt(123)
    with pytest.raises(ValueError):
        validate_prompt(None)


# ---------------------------------------------------------------------------
# Validation: system prompt
# ---------------------------------------------------------------------------


def test_validate_system_prompt_none_ok():
    """Test that validate_system_prompt() accepts None."""
    assert validate_system_prompt(None) is None


def test_validate_system_prompt_valid_ok():
    """Test that validate_system_prompt() accepts valid strings."""
    assert validate_system_prompt("be concise") == "be concise"
    assert validate_system_prompt("संक्षिप्त रहें") == "संक्षिप्त रहें"


@pytest.mark.parametrize("bad", ["", "   ", "\t"])
def test_validate_system_prompt_rejects_empty(bad):
    """Test that validate_system_prompt() rejects empty/whitespace."""
    with pytest.raises(ValueError):
        validate_system_prompt(bad)


def test_validate_system_prompt_rejects_non_string():
    """Test that validate_system_prompt() rejects non-strings."""
    with pytest.raises(ValueError):
        validate_system_prompt(123)


def test_request_accepts_system_prompt():
    """Test that LLMRequest accepts a system prompt."""
    request = LLMRequest(prompt="q", system_prompt="answer in hindi")
    assert request.system_prompt == "answer in hindi"


# ---------------------------------------------------------------------------
# Validation: generation parameters
# ---------------------------------------------------------------------------


def test_validate_max_tokens_ok():
    """Test that validate_max_tokens() accepts valid values."""
    assert validate_max_tokens(None) is None
    assert validate_max_tokens(10) == 10
    assert validate_max_tokens(1) == 1


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "10"])
def test_validate_max_tokens_invalid(bad):
    """Test that validate_max_tokens() rejects invalid values."""
    with pytest.raises(ValueError):
        validate_max_tokens(bad)


def test_validate_temperature_ok():
    """Test that validate_temperature() accepts valid values."""
    assert validate_temperature(None) is None
    assert validate_temperature(0.0) == 0.0
    assert validate_temperature(2.0) == 2.0


@pytest.mark.parametrize("bad", [-0.1, 2.1, "hot", True])
def test_validate_temperature_invalid(bad):
    """Test that validate_temperature() rejects invalid values."""
    with pytest.raises(ValueError):
        validate_temperature(bad)


def test_validate_top_p_ok():
    """Test that validate_top_p() accepts valid values."""
    assert validate_top_p(None) is None
    assert validate_top_p(0.0) == 0.0
    assert validate_top_p(1.0) == 1.0


@pytest.mark.parametrize("bad", [-0.1, 1.1, "top", True])
def test_validate_top_p_invalid(bad):
    """Test that validate_top_p() rejects invalid values."""
    with pytest.raises(ValueError):
        validate_top_p(bad)


def test_validate_generated_text_ok():
    """Test that validate_generated_text() accepts valid text."""
    assert validate_generated_text("answer") == "answer"


@pytest.mark.parametrize("bad", ["", "   ", "\t", 123, None])
def test_validate_generated_text_invalid(bad):
    """Test that validate_generated_text() rejects invalid text."""
    with pytest.raises(ValueError):
        validate_generated_text(bad)


# ---------------------------------------------------------------------------
# Validation: batch
# ---------------------------------------------------------------------------


def test_validate_batch_ok():
    """Test that validate_batch() accepts a valid list."""
    requests = [LLMRequest(prompt="a"), LLMRequest(prompt="b")]
    assert validate_batch(requests) == requests


def test_validate_batch_empty_raises():
    """Test that validate_batch() rejects an empty batch."""
    with pytest.raises(ValueError):
        validate_batch([])


def test_validate_batch_rejects_non_list():
    """Test that validate_batch() rejects non-list inputs."""
    with pytest.raises(ValueError):
        validate_batch("not a list")


def test_validate_batch_rejects_invalid_item():
    """Test that validate_batch() rejects items without a prompt."""

    class NoPrompt:
        pass

    with pytest.raises(ValueError):
        validate_batch([LLMRequest(prompt="ok"), NoPrompt()])


def test_validate_batch_rejects_empty_prompt_item():
    """Test that validate_batch() rejects items with empty prompts."""
    with pytest.raises(ValueError):
        validate_batch([LLMRequest(prompt="ok"), LLMRequest(prompt="   ")])


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


def test_request_defaults():
    """Test that LLMRequest has safe defaults."""
    request = LLMRequest(prompt="hello")
    assert request.prompt == "hello"
    assert request.system_prompt is None
    assert request.max_tokens is None
    assert request.temperature is None
    assert request.top_p is None


def test_request_explicit_values():
    """Test that LLMRequest accepts explicit values."""
    request = LLMRequest(
        prompt="q",
        system_prompt="sys",
        max_tokens=100,
        temperature=0.5,
        top_p=0.9,
    )
    assert request.max_tokens == 100
    assert request.temperature == 0.5
    assert request.top_p == 0.9


def test_request_rejects_empty_prompt():
    """Test that LLMRequest rejects empty/whitespace prompts."""
    with pytest.raises(ValueError):
        LLMRequest(prompt="")
    with pytest.raises(ValueError):
        LLMRequest(prompt="   ")


def test_request_rejects_missing_prompt():
    """Test that LLMRequest requires a prompt."""
    with pytest.raises(ValueError):
        LLMRequest()


def test_request_rejects_whitespace_system_prompt():
    """Test that LLMRequest rejects whitespace-only system prompts."""
    with pytest.raises(ValueError):
        LLMRequest(prompt="q", system_prompt="   ")


@pytest.mark.parametrize("kwarg", [("max_tokens", 0), ("max_tokens", -1), ("temperature", -0.1), ("temperature", 2.1), ("top_p", -0.1), ("top_p", 1.1)])
def test_request_rejects_invalid_parameters(kwarg):
    """Test that LLMRequest rejects invalid generation parameters."""
    name, value = kwarg
    with pytest.raises(ValueError):
        LLMRequest(prompt="q", **{name: value})


def test_request_hindi_prompt_ok():
    """Test that LLMRequest accepts Hindi prompts."""
    request = LLMRequest(prompt="गोवा की राजधानी क्या है?")
    assert request.prompt == "गोवा की राजधानी क्या है?"


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


def test_response_defaults():
    """Test that LLMResponse has safe defaults."""
    response = LLMResponse(text="answer")
    assert response.text == "answer"
    assert response.model is None
    assert response.provider is None
    assert response.finish_reason == FinishReason.STOP
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.latency_ms is None


def test_response_rejects_empty_text():
    """Test that LLMResponse rejects empty/whitespace text."""
    with pytest.raises(ValueError):
        LLMResponse(text="")
    with pytest.raises(ValueError):
        LLMResponse(text="   ")


def test_response_rejects_missing_text():
    """Test that LLMResponse requires text."""
    with pytest.raises(ValueError):
        LLMResponse()


def test_response_rejects_negative_latency():
    """Test that LLMResponse rejects negative latency."""
    with pytest.raises(ValueError):
        LLMResponse(text="x", latency_ms=-1.0)


def test_response_explicit_fields():
    """Test that LLMResponse accepts explicit fields."""
    response = LLMResponse(
        text="done",
        model="m",
        provider="fake",
        finish_reason=FinishReason.LENGTH,
        usage=LLMUsage(prompt_tokens=5, completion_tokens=7),
        latency_ms=3.2,
    )
    assert response.finish_reason == FinishReason.LENGTH
    assert response.usage.total_tokens == 12
    assert response.latency_ms == 3.2


# ---------------------------------------------------------------------------
# Usage model
# ---------------------------------------------------------------------------


def test_usage_total_tokens_computed():
    """Test that total_tokens is computed from prompt + completion."""
    usage = LLMUsage(prompt_tokens=3, completion_tokens=4)
    assert usage.total_tokens == 7


def test_usage_defaults_zero():
    """Test that LLMUsage defaults to zero tokens."""
    usage = LLMUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_usage_rejects_negative_tokens():
    """Test that LLMUsage rejects negative token counts."""
    with pytest.raises(ValueError):
        LLMUsage(prompt_tokens=-1)
    with pytest.raises(ValueError):
        LLMUsage(completion_tokens=-2)


def test_finish_reason_values():
    """Test FinishReason enum values."""
    assert FinishReason.STOP.value == "stop"
    assert FinishReason.LENGTH.value == "length"


# ---------------------------------------------------------------------------
# LLMConfig model
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Test that LLMConfig has safe defaults."""
    config = LLMConfig()
    assert config.provider == LLMProvider.FAKE
    assert config.model_name is None
    assert config.max_tokens is None
    assert config.temperature == 0.7
    assert config.top_p is None
    assert config.timeout_seconds == 30.0


def test_config_explicit_values():
    """Test that LLMConfig accepts explicit values."""
    config = LLMConfig(
        provider=LLMProvider.OPENAI_COMPATIBLE,
        model_name="some-model",
        max_tokens=512,
        temperature=0.2,
        top_p=0.95,
        timeout_seconds=60.0,
    )
    assert config.provider == LLMProvider.OPENAI_COMPATIBLE
    assert config.model_name == "some-model"
    assert config.max_tokens == 512
    assert config.temperature == 0.2
    assert config.top_p == 0.95
    assert config.timeout_seconds == 60.0


def test_config_rejects_empty_model_name():
    """Test that LLMConfig rejects empty/whitespace model names."""
    with pytest.raises(ValueError):
        LLMConfig(model_name="")
    with pytest.raises(ValueError):
        LLMConfig(model_name="   ")


def test_config_rejects_invalid_parameters():
    """Test that LLMConfig rejects invalid generation parameters."""
    with pytest.raises(ValueError):
        LLMConfig(max_tokens=0)
    with pytest.raises(ValueError):
        LLMConfig(temperature=-0.1)
    with pytest.raises(ValueError):
        LLMConfig(temperature=2.1)
    with pytest.raises(ValueError):
        LLMConfig(top_p=1.1)
    with pytest.raises(ValueError):
        LLMConfig(timeout_seconds=0)
    with pytest.raises(ValueError):
        LLMConfig(timeout_seconds=-5.0)


def test_config_provider_values():
    """Test that the provider enum values match expected strings."""
    assert LLMProvider.FAKE.value == "fake"
    assert LLMProvider.OPENAI_COMPATIBLE.value == "openai_compatible"
    assert LLMProvider.GEMINI.value == "gemini"
    assert LLMProvider.LOCAL.value == "local"


def test_config_can_drive_fake_llm_creation():
    """Test that a config with the fake provider creates a FakeLLM."""
    config = LLMConfig(model_name="cfg-model", max_tokens=64)
    llm = FakeLLM(model_name=config.model_name, max_tokens=config.max_tokens)
    assert llm.model_name == "cfg-model"
    assert llm.max_tokens == 64
    assert llm.provider == "fake"


# ---------------------------------------------------------------------------
# No network / no model download
# ---------------------------------------------------------------------------


def test_fake_llm_never_touches_network(monkeypatch):
    """Test that FakeLLM works with all network access blocked."""
    def deny_connect(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny_connect)

    llm = FakeLLM()
    response = llm.generate(LLMRequest(prompt="offline test"))
    batch = llm.generate_batch([LLMRequest(prompt="a"), LLMRequest(prompt="b")])

    assert response.text
    assert len(batch) == 2


def test_llm_package_imports_no_external_libraries():
    """Test that the LLM package imports without any external SDKs."""
    import importlib

    forbidden_prefixes = (
        "openai",
        "anthropic",
        "google",
        "transformers",
        "torch",
        "numpy",
        "faiss",
    )

    module_names = (
        "app.llm",
        "app.llm.base",
        "app.llm.fake",
        "app.llm.config",
        "app.llm.models",
        "app.llm.types",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module is not None

        # No name in the LLM package namespace may originate from an
        # external SDK/ML library (checked directly, immune to other test
        # modules importing those libraries first)
        for attr_name, attr in vars(module).items():
            origin = getattr(attr, "__module__", None)
            if origin is not None and origin.startswith(forbidden_prefixes):
                raise AssertionError(
                    f"{module_name} exposes {attr_name} from forbidden "
                    f"library '{origin}'"
                )


# ---------------------------------------------------------------------------
# Unicode / Hindi text support
# ---------------------------------------------------------------------------


def test_unicode_hindi_prompt_support():
    """Test that Hindi (Devanagari) prompts generate correctly."""
    llm = FakeLLM()
    hindi_prompt = "गोवा में पर्यटन एक प्रमुख उद्योग है"
    response = llm.generate(LLMRequest(prompt=hindi_prompt))
    assert isinstance(response.text, str)
    assert response.text.strip()


def test_unicode_hindi_deterministic():
    """Test that Hindi prompt generation is deterministic."""
    llm = FakeLLM()
    hindi_prompt = "भारत की राजधानी नई दिल्ली है"
    assert llm.generate(LLMRequest(prompt=hindi_prompt)).text == \
        llm.generate(LLMRequest(prompt=hindi_prompt)).text


def test_unicode_batch_preserves_order():
    """Test that a mixed Hindi/English batch preserves order."""
    llm = FakeLLM()
    prompts = ["English one", "हिंदी पाठ", "mixed text with 123", "होटल"]
    requests = [LLMRequest(prompt=p) for p in prompts]
    responses = llm.generate_batch(requests)
    expected = [llm.generate(r).text for r in requests]
    assert [r.text for r in responses] == expected


def test_hindi_and_english_different_text():
    """Test that Hindi and its English translation map to different text."""
    llm = FakeLLM()
    hindi = llm.generate(LLMRequest(prompt="गोवा")).text
    english = llm.generate(LLMRequest(prompt="goa")).text
    assert hindi != english


# ---------------------------------------------------------------------------
# Generic interface shape for future providers
# ---------------------------------------------------------------------------


def test_future_provider_duck_typing_compatible():
    """Test that the base interface is duck-typed for future providers.

    Simulates a future OpenAI-compatible-style provider wrapping the
    interface: it only needs generate/generate_batch/model_name/provider.
    """

    class FakeOpenAIProvider:
        """Stand-in for a future production provider."""

        def __init__(self, model_name: str = "gpt-x"):
            self._model_name = model_name

        @property
        def model_name(self):
            return self._model_name

        @property
        def provider(self):
            return "openai_compatible"

        def generate(self, request):
            return LLMResponse(text=f"response for {request.prompt}")

        def generate_batch(self, requests):
            return [self.generate(r) for r in requests]

    provider = FakeOpenAIProvider(model_name="gpt-x")
    assert provider.model_name == "gpt-x"
    assert provider.provider == "openai_compatible"
    assert provider.generate(LLMRequest(prompt="hi")).text.startswith("response for")
    assert len(provider.generate_batch([LLMRequest(prompt="a"), LLMRequest(prompt="b")])) == 2

    # Protocol-compatible shapes pass the shared validators too
    validated = validate_batch([LLMRequest(prompt="a"), LLMRequest(prompt="b")])
    assert len(validated) == 2
