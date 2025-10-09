"""`optimize.llm` 모듈에 대한 테스트."""
from types import SimpleNamespace

import optuna
from optuna.trial import TrialState

from optimize.llm import LLMSuggestions, generate_llm_candidates, _validate_candidate
from optimize.search_spaces import SpaceSpec


def test_validate_candidate_accepts_string_choices() -> None:
    space: SpaceSpec = {
        "optimizer": {"type": "str", "values": ["Adam", "SGD", "RMSprop"]}
    }
    candidate = {"optimizer": " adam "}

    validated = _validate_candidate(candidate, space)

    assert validated == {"optimizer": "Adam"}


def test_generate_llm_candidates_filters_unsupported_kwargs(monkeypatch) -> None:
    space: SpaceSpec = {"x": {"type": "int", "min": 0, "max": 5}}
    trial = optuna.trial.create_trial(
        params={"x": 1},
        distributions={"x": optuna.distributions.IntDistribution(0, 5)},
        value=1.0,
        state=TrialState.COMPLETE,
    )

    class DummyResponse:
        def __init__(self):
            self.text = "{\"candidates\": [{\"x\": 1}], \"insights\": [\"ok\"]}"

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise TypeError("generate_content() got an unexpected keyword argument 'system_instruction'")
            return DummyResponse()

    dummy_models = DummyModels()
    dummy_client = SimpleNamespace(models=dummy_models)
    monkeypatch.setattr(
        "optimize.llm.genai",
        SimpleNamespace(Client=lambda api_key: dummy_client, types=None),
    )
    monkeypatch.setattr(
        "optimize.llm.optuna.importance.get_param_importances",
        lambda study: {"x": 0.9},
    )

    config = {
        "enabled": True,
        "api_key": "dummy",
        "count": 1,
        "top_n": 1,
        "system_instruction": "Focus on profit factor",
        "candidate_count": 2,
    }

    suggestions: LLMSuggestions = generate_llm_candidates(space, [trial], config)

    assert suggestions.candidates == [{"x": 1}]
    assert suggestions.insights == ["ok"]
    assert len(dummy_models.calls) == 2
    assert "system_instruction" in dummy_models.calls[0]
    assert "system_instruction" not in dummy_models.calls[1]


def test_generate_llm_candidates_switches_model_on_permission_error(monkeypatch) -> None:
    space: SpaceSpec = {"x": {"type": "int", "min": 0, "max": 5}}
    trial = optuna.trial.create_trial(
        params={"x": 1},
        distributions={"x": optuna.distributions.IntDistribution(0, 5)},
        value=1.0,
        state=TrialState.COMPLETE,
    )

    class DummyResponse:
        def __init__(self):
            self.text = "{\"candidates\": [{\"x\": 1}], \"insights\": []}"

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("model") == "gemini-2.5-flash":
                raise RuntimeError("PERMISSION_DENIED: model not available")
            return DummyResponse()

    dummy_models = DummyModels()
    dummy_client = SimpleNamespace(models=dummy_models)
    monkeypatch.setattr(
        "optimize.llm.genai",
        SimpleNamespace(Client=lambda api_key: dummy_client, types=None),
    )
    monkeypatch.setattr(
        "optimize.llm.optuna.importance.get_param_importances",
        lambda study: {"x": 1.0},
    )

    config = {
        "enabled": True,
        "api_key": "dummy",
        "model": "gemini-2.5-flash",
        "fallback_models": ["gemini-2.0-flash"],
        "count": 1,
        "top_n": 1,
    }

    suggestions: LLMSuggestions = generate_llm_candidates(space, [trial], config)

    assert suggestions.candidates == [{"x": 1}]
    assert len(dummy_models.calls) == 2
    assert dummy_models.calls[0]["model"] == "gemini-2.5-flash"
    assert dummy_models.calls[1]["model"] == "gemini-2.0-flash"
