import numpy as np
import optuna

from optimize.search_spaces import mutate_around, sample_parameters


def test_sample_parameters_skips_requires_when_condition_false():
    space = {
        "useStopLoss": {"type": "bool"},
        "stopLookback": {"type": "int", "min": 2, "max": 10, "step": 2, "requires": "useStopLoss"},
    }
    trial = optuna.trial.FixedTrial({"useStopLoss": False})

    params = sample_parameters(trial, space)

    assert params["useStopLoss"] is False
    assert "stopLookback" not in params


def test_sample_parameters_emits_requires_when_condition_true():
    space = {
        "useStopLoss": {"type": "bool"},
        "stopLookback": {"type": "int", "min": 2, "max": 10, "step": 2, "requires": "useStopLoss"},
    }
    trial = optuna.trial.FixedTrial({"useStopLoss": True, "stopLookback": 6})

    params = sample_parameters(trial, space)

    assert params["useStopLoss"] is True
    assert params["stopLookback"] == 6


def test_mutate_around_respects_integer_steps():
    space = {"foo": {"type": "int", "min": 3, "max": 11, "step": 2}}
    params = {"foo": 7}
    legal_values = {3, 5, 7, 9, 11}

    rng = np.random.default_rng(123)
    for _ in range(200):
        mutated = mutate_around(params, space, scale=0.5, rng=rng)
        assert mutated["foo"] in legal_values
