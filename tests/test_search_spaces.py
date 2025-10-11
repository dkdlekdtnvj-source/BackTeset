import numpy as np
import optuna

from optimize.search_spaces import mutate_around, random_parameters, sample_parameters


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


def test_random_parameters_respects_requires_and_steps():
    space = {
        "useStopLoss": {"type": "bool"},
        "stopLookback": {
            "type": "int",
            "min": 2,
            "max": 10,
            "step": 2,
            "requires": "useStopLoss",
        },
        "signal": {"type": "float", "min": 1.0, "max": 2.0, "step": 0.5},
        "mode": {"type": "choice", "values": ["A", "B"]},
    }

    rng = np.random.default_rng(0)
    seen_stop_values = set()
    for _ in range(100):
        params = random_parameters(space, rng=rng)
        assert params["signal"] in {1.0, 1.5, 2.0}
        assert params["mode"] in {"A", "B"}
        if params["useStopLoss"]:
            assert params["stopLookback"] in {2, 4, 6, 8, 10}
            seen_stop_values.add(params["stopLookback"])
        else:
            assert "stopLookback" not in params
    assert seen_stop_values  # 적어도 한 번은 조건부 파라미터가 생성되어야 함
