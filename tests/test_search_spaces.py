import optuna

from optimize.search_spaces import sample_parameters


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
