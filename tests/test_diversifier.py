from types import SimpleNamespace

from optuna.trial import TrialState

from optimize.run import TrialDiversifier


class DummyStudy:
    def __init__(self) -> None:
        self.enqueued = []

    def enqueue_trial(self, params, skip_if_exists=True):  # noqa: D401 - 단순 저장
        self.enqueued.append(params)


def _trial(params, number):
    return SimpleNamespace(state=TrialState.COMPLETE, params=params, number=number)


def test_diversifier_enqueues_after_similarity_streak():
    space = {
        "a": {"type": "int", "min": 1, "max": 5, "step": 1},
        "timeframe": {"type": "choice", "values": ["1m", "3m", "5m"]},
    }
    config = {
        "max_consecutive": 3,
        "similarity_threshold": 0.9,
        "jump_trials": 2,
        "history_bias": 0.0,
        "jump_scale": 0.0,
        "timeframe_cycle": [
            {"timeframe": "1m", "repeat": 1},
            {"timeframe": "3m", "repeat": 1},
        ],
    }
    diversifier = TrialDiversifier(space, config, param_order=list(space.keys()), seed=1)
    study = DummyStudy()

    for idx in range(3):
        diversifier(study, _trial({"a": 2, "timeframe": "1m"}, idx))

    assert len(study.enqueued) == 2
    assert {entry["timeframe"] for entry in study.enqueued} == {"1m", "3m"}


def test_diversifier_respects_cooldown():
    space = {"a": {"type": "int", "min": 1, "max": 5, "step": 1}}
    config = {
        "max_consecutive": 2,
        "similarity_threshold": 0.9,
        "jump_trials": 1,
        "cooldown": 2,
        "jump_scale": 0.0,
        "history_bias": 0.0,
    }
    diversifier = TrialDiversifier(space, config, seed=123)
    study = DummyStudy()

    diversifier(study, _trial({"a": 1}, 0))
    diversifier(study, _trial({"a": 1}, 1))
    assert len(study.enqueued) == 1

    # Cooldown 2 trials → 다음 호출은 큐를 추가하지 않아야 한다.
    diversifier(study, _trial({"a": 1}, 2))
    assert len(study.enqueued) == 1
