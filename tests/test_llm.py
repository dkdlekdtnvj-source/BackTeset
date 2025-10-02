"""`optimize.llm` 모듈에 대한 테스트."""
from optimize.llm import _validate_candidate
from optimize.search_spaces import SpaceSpec


def test_validate_candidate_accepts_string_choices() -> None:
    space: SpaceSpec = {
        "optimizer": {"type": "str", "values": ["Adam", "SGD", "RMSprop"]}
    }
    candidate = {"optimizer": " adam "}

    validated = _validate_candidate(candidate, space)

    assert validated == {"optimizer": "Adam"}
