from mate.trajectory.datatypes import InteractionRecord, ModelResponse
from mate.trajectory.replay_cache import ReplayCache


def _make_record(
    agent_role: str,
    turn_index: int,
    response_text: str = "cached",
) -> InteractionRecord:
    return InteractionRecord(
        agent_role=agent_role,
        turn_index=turn_index,
        timestamp=float(turn_index),
        messages=[{"role": "user", "content": f"turn {turn_index}"}],
        generation_params={},
        response_text=response_text,
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
        episode_id="pilot-001",
        metadata={},
    )


def test_replay_cache_from_buffer_and_lookup() -> None:
    buffer = [
        _make_record("searcher", 1, response_text="search-result"),
        _make_record("verifier", 0, response_text="verified"),
    ]

    cache = ReplayCache.from_buffer(buffer)

    assert cache.lookup("verifier", 0) == ModelResponse(
        content="verified",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )


def test_replay_cache_miss() -> None:
    cache = ReplayCache.from_buffer([_make_record("verifier", 0)])

    assert cache.lookup("searcher", 0) is None


def test_replay_cache_truncated_at_branch_point() -> None:
    buffer = [
        _make_record("verifier", 0, response_text="turn-0"),
        _make_record("searcher", 1, response_text="turn-1"),
        _make_record("verifier", 2, response_text="turn-2"),
        _make_record("answerer", 3, response_text="turn-3"),
    ]

    cache = ReplayCache.from_buffer(buffer, branch_at_global_position=2)

    assert cache.lookup("verifier", 0) == ModelResponse(
        content="turn-0",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )
    assert cache.lookup("searcher", 1) == ModelResponse(
        content="turn-1",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
        finish_reason="stop",
    )
    assert cache.lookup("verifier", 2) is None
    assert cache.lookup("answerer", 3) is None


def test_replay_cache_size() -> None:
    cache = ReplayCache.from_buffer(
        [
            _make_record("verifier", 0),
            _make_record("searcher", 1),
        ]
    )

    assert len(cache) == 2
