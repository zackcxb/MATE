from __future__ import annotations

from scripts._trajectory_utils import (
    build_integrity_report,
    collect_all_turns,
    collect_tree_rewards,
    compute_prefix_sharing,
)


def test_collect_all_turns_flattens_roles_and_sorts_by_timestamp_then_turn_index() -> None:
    episode_payload = {
        "trajectory": {
            "agent_trajectories": {
                "answerer": [
                    {
                        "turn_index": 1,
                        "timestamp": 3.0,
                        "response_text": "later",
                    }
                ],
                "verifier": [
                    {
                        "turn_index": 2,
                        "timestamp": 1.0,
                        "response_text": "same-ts-later-index",
                    },
                    {
                        "turn_index": 0,
                        "timestamp": 1.0,
                        "response_text": "first",
                    },
                ],
            }
        }
    }

    turns = collect_all_turns(episode_payload)

    assert [turn["agent_role"] for turn in turns] == ["verifier", "verifier", "answerer"]
    assert [turn["response_text"] for turn in turns] == ["first", "same-ts-later-index", "later"]


def test_build_integrity_report_counts_missing_tokens_and_non_monotonic_context() -> None:
    episode_payload = {
        "trajectory": {
            "agent_trajectories": {
                "verifier": [
                    {
                        "turn_index": 0,
                        "timestamp": 1.0,
                        "messages": [{"role": "user", "content": "a"}],
                        "token_ids": [1, 2],
                        "logprobs": [-0.1],
                    },
                    {
                        "turn_index": 1,
                        "timestamp": 2.0,
                        "messages": [],
                        "token_ids": None,
                        "logprobs": None,
                    },
                ]
            }
        }
    }

    report = build_integrity_report(episode_payload)

    assert report["token_ids_none_turns"] == 1
    assert report["token_logprobs_mismatch_turns"] == 2
    assert report["context_monotonic_by_agent"] == {"verifier": False}


def test_collect_tree_rewards_gathers_finite_rewards_from_pilot_and_branches() -> None:
    tree_payload = {
        "pilot_result": {"final_reward": 1.0},
        "branch_results": [
            {"episode_result": {"final_reward": 0.0}},
            {"episode_result": {"final_reward": None}},
            {"episode_result": {"final_reward": float("nan")}},
        ],
    }

    rewards = collect_tree_rewards(tree_payload)

    assert rewards == [1.0, 0.0]


def test_compute_prefix_sharing_uses_replayed_token_count_over_total_branch_tokens() -> None:
    tree_payload = {
        "branch_results": [
            {
                "episode_result": {
                    "trajectory": {
                        "agent_trajectories": {
                            "verifier": [
                                {
                                    "turn_index": 0,
                                    "timestamp": 1.0,
                                    "token_ids": [1, 2],
                                    "metadata": {"replayed": True},
                                },
                                {
                                    "turn_index": 1,
                                    "timestamp": 2.0,
                                    "token_ids": [3],
                                    "metadata": {},
                                },
                            ]
                        }
                    }
                }
            },
            {
                "episode_result": {
                    "trajectory": {
                        "agent_trajectories": {
                            "answerer": [
                                {
                                    "turn_index": 0,
                                    "timestamp": 3.0,
                                    "token_ids": [4, 5, 6],
                                    "metadata": {"replayed": True},
                                }
                            ]
                        }
                    }
                }
            },
        ]
    }

    stats = compute_prefix_sharing(tree_payload)

    assert stats["replayed_tokens"] == 5
    assert stats["total_branch_tokens"] == 6
    assert stats["prefix_sharing_rate"] == 5 / 6
