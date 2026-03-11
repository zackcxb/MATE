from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


def extract_tag(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def preview_text(text: str, limit: int = 60) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def collect_all_turns(episode_payload: dict[str, Any]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    trajectories = episode_payload.get("trajectory", {}).get("agent_trajectories", {})
    if not isinstance(trajectories, dict):
        return turns
    for role, role_turns in trajectories.items():
        if not isinstance(role_turns, list):
            continue
        for turn in role_turns:
            if not isinstance(turn, dict):
                continue
            item = dict(turn)
            item["agent_role"] = role
            turns.append(item)

    turns.sort(
        key=lambda item: (
            safe_float(item.get("timestamp", 0.0), default=0.0),
            safe_int(item.get("turn_index", 0), default=0),
        )
    )
    return turns


def message_len(turn: dict[str, Any]) -> int:
    messages = turn.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def is_token_logprob_consistent(turn: dict[str, Any]) -> tuple[bool, int | None, int | None]:
    token_ids = turn.get("token_ids")
    logprobs = turn.get("logprobs")
    token_len = len(token_ids) if isinstance(token_ids, list) else None
    logprob_len = len(logprobs) if isinstance(logprobs, list) else None
    if token_len is None or logprob_len is None:
        return False, token_len, logprob_len
    return token_len == logprob_len, token_len, logprob_len


def build_integrity_report(episode_payload: dict[str, Any]) -> dict[str, Any]:
    turns = collect_all_turns(episode_payload)
    token_ids_none_turns = 0
    mismatch_turns = 0

    context_monotonic_by_agent: dict[str, bool] = {}
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for turn in turns:
        by_agent[turn["agent_role"]].append(turn)
        if turn.get("token_ids") is None:
            token_ids_none_turns += 1
        is_ok, _, _ = is_token_logprob_consistent(turn)
        if not is_ok:
            mismatch_turns += 1

    for role, role_turns in by_agent.items():
        role_turns.sort(key=lambda item: safe_int(item.get("turn_index", 0), default=0))
        msg_sizes = [message_len(turn) for turn in role_turns]
        is_monotonic = all(a <= b for a, b in zip(msg_sizes, msg_sizes[1:]))
        context_monotonic_by_agent[role] = is_monotonic

    return {
        "token_ids_none_turns": token_ids_none_turns,
        "token_logprobs_mismatch_turns": mismatch_turns,
        "context_monotonic_by_agent": context_monotonic_by_agent,
    }


def timeline_lines(episode_payload: dict[str, Any], preview_limit: int = 60) -> list[str]:
    turns = collect_all_turns(episode_payload)
    lines: list[str] = []
    for idx, turn in enumerate(turns):
        role = turn.get("agent_role", "unknown")
        text = str(turn.get("response_text", ""))
        detail = ""

        verify = extract_tag(text, "verify")
        search = extract_tag(text, "search")
        answer = extract_tag(text, "answer")

        if verify:
            detail = f"decision={verify}"
        elif search:
            detail = f"query={search}"
        elif answer:
            detail = f"answer={answer}"

        suffix = f" ({detail})" if detail else ""
        lines.append(f"  [{idx}] {role} -> \"{preview_text(text, preview_limit)}\"{suffix}")
    return lines


def collect_tree_rewards(tree_payload: dict[str, Any]) -> list[float]:
    rewards: list[float] = []
    pilot_reward = tree_payload.get("pilot_result", {}).get("final_reward")
    if pilot_reward is not None and isinstance(pilot_reward, (int, float)) and math.isfinite(pilot_reward):
        rewards.append(float(pilot_reward))
    for branch in tree_payload.get("branch_results", []):
        branch_reward = branch.get("episode_result", {}).get("final_reward")
        if branch_reward is not None and isinstance(branch_reward, (int, float)) and math.isfinite(branch_reward):
            rewards.append(float(branch_reward))
    return rewards


def compute_prefix_sharing(tree_payload: dict[str, Any]) -> dict[str, Any]:
    branches = tree_payload.get("branch_results", [])
    total_tokens = 0
    replayed_tokens = 0

    for branch in branches:
        branch_turns = collect_all_turns(branch.get("episode_result", {}))
        for turn in branch_turns:
            token_ids = turn.get("token_ids")
            token_count = len(token_ids) if isinstance(token_ids, list) else 0
            total_tokens += token_count
            metadata = turn.get("metadata", {})
            if metadata.get("replayed") is True:
                replayed_tokens += token_count

    rate = replayed_tokens / total_tokens if total_tokens > 0 else 0.0
    return {
        "replayed_tokens": replayed_tokens,
        "total_branch_tokens": total_tokens,
        "prefix_sharing_rate": rate,
    }
