#!/usr/bin/env python3
"""Trajectory JSON 可视化与完整性审查。"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 run_real_validation 输出")
    parser.add_argument("--input", type=Path, required=True, help="阶段 1 输出 JSON 文件")
    parser.add_argument("--html", type=Path, default=None, help="可选 HTML 报告输出路径")
    parser.add_argument("--limit", type=int, default=10, help="终端最多展示多少条 episode")
    return parser.parse_args()


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _preview(text: str, limit: int = 60) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _flatten_turns(episode: dict[str, Any]) -> list[dict[str, Any]]:
    trajectories = episode.get("trajectory", {}).get("agent_trajectories", {})
    turns: list[dict[str, Any]] = []
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
    turns.sort(key=lambda x: (_safe_float(x.get("timestamp", 0.0)), _safe_int(x.get("turn_index", 0))))
    return turns


def _message_len(turn: dict[str, Any]) -> int:
    messages = turn.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def _is_token_logprob_consistent(turn: dict[str, Any]) -> tuple[bool, int | None, int | None]:
    token_ids = turn.get("token_ids")
    logprobs = turn.get("logprobs")
    token_len = len(token_ids) if isinstance(token_ids, list) else None
    logprob_len = len(logprobs) if isinstance(logprobs, list) else None
    if token_len is None or logprob_len is None:
        return False, token_len, logprob_len
    return token_len == logprob_len, token_len, logprob_len


def _build_integrity_report(episode: dict[str, Any]) -> dict[str, Any]:
    turns = _flatten_turns(episode)
    token_ids_none_turns = 0
    mismatch_turns = 0

    context_monotonic_by_agent: dict[str, bool] = {}
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for turn in turns:
        by_agent[turn["agent_role"]].append(turn)
        if turn.get("token_ids") is None:
            token_ids_none_turns += 1
        is_ok, _, _ = _is_token_logprob_consistent(turn)
        if not is_ok:
            mismatch_turns += 1

    for role, role_turns in by_agent.items():
        role_turns.sort(key=lambda x: int(x.get("turn_index", 0)))
        msg_sizes = [_message_len(turn) for turn in role_turns]
        is_monotonic = all(a <= b for a, b in zip(msg_sizes, msg_sizes[1:]))
        context_monotonic_by_agent[role] = is_monotonic

    return {
        "token_ids_none_turns": token_ids_none_turns,
        "token_logprobs_mismatch_turns": mismatch_turns,
        "context_monotonic_by_agent": context_monotonic_by_agent,
    }


def _timeline_lines(episode: dict[str, Any]) -> list[str]:
    turns = _flatten_turns(episode)
    lines: list[str] = []
    for idx, turn in enumerate(turns):
        role = turn.get("agent_role", "unknown")
        text = str(turn.get("response_text", ""))
        detail = ""

        verify = _extract_tag(text, "verify")
        search = _extract_tag(text, "search")
        answer = _extract_tag(text, "answer")

        if verify:
            detail = f"decision={verify}"
        elif search:
            detail = f"query={search}"
        elif answer:
            detail = f"answer={answer}"

        suffix = f" ({detail})" if detail else ""
        lines.append(f"  [{idx}] {role} -> \"{_preview(text)}\"{suffix}")
    return lines


def _terminal_report(payload: dict[str, Any], limit: int) -> str:
    episodes = payload.get("episodes", [])
    summary = payload.get("summary", {})
    run_mode = payload.get("run_mode", "unknown")

    lines: list[str] = []
    lines.append("=== 轨迹可视化 ===")
    lines.append(f"run_mode: {run_mode}")
    lines.append(f"total_episodes: {summary.get('total_episodes', len(episodes))}")
    lines.append(f"success_rate: {summary.get('success_rate', 0):.2%}")
    lines.append(f"avg_turns: {summary.get('avg_turns', 0):.2f}")
    lines.append("")

    for idx, episode in enumerate(episodes[:limit]):
        episode_id = episode.get("trajectory", {}).get("episode_id", "unknown")
        final_reward = episode.get("final_reward")
        turns = _flatten_turns(episode)
        lines.append(f"Episode {episode_id} (reward={final_reward}, {len(turns)} turns):")
        lines.extend(_timeline_lines(episode))

        integrity = _build_integrity_report(episode)
        token_none = integrity["token_ids_none_turns"]
        mismatch = integrity["token_logprobs_mismatch_turns"]

        if token_none > 0:
            lines.append(f"  {RED}WARNING token_ids=None turns: {token_none}{RESET}")
        if mismatch > 0:
            lines.append(f"  {RED}WARNING token/logprobs mismatch turns: {mismatch}{RESET}")

        non_mono_roles = [
            role for role, ok in integrity["context_monotonic_by_agent"].items() if not ok
        ]
        if non_mono_roles:
            roles = ", ".join(non_mono_roles)
            lines.append(f"  {YELLOW}WARNING context 非单调 agent: {roles}{RESET}")

        lines.append("  turn integrity:")
        for turn in turns:
            role = turn.get("agent_role", "unknown")
            turn_idx = turn.get("turn_index", "?")
            is_ok, token_len, logprob_len = _is_token_logprob_consistent(turn)
            msg_len = _message_len(turn)
            status = "OK" if is_ok else "WARN"
            lines.append(
                f"    - {role}[{turn_idx}] token_len={token_len} logprob_len={logprob_len} msg_len={msg_len} status={status}"
            )

        lines.append("")

    if len(episodes) > limit:
        lines.append(f"... 省略 {len(episodes) - limit} 条 episode（可用 --limit 调整）")

    lines.append("=== 统计摘要 ===")
    lines.append(f"成功/失败: {summary.get('success_count', 0)}/{summary.get('failure_count', 0)}")

    agent_calls = summary.get("agent_call_counts", {})
    lines.append(f"agent 调用次数: {json.dumps(agent_calls, ensure_ascii=False)}")

    agent_tokens = summary.get("agent_avg_generated_tokens", {})
    lines.append(f"agent 平均生成 token: {json.dumps(agent_tokens, ensure_ascii=False)}")

    return "\n".join(lines)


def _html_report(payload: dict[str, Any], limit: int) -> str:
    episodes = payload.get("episodes", [])
    summary = payload.get("summary", {})

    blocks: list[str] = []
    blocks.append("<h1>Trajectory Validation Report</h1>")
    blocks.append("<h2>Summary</h2>")
    blocks.append("<ul>")
    blocks.append(f"<li>run_mode: {html.escape(str(payload.get('run_mode', 'unknown')))}</li>")
    blocks.append(f"<li>total_episodes: {html.escape(str(summary.get('total_episodes', len(episodes))))}</li>")
    success_rate = f"{summary.get('success_rate', 0):.2%}"
    avg_turns = f"{summary.get('avg_turns', 0):.2f}"
    blocks.append(f"<li>success_rate: {html.escape(success_rate)}</li>")
    blocks.append(f"<li>avg_turns: {html.escape(avg_turns)}</li>")
    blocks.append("</ul>")

    for episode in episodes[:limit]:
        episode_id = episode.get("trajectory", {}).get("episode_id", "unknown")
        final_reward = episode.get("final_reward")
        turns = _flatten_turns(episode)
        integrity = _build_integrity_report(episode)

        blocks.append(f"<h3>Episode {html.escape(str(episode_id))}</h3>")
        blocks.append(f"<p>reward={html.escape(str(final_reward))}, turns={len(turns)}</p>")
        blocks.append("<pre>")
        for line in _timeline_lines(episode):
            blocks.append(html.escape(line))
        blocks.append("</pre>")

        warn_items: list[str] = []
        if integrity["token_ids_none_turns"] > 0:
            warn_items.append(f"token_ids=None: {integrity['token_ids_none_turns']}")
        if integrity["token_logprobs_mismatch_turns"] > 0:
            warn_items.append(f"token/logprobs mismatch: {integrity['token_logprobs_mismatch_turns']}")

        non_mono = [
            role for role, ok in integrity["context_monotonic_by_agent"].items() if not ok
        ]
        if non_mono:
            warn_items.append(f"context non-monotonic: {', '.join(non_mono)}")

        if warn_items:
            blocks.append("<p style='color:#b00020'><strong>Warnings:</strong> " + html.escape("; ".join(warn_items)) + "</p>")
        else:
            blocks.append("<p style='color:#0a7d2e'><strong>Warnings:</strong> none</p>")

        blocks.append("<table border='1' cellspacing='0' cellpadding='6'>")
        blocks.append("<tr><th>agent</th><th>turn</th><th>token_len</th><th>logprob_len</th><th>msg_len</th><th>status</th></tr>")
        for turn in turns:
            is_ok, token_len, logprob_len = _is_token_logprob_consistent(turn)
            status = "OK" if is_ok else "WARN"
            blocks.append(
                "<tr>"
                f"<td>{html.escape(str(turn.get('agent_role', 'unknown')))}</td>"
                f"<td>{html.escape(str(turn.get('turn_index', '?')))}</td>"
                f"<td>{html.escape(str(token_len))}</td>"
                f"<td>{html.escape(str(logprob_len))}</td>"
                f"<td>{html.escape(str(_message_len(turn)))}</td>"
                f"<td>{html.escape(status)}</td>"
                "</tr>"
            )
        blocks.append("</table>")

    body = "\n".join(blocks)
    doc = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Trajectory Report</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 24px; }}
    h1, h2, h3 {{ margin-top: 1.2em; }}
    pre {{ background: #f6f8fa; padding: 12px; border-radius: 8px; }}
    table {{ margin: 8px 0 20px 0; width: 100%; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    return doc


def _compute_extra_stats(payload: dict[str, Any]) -> dict[str, float]:
    episodes = payload.get("episodes", [])
    turn_counts: list[int] = []
    token_lengths: list[int] = []
    for episode in episodes:
        turns = _flatten_turns(episode)
        turn_counts.append(len(turns))
        for turn in turns:
            token_ids = turn.get("token_ids")
            if isinstance(token_ids, list):
                token_lengths.append(len(token_ids))
    return {
        "episode_avg_turns": statistics.mean(turn_counts) if turn_counts else 0.0,
        "turn_avg_tokens": statistics.mean(token_lengths) if token_lengths else 0.0,
    }


def main() -> int:
    args = _parse_args()
    if args.limit < 1:
        raise ValueError("--limit 必须 >= 1")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    text = _terminal_report(payload, limit=args.limit)
    print(text)

    if args.html is not None:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(_html_report(payload, limit=args.limit), encoding="utf-8")
        print(f"\nHTML 报告已写入: {args.html}")

    extra = _compute_extra_stats(payload)
    print(
        "\n附加统计: "
        f"episode_avg_turns={extra['episode_avg_turns']:.2f}, "
        f"turn_avg_tokens={extra['turn_avg_tokens']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
