#!/usr/bin/env python3
"""Trajectory JSON 可视化与完整性审查。

支持 V0.1 (schema v1.0) 和 V0.2 (schema v2.0) 两种格式。

V0.1: trajectory_validation.json — episodes[] 列表
V0.2: tree_validation_trajectories.json — smoke/replay/multi_prompt 树结构
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scripts._trajectory_utils import (
    build_integrity_report as _build_integrity_report,
    collect_all_turns as _flatten_turns,
    collect_all_turns as _v2_collect_all_turns,
    collect_tree_rewards as _v2_collect_tree_rewards,
    compute_prefix_sharing as _v2_prefix_sharing,
    extract_tag as _extract_tag,
    is_token_logprob_consistent as _is_token_logprob_consistent,
    message_len as _message_len,
    preview_text as _preview,
    safe_float as _safe_float,
    safe_int as _safe_int,
    timeline_lines as _timeline_lines,
)

RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# V0.2 preview limit (response text truncation)
# ---------------------------------------------------------------------------
_V2_PREVIEW_LIMIT = 120


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 run_real_validation / run_tree_validation 输出")
    parser.add_argument("--input", type=Path, required=True, help="轨迹 JSON 文件（v1.0 或 v2.0）")
    parser.add_argument("--html", type=Path, default=None, help="可选 HTML 报告输出路径")
    parser.add_argument("--limit", type=int, default=5, help="最多展示多少条 episode/branch/tree")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Schema version detection
# ---------------------------------------------------------------------------

def _detect_schema(payload: dict[str, Any]) -> str:
    """返回 '1.0' 或 '2.0'。"""
    explicit = payload.get("schema_version")
    if explicit is not None:
        return str(explicit)
    # V0.1 has episodes[]; V0.2 has smoke/replay/multi_prompt sections
    if "episodes" in payload:
        return "1.0"
    if any(k in payload for k in ("smoke", "replay", "multi_prompt")):
        return "2.0"
    # fallback: treat as v1.0
    return "1.0"


# ===================================================================
# V0.1 terminal + HTML (original code, unchanged logic)
# ===================================================================

def _terminal_report_v1(payload: dict[str, Any], limit: int) -> str:
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


def _html_report_v1(payload: dict[str, Any], limit: int) -> str:
    episodes = payload.get("episodes", [])
    summary = payload.get("summary", {})

    blocks: list[str] = []
    blocks.append("<h1>Trajectory Validation Report</h1>")
    blocks.append("<h2>Summary</h2>")
    blocks.append("<ul>")
    blocks.append(f"<li>run_mode: {html_mod.escape(str(payload.get('run_mode', 'unknown')))}</li>")
    blocks.append(f"<li>total_episodes: {html_mod.escape(str(summary.get('total_episodes', len(episodes))))}</li>")
    success_rate = f"{summary.get('success_rate', 0):.2%}"
    avg_turns = f"{summary.get('avg_turns', 0):.2f}"
    blocks.append(f"<li>success_rate: {html_mod.escape(success_rate)}</li>")
    blocks.append(f"<li>avg_turns: {html_mod.escape(avg_turns)}</li>")
    blocks.append("</ul>")

    for episode in episodes[:limit]:
        episode_id = episode.get("trajectory", {}).get("episode_id", "unknown")
        final_reward = episode.get("final_reward")
        turns = _flatten_turns(episode)
        integrity = _build_integrity_report(episode)

        blocks.append(f"<h3>Episode {html_mod.escape(str(episode_id))}</h3>")
        blocks.append(f"<p>reward={html_mod.escape(str(final_reward))}, turns={len(turns)}</p>")
        blocks.append("<pre>")
        for line in _timeline_lines(episode):
            blocks.append(html_mod.escape(line))
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
            blocks.append("<p style='color:#b00020'><strong>Warnings:</strong> " + html_mod.escape("; ".join(warn_items)) + "</p>")
        else:
            blocks.append("<p style='color:#0a7d2e'><strong>Warnings:</strong> none</p>")

        blocks.append("<table border='1' cellspacing='0' cellpadding='6'>")
        blocks.append("<tr><th>agent</th><th>turn</th><th>token_len</th><th>logprob_len</th><th>msg_len</th><th>status</th></tr>")
        for turn in turns:
            is_ok, token_len, logprob_len = _is_token_logprob_consistent(turn)
            status = "OK" if is_ok else "WARN"
            blocks.append(
                "<tr>"
                f"<td>{html_mod.escape(str(turn.get('agent_role', 'unknown')))}</td>"
                f"<td>{html_mod.escape(str(turn.get('turn_index', '?')))}</td>"
                f"<td>{html_mod.escape(str(token_len))}</td>"
                f"<td>{html_mod.escape(str(logprob_len))}</td>"
                f"<td>{html_mod.escape(str(_message_len(turn)))}</td>"
                f"<td>{html_mod.escape(status)}</td>"
                "</tr>"
            )
        blocks.append("</table>")

    return "\n".join(blocks)


def _compute_extra_stats_v1(payload: dict[str, Any]) -> dict[str, float]:
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


# ===================================================================
# V0.2 helpers
# ===================================================================

def _v2_tree_terminal_lines(
    tree_payload: dict[str, Any],
    limit: int,
    *,
    label: str = "",
) -> list[str]:
    """生成单棵树的终端诊断行。"""
    lines: list[str] = []

    prompt = tree_payload.get("prompt", "")
    meta = tree_payload.get("tree_metadata", {})
    pilot = tree_payload.get("pilot_result", {})
    branches = tree_payload.get("branch_results", [])
    rewards = _v2_collect_tree_rewards(tree_payload)

    pilot_reward = pilot.get("final_reward")
    pilot_turns = _v2_collect_all_turns(pilot)
    n_branches = len(branches)

    # ---- tree-level summary ----
    header = f"--- 树 {label}" if label else "--- 树"
    lines.append(header)
    lines.append(f"  prompt: \"{_preview(prompt, _V2_PREVIEW_LIMIT)}\"")
    lines.append(f"  pilot turns: {len(pilot_turns)}, pilot reward: {pilot_reward}")
    lines.append(f"  分支数: {n_branches}, k_branches: {meta.get('k_branches', '?')}")
    if rewards:
        r_mean = statistics.mean(rewards)
        r_min = min(rewards)
        r_max = max(rewards)
        lines.append(f"  reward: mean={r_mean:.3f} min={r_min:.3f} max={r_max:.3f} (n={len(rewards)})")

    # ---- prefix sharing ----
    prefix = _v2_prefix_sharing(tree_payload)
    lines.append(
        f"  前缀共享: replayed_tokens={prefix['replayed_tokens']} / "
        f"total_branch_tokens={prefix['total_branch_tokens']} "
        f"rate={prefix['prefix_sharing_rate']:.2%}"
    )

    # ---- pilot timeline ----
    lines.append("  Pilot timeline:")
    lines.extend(["  " + l for l in _timeline_lines(pilot, preview_limit=_V2_PREVIEW_LIMIT)])

    # ---- pilot integrity ----
    pilot_integrity = _build_integrity_report(pilot)
    _v2_append_integrity_warnings(lines, pilot_integrity, indent="  ", label="pilot")
    _v2_append_turn_integrity(lines, pilot_turns, indent="    ")

    # ---- branch diagnostics ----
    shown_branches = branches[:limit]
    for bidx, branch in enumerate(shown_branches):
        bt = branch.get("branch_turn", "?")
        ba = branch.get("branch_agent_role", "?")
        br_ep = branch.get("episode_result", {})
        br_reward = br_ep.get("final_reward")
        br_turns = _v2_collect_all_turns(br_ep)

        lines.append(f"  Branch [{bidx}] branch_turn={bt} role={ba} reward={br_reward} turns={len(br_turns)}")

        # replay marker check
        replayed_ok = 0
        replayed_bad = 0
        for pos, turn in enumerate(br_turns):
            metadata = turn.get("metadata", {})
            is_replayed = metadata.get("replayed") is True
            if isinstance(bt, int) and pos < bt:
                if is_replayed:
                    replayed_ok += 1
                else:
                    replayed_bad += 1

        if isinstance(bt, int):
            expected_replayed = bt
            lines.append(
                f"    replayed 标记: {replayed_ok}/{expected_replayed} OK"
                + (f", {RED}{replayed_bad} 缺失{RESET}" if replayed_bad > 0 else "")
            )

        # branch turn-by-turn timeline
        for pos, turn in enumerate(br_turns):
            role = turn.get("agent_role", "unknown")
            text = str(turn.get("response_text", ""))
            metadata = turn.get("metadata", {})
            is_replayed = metadata.get("replayed") is True
            marker = f"{GREEN}R{RESET}" if is_replayed else f"{YELLOW}F{RESET}"

            detail = ""
            verify = _extract_tag(text, "verify")
            search = _extract_tag(text, "search")
            answer = _extract_tag(text, "answer")
            if verify:
                detail = f"decision={verify}"
            elif search:
                detail = f"query={_preview(search, 60)}"
            elif answer:
                detail = f"answer={_preview(answer, 60)}"
            suffix = f" ({detail})" if detail else ""

            tids = turn.get("token_ids")
            tok_n = len(tids) if isinstance(tids, list) else 0
            lines.append(
                f"    [{marker}][{pos}] {role} tokens={tok_n} -> "
                f"\"{_preview(text, _V2_PREVIEW_LIMIT)}\"{suffix}"
            )

        # branch integrity
        br_integrity = _build_integrity_report(br_ep)
        _v2_append_integrity_warnings(lines, br_integrity, indent="    ", label=f"branch[{bidx}]")

    omitted = n_branches - len(shown_branches)
    if omitted > 0:
        lines.append(f"  [{omitted} 条分支已省略]")

    return lines


def _v2_append_integrity_warnings(
    lines: list[str],
    integrity: dict[str, Any],
    *,
    indent: str = "  ",
    label: str = "",
) -> None:
    prefix = f"{indent}{label} " if label else indent
    token_none = integrity["token_ids_none_turns"]
    mismatch = integrity["token_logprobs_mismatch_turns"]
    if token_none > 0:
        lines.append(f"{prefix}{RED}WARNING token_ids=None turns: {token_none}{RESET}")
    if mismatch > 0:
        lines.append(f"{prefix}{RED}WARNING token/logprobs mismatch turns: {mismatch}{RESET}")
    non_mono_roles = [
        role for role, ok in integrity["context_monotonic_by_agent"].items() if not ok
    ]
    if non_mono_roles:
        roles_str = ", ".join(non_mono_roles)
        lines.append(f"{prefix}{YELLOW}WARNING context 非单调 agent: {roles_str}{RESET}")


def _v2_append_turn_integrity(
    lines: list[str],
    turns: list[dict[str, Any]],
    *,
    indent: str = "    ",
) -> None:
    for turn in turns:
        role = turn.get("agent_role", "unknown")
        turn_idx = turn.get("turn_index", "?")
        is_ok, token_len, logprob_len = _is_token_logprob_consistent(turn)
        msg_len = _message_len(turn)
        status = "OK" if is_ok else "WARN"
        lines.append(
            f"{indent}- {role}[{turn_idx}] token_len={token_len} logprob_len={logprob_len} "
            f"msg_len={msg_len} status={status}"
        )


# ---------------------------------------------------------------------------
# V0.2 terminal report
# ---------------------------------------------------------------------------

def _terminal_report_v2(payload: dict[str, Any], limit: int) -> str:
    run_mode = payload.get("run_mode", "unknown")
    lines: list[str] = []
    lines.append("=== V0.2 树状轨迹可视化 ===")
    lines.append(f"schema_version: {payload.get('schema_version', '2.0')}")
    lines.append(f"run_mode: {run_mode}")
    lines.append(f"timestamp: {payload.get('timestamp', 'N/A')}")
    lines.append("")

    section_count = 0

    # ---- smoke ----
    smoke = payload.get("smoke")
    if isinstance(smoke, dict) and smoke:
        lines.append("== Round 1: Smoke ==")
        lines.extend(_v2_tree_terminal_lines(smoke, limit=limit, label="smoke"))
        lines.append("")
        section_count += 1

    # ---- replay ----
    replay = payload.get("replay")
    if isinstance(replay, dict) and replay:
        lines.append("== Round 2: Replay ==")
        lines.extend(_v2_tree_terminal_lines(replay, limit=limit, label="replay"))
        lines.append("")
        section_count += 1

    # ---- multi_prompt ----
    multi = payload.get("multi_prompt")
    if isinstance(multi, list) and multi:
        lines.append("== Round 3: Multi-prompt ==")
        shown_trees = multi[:limit]
        all_rewards: list[float] = []
        prefix_rates: list[float] = []

        for tidx, tree_entry in enumerate(shown_trees):
            tp = tree_entry.get("tree_payload", {})
            pidx = tree_entry.get("prompt_index", tidx)
            tree_label = f"multi[{pidx}]"
            lines.extend(_v2_tree_terminal_lines(tp, limit=limit, label=tree_label))
            all_rewards.extend(_v2_collect_tree_rewards(tp))
            ps = _v2_prefix_sharing(tp)
            prefix_rates.append(ps["prefix_sharing_rate"])
            lines.append("")

        omitted_trees = len(multi) - len(shown_trees)
        if omitted_trees > 0:
            lines.append(f"[{omitted_trees} 棵树已省略（可用 --limit 调整）]")

        # aggregate stats
        lines.append("  Multi-prompt 聚合统计:")
        lines.append(f"    总树数: {len(multi)}")
        if all_rewards:
            lines.append(f"    reward 均值: {statistics.mean(all_rewards):.3f}")
        if prefix_rates:
            lines.append(f"    前缀共享率均值: {statistics.mean(prefix_rates):.2%}")
        lines.append("")
        section_count += 1

    if section_count == 0:
        lines.append("未找到 smoke/replay/multi_prompt 数据段。")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# V0.2 HTML report
# ---------------------------------------------------------------------------

def _v2_tree_html_blocks(
    tree_payload: dict[str, Any],
    limit: int,
    *,
    label: str = "",
) -> list[str]:
    """为单棵树生成 HTML 片段。"""
    blocks: list[str] = []

    prompt = tree_payload.get("prompt", "")
    meta = tree_payload.get("tree_metadata", {})
    pilot = tree_payload.get("pilot_result", {})
    branches = tree_payload.get("branch_results", [])
    rewards = _v2_collect_tree_rewards(tree_payload)
    prefix = _v2_prefix_sharing(tree_payload)

    pilot_reward = pilot.get("final_reward")
    pilot_turns = _v2_collect_all_turns(pilot)
    n_branches = len(branches)

    heading = f"Tree {html_mod.escape(label)}" if label else "Tree"
    blocks.append(f"<h3>{heading}</h3>")
    blocks.append("<ul>")
    blocks.append(f"<li>prompt: {html_mod.escape(_preview(prompt, _V2_PREVIEW_LIMIT))}</li>")
    blocks.append(f"<li>pilot turns: {len(pilot_turns)}, pilot reward: {html_mod.escape(str(pilot_reward))}</li>")
    blocks.append(f"<li>branches: {n_branches}, k_branches: {html_mod.escape(str(meta.get('k_branches', '?')))}</li>")
    if rewards:
        r_mean = statistics.mean(rewards)
        blocks.append(f"<li>reward mean={r_mean:.3f} min={min(rewards):.3f} max={max(rewards):.3f} (n={len(rewards)})</li>")
    blocks.append(
        f"<li>prefix sharing: replayed={prefix['replayed_tokens']} / total={prefix['total_branch_tokens']} "
        f"rate={prefix['prefix_sharing_rate']:.2%}</li>"
    )
    blocks.append("</ul>")

    # pilot timeline
    blocks.append("<h4>Pilot Timeline</h4>")
    blocks.append("<pre>")
    for line in _timeline_lines(pilot, preview_limit=_V2_PREVIEW_LIMIT):
        blocks.append(html_mod.escape(line))
    blocks.append("</pre>")

    # pilot integrity table
    pilot_integrity = _build_integrity_report(pilot)
    blocks.extend(_v2_html_integrity_section(pilot_turns, pilot_integrity, label="Pilot"))

    # branches
    shown = branches[:limit]
    for bidx, branch in enumerate(shown):
        bt = branch.get("branch_turn", "?")
        ba = branch.get("branch_agent_role", "?")
        br_ep = branch.get("episode_result", {})
        br_reward = br_ep.get("final_reward")
        br_turns = _v2_collect_all_turns(br_ep)

        blocks.append(f"<h4>Branch [{bidx}] turn={html_mod.escape(str(bt))} role={html_mod.escape(str(ba))} "
                       f"reward={html_mod.escape(str(br_reward))}</h4>")

        # replay summary
        replayed_ok = 0
        replayed_bad = 0
        if isinstance(bt, int):
            for pos, turn in enumerate(br_turns):
                md = turn.get("metadata", {})
                if pos < bt:
                    if md.get("replayed") is True:
                        replayed_ok += 1
                    else:
                        replayed_bad += 1

            color = "#0a7d2e" if replayed_bad == 0 else "#b00020"
            blocks.append(
                f"<p style='color:{color}'>replayed: {replayed_ok}/{bt} OK"
                + (f", {replayed_bad} missing" if replayed_bad > 0 else "")
                + "</p>"
            )

        # branch timeline
        blocks.append("<pre>")
        for pos, turn in enumerate(br_turns):
            turn_role = turn.get("agent_role", "unknown")
            text = str(turn.get("response_text", ""))
            md = turn.get("metadata", {})
            is_rep = md.get("replayed") is True
            marker = "R" if is_rep else "F"
            tids = turn.get("token_ids")
            tok_n = len(tids) if isinstance(tids, list) else 0

            detail = ""
            verify = _extract_tag(text, "verify")
            search = _extract_tag(text, "search")
            answer = _extract_tag(text, "answer")
            if verify:
                detail = f" (decision={html_mod.escape(verify)})"
            elif search:
                detail = f" (query={html_mod.escape(_preview(search, 60))})"
            elif answer:
                detail = f" (answer={html_mod.escape(_preview(answer, 60))})"

            blocks.append(
                f"  [{marker}][{pos}] {html_mod.escape(turn_role)} tokens={tok_n} -&gt; "
                f"&quot;{html_mod.escape(_preview(text, _V2_PREVIEW_LIMIT))}&quot;{detail}"
            )
        blocks.append("</pre>")

        br_integrity = _build_integrity_report(br_ep)
        blocks.extend(_v2_html_integrity_section(br_turns, br_integrity, label=f"Branch[{bidx}]"))

    omitted = n_branches - len(shown)
    if omitted > 0:
        blocks.append(f"<p><em>[{omitted} branches omitted]</em></p>")

    return blocks


def _v2_html_integrity_section(
    turns: list[dict[str, Any]],
    integrity: dict[str, Any],
    *,
    label: str = "",
) -> list[str]:
    blocks: list[str] = []
    warn_items: list[str] = []
    if integrity["token_ids_none_turns"] > 0:
        warn_items.append(f"token_ids=None: {integrity['token_ids_none_turns']}")
    if integrity["token_logprobs_mismatch_turns"] > 0:
        warn_items.append(f"token/logprobs mismatch: {integrity['token_logprobs_mismatch_turns']}")
    non_mono = [role for role, ok in integrity["context_monotonic_by_agent"].items() if not ok]
    if non_mono:
        warn_items.append(f"context non-monotonic: {', '.join(non_mono)}")

    if warn_items:
        blocks.append(f"<p style='color:#b00020'><strong>{html_mod.escape(label)} Warnings:</strong> "
                       + html_mod.escape("; ".join(warn_items)) + "</p>")
    else:
        blocks.append(f"<p style='color:#0a7d2e'><strong>{html_mod.escape(label)} Warnings:</strong> none</p>")

    blocks.append("<table border='1' cellspacing='0' cellpadding='6'>")
    blocks.append("<tr><th>agent</th><th>turn</th><th>token_len</th><th>logprob_len</th><th>msg_len</th><th>status</th></tr>")
    for turn in turns:
        is_ok, token_len, logprob_len = _is_token_logprob_consistent(turn)
        status = "OK" if is_ok else "WARN"
        blocks.append(
            "<tr>"
            f"<td>{html_mod.escape(str(turn.get('agent_role', 'unknown')))}</td>"
            f"<td>{html_mod.escape(str(turn.get('turn_index', '?')))}</td>"
            f"<td>{html_mod.escape(str(token_len))}</td>"
            f"<td>{html_mod.escape(str(logprob_len))}</td>"
            f"<td>{html_mod.escape(str(_message_len(turn)))}</td>"
            f"<td>{html_mod.escape(status)}</td>"
            "</tr>"
        )
    blocks.append("</table>")
    return blocks


def _html_report_v2(payload: dict[str, Any], limit: int) -> str:
    blocks: list[str] = []
    blocks.append("<h1>V0.2 Tree Trajectory Report</h1>")
    blocks.append("<h2>Summary</h2>")
    blocks.append("<ul>")
    blocks.append(f"<li>schema_version: {html_mod.escape(str(payload.get('schema_version', '2.0')))}</li>")
    blocks.append(f"<li>run_mode: {html_mod.escape(str(payload.get('run_mode', 'unknown')))}</li>")
    blocks.append(f"<li>timestamp: {html_mod.escape(str(payload.get('timestamp', 'N/A')))}</li>")
    blocks.append("</ul>")

    smoke = payload.get("smoke")
    if isinstance(smoke, dict) and smoke:
        blocks.append("<h2>Round 1: Smoke</h2>")
        blocks.extend(_v2_tree_html_blocks(smoke, limit=limit, label="smoke"))

    replay = payload.get("replay")
    if isinstance(replay, dict) and replay:
        blocks.append("<h2>Round 2: Replay</h2>")
        blocks.extend(_v2_tree_html_blocks(replay, limit=limit, label="replay"))

    multi = payload.get("multi_prompt")
    if isinstance(multi, list) and multi:
        blocks.append("<h2>Round 3: Multi-prompt</h2>")
        shown = multi[:limit]
        for tidx, tree_entry in enumerate(shown):
            tp = tree_entry.get("tree_payload", {})
            pidx = tree_entry.get("prompt_index", tidx)
            blocks.extend(_v2_tree_html_blocks(tp, limit=limit, label=f"multi[{pidx}]"))
        omitted = len(multi) - len(shown)
        if omitted > 0:
            blocks.append(f"<p><em>[{omitted} trees omitted]</em></p>")

    body = "\n".join(blocks)
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>V0.2 Tree Trajectory Report</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 24px; }}
    h1, h2, h3, h4 {{ margin-top: 1.2em; }}
    pre {{ background: #f6f8fa; padding: 12px; border-radius: 8px; }}
    table {{ margin: 8px 0 20px 0; width: 100%; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _compute_extra_stats_v2(payload: dict[str, Any]) -> dict[str, float]:
    """V0.2 附加统计: 汇总所有 tree_payload 的 turn / token 统计。"""
    turn_counts: list[int] = []
    token_lengths: list[int] = []

    def _collect_from_tree(tp: dict[str, Any]) -> None:
        if not isinstance(tp, dict):
            return
        # pilot
        pilot = tp.get("pilot_result", {})
        pilot_turns = _v2_collect_all_turns(pilot)
        turn_counts.append(len(pilot_turns))
        for turn in pilot_turns:
            tids = turn.get("token_ids")
            if isinstance(tids, list):
                token_lengths.append(len(tids))
        # branches
        for branch in tp.get("branch_results", []):
            br_turns = _v2_collect_all_turns(branch.get("episode_result", {}))
            turn_counts.append(len(br_turns))
            for turn in br_turns:
                tids = turn.get("token_ids")
                if isinstance(tids, list):
                    token_lengths.append(len(tids))

    smoke = payload.get("smoke")
    if isinstance(smoke, dict):
        _collect_from_tree(smoke)

    replay = payload.get("replay")
    if isinstance(replay, dict):
        _collect_from_tree(replay)

    multi = payload.get("multi_prompt")
    if isinstance(multi, list):
        for entry in multi:
            _collect_from_tree(entry.get("tree_payload", {}))

    return {
        "episode_avg_turns": statistics.mean(turn_counts) if turn_counts else 0.0,
        "turn_avg_tokens": statistics.mean(token_lengths) if token_lengths else 0.0,
    }


# ===================================================================
# Unified HTML wrapper
# ===================================================================

def _wrap_html(body_html: str, title: str = "Trajectory Report") -> str:
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html_mod.escape(title)}</title>
  <style>
    body {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; margin: 24px; }}
    h1, h2, h3, h4 {{ margin-top: 1.2em; }}
    pre {{ background: #f6f8fa; padding: 12px; border-radius: 8px; }}
    table {{ margin: 8px 0 20px 0; width: 100%; }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""


# ===================================================================
# Entrypoint
# ===================================================================

def main() -> int:
    args = _parse_args()
    if args.limit < 1:
        raise ValueError("--limit 必须 >= 1")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    schema = _detect_schema(payload)

    if schema.startswith("2"):
        # V0.2
        text = _terminal_report_v2(payload, limit=args.limit)
        print(text)

        if args.html is not None:
            args.html.parent.mkdir(parents=True, exist_ok=True)
            args.html.write_text(_html_report_v2(payload, limit=args.limit), encoding="utf-8")
            print(f"\nHTML 报告已写入: {args.html}")

        extra = _compute_extra_stats_v2(payload)
    else:
        # V0.1
        text = _terminal_report_v1(payload, limit=args.limit)
        print(text)

        if args.html is not None:
            args.html.parent.mkdir(parents=True, exist_ok=True)
            body = _html_report_v1(payload, limit=args.limit)
            args.html.write_text(_wrap_html(body, "Trajectory Report"), encoding="utf-8")
            print(f"\nHTML 报告已写入: {args.html}")

        extra = _compute_extra_stats_v1(payload)

    print(
        "\n附加统计: "
        f"episode_avg_turns={extra['episode_avg_turns']:.2f}, "
        f"turn_avg_tokens={extra['turn_avg_tokens']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
