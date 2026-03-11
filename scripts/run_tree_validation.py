#!/usr/bin/env python3
"""
V0.2 树状分支验证脚本。

使用 tree_rollout 对 DrMAS Search MAS 进行树结构轨迹采集，
验证树结构完整性、重放标记正确性以及前缀共享指标。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import tempfile
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_real_validation as _v0
from scripts._trajectory_utils import (
    collect_all_turns as _collect_all_turns,
    collect_tree_rewards as _collect_tree_rewards,
    compute_prefix_sharing as _compute_prefix_sharing,
)

from mate.trajectory import (
    AgentPipeConfig,
    FunctionRewardProvider,
    InferenceBackend,
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
    TreeEpisodeResult,
    BranchResult,
    VLLMBackend,
    tree_rollout,
    parallel_rollout,
)

DEFAULT_OUTPUT = Path("artifacts/tree_validation.json")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_tree_structure(tree_payload: dict, roles: list[str]) -> dict:
    """Check pilot completeness, branch fields, tree_metadata, episode_id uniqueness."""
    issues: list[str] = []
    checks: dict[str, Any] = {}

    # --- pilot ---
    pilot = tree_payload.get("pilot_result", {})
    pilot_status = pilot.get("status")
    checks["pilot_status"] = pilot_status
    if pilot_status != "success":
        issues.append(f"pilot status={pilot_status}, expected 'success'")

    pilot_traj = pilot.get("trajectory", {})
    pilot_agents = pilot_traj.get("agent_trajectories", {})
    missing_roles = [r for r in roles if r not in pilot_agents or not pilot_agents[r]]
    checks["pilot_has_all_roles"] = len(missing_roles) == 0
    checks["pilot_present_roles"] = [r for r in roles if r in pilot_agents and pilot_agents[r]]
    if missing_roles:
        # Not a hard failure: MAS may legitimately skip roles (e.g. searcher skipped
        # when verifier immediately approves).  Record as warning, not issue.
        checks["pilot_missing_roles_warning"] = missing_roles

    pilot_reward = pilot.get("final_reward")
    reward_ok = (
        pilot_reward is not None
        and isinstance(pilot_reward, (int, float))
        and math.isfinite(pilot_reward)
        and 0.0 <= pilot_reward <= 1.0
    )
    checks["pilot_reward_valid"] = reward_ok
    if not reward_ok:
        issues.append(f"pilot final_reward={pilot_reward} not finite in [0,1]")

    # count pilot turns (all turns sorted by timestamp)
    pilot_turns_list = _collect_all_turns(pilot)
    pilot_turn_count = len(pilot_turns_list)
    checks["pilot_turn_count"] = pilot_turn_count

    pilot_episode_id = pilot_traj.get("episode_id")

    # --- branches ---
    branches = tree_payload.get("branch_results", [])
    branch_issues: list[str] = []
    all_episode_ids = [pilot_episode_id] if pilot_episode_id else []

    for idx, branch in enumerate(branches):
        bt = branch.get("branch_turn")
        ba = branch.get("branch_agent_role")
        pid = branch.get("parent_episode_id")

        if bt is None or not isinstance(bt, int) or bt < 0 or bt >= pilot_turn_count:
            branch_issues.append(f"branch[{idx}] branch_turn={bt} out of [0, {pilot_turn_count})")
        # branch_agent_role should match a role present in the pilot (not necessarily all configured roles)
        pilot_present_roles = [r for r in roles if r in pilot_agents and pilot_agents[r]]
        if ba is None or (pilot_present_roles and ba not in pilot_present_roles):
            branch_issues.append(f"branch[{idx}] branch_agent_role={ba} not in pilot roles {pilot_present_roles}")
        if pid != pilot_episode_id:
            branch_issues.append(
                f"branch[{idx}] parent_episode_id={pid} != pilot episode_id={pilot_episode_id}"
            )

        br_ep = branch.get("episode_result", {})
        br_traj = br_ep.get("trajectory", {})
        br_eid = br_traj.get("episode_id")
        if br_eid:
            all_episode_ids.append(br_eid)

    checks["branch_count"] = len(branches)
    checks["branch_field_issues"] = branch_issues
    if branch_issues:
        issues.extend(branch_issues)

    # --- tree_metadata ---
    meta = tree_payload.get("tree_metadata", {})
    required_meta_keys = {"n_branch_points", "k_branches", "total_branches_collected", "pilot_total_turns"}
    missing_meta = required_meta_keys - set(meta.keys())
    checks["tree_metadata_complete"] = len(missing_meta) == 0
    if missing_meta:
        issues.append(f"tree_metadata missing keys: {missing_meta}")

    # --- episode_id uniqueness ---
    unique = len(set(all_episode_ids)) == len(all_episode_ids)
    checks["episode_ids_unique"] = unique
    checks["total_episode_ids"] = len(all_episode_ids)
    if not unique:
        issues.append(f"episode_id collision: {len(all_episode_ids)} total, {len(set(all_episode_ids))} unique")

    checks["issues"] = issues
    checks["passed"] = len(issues) == 0
    return checks


def _validate_replay_markers(
    branch_payload: dict,
    branch_turn: int,
    pilot_turns_by_position: list[dict],
) -> dict:
    """Check replayed metadata on branch turns matches branch_turn position."""
    issues: list[str] = []
    checks: dict[str, Any] = {"branch_turn": branch_turn}

    br_ep = branch_payload.get("episode_result", {})
    branch_turns = _collect_all_turns(br_ep)
    checks["branch_total_turns"] = len(branch_turns)

    replayed_count = 0
    fresh_count = 0
    text_match_failures: list[str] = []

    for pos, turn in enumerate(branch_turns):
        metadata = turn.get("metadata", {})
        is_replayed = metadata.get("replayed") is True

        if pos < branch_turn:
            # Should be replayed
            if not is_replayed:
                issues.append(
                    f"turn[{pos}] (role={turn.get('agent_role')}) expected replayed=True "
                    f"but got metadata={metadata}"
                )
            else:
                replayed_count += 1
                # check response text matches pilot
                if pos < len(pilot_turns_by_position):
                    pilot_text = pilot_turns_by_position[pos].get("response_text", "")
                    branch_text = turn.get("response_text", "")
                    if branch_text != pilot_text:
                        text_match_failures.append(
                            f"turn[{pos}] text mismatch: "
                            f"pilot={pilot_text[:60]!r}... vs branch={branch_text[:60]!r}..."
                        )
        else:
            # Should NOT be replayed
            if is_replayed:
                issues.append(
                    f"turn[{pos}] (role={turn.get('agent_role')}) should NOT be replayed "
                    f"at pos >= branch_turn={branch_turn}"
                )
            fresh_count += 1

    checks["replayed_count"] = replayed_count
    checks["fresh_count"] = fresh_count
    checks["expected_replayed"] = branch_turn
    checks["text_match_failures"] = text_match_failures
    if text_match_failures:
        issues.extend(text_match_failures)

    checks["issues"] = issues
    checks["passed"] = len(issues) == 0
    return checks


def _compute_reward_stats(rewards: list[float]) -> dict:
    """mean, variance, min, max, success_rate."""
    if not rewards:
        return {
            "mean": None, "variance": None,
            "min": None, "max": None, "success_rate": 0.0, "count": 0,
        }
    return {
        "mean": statistics.mean(rewards),
        "variance": statistics.variance(rewards) if len(rewards) > 1 else 0.0,
        "min": min(rewards),
        "max": max(rewards),
        "success_rate": sum(1 for r in rewards if r == 1.0) / len(rewards),
        "count": len(rewards),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V0.2 树状分支验证")

    # Inherited from V0
    parser.add_argument("--config", type=Path, default=None, help="OrchRL Search MAS YAML 配置")
    parser.add_argument("--vllm-url", type=str, default="http://127.0.0.1:8000", help="vLLM 服务地址")
    parser.add_argument("--model", type=str, default=None, help="模型名称（默认从配置读取）")
    parser.add_argument(
        "--prompts-file", type=Path, default=_v0.DEFAULT_PROMPTS_FILE,
        help="测试题目 parquet 文件",
    )
    parser.add_argument("--n-prompts", type=int, default=3, help="抽样题目数")
    parser.add_argument("--max-concurrent", type=int, default=2, help="最大并发 episode (parallel_rollout)")
    parser.add_argument("--force-mock", action="store_true", help="强制 mock 模式")
    parser.add_argument("--mas-work-dir", type=Path, default=None, help="MAS 运行目录")
    parser.add_argument("--mas-command-template", type=str, default=None, help="MAS 命令模板")

    # New for V0.2
    parser.add_argument("--k-branches", type=int, default=3, help="每个分支点的分支数")
    parser.add_argument(
        "--max-concurrent-branches", type=int, default=None,
        help="tree_rollout 最大并发分支数",
    )
    parser.add_argument("--compare", action="store_true", help="启用 parallel_rollout 对比模式")
    parser.add_argument("--compare-n-samples", type=int, default=2, help="对比模式每题采样数")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 文件")
    parser.add_argument("--timeout", type=float, default=180.0, help="单 episode 超时时间")
    parser.add_argument(
        "--search-health-url", type=str,
        default="http://127.0.0.1:8010/retrieve",
        help="检索服务健康检查 URL",
    )
    parser.add_argument(
        "--save-trajectories", action="store_true",
        help="保存完整 trajectory 到 <output>_trajectories.json",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Round runners
# ---------------------------------------------------------------------------

async def _run_round_smoke(
    prompt: str,
    reward_provider: FunctionRewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    roles: list[str],
) -> dict[str, Any]:
    """Round 1: Smoke — 结构完整性。k=2, max_concurrent=1."""
    tree_result = await tree_rollout(
        prompt=prompt,
        reward_provider=reward_provider,
        config=config,
        backend=backend,
        k_branches=2,
        max_concurrent_branches=1,
    )
    tree_payload = asdict(tree_result)
    structure = _validate_tree_structure(tree_payload, roles)
    prefix = _compute_prefix_sharing(tree_payload)
    rewards = _collect_tree_rewards(tree_payload)
    return {
        "status": "pass" if structure["passed"] else "fail",
        "validations": {
            "structure": structure,
            "prefix_sharing": prefix,
            "reward_stats": _compute_reward_stats(rewards),
        },
        "tree_payload": tree_payload,
        "error": None,
    }


async def _run_round_replay(
    prompt: str,
    reward_provider: FunctionRewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    roles: list[str],
) -> dict[str, Any]:
    """Round 2: Replay — 重放标记验证。k=2, independent call."""
    tree_result = await tree_rollout(
        prompt=prompt,
        reward_provider=reward_provider,
        config=config,
        backend=backend,
        k_branches=2,
        max_concurrent_branches=1,
    )
    tree_payload = asdict(tree_result)

    # build pilot turns by global position
    pilot_ep = tree_payload.get("pilot_result", {})
    pilot_turns_by_position = _collect_all_turns(pilot_ep)

    # validate replay markers on each branch
    branch_validations: list[dict] = []
    all_passed = True
    for idx, branch in enumerate(tree_payload.get("branch_results", [])):
        bt = branch.get("branch_turn", 0)
        rv = _validate_replay_markers(branch, bt, pilot_turns_by_position)
        rv["branch_index"] = idx
        branch_validations.append(rv)
        if not rv["passed"]:
            all_passed = False

    prefix = _compute_prefix_sharing(tree_payload)
    structure = _validate_tree_structure(tree_payload, roles)
    rewards = _collect_tree_rewards(tree_payload)

    overall_passed = all_passed and structure["passed"]
    return {
        "status": "pass" if overall_passed else "fail",
        "validations": {
            "structure": structure,
            "replay_markers": branch_validations,
            "prefix_sharing": prefix,
            "reward_stats": _compute_reward_stats(rewards),
        },
        "tree_payload": tree_payload,
        "error": None,
    }


async def _run_round_multi_prompt(
    prompts: list[str],
    reward_provider: FunctionRewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    roles: list[str],
    k_branches: int,
    max_concurrent_branches: int | None,
    save_trajectories: bool = False,
) -> dict[str, Any]:
    """Round 3: Multi-prompt — 多提示词稳定性。"""
    trees: list[dict[str, Any]] = []
    all_rewards: list[float] = []
    total_episodes = 0
    total_branches = 0
    prefix_rates: list[float] = []
    success_count = 0

    for pidx, prompt in enumerate(prompts):
        tree_entry: dict[str, Any] = {"prompt_index": pidx, "prompt": prompt[:120]}
        try:
            tree_result = await tree_rollout(
                prompt=prompt,
                reward_provider=reward_provider,
                config=config,
                backend=backend,
                k_branches=k_branches,
                max_concurrent_branches=max_concurrent_branches,
            )
            tp = asdict(tree_result)

            struct = _validate_tree_structure(tp, roles)

            # replay markers
            pilot_turns_pos = _collect_all_turns(tp.get("pilot_result", {}))
            replay_checks: list[dict] = []
            for bidx, br in enumerate(tp.get("branch_results", [])):
                rv = _validate_replay_markers(br, br.get("branch_turn", 0), pilot_turns_pos)
                rv["branch_index"] = bidx
                replay_checks.append(rv)

            prefix = _compute_prefix_sharing(tp)
            rewards = _collect_tree_rewards(tp)

            n_branches = len(tp.get("branch_results", []))
            n_ep = 1 + n_branches  # pilot + branches
            total_episodes += n_ep
            total_branches += n_branches
            all_rewards.extend(rewards)
            prefix_rates.append(prefix["prefix_sharing_rate"])

            tree_passed = struct["passed"] and all(r["passed"] for r in replay_checks)
            if tree_passed:
                success_count += 1

            tree_entry["status"] = "pass" if tree_passed else "fail"
            tree_entry["structure"] = struct
            tree_entry["replay_markers"] = replay_checks
            tree_entry["prefix_sharing"] = prefix
            tree_entry["reward_stats"] = _compute_reward_stats(rewards)
            tree_entry["n_episodes"] = n_ep
            tree_entry["error"] = None
            if save_trajectories:
                tree_entry["tree_payload"] = tp
        except Exception as exc:
            tree_entry["status"] = "error"
            tree_entry["error"] = f"{type(exc).__name__}: {exc}"
            tree_entry["traceback"] = traceback.format_exc()

        trees.append(tree_entry)

    n_trees = len(trees)
    avg_branches = total_branches / n_trees if n_trees else 0.0
    avg_prefix = _v0._safe_mean(prefix_rates)

    all_passed = all(t.get("status") == "pass" for t in trees)
    return {
        "status": "pass" if all_passed else "fail",
        "trees": trees,
        "aggregate": {
            "total_trees": n_trees,
            "success_trees": success_count,
            "total_episodes": total_episodes,
            "avg_branches_per_tree": avg_branches,
            "avg_prefix_sharing_rate": avg_prefix,
            "reward_distribution": _compute_reward_stats(all_rewards),
        },
        "error": None,
    }


async def _run_comparison(
    prompts: list[str],
    reward_provider: FunctionRewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    n_samples: int,
    max_concurrent: int,
) -> dict[str, Any]:
    """Optional comparison: parallel_rollout vs tree_rollout reward distribution."""
    try:
        results = await parallel_rollout(
            prompts=prompts,
            reward_provider=reward_provider,
            config=config,
            backend=backend,
            n_samples_per_prompt=n_samples,
            max_concurrent=max_concurrent,
        )
        rewards: list[float] = []
        for r in results:
            if r.final_reward is not None:
                v = _v0._safe_float(r.final_reward)
                if math.isfinite(v):
                    rewards.append(v)
        return {
            "status": "pass",
            "total_episodes": len(results),
            "reward_stats": _compute_reward_stats(rewards),
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    env_report: dict[str, Any] = {}
    mock_runtime: tempfile.TemporaryDirectory[str] | None = None

    try:
        # ---- environment detection (mirrors V0) ----
        vllm_ok, vllm_payload, vllm_error = _v0._safe_json_get(
            f"{args.vllm_url.rstrip('/')}/v1/models"
        )
        env_report["vllm"] = {
            "available": vllm_ok,
            "url": args.vllm_url,
            "error": vllm_error,
            "models": vllm_payload,
        }

        search_ok, search_payload, search_error = _v0._safe_json_get(args.search_health_url)
        if not search_ok:
            search_ok, search_payload, search_error = _v0._safe_json_post(args.search_health_url)
        env_report["search_service"] = {
            "available": search_ok,
            "url": args.search_health_url,
            "error": search_error,
        }

        config_template, config_source = _v0._load_config(args.config)
        prompts, prompts_meta = _v0._load_prompts(args.prompts_file.expanduser(), args.n_prompts)
        env_report["prompts"] = prompts_meta

        prompt_list = [item["prompt"] for item in prompts]
        expected_by_prompt = {item["prompt"]: item.get("expected", "") for item in prompts}

        real_runner_root, real_runner_command = _v0._find_real_runner(args.config, args.mas_work_dir)
        env_report["orchrl_runner"] = {
            "available": real_runner_root is not None,
            "root": str(real_runner_root) if real_runner_root else None,
        }

        search_cfg = config_template.get("search")
        search_provider = (
            search_cfg.get("provider")
            if isinstance(search_cfg, dict) and isinstance(search_cfg.get("provider"), str)
            else None
        )
        search_required = (search_provider or "http").lower() != "disabled"

        use_mock = (
            args.force_mock
            or (not vllm_ok)
            or (real_runner_root is None)
            or (search_required and not search_ok)
        )

        search_root_for_reward = real_runner_root
        runner_root = real_runner_root
        runner_command = real_runner_command
        run_mode_reasons: list[str] = []

        if args.force_mock:
            run_mode_reasons.append("--force-mock")
        if not vllm_ok:
            run_mode_reasons.append("vLLM 不可用")
        if runner_root is None:
            run_mode_reasons.append("OrchRL Search MAS 不可用")
        if search_required and not search_ok:
            run_mode_reasons.append("检索服务不可用")

        if use_mock:
            mock_runtime = _v0._prepare_mock_runtime()
            runner_root = Path(mock_runtime.name)
            runner_command = (
                f"{sys.executable} scripts/run_search_mas.py"
                f" --config {{config_path}} --question {{prompt}}"
            )
            mock_search_cfg = config_template.setdefault("search", {})
            if isinstance(mock_search_cfg, dict):
                mock_search_cfg["provider"] = "disabled"
        else:
            # Set retrieval service URL for real mode
            os.environ.setdefault(
                "SEARCH_MAS_RETRIEVAL_SERVICE_URL", args.search_health_url
            )
            # CRITICAL: Remove LLM env vars that would override the monitor URL.
            # The OrchRL MAS config loader (_apply_llm_env_overrides) reads
            # SEARCH_MAS_LLM_BASE_URL and overwrites llm.base_url in the config,
            # which conflicts with AgentPipe's monitor proxy.
            for env_key in (
                "SEARCH_MAS_LLM_BASE_URL", "OPENAI_BASE_URL",
                "SEARCH_MAS_VERIFIER_LLM_BASE_URL",
                "SEARCH_MAS_SEARCHER_LLM_BASE_URL",
                "SEARCH_MAS_ANSWERER_LLM_BASE_URL",
            ):
                os.environ.pop(env_key, None)

        if runner_command is None or runner_root is None:
            raise RuntimeError("无法确定 MAS 运行命令")

        checker, checker_source = _v0._load_is_search_answer_correct(search_root_for_reward)

        agents_cfg = config_template.get("agents")
        if not isinstance(agents_cfg, dict) or not agents_cfg:
            agents_cfg = {role: {} for role in _v0.DEFAULT_ROLES}
            config_template["agents"] = agents_cfg
        roles = list(agents_cfg.keys())

        llm_cfg = config_template.get("llm") if isinstance(config_template.get("llm"), dict) else {}
        configured_model = llm_cfg.get("model") if isinstance(llm_cfg, dict) else None
        model_name, model_source = _v0._resolve_model_name(
            cli_model=args.model,
            configured_model=configured_model if isinstance(configured_model, str) else None,
            vllm_payload=vllm_payload,
            config_path=args.config.expanduser().resolve() if args.config is not None else None,
        )
        env_report["model_path"] = {
            "resolved": model_name,
            "source": model_source,
            "available_vllm_models": _v0._iter_vllm_model_ids(vllm_payload),
        }

        # ---- build backend + config ----
        if use_mock:
            backend: InferenceBackend = _v0.ScriptedBackend(expected_by_prompt=expected_by_prompt)
            model_mapping = {role: ModelMappingEntry(actual_model=None) for role in roles}
        else:
            if model_name and Path(model_name).exists():
                backend = VLLMBackend.with_tokenizer(
                    backend_url=args.vllm_url,
                    model_path=model_name,
                    actual_model=model_name,
                )
            else:
                backend = VLLMBackend(backend_url=args.vllm_url, actual_model=model_name)
            model_mapping = {role: ModelMappingEntry(actual_model=model_name) for role in roles}

        pipe_config = AgentPipeConfig(
            mas_command_template=args.mas_command_template or runner_command,
            config_template=config_template,
            model_mapping=model_mapping,
            timeout=args.timeout,
            mas_work_dir=runner_root,
        )

        reward_provider = _v0._build_reward_provider(
            expected_by_prompt=expected_by_prompt, checker=checker
        )

        first_prompt = prompt_list[0]

        # ================================================================
        # Round 1: Smoke
        # ================================================================
        round_smoke: dict[str, Any]
        try:
            round_smoke = await _run_round_smoke(
                first_prompt, reward_provider, pipe_config, backend, roles
            )
        except Exception as exc:
            round_smoke = {
                "status": "error",
                "validations": {},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

        # ================================================================
        # Round 2: Replay
        # ================================================================
        round_replay: dict[str, Any]
        try:
            round_replay = await _run_round_replay(
                first_prompt, reward_provider, pipe_config, backend, roles
            )
        except Exception as exc:
            round_replay = {
                "status": "error",
                "validations": {},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

        # ================================================================
        # Round 3: Multi-prompt
        # ================================================================
        round_multi: dict[str, Any]
        try:
            round_multi = await _run_round_multi_prompt(
                prompt_list, reward_provider, pipe_config, backend, roles,
                k_branches=args.k_branches,
                max_concurrent_branches=args.max_concurrent_branches,
                save_trajectories=args.save_trajectories,
            )
        except Exception as exc:
            round_multi = {
                "status": "error",
                "trees": [],
                "aggregate": {},
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

        # ================================================================
        # Optional: Comparison
        # ================================================================
        comparison: dict[str, Any] | None = None
        if args.compare:
            try:
                comparison = await _run_comparison(
                    prompt_list, reward_provider, pipe_config, backend,
                    n_samples=args.compare_n_samples,
                    max_concurrent=args.max_concurrent,
                )
            except Exception as exc:
                comparison = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }

        # ================================================================
        # Assemble report
        # ================================================================
        # Strip bulky tree_payload from round dicts before writing report
        smoke_report = {k: v for k, v in round_smoke.items() if k != "tree_payload"}
        replay_report = {k: v for k, v in round_replay.items() if k != "tree_payload"}

        # For multi_prompt, strip tree_payload from individual tree entries
        # (these are only present when save_trajectories is enabled)
        multi_report = dict(round_multi)
        if "trees" in multi_report:
            multi_report["trees"] = [
                {k: v for k, v in t.items() if k != "tree_payload"}
                for t in multi_report["trees"]
            ]

        round_statuses = {
            "smoke": round_smoke["status"],
            "replay": round_replay["status"],
            "multi_prompt": round_multi["status"],
        }
        all_passed = all(s == "pass" for s in round_statuses.values())

        aggregate = round_multi.get("aggregate", {})

        report: dict[str, Any] = {
            "schema_version": "2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_mode": "mock" if use_mock else "real",
            "run_mode_reasons": run_mode_reasons,
            "environment": env_report,
            "input": {
                "n_prompts": args.n_prompts,
                "k_branches": args.k_branches,
                "max_concurrent_branches": args.max_concurrent_branches,
                "compare_mode": args.compare,
                "compare_n_samples": args.compare_n_samples if args.compare else None,
                "timeout": args.timeout,
                "model": model_name,
                "vllm_url": args.vllm_url,
                "prompts_file": str(args.prompts_file.expanduser()),
            },
            "rounds": {
                "smoke": smoke_report,
                "replay": replay_report,
                "multi_prompt": multi_report,
            },
            "comparison": comparison,
            "summary": {
                "all_rounds_passed": all_passed,
                "round_statuses": round_statuses,
            },
        }

        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # ---- save trajectories ----
        if args.save_trajectories:
            trajectories: dict[str, Any] = {
                "schema_version": "2.0",
                "timestamp": report["timestamp"],
                "run_mode": report["run_mode"],
            }
            # Smoke & Replay carry tree_payload from their round dicts
            if "tree_payload" in round_smoke:
                trajectories["smoke"] = round_smoke["tree_payload"]
            if "tree_payload" in round_replay:
                trajectories["replay"] = round_replay["tree_payload"]
            # Multi-prompt trees carry tree_payload per entry
            multi_trajs = []
            for t in round_multi.get("trees", []):
                tp = t.get("tree_payload")
                if tp is not None:
                    multi_trajs.append({
                        "prompt_index": t.get("prompt_index"),
                        "prompt": t.get("prompt"),
                        "tree_payload": tp,
                    })
            if multi_trajs:
                trajectories["multi_prompt"] = multi_trajs

            traj_path = output_path.with_name(
                output_path.stem + "_trajectories" + output_path.suffix
            )
            traj_path.write_text(
                json.dumps(trajectories, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"Trajectory 文件: {traj_path}")

        # ---- console summary ----
        print("=== V0.2 树状分支验证摘要 ===")
        print(f"模式: {'mock 模式' if use_mock else 'real 模式'}")
        if run_mode_reasons:
            print(f"切换原因: {', '.join(run_mode_reasons)}")
        print(f"Round 1 (Smoke): {round_smoke['status'].upper()}")
        print(f"Round 2 (Replay): {round_replay['status'].upper()}")
        print(f"Round 3 (Multi-prompt): {round_multi['status'].upper()}")

        total_trees = aggregate.get("total_trees", 0)
        success_trees = aggregate.get("success_trees", 0)
        total_episodes = aggregate.get("total_episodes", 0)
        avg_prefix = aggregate.get("avg_prefix_sharing_rate", 0.0)
        reward_dist = aggregate.get("reward_distribution", {})
        reward_mean = reward_dist.get("mean")

        print(f"  成功树数: {success_trees}/{total_trees}")
        print(f"  总 episode 数: {total_episodes} (pilot + branches)")
        print(f"  前缀共享率: {avg_prefix * 100:.1f}%")
        print(f"  reward 均值: {reward_mean:.2f}" if reward_mean is not None else "  reward 均值: N/A")
        print(f"对比模式: {'启用' if args.compare else '未启用'}")
        if comparison is not None:
            cmp_status = comparison.get("status", "N/A")
            cmp_reward = comparison.get("reward_stats", {}).get("mean")
            print(f"  对比 status: {cmp_status}")
            if cmp_reward is not None:
                print(f"  对比 reward 均值: {cmp_reward:.2f}")
        print(f"输出文件: {output_path}")

        return 0 if all_passed else 1

    finally:
        if mock_runtime is not None:
            mock_runtime.cleanup()


def main() -> int:
    args = _parse_args()
    if args.k_branches < 1:
        raise ValueError("--k-branches 必须 >= 1")
    if args.n_prompts < 1:
        raise ValueError("--n-prompts 必须 >= 1")
    if args.max_concurrent_branches is not None and args.max_concurrent_branches < 1:
        raise ValueError("--max-concurrent-branches 必须 >= 1")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
