#!/usr/bin/env python3
"""Demo: run a 3-agent Search MAS episode via MATE trajectory engine.

Collects full multi-agent trajectories using a real vLLM backend,
then displays the structured result layer by layer.

Usage:
    # Single question
    python scripts/demo_search_mas.py --question "What is the capital of France?"

    # Multiple questions (run in parallel)
    python scripts/demo_search_mas.py \
        --question "What is the capital of France?" "Who invented the telephone?" \
        --max-concurrent 2

    # With explicit options
    python scripts/demo_search_mas.py \
        --question "When was the first iPhone released?" \
        --vllm-url http://127.0.0.1:8000 \
        --model /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
        --show-mapping

    # Using a config file
    python scripts/demo_search_mas.py \
        --config artifacts/search_mas_real_nosearch.yaml \
        --question "What is the speed of light?"

    # Tree rollout mode
    python scripts/demo_search_mas.py \
        --question "Who painted the Mona Lisa?" \
        --tree --k-branches 2

    # Save trajectories to file
    python scripts/demo_search_mas.py \
        --question "What is the capital of France?" "Who invented the telephone?" \
        --output artifacts/demo_episodes.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mate.trajectory import (
    AgentPipeConfig,
    EpisodeResult,
    FunctionRewardProvider,
    ModelMappingEntry,
    VLLMBackend,
    parallel_rollout,
    tree_rollout,
)
from mate.trajectory.datatypes import EpisodeTrajectory, TreeEpisodeResult
from mate.trajectory.display import (
    format_episode,
    format_training_mapping,
    format_tree,
)

DEFAULT_VLLM_URL = "http://127.0.0.1:8000"
DEFAULT_MAS_WORK_DIR = "/home/cxb/OrchRL/examples/mas_app/search"
DEFAULT_ROLES = ["verifier", "searcher", "answerer"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: MATE trajectory engine with a 3-agent Search MAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--question", nargs="+", required=True, metavar="Q",
                        help="One or more questions to ask the MAS (run in parallel when multiple)")
    parser.add_argument("--vllm-url", default=DEFAULT_VLLM_URL, help="vLLM server URL")
    parser.add_argument("--model", default=None, help="Model name or path (auto-detected from vLLM if omitted)")
    parser.add_argument("--config", type=Path, default=None, help="Search MAS YAML config")
    parser.add_argument("--mas-work-dir", type=Path, default=None, help="MAS working directory")
    parser.add_argument("--timeout", type=float, default=180.0, help="Episode timeout in seconds")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Max concurrent episodes (default: unlimited)")
    parser.add_argument("--show-mapping", action="store_true", help="Show MATE → VERL field mapping table")
    parser.add_argument("--tree", action="store_true", help="Run tree rollout instead of single episode")
    parser.add_argument("--k-branches", type=int, default=2, help="Branches per turn in tree mode")
    parser.add_argument("--output", type=Path, default=None, help="Save trajectory/trajectories JSON to file")
    return parser.parse_args()


def _resolve_model(vllm_url: str, cli_model: str | None, config: dict[str, Any] | None) -> str | None:
    """Resolve model name: CLI > config > auto-detect from vLLM."""
    if cli_model:
        return cli_model

    if config:
        llm_cfg = config.get("llm", {})
        if isinstance(llm_cfg, dict) and isinstance(llm_cfg.get("model"), str):
            return llm_cfg["model"]

    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{vllm_url.rstrip('/')}/v1/models")
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data and isinstance(data[0], dict):
                return data[0].get("id")
    except Exception:
        pass

    return None


def _load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return _default_config()
    import yaml
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must be a YAML mapping, got {type(loaded).__name__}")
    return loaded


def _default_config() -> dict[str, Any]:
    return {
        "application": {
            "type": "search",
            "max_turns": 4,
            "force_final_answer_on_max_turn": True,
        },
        "llm": {
            "base_url": "http://placeholder/v1",
            "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
            "timeout": 120,
            "max_retries": 3,
            "retry_backoff_sec": 1.0,
        },
        "search": {"provider": "disabled"},
        "agents": {
            "verifier": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 1024},
            "searcher": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 1024},
            "answerer": {"temperature": 0.4, "top_p": 0.95, "max_tokens": 1536},
        },
    }


def _find_mas_work_dir(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(DEFAULT_MAS_WORK_DIR),
        Path(os.environ.get("ORCHRL_SEARCH_DIR", "")),
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and (candidate / "scripts" / "run_search_mas.py").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find OrchRL Search MAS. Provide --mas-work-dir or set ORCHRL_SEARCH_DIR."
    )


def _build_reward_fn(expected: str | None = None):
    """Simple reward: 1.0 if answerer produced <answer>...</answer>, else 0.0."""
    def _reward(trajectory: EpisodeTrajectory) -> dict[str, Any]:
        answer_turns = trajectory.agent_trajectories.get("answerer", [])
        if not answer_turns:
            return {"agent_rewards": {}, "final_reward": 0.0}
        last_text = answer_turns[-1].response_text.lower()
        has_answer = "<answer>" in last_text and "</answer>" in last_text
        final_reward = 1.0 if has_answer else 0.0
        return {
            "agent_rewards": {role: final_reward for role in trajectory.agent_trajectories},
            "final_reward": final_reward,
        }
    return FunctionRewardProvider(_reward)


async def _run_episodes(
    questions: list[str],
    pipe_config: AgentPipeConfig,
    backend: VLLMBackend,
    max_concurrent: int | None,
) -> list[EpisodeResult]:
    results = await parallel_rollout(
        prompts=questions,
        reward_provider=_build_reward_fn(),
        config=pipe_config,
        backend=backend,
        n_samples_per_prompt=1,
        max_concurrent=max_concurrent,
    )
    if not results:
        raise RuntimeError("Episode collection failed — no results returned")
    return results


async def _run_tree_episode(
    question: str,
    pipe_config: AgentPipeConfig,
    backend: VLLMBackend,
    k_branches: int,
) -> TreeEpisodeResult:
    return await tree_rollout(
        prompt=question,
        reward_provider=_build_reward_fn(),
        config=pipe_config,
        backend=backend,
        k_branches=k_branches,
        max_concurrent_branches=2,
    )


def _save_output(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved trajectory JSON → {output_path}")


async def _main() -> int:
    args = _parse_args()

    # ── Resolve runtime ──
    config = _load_config(args.config)
    model_name = _resolve_model(args.vllm_url, args.model, config)
    mas_work_dir = _find_mas_work_dir(args.mas_work_dir)

    if model_name and config.get("llm") and isinstance(config["llm"], dict):
        config["llm"]["model"] = model_name

    questions: list[str] = args.question

    print(f"\n  vLLM URL     : {args.vllm_url}")
    print(f"  Model        : {model_name or '(auto-detect)'}")
    print(f"  MAS work dir : {mas_work_dir}")
    if len(questions) == 1:
        print(f"  Question     : {questions[0]}")
    else:
        print(f"  Questions    : {len(questions)} prompts")
        for i, q in enumerate(questions, 1):
            print(f"    [{i}] {q}")
    print(f"  Mode         : {'tree (k={})'.format(args.k_branches) if args.tree else 'parallel rollout'}")
    if not args.tree and args.max_concurrent:
        print(f"  Concurrency  : {args.max_concurrent}")
    print()

    # ── Build pipeline ──
    roles = list((config.get("agents") or {}).keys()) or DEFAULT_ROLES
    model_mapping = {role: ModelMappingEntry(actual_model=model_name) for role in roles}

    mas_command = f"{sys.executable} scripts/run_search_mas.py --config {{config_path}} --question {{prompt}}"

    use_tokenizer = model_name and Path(model_name).exists()
    if use_tokenizer:
        backend = VLLMBackend.with_tokenizer(
            backend_url=args.vllm_url,
            model_path=model_name,
            actual_model=model_name,
            timeout=args.timeout,
        )
        print(f"  Backend      : VLLMBackend with local tokenizer (canonical prompt_ids enabled)")
    else:
        backend = VLLMBackend(
            backend_url=args.vllm_url,
            actual_model=model_name,
            timeout=args.timeout,
        )
        print(f"  Backend      : VLLMBackend (HTTP-only, prompt_ids from logprobs)")

    renderer = backend._renderer if hasattr(backend, "_renderer") else None

    pipe_config = AgentPipeConfig(
        mas_command_template=mas_command,
        config_template=config,
        model_mapping=model_mapping,
        timeout=args.timeout,
        mas_work_dir=mas_work_dir,
        renderer=renderer,
    )

    print()
    print("  Starting episode collection...")
    print()

    # ── Run ──
    if args.tree:
        if len(questions) > 1:
            print("  Note: --tree mode runs one episode at a time; using the first question only.\n")
        tree_result = await _run_tree_episode(questions[0], pipe_config, backend, args.k_branches)
        print(format_tree(tree_result, expand_pilot=True, show_mapping=args.show_mapping))
        if args.output:
            _save_output(asdict(tree_result), args.output)
    else:
        results = await _run_episodes(questions, pipe_config, backend, args.max_concurrent)
        n = len(results)
        for idx, result in enumerate(results):
            if n > 1:
                print(f"\n{'═' * 60}")
                print(f"  Episode {idx + 1} / {n}  ·  {questions[idx]}")
                print(f"{'═' * 60}")
            print(format_episode(result, show_mapping=args.show_mapping))
        if args.output:
            payload = asdict(results[0]) if n == 1 else [asdict(r) for r in results]
            _save_output(payload, args.output)

    print()
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
