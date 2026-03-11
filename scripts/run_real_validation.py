#!/usr/bin/env python3
"""
真实环境端到端验证脚本。

使用 AgentPipe + parallel_rollout 对 DrMAS Search 任务进行轨迹采集。
输出采集到的轨迹数据（JSON 格式）供后续可视化和审查。
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import importlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import tempfile
import threading
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._trajectory_utils import (
    collect_all_turns as _collect_all_turns,
    extract_tag as _extract_tag,
    safe_float as _safe_float,
)

from mate.trajectory import (
    AgentPipeConfig,
    FunctionRewardProvider,
    InferenceBackend,
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
    VLLMBackend,
    parallel_rollout,
)

DEFAULT_ROLES = ["verifier", "searcher", "answerer"]
_PROMPTS_FILE_CANDIDATES = [
    Path("/data0/dataset/multi_agent_rl/DrMAS/mas_app/search/data/drmas_search_mas/test_sampled.parquet"),
    Path("~/data/drmas_search_mas/test_sampled.parquet").expanduser(),
]
DEFAULT_PROMPTS_FILE = next(
    (path for path in _PROMPTS_FILE_CANDIDATES if path.exists()),
    _PROMPTS_FILE_CANDIDATES[0],
)
DEFAULT_OUTPUT = Path("artifacts/trajectory_validation.json")


class ScriptedBackend(InferenceBackend):
    """Mock backend for unavailable real services."""

    def __init__(self, expected_by_prompt: dict[str, Any]) -> None:
        self._expected_by_prompt = expected_by_prompt
        self._verifier_turns: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    async def generate(self, request: ModelRequest) -> ModelResponse:
        prompt_key = _extract_prompt_from_messages(request.messages) or "default"

        if request.agent_role == "verifier":
            with self._lock:
                idx = self._verifier_turns[prompt_key]
                self._verifier_turns[prompt_key] += 1
            if idx == 0:
                content = (
                    "Need additional evidence before final answer.\n"
                    "<verify>no</verify>"
                )
            else:
                content = (
                    "Now I have enough evidence for final answer.\n"
                    "<verify>yes</verify>"
                )
        elif request.agent_role == "searcher":
            content = (
                "Searching the web for grounding evidence.\n"
                "<search>mock query from prompt</search>"
            )
        elif request.agent_role == "answerer":
            expected = _expected_to_answer_text(self._expected_by_prompt.get(prompt_key, "42"))
            content = f"The answer is {expected}.\n<answer>{expected}</answer>"
        else:
            content = "unknown agent"

        token_count = max(1, len(content.split()))
        return ModelResponse(
            content=content,
            token_ids=list(range(1, token_count + 1)),
            logprobs=[-0.1] * token_count,
            finish_reason="stop",
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory Engine 真实环境验证")
    parser.add_argument("--config", type=Path, default=None, help="OrchRL Search MAS YAML 配置")
    parser.add_argument("--vllm-url", type=str, default="http://127.0.0.1:8000", help="vLLM 服务地址")
    parser.add_argument("--model", type=str, default=None, help="模型名称（默认从配置读取）")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="测试题目 parquet 文件",
    )
    parser.add_argument("--n-prompts", type=int, default=5, help="抽样题目数")
    parser.add_argument("--n-samples", type=int, default=2, help="每题 episode 采样数")
    parser.add_argument("--max-concurrent", type=int, default=2, help="最大并发 episode")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 文件")

    parser.add_argument(
        "--search-health-url",
        type=str,
        default="http://127.0.0.1:18080/retrieve",
        help="检索服务健康检查 URL",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="单 episode 超时时间")
    parser.add_argument("--force-mock", action="store_true", help="强制 mock 模式")
    parser.add_argument("--mas-work-dir", type=Path, default=None, help="MAS 运行目录（可选）")
    parser.add_argument(
        "--mas-command-template",
        type=str,
        default=None,
        help="MAS 命令模板，需包含 {config_path} 和 {prompt}",
    )
    return parser.parse_args()


def _safe_json_get(url: str, timeout: float = 5.0) -> tuple[bool, Any, str | None]:
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return True, response.json(), None
    except Exception as exc:
        return False, None, str(exc)


def _safe_json_post(url: str, payload: dict | None = None, timeout: float = 5.0) -> tuple[bool, Any, str | None]:
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload or {"query": "health_check"})
            response.raise_for_status()
            return True, response.json(), None
    except Exception as exc:
        return False, None, str(exc)


def _load_config(config_path: Path | None) -> tuple[dict[str, Any], str]:
    if config_path is None:
        return _default_config(), "default_inline"

    if not config_path.exists():
        raise FileNotFoundError(f"config 不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config 必须是 YAML object，当前为 {type(loaded).__name__}")
    return loaded, str(config_path)


def _default_config() -> dict[str, Any]:
    return {
        "application": {
            "type": "search",
            "max_turns": 4,
            "force_final_answer_on_max_turn": True,
        },
        "llm": {
            "base_url": "http://placeholder/v1",
            "api_key": "EMPTY",
            "timeout": 30,
            "max_retries": 1,
            "retry_backoff_sec": 0.1,
            "model": "Qwen3-4B-Instruct-2507",
        },
        "search": {
            "provider": "disabled",
        },
        "agents": {
            "verifier": {"temperature": 0.2, "max_tokens": 512},
            "searcher": {"temperature": 0.6, "max_tokens": 512},
            "answerer": {"temperature": 0.4, "max_tokens": 512},
        },
    }


def _extract_str(record: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_expected(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key not in record:
            continue
        value = _normalize_expected_value(record[key])
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
            continue
        if isinstance(value, (list, tuple)):
            values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                return values
            continue
        if isinstance(value, dict) and "target" in value:
            target = _normalize_expected_value(value["target"])
            if target is not None:
                return target
        return value
    return ""


def _normalize_expected_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text[0] in "[{(":
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return text
            return _normalize_expected_value(parsed)
        return text
    if isinstance(value, list):
        normalized = [_normalize_expected_value(item) for item in value]
        return [item for item in normalized if item not in (None, "", [])]
    if isinstance(value, tuple):
        normalized = [_normalize_expected_value(item) for item in value]
        return tuple(item for item in normalized if item not in (None, "", []))
    if isinstance(value, dict):
        return {
            key: _normalize_expected_value(item)
            for key, item in value.items()
        }
    return value


def _expected_to_answer_text(expected: Any) -> str:
    if isinstance(expected, str):
        return expected.strip() or "42"
    if isinstance(expected, (list, tuple)):
        for item in expected:
            text = str(item).strip()
            if text:
                return text
        return "42"
    if expected is None:
        return "42"
    text = str(expected).strip()
    return text or "42"


def _load_prompts(prompts_file: Path, n_prompts: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if n_prompts < 1:
        raise ValueError("n_prompts 必须 >= 1")

    if not prompts_file.exists():
        fallback = [{"prompt": f"What is 6 * 7? [sample {i}]", "expected": "42"} for i in range(n_prompts)]
        return fallback, {
            "source": "inline_fallback",
            "reason": f"prompts 文件不存在: {prompts_file}",
            "available": False,
        }

    table = pq.read_table(prompts_file)
    rows = table.to_pylist()

    prompt_keys = ["question", "prompt", "query", "problem", "input"]
    answer_keys = [
        "answer",
        "expected_answer",
        "expected_answers",
        "gold_answer",
        "golden_answers",
        "target",
        "label",
    ]

    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = _extract_str(row, prompt_keys)
        if not prompt:
            continue
        expected = _extract_expected(row, answer_keys)
        selected.append({"prompt": prompt, "expected": expected})
        if len(selected) >= n_prompts:
            break

    if not selected:
        fallback = [{"prompt": f"What is 6 * 7? [sample {i}]", "expected": "42"} for i in range(n_prompts)]
        return fallback, {
            "source": "inline_fallback",
            "reason": f"parquet 未解析到可用题目: {prompts_file}",
            "available": False,
            "row_count": len(rows),
        }

    return selected, {
        "source": str(prompts_file),
        "available": True,
        "row_count": len(rows),
        "selected_count": len(selected),
    }


def _extract_prompt_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"[\t\n\r]", " ", cleaned)
    return cleaned.strip()


def _load_is_search_answer_correct(search_root: Path | None) -> tuple[Callable[[str, Any], bool], str]:
    module_candidates = [
        "search_mas.reward",
        "search_mas.utils.reward",
        "search_mas.utils.eval",
        "search_mas.apps.search.evaluator",
        "search.reward",
    ]

    if search_root is not None and search_root.exists():
        sys.path.insert(0, str(search_root))
        sys.path.insert(0, str(search_root.parent))

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        func = getattr(module, "is_search_answer_correct", None)
        if callable(func):
            return _wrap_checker(func), f"{module_name}.is_search_answer_correct"

    if search_root is not None and search_root.exists():
        file_candidates = [
            search_root / "reward.py",
            search_root / "search_mas" / "reward.py",
            search_root / "search_mas" / "utils" / "reward.py",
            search_root / "search_mas" / "utils" / "eval.py",
            search_root / "search_mas" / "apps" / "search" / "evaluator.py",
        ]
        for file_path in file_candidates:
            if not file_path.exists():
                continue
            module_name = f"_mate_orchrl_reward_{file_path.stem}_{abs(hash(file_path))}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception:
                continue
            func = getattr(module, "is_search_answer_correct", None)
            if callable(func):
                return _wrap_checker(func), f"{file_path}.is_search_answer_correct"

    def _fallback(predicted: str, expected: Any) -> bool:
        expected_text = _expected_to_answer_text(expected)
        if expected_text.strip():
            return _normalize_text(predicted) == _normalize_text(expected_text)
        return bool(predicted.strip())

    return _fallback, "fallback_exact_match"


def _wrap_checker(func: Callable[..., Any]) -> Callable[[str, Any], bool]:
    def _checker(predicted: str, expected: Any) -> bool:
        attempts = [
            lambda: func(predicted, expected),
            lambda: func(predicted=predicted, expected_answer=expected),
            lambda: func(predicted_answer=predicted, expected_answer=expected),
        ]
        for attempt in attempts:
            try:
                value = attempt()
            except TypeError:
                continue
            except Exception:
                return False
            return bool(value)
        return False

    return _checker


def _prepare_mock_runtime() -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="mate_mock_search_")
    root = Path(temp_dir.name)
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    runner_code = '''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import yaml


def _chat(base_url, agent, messages, agent_cfg):
    payload = {
        "model": agent,
        "messages": messages,
        "temperature": agent_cfg.get("temperature", 0.2),
        "max_tokens": agent_cfg.get("max_tokens", 256),
    }
    req = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _assistant_message(content):
    return {"role": "assistant", "content": content}


def _extract_content(resp):
    return resp["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_url = cfg["llm"]["base_url"]
    agents_cfg = cfg.get("agents", {})

    history = [{"role": "user", "content": args.question}]
    verifier_resp = _chat(base_url, "verifier", history, agents_cfg.get("verifier", {}))
    verifier_text = _extract_content(verifier_resp)
    history.append(_assistant_message(verifier_text))

    if "<verify>yes</verify>" not in verifier_text.lower():
        searcher_resp = _chat(base_url, "searcher", history, agents_cfg.get("searcher", {}))
        searcher_text = _extract_content(searcher_resp)
        history.append(_assistant_message(searcher_text))

        verifier_resp2 = _chat(base_url, "verifier", history, agents_cfg.get("verifier", {}))
        verifier_text2 = _extract_content(verifier_resp2)
        history.append(_assistant_message(verifier_text2))

    answerer_resp = _chat(base_url, "answerer", history, agents_cfg.get("answerer", {}))
    answer_text = _extract_content(answerer_resp)

    print(f"final_answer: {answer_text}")


if __name__ == "__main__":
    main()
'''
    runner_path = scripts_dir / "run_search_mas.py"
    runner_path.write_text(runner_code, encoding="utf-8")
    runner_path.chmod(0o755)
    return temp_dir


def _find_real_runner(config_path: Path | None, explicit_work_dir: Path | None) -> tuple[Path | None, str | None]:
    candidate_roots: list[Path] = []
    if explicit_work_dir is not None:
        candidate_roots.append(explicit_work_dir)
    if config_path is not None:
        candidate_roots.append(config_path.parent)
        candidate_roots.append(config_path.parent.parent)
    env_root = os.environ.get("ORCHRL_SEARCH_DIR")
    if env_root:
        candidate_roots.append(Path(env_root))
    candidate_roots.append(Path("/home/cxb/OrchRL/examples/mas_app/search"))

    for root in candidate_roots:
        script = root / "scripts" / "run_search_mas.py"
        if script.exists():
            command = f"{sys.executable} scripts/run_search_mas.py --config {{config_path}} --question {{prompt}}"
            return root, command

    try:
        spec = importlib.util.find_spec("search_mas.scripts.run_search_mas")
    except Exception:
        spec = None

    if spec is not None:
        root = Path.cwd()
        command = f"{sys.executable} -m search_mas.scripts.run_search_mas --config {{config_path}} --question {{prompt}}"
        return root, command

    return None, None


def _build_reward_provider(
    expected_by_prompt: dict[str, Any],
    checker: Callable[[str, Any], bool],
) -> FunctionRewardProvider:
    def _reward(trajectory: Any) -> dict[str, Any]:
        answer_turns = trajectory.agent_trajectories.get("answerer", [])
        predicted = ""
        if answer_turns:
            predicted = _extract_tag(answer_turns[-1].response_text, "answer") or answer_turns[-1].response_text

        prompt_hint = ""
        for turns in trajectory.agent_trajectories.values():
            for turn in turns:
                prompt_hint = _extract_prompt_from_messages(turn.messages) or prompt_hint
                if prompt_hint:
                    break
            if prompt_hint:
                break

        expected = expected_by_prompt.get(prompt_hint)
        if expected is None:
            for key, val in expected_by_prompt.items():
                if key and key in prompt_hint:
                    expected = val
                    break
            else:
                expected = ""
        expected = _normalize_expected_value(expected)
        is_correct = checker(predicted, expected)
        final_reward = 1.0 if is_correct else 0.0
        return {
            "agent_rewards": {
                role: final_reward for role in trajectory.agent_trajectories
            },
            "final_reward": final_reward,
        }

    return FunctionRewardProvider(_reward)


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def _iter_vllm_model_ids(vllm_payload: Any) -> list[str]:
    if not isinstance(vllm_payload, dict):
        return []
    data = vllm_payload.get("data")
    if not isinstance(data, list):
        return []
    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_ids.append(model_id.strip())
    return model_ids


def _resolve_model_name(
    cli_model: str | None,
    configured_model: str | None,
    vllm_payload: Any,
    config_path: Path | None,
) -> tuple[str | None, str]:
    if cli_model:
        return cli_model, "cli"

    available_model_ids = _iter_vllm_model_ids(vllm_payload)
    resolved_config_path: str | None = None
    if configured_model:
        candidate = configured_model.strip()
        if candidate and config_path is not None and not Path(candidate).is_absolute():
            resolved_path = (config_path.parent / candidate).resolve()
            resolved_config_path = str(resolved_path)
        candidate_variants = [item for item in [configured_model.strip(), resolved_config_path] if item]
        for candidate_variant in candidate_variants:
            if candidate_variant in available_model_ids:
                return candidate_variant, "config"

    if len(available_model_ids) == 1:
        if configured_model:
            return available_model_ids[0], "vllm_models[0].id_fallback_from_invalid_config"
        return available_model_ids[0], "vllm_models[0].id"

    if configured_model:
        if resolved_config_path and Path(resolved_config_path).exists():
            return resolved_config_path, "config"
        candidate = configured_model.strip()
        if candidate and Path(candidate).exists():
            return candidate, "config"
        return configured_model.strip() or None, "config_unverified"

    if available_model_ids:
        return None, "unresolved_multiple_vllm_models"

    return None, "unresolved"


async def _run(args: argparse.Namespace) -> int:
    env_report: dict[str, Any] = {}
    mock_runtime: tempfile.TemporaryDirectory[str] | None = None

    try:
        vllm_ok, vllm_payload, vllm_error = _safe_json_get(f"{args.vllm_url.rstrip('/')}/v1/models")
        env_report["vllm"] = {
            "available": vllm_ok,
            "url": args.vllm_url,
            "error": vllm_error,
            "models": vllm_payload,
        }

        search_ok, search_payload, search_error = _safe_json_get(args.search_health_url)
        if not search_ok:
            search_ok, search_payload, search_error = _safe_json_post(args.search_health_url)
        env_report["search_service"] = {
            "available": search_ok,
            "url": args.search_health_url,
            "error": search_error,
            "payload": search_payload,
        }

        prepare_script_candidates = [
            Path("prepare_drmas_search_data.py").resolve(),
            Path("/home/cxb/OrchRL/examples/mas_app/search/scripts/prepare_drmas_search_data.py"),
        ]
        existing_prepare_script = next(
            (path for path in prepare_script_candidates if path.exists()),
            prepare_script_candidates[0],
        )
        env_report["prepare_data_script"] = {
            "path": str(existing_prepare_script),
            "exists": existing_prepare_script.exists(),
        }

        config_template, config_source = _load_config(args.config)
        prompts, prompts_meta = _load_prompts(args.prompts_file.expanduser(), args.n_prompts)
        env_report["prompts"] = prompts_meta

        prompt_list = [item["prompt"] for item in prompts]
        expected_by_prompt = {item["prompt"]: item.get("expected", "") for item in prompts}

        real_runner_root, real_runner_command = _find_real_runner(args.config, args.mas_work_dir)
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

        use_mock = args.force_mock or (not vllm_ok) or (real_runner_root is None) or (search_required and not search_ok)

        search_root_for_reward = real_runner_root
        runner_root = real_runner_root
        runner_command = real_runner_command
        run_mode_reason: list[str] = []

        if args.force_mock:
            run_mode_reason.append("--force-mock")
        if not vllm_ok:
            run_mode_reason.append("vLLM 不可用")
        if runner_root is None:
            run_mode_reason.append("OrchRL Search MAS 不可用")
        if search_required and not search_ok:
            run_mode_reason.append("检索服务不可用")

        if use_mock:
            mock_runtime = _prepare_mock_runtime()
            runner_root = Path(mock_runtime.name)
            runner_command = f"{sys.executable} scripts/run_search_mas.py --config {{config_path}} --question {{prompt}}"
            mock_search_cfg = config_template.setdefault("search", {})
            if isinstance(mock_search_cfg, dict):
                mock_search_cfg["provider"] = "disabled"

        if runner_command is None or runner_root is None:
            raise RuntimeError("无法确定 MAS 运行命令")

        checker, checker_source = _load_is_search_answer_correct(search_root_for_reward)

        agents_cfg = config_template.get("agents")
        if not isinstance(agents_cfg, dict) or not agents_cfg:
            agents_cfg = {role: {} for role in DEFAULT_ROLES}
            config_template["agents"] = agents_cfg
        roles = list(agents_cfg.keys())

        llm_cfg = config_template.get("llm") if isinstance(config_template.get("llm"), dict) else {}
        configured_model = llm_cfg.get("model") if isinstance(llm_cfg, dict) else None
        model_name, model_source = _resolve_model_name(
            cli_model=args.model,
            configured_model=configured_model if isinstance(configured_model, str) else None,
            vllm_payload=vllm_payload,
            config_path=args.config.expanduser().resolve() if args.config is not None else None,
        )
        env_report["model_path"] = {
            "requested": args.model,
            "configured": configured_model if isinstance(configured_model, str) else None,
            "resolved": model_name,
            "source": model_source,
            "exists": bool(model_name) and Path(model_name).exists(),
            "available_vllm_models": _iter_vllm_model_ids(vllm_payload),
        }

        if use_mock:
            backend: InferenceBackend = ScriptedBackend(expected_by_prompt=expected_by_prompt)
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

        reward_provider = _build_reward_provider(expected_by_prompt=expected_by_prompt, checker=checker)
        results = await parallel_rollout(
            prompts=prompt_list,
            reward_provider=reward_provider,
            config=pipe_config,
            backend=backend,
            n_samples_per_prompt=args.n_samples,
            max_concurrent=args.max_concurrent,
        )

        episodes_payload: list[dict[str, Any]] = []
        episode_ids: list[str] = []
        token_ids_none_count = 0
        length_mismatch_count = 0
        turns_per_episode: list[float] = []
        success_count = 0
        final_rewards: list[float] = []
        agent_call_counts: dict[str, int] = defaultdict(int)
        agent_token_sizes: dict[str, list[int]] = defaultdict(list)

        for result in results:
            payload = asdict(result)
            turns = _collect_all_turns(payload)

            has_required_agents = all(
                role in payload["trajectory"]["agent_trajectories"] and payload["trajectory"]["agent_trajectories"][role]
                for role in roles
            )

            for turn in turns:
                token_ids = turn.get("token_ids")
                logprobs = turn.get("logprobs")
                if token_ids is None:
                    token_ids_none_count += 1
                if isinstance(token_ids, list):
                    agent_token_sizes[turn["agent_role"]].append(len(token_ids))
                if isinstance(token_ids, list) and isinstance(logprobs, list) and len(token_ids) != len(logprobs):
                    length_mismatch_count += 1
                agent_call_counts[turn["agent_role"]] += 1

            episode_id = str(payload["trajectory"]["episode_id"])
            episode_ids.append(episode_id)
            if result.final_reward is not None:
                reward_value = _safe_float(result.final_reward, default=0.0)
                final_rewards.append(reward_value)
                if reward_value == 1.0:
                    success_count += 1

            turns_per_episode.append(float(len(turns)))

            payload["checks"] = {
                "required_agents_complete": has_required_agents,
                "turn_count": len(turns),
                "token_ids_none_turns": sum(1 for turn in turns if turn.get("token_ids") is None),
                "token_logprobs_length_mismatch_turns": sum(
                    1
                    for turn in turns
                    if isinstance(turn.get("token_ids"), list)
                    and isinstance(turn.get("logprobs"), list)
                    and len(turn.get("token_ids", [])) != len(turn.get("logprobs", []))
                ),
            }
            episodes_payload.append(payload)

        unique_episode_ids = len(set(episode_ids)) == len(episode_ids)

        total_episodes = len(results)
        avg_turns = _safe_mean(turns_per_episode)
        success_rate = (success_count / total_episodes) if total_episodes else 0.0

        agent_avg_tokens = {
            role: _safe_mean([_safe_float(v) for v in values])
            for role, values in agent_token_sizes.items()
        }

        reward_reasonable = all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in final_rewards)

        report = {
            "schema_version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_mode": "mock" if use_mock else "real",
            "run_mode_reasons": run_mode_reason,
            "config_source": config_source,
            "environment": env_report,
            "reward_checker_source": checker_source,
            "input": {
                "n_prompts": args.n_prompts,
                "n_samples": args.n_samples,
                "max_concurrent": args.max_concurrent,
                "timeout": args.timeout,
                "model": model_name,
                "vllm_url": args.vllm_url,
                "prompts_file": str(args.prompts_file.expanduser()),
            },
            "summary": {
                "total_episodes": total_episodes,
                "success_count": success_count,
                "failure_count": total_episodes - success_count,
                "success_rate": success_rate,
                "avg_turns": avg_turns,
                "agent_call_counts": dict(agent_call_counts),
                "agent_avg_generated_tokens": agent_avg_tokens,
                "validation": {
                    "unique_episode_ids": unique_episode_ids,
                    "token_ids_none_turns": token_ids_none_count,
                    "token_logprobs_length_mismatch_turns": length_mismatch_count,
                    "required_roles": roles,
                    "reward_reasonableness": {
                        "in_unit_interval": reward_reasonable,
                        "min_final_reward": min(final_rewards) if final_rewards else None,
                        "max_final_reward": max(final_rewards) if final_rewards else None,
                    },
                },
            },
            "episodes": episodes_payload,
        }

        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("=== 真实环境验证摘要 ===")
        print(f"模式: {'mock 模式' if use_mock else 'real 模式'}")
        if run_mode_reason:
            print(f"切换原因: {', '.join(run_mode_reason)}")
        print(f"episode 总数: {total_episodes}")
        print(f"成功率: {success_rate:.2%} ({success_count}/{total_episodes})")
        print(f"平均 turn 数: {avg_turns:.2f}")
        print(f"episode_id 全局唯一: {unique_episode_ids}")
        print(f"token_ids=None turn 数: {token_ids_none_count}")
        print(f"token/logprobs 长度不一致 turn 数: {length_mismatch_count}")
        print(f"reward 合理性（[0,1]）: {reward_reasonable}")
        print(f"输出文件: {output_path}")

        return 0
    finally:
        if mock_runtime is not None:
            mock_runtime.cleanup()


def main() -> int:
    args = _parse_args()
    if args.n_samples < 1:
        raise ValueError("--n-samples 必须 >= 1")
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent 必须 >= 1")
    if args.n_prompts < 1:
        raise ValueError("--n-prompts 必须 >= 1")

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
