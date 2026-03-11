import os
import subprocess
import sys
from pathlib import Path

import pytest

from mate.trajectory.backend import InferenceBackend
from mate.trajectory.datatypes import (
    EpisodeResult,
    EpisodeTrajectory,
    ModelMappingEntry,
    ModelRequest,
    ModelResponse,
)
from mate.trajectory.pipe import AgentPipe, AgentPipeConfig
from mate.trajectory.reward import FunctionRewardProvider


DEFAULT_ORCHRL_SEARCH_DIR = "/home/cxb/OrchRL/examples/mas_app/search"
ORCHRL_SEARCH_DIR = os.environ.get("ORCHRL_SEARCH_DIR", DEFAULT_ORCHRL_SEARCH_DIR)

SCRIPTED_RESPONSES = {
    "verifier": "Based on current context, I can answer directly.\n<verify>yes</verify>",
    "searcher": "I will search for more evidence.\n<search>test query</search>",
    "answerer": "The answer is 42.\n<answer>42</answer>",
}


class ScriptedBackend(InferenceBackend):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        content = SCRIPTED_RESPONSES.get(request.agent_role, "unknown agent")
        token_count = max(1, len(content.split()))
        return ModelResponse(
            content=content,
            token_ids=list(range(1, token_count + 1)),
            logprobs=[-0.1] * token_count,
            finish_reason="stop",
        )


def _reward_on_answer_tag(trajectory: EpisodeTrajectory) -> dict[str, object]:
    answer_turns = trajectory.agent_trajectories.get("answerer", [])
    if not answer_turns:
        return {"agent_rewards": {}, "final_reward": 0.0}

    answer_text = answer_turns[-1].response_text.lower()
    has_answer_tag = "<answer>" in answer_text and "</answer>" in answer_text
    final_reward = 1.0 if has_answer_tag else 0.0
    return {
        "agent_rewards": {
            role: final_reward for role in trajectory.agent_trajectories
        },
        "final_reward": final_reward,
    }


def _probe_output_tail(probe: subprocess.CompletedProcess[str]) -> str:
    return (probe.stderr or probe.stdout).strip()[-300:]


def _ensure_orchrl_runtime_ready() -> None:
    work_dir = Path(ORCHRL_SEARCH_DIR)
    run_script = work_dir / "scripts" / "run_search_mas.py"
    if not run_script.exists():
        pytest.skip(f"OrchRL run script not found: {run_script}")

    try:
        probe = subprocess.run(
            [sys.executable, str(run_script), "--help"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"OrchRL environment is not runnable: {exc}")

    if probe.returncode != 0:
        pytest.skip(
            "OrchRL environment probe failed "
            f"(exit={probe.returncode}): {_probe_output_tail(probe)}"
        )

    try:
        import_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, '.'); import search_mas.apps.search.app",
            ],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"OrchRL import probe could not run: {exc}")

    if import_probe.returncode != 0:
        pytest.skip(
            "OrchRL import probe failed "
            f"(exit={import_probe.returncode}): {_probe_output_tail(import_probe)}"
        )


def test_ensure_orchrl_runtime_ready_skips_when_import_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work_dir = tmp_path / "search"
    run_script = work_dir / "scripts" / "run_search_mas.py"
    run_script.parent.mkdir(parents=True)
    run_script.write_text("print('placeholder')\n")
    monkeypatch.setattr(
        sys.modules[__name__], "ORCHRL_SEARCH_DIR", str(work_dir)
    )

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if "--help" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="usage: run_search_mas.py",
                stderr="",
            )

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="SyntaxError: bad import",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(pytest.skip.Exception, match="import probe failed|bad import"):
        _ensure_orchrl_runtime_ready()

    assert calls[0][-1] == "--help"
    assert (
        calls[1][-1]
        == "import sys; sys.path.insert(0, '.'); import search_mas.apps.search.app"
    )


@pytest.fixture
def orchrl_config() -> dict[str, object]:
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
            "model": "default",
        },
        "search": {"provider": "disabled"},
        "agents": {
            "verifier": {"temperature": 0.2, "max_tokens": 512},
            "searcher": {"temperature": 0.6, "max_tokens": 512},
            "answerer": {"temperature": 0.4, "max_tokens": 512},
        },
    }


@pytest.mark.skipif(
    Path(ORCHRL_SEARCH_DIR).exists() is False,
    reason="OrchRL Search MAS not available at expected path",
)
async def test_orchrl_search_mas_e2e(orchrl_config: dict[str, object]) -> None:
    _ensure_orchrl_runtime_ready()

    pipe_config = AgentPipeConfig(
        mas_command_template=(
            f"{sys.executable} scripts/run_search_mas.py"
            " --config {config_path}"
            " --question {prompt}"
        ),
        config_template=orchrl_config,
        model_mapping={
            "verifier": ModelMappingEntry(),
            "searcher": ModelMappingEntry(),
            "answerer": ModelMappingEntry(),
        },
        timeout=60.0,
        mas_work_dir=ORCHRL_SEARCH_DIR,
    )

    pipe = AgentPipe(config=pipe_config, backend=ScriptedBackend())
    result = await pipe.run(
        prompt="What is the meaning of life?",
        reward_provider=FunctionRewardProvider(_reward_on_answer_tag),
    )

    assert isinstance(result, EpisodeResult)
    assert result.trajectory.episode_id
    assert "verifier" in result.trajectory.agent_trajectories
    assert "answerer" in result.trajectory.agent_trajectories

    verifier_turns = result.trajectory.agent_trajectories["verifier"]
    assert len(verifier_turns) >= 1
    assert "<verify>yes</verify>" in verifier_turns[0].response_text.lower()

    answerer_turns = result.trajectory.agent_trajectories["answerer"]
    assert len(answerer_turns) >= 1
    assert "<answer>42</answer>" in answerer_turns[0].response_text.lower()

    assert result.final_reward == 1.0
    assert result.rewards["verifier"] == 1.0
    assert result.rewards["answerer"] == 1.0
    assert result.metadata.get("exit_code") == 0
