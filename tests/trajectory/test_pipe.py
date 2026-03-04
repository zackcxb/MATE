import sys
from pathlib import Path

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


class EchoBackend(InferenceBackend):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"echo:{request.agent_role}",
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            finish_reason="stop",
        )


def _reward_fn(trajectory: EpisodeTrajectory) -> dict[str, object]:
    return {
        "agent_rewards": {role: 1.0 for role in trajectory.agent_trajectories},
        "final_reward": 1.0,
    }


async def test_agent_pipe_end_to_end(tmp_path: Path) -> None:
    script = tmp_path / "tiny_mas.py"
    script.write_text(
        """
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
base_url = config["llm"]["base_url"]
prompt = sys.argv[2]

for role in ("verifier", "answerer"):
    payload = json.dumps(
        {
            "model": role,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        data = json.loads(response.read().decode("utf-8"))
    _ = data["choices"][0]["message"]["content"]
""".strip(),
        encoding="utf-8",
    )

    config_template = {
        "llm": {
            "base_url": "http://placeholder/v1",
            "model": "default-model",
        },
        "agents": {
            "verifier": {"temperature": 0.2},
            "answerer": {"temperature": 0.4},
        },
    }
    config = AgentPipeConfig(
        mas_command_template=f"{sys.executable} {script} {{config_path}} {{prompt}}",
        config_template=config_template,
        model_mapping={
            "verifier": ModelMappingEntry(),
            "answerer": ModelMappingEntry(),
        },
        timeout=30.0,
    )
    pipe = AgentPipe(config=config, backend=EchoBackend())
    reward_provider = FunctionRewardProvider(_reward_fn)

    result = await pipe.run(prompt="What is 2 + 2 ?", reward_provider=reward_provider)

    assert isinstance(result, EpisodeResult)
    assert set(result.trajectory.agent_trajectories.keys()) == {"verifier", "answerer"}
    assert len(result.trajectory.agent_trajectories["verifier"]) == 1
    assert len(result.trajectory.agent_trajectories["answerer"]) == 1
    assert result.trajectory.agent_trajectories["verifier"][0].response_text == "echo:verifier"
    assert result.trajectory.agent_trajectories["answerer"][0].response_text == "echo:answerer"
    assert result.trajectory.agent_trajectories["verifier"][0].messages[0]["content"] == "What is 2 + 2 ?"
    assert result.rewards["verifier"] == 1.0
    assert result.rewards["answerer"] == 1.0
    assert result.final_reward == 1.0
    assert result.metadata["exit_code"] == 0
