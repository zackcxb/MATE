from __future__ import annotations

from typing import Any, Callable, Protocol

from .datatypes import EpisodeResult, EpisodeTrajectory


class RewardProvider(Protocol):
    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]: ...


class FunctionRewardProvider:
    def __init__(self, func: Callable[[EpisodeTrajectory], dict[str, Any]]):
        self._func = func

    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]:
        return self._func(trajectory)


class RewardWorker:
    def compute(self, trajectory: EpisodeTrajectory, provider: RewardProvider) -> EpisodeResult:
        result = provider.compute(trajectory)
        return EpisodeResult(
            trajectory=trajectory,
            rewards=result.get("agent_rewards", {}),
            final_reward=result.get("final_reward"),
            metadata={},
        )
