from __future__ import annotations

from .datatypes import EpisodeTrajectory, InteractionRecord, TurnData


class TrajectoryCollector:
    def build(self, buffer: list[InteractionRecord], episode_id: str) -> EpisodeTrajectory:
        grouped: dict[str, list[TurnData]] = {}
        for record in buffer:
            turn = self._to_turn_data(record)
            grouped.setdefault(record.agent_role, []).append(turn)

        for turns in grouped.values():
            turns.sort(key=lambda turn: turn.turn_index)

        return EpisodeTrajectory(episode_id=episode_id, agent_trajectories=grouped, metadata={})

    @staticmethod
    def _to_turn_data(record: InteractionRecord) -> TurnData:
        return TurnData(
            agent_role=record.agent_role,
            turn_index=record.turn_index,
            messages=record.messages,
            response_text=record.response_text,
            token_ids=record.token_ids,
            logprobs=record.logprobs,
            finish_reason=record.finish_reason,
            timestamp=record.timestamp,
            metadata=record.metadata,
        )
