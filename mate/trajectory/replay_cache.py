from __future__ import annotations

from mate.trajectory.datatypes import InteractionRecord, ModelResponse


class ReplayCache:
    def __init__(self, entries: dict[tuple[str, int], ModelResponse]) -> None:
        self._entries = entries

    @classmethod
    def from_buffer(
        cls,
        buffer: list[InteractionRecord],
        branch_at_global_position: int | None = None,
    ) -> "ReplayCache":
        sorted_buffer = sorted(buffer, key=lambda record: record.timestamp)
        if branch_at_global_position is not None:
            sorted_buffer = sorted_buffer[:branch_at_global_position]

        entries: dict[tuple[str, int], ModelResponse] = {}
        for record in sorted_buffer:
            entries[(record.agent_role, record.turn_index)] = ModelResponse(
                content=record.response_text,
                token_ids=record.token_ids,
                logprobs=record.logprobs,
                finish_reason=record.finish_reason,
            )

        return cls(entries)

    def lookup(self, agent_role: str, turn_index: int) -> ModelResponse | None:
        return self._entries.get((agent_role, turn_index))

    def __len__(self) -> int:
        return len(self._entries)
