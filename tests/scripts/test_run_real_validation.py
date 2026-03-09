import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_real_validation as validation

ORCHRL_SEARCH_ROOT = Path("/home/cxb/OrchRL/examples/mas_app/search")


def test_extract_expected_prefers_structured_answers_over_stringified_list() -> None:
    record = {
        "expected_answer": "['Wilhelm Conrad Röntgen']",
        "expected_answers": ["Wilhelm Conrad Röntgen"],
    }

    value = validation._extract_expected(
        record,
        ["expected_answer", "expected_answers"],
    )

    assert value == ["Wilhelm Conrad Röntgen"]


def test_build_reward_provider_accepts_stringified_answer_lists() -> None:
    reward_provider = validation._build_reward_provider(
        expected_by_prompt={
            "who got the first nobel prize in physics?": "['Wilhelm Conrad Röntgen']",
        },
        checker=lambda predicted, expected: (
            predicted == "Wilhelm Röntgen"
            and expected == ["Wilhelm Conrad Röntgen"]
        ),
    )
    trajectory = SimpleNamespace(
        agent_trajectories={
            "answerer": [
                SimpleNamespace(
                    response_text="<answer>Wilhelm Röntgen</answer>",
                    messages=[
                        {
                            "role": "user",
                            "content": "who got the first nobel prize in physics?",
                        }
                    ],
                )
            ]
        }
    )

    result = reward_provider.compute(trajectory)

    assert result["final_reward"] == 1.0
    assert result["agent_rewards"]["answerer"] == 1.0


@pytest.mark.skipif(
    ORCHRL_SEARCH_ROOT.exists() is False,
    reason="OrchRL Search MAS not available at expected path",
)
def test_build_reward_provider_works_with_real_orchrl_checker() -> None:
    checker, source = validation._load_is_search_answer_correct(ORCHRL_SEARCH_ROOT)
    reward_provider = validation._build_reward_provider(
        expected_by_prompt={
            "who got the first nobel prize in physics?": "['Wilhelm Conrad Röntgen']",
        },
        checker=checker,
    )
    trajectory = SimpleNamespace(
        agent_trajectories={
            "answerer": [
                SimpleNamespace(
                    response_text="<answer>Wilhelm Conrad Röntgen</answer>",
                    messages=[
                        {
                            "role": "user",
                            "content": "who got the first nobel prize in physics?",
                        }
                    ],
                )
            ]
        }
    )

    result = reward_provider.compute(trajectory)

    assert "is_search_answer_correct" in source
    assert result["final_reward"] == 1.0
    assert result["agent_rewards"]["answerer"] == 1.0


def test_resolve_model_name_prefers_live_vllm_model_when_configured_path_missing() -> None:
    model_name, source = validation._resolve_model_name(
        cli_model=None,
        configured_model="/data1/lll/models/Qwen3-4B-Instruct-2507",
        vllm_payload={
            "data": [
                {
                    "id": "/data1/models/Qwen/Qwen3-4B-Instruct-2507",
                }
            ]
        },
        config_path=None,
    )

    assert model_name == "/data1/models/Qwen/Qwen3-4B-Instruct-2507"
    assert source == "vllm_models[0].id_fallback_from_invalid_config"


def test_resolve_model_name_does_not_silently_switch_when_multiple_vllm_models_exist() -> None:
    model_name, source = validation._resolve_model_name(
        cli_model=None,
        configured_model="/missing/model",
        vllm_payload={
            "data": [
                {"id": "model-a"},
                {"id": "model-b"},
            ]
        },
        config_path=None,
    )

    assert model_name == "/missing/model"
    assert source == "config_unverified"


def test_resolve_model_name_resolves_relative_model_path_from_config_directory(tmp_path) -> None:
    model_dir = tmp_path / "models" / "Qwen3-4B-Instruct-2507"
    model_dir.mkdir(parents=True)
    config_path = tmp_path / "configs" / "search.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("llm: {}\n", encoding="utf-8")

    model_name, source = validation._resolve_model_name(
        cli_model=None,
        configured_model="../models/Qwen3-4B-Instruct-2507",
        vllm_payload={"data": []},
        config_path=config_path,
    )

    assert model_name == str(model_dir.resolve())
    assert source == "config"
