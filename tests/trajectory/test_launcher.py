import shlex
import sys

import yaml

from mate.trajectory.launcher import MASLauncher


def _make_template() -> dict:
    return {
        "llm": {
            "base_url": "http://original:8000/v1",
            "model": "Qwen3-4B",
            "max_tokens": 1024,
        },
        "agents": {
            "verifier": {"temperature": 0.2},
            "searcher": {"temperature": 0.6},
            "answerer": {"temperature": 0.4},
        },
        "metadata": {
            "experiment": "task-5",
        },
    }


def test_prepare_config_replaces_base_url():
    launcher = MASLauncher()
    try:
        out = launcher.prepare_config(
            config_template=_make_template(),
            monitor_url="http://127.0.0.1:19000/v1",
            agent_roles=["verifier", "searcher", "answerer"],
        )
        result = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert result["llm"]["base_url"] == "http://127.0.0.1:19000/v1"
    finally:
        launcher.cleanup()


def test_prepare_config_injects_agent_model_names():
    launcher = MASLauncher()
    try:
        out = launcher.prepare_config(
            config_template=_make_template(),
            monitor_url="http://127.0.0.1:19000/v1",
            agent_roles=["verifier", "searcher", "answerer"],
        )
        result = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert result["agents"]["verifier"]["model"] == "verifier"
        assert result["agents"]["searcher"]["model"] == "searcher"
        assert result["agents"]["answerer"]["model"] == "answerer"
    finally:
        launcher.cleanup()


def test_prepare_config_preserves_other_fields():
    template = _make_template()
    launcher = MASLauncher()
    try:
        out = launcher.prepare_config(
            config_template=template,
            monitor_url="http://127.0.0.1:19000/v1",
            agent_roles=["verifier", "searcher", "answerer"],
        )
        result = yaml.safe_load(out.read_text(encoding="utf-8"))
        assert result["llm"]["model"] == "Qwen3-4B"
        assert result["llm"]["max_tokens"] == 1024
        assert result["agents"]["verifier"]["temperature"] == 0.2
        assert result["agents"]["searcher"]["temperature"] == 0.6
        assert result["metadata"]["experiment"] == "task-5"
        assert "model" not in template["agents"]["verifier"]
    finally:
        launcher.cleanup()


def test_launch_and_wait_success():
    launcher = MASLauncher()
    command = f"{shlex.quote(sys.executable)} -c \"print('hello')\""
    process = launcher.launch(command=command)
    exit_code = launcher.wait(process, timeout=10.0)
    assert exit_code == 0


def test_launch_and_wait_timeout_kills_process():
    launcher = MASLauncher()
    command = f"{shlex.quote(sys.executable)} -c \"import time; time.sleep(60)\""
    process = launcher.launch(command=command)
    exit_code = launcher.wait(process, timeout=0.3)
    assert exit_code != 0
    assert process.poll() is not None
