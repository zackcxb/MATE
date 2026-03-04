from __future__ import annotations

import copy
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


class MASLauncher:
    def __init__(self, work_dir: str | Path | None = None) -> None:
        self._work_dir = Path(work_dir) if work_dir is not None else None
        self._temp_files: list[Path] = []

    def prepare_config(
        self,
        config_template: dict[str, Any],
        monitor_url: str,
        agent_roles: list[str],
    ) -> Path:
        config = copy.deepcopy(config_template)

        llm_cfg = config.setdefault("llm", {})
        if isinstance(llm_cfg, dict):
            llm_cfg["base_url"] = monitor_url

        agents_cfg = config.setdefault("agents", {})
        if not isinstance(agents_cfg, dict):
            agents_cfg = {}
            config["agents"] = agents_cfg

        for role in agent_roles:
            role_cfg = agents_cfg.setdefault(role, {})
            if not isinstance(role_cfg, dict):
                role_cfg = {}
                agents_cfg[role] = role_cfg
            role_cfg["model"] = role

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="mate_mas_",
            dir=str(self._work_dir) if self._work_dir else None,
            delete=False,
            encoding="utf-8",
        )
        try:
            yaml.safe_dump(config, temp_file, sort_keys=False)
        finally:
            temp_file.close()

        config_path = Path(temp_file.name)
        self._temp_files.append(config_path)
        return config_path

    def launch(
        self,
        command: str,
        env_vars: dict[str, str] | None = None,
    ) -> subprocess.Popen[str]:
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        return subprocess.Popen(
            command,
            shell=True,
            cwd=str(self._work_dir) if self._work_dir else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def wait(
        self,
        process: subprocess.Popen[Any],
        timeout: float | None = None,
    ) -> int:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()

    def cleanup(self) -> None:
        for path in self._temp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self._temp_files.clear()
