# 向 OrchRL 提交 Trajectory Engine PR — Coding Agent Prompt

> 粘贴到 coding agent 窗口中执行。

---

## 角色

你是一位代码移植工程师，负责将 MATE-reboot 仓库中的 Trajectory Engine 核心代码提取、重命名并提交到 OrchRL 仓库作为 PR。

## 前置阅读

1. `/home/cxb/MATE-reboot/AGENTS.md` — 项目治理规则
2. `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` — V0 架构设计（理解模块职责）
3. `/home/cxb/OrchRL/README.md` — OrchRL 仓库定位

## 背景

MATE-reboot 仓库中的 Trajectory Engine（`mate/trajectory/`）已完成 V0 实现和真实环境验证（66 tests passing）。训练侧同事已通过适配方式完成端到端联调。现在需要将核心代码提取到 OrchRL 仓库中上库。

**关键约定**：
- 代码放在 `OrchRL/trajectory/`（仓库顶层的 `trajectory` 目录）
- 包名从 `mate.trajectory` 改为 `trajectory`
- 基于 OrchRL `main` 分支（当前在 `87a8256`）创建 PR 分支
- 不包含本地开发临时文档、验证脚本和 artifacts

## 源码和目标映射

### 核心源码（必须迁移）

| 源路径（MATE-reboot） | 目标路径（OrchRL） | 说明 |
|---|---|---|
| `mate/trajectory/__init__.py` | `trajectory/__init__.py` | 导出符号，无需修改 import |
| `mate/trajectory/datatypes.py` | `trajectory/datatypes.py` | 数据类型，无内部依赖 |
| `mate/trajectory/backend.py` | `trajectory/backend.py` | InferenceBackend + VLLMBackend |
| `mate/trajectory/monitor.py` | `trajectory/monitor.py` | ModelMonitor HTTP 代理 |
| `mate/trajectory/launcher.py` | `trajectory/launcher.py` | MASLauncher 进程管理 |
| `mate/trajectory/collector.py` | `trajectory/collector.py` | TrajectoryCollector |
| `mate/trajectory/reward.py` | `trajectory/reward.py` | RewardWorker + RewardProvider |
| `mate/trajectory/pipe.py` | `trajectory/pipe.py` | AgentPipe 顶层编排 |
| `mate/trajectory/parallel.py` | `trajectory/parallel.py` | parallel_rollout |

### 测试（必须迁移）

| 源路径（MATE-reboot） | 目标路径（OrchRL） | 说明 |
|---|---|---|
| `tests/trajectory/test_datatypes.py` | `tests/trajectory/test_datatypes.py` | |
| `tests/trajectory/test_backend.py` | `tests/trajectory/test_backend.py` | |
| `tests/trajectory/test_monitor.py` | `tests/trajectory/test_monitor.py` | |
| `tests/trajectory/test_launcher.py` | `tests/trajectory/test_launcher.py` | |
| `tests/trajectory/test_collector.py` | `tests/trajectory/test_collector.py` | |
| `tests/trajectory/test_reward.py` | `tests/trajectory/test_reward.py` | |
| `tests/trajectory/test_pipe.py` | `tests/trajectory/test_pipe.py` | |
| `tests/trajectory/test_parallel.py` | `tests/trajectory/test_parallel.py` | |

**用例整改**：迁移过程中可以对用例进行梳理，仅保留适合加入流水线测试的用例，去除仅适合用于TDD开发的用例。

**不迁移**：`tests/trajectory/test_orchrl_integration.py`（依赖本地 OrchRL 路径）、`tests/scripts/`、`scripts/`、`artifacts/`、`docs/`。

### 新增文件

| 目标路径（OrchRL） | 说明 |
|---|---|
| `tests/trajectory/__init__.py` | 空 `__init__.py`，使 pytest 发现测试 |
| `tests/__init__.py` | 空 `__init__.py` |

## 任务步骤

### 第一步：在 OrchRL 创建 PR 分支

```bash
cd /home/cxb/OrchRL
git checkout main
git checkout -b feat/trajectory-engine
```

### 第二步：复制核心源码

将 `mate/trajectory/` 的 9 个 Python 文件复制到 `trajectory/`。

```bash
mkdir -p trajectory
cp /home/cxb/MATE-reboot/mate/trajectory/*.py trajectory/
```

### 第三步：修改 import 路径

所有文件中的 `mate.trajectory` 引用都是**相对 import**（`from .datatypes import ...`），所以核心源码不需要改 import 路径。

但需要检查并确认：
1. 所有 `from .xxx import` 形式的相对 import 在新位置仍然正确（应该没问题，因为目录结构保持一致）
2. 没有任何 `from mate.trajectory` 或 `import mate` 的绝对引用

### 第四步：复制并修改测试文件

```bash
mkdir -p tests/trajectory
touch tests/__init__.py
touch tests/trajectory/__init__.py
cp /home/cxb/MATE-reboot/tests/trajectory/test_datatypes.py tests/trajectory/
cp /home/cxb/MATE-reboot/tests/trajectory/test_backend.py tests/trajectory/
cp /home/cxb/MATE-reboot/tests/trajectory/test_monitor.py tests/trajectory/
cp /home/cxb/MATE-reboot/tests/trajectory/test_launcher.py tests/trajectory/
cp /home/cxb/MATE-reboot/tests/trajectory/test_collector.py tests/trajectory/
cp /home/cxb/MATE-reboot/tests/trajectory/test_reward.py tests/trajectory/
cp /home/cxb/MATE-reboot/tests/trajectory/test_pipe.py tests/trajectory/
cp /home/cxb/MATE-reboot/tests/trajectory/test_parallel.py tests/trajectory/
```

然后在所有测试文件中，将 `from mate.trajectory` 改为 `from trajectory`。具体来说：
- `from mate.trajectory import ...` → `from trajectory import ...`
- `from mate.trajectory.xxx import ...` → `from trajectory.xxx import ...`

检查每个测试文件的 import 部分，确认全部改完。并对用例进行整改，删去无需加入流水线的用例。

### 第五步：编写精简示例脚本

在 `examples/trajectory/` 下创建一个 ~100 行的示例脚本 `run_trajectory_example.py`，展示核心用法：

```python
#!/usr/bin/env python3
"""
Trajectory Engine 使用示例。

展示如何使用 AgentPipe + parallel_rollout 对 DrMAS Search MAS 进行轨迹采集。
需要：1) vLLM 服务在线  2) OrchRL Search MAS 可用  3) 检索服务在线

用法：
    python examples/trajectory/run_trajectory_example.py \
        --vllm-url http://127.0.0.1:8000 \
        --model Qwen3-4B-Instruct-2507 \
        --mas-dir /path/to/OrchRL/examples/mas_app/search \
        --config /path/to/search_mas_example.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from trajectory import (
    AgentPipeConfig,
    FunctionRewardProvider,
    ModelMappingEntry,
    VLLMBackend,
    parallel_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory Engine example")
    parser.add_argument("--vllm-url", type=str, required=True, help="vLLM service URL")
    parser.add_argument("--model", type=str, required=True, help="Model name served by vLLM")
    parser.add_argument("--mas-dir", type=Path, required=True, help="OrchRL Search MAS directory")
    parser.add_argument("--config", type=Path, required=True, help="Search MAS YAML config")
    parser.add_argument("--prompt", type=str, default="who got the first nobel prize in physics?")
    parser.add_argument("--n-samples", type=int, default=2, help="Episodes per prompt")
    parser.add_argument("--output", type=Path, default=Path("trajectory_output.json"))
    return parser.parse_args()


def build_reward_provider() -> FunctionRewardProvider:
    import re

    def _reward(trajectory):
    """参考DrMAS的compute_score实现reward函数"""

    return FunctionRewardProvider(_reward)


async def main() -> None:
    args = parse_args()

    import yaml
    config_template = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    backend = VLLMBackend(backend_url=args.vllm_url, actual_model=args.model)
    roles = list(config_template.get("agents", {}).keys()) or ["verifier", "searcher", "answerer"]

    pipe_config = AgentPipeConfig(
        mas_command_template=(
            f"{sys.executable} scripts/run_search_mas.py "
            "--config {config_path} --question {prompt}"
        ),
        config_template=config_template,
        model_mapping={role: ModelMappingEntry(actual_model=args.model) for role in roles},
        timeout=180.0,
        mas_work_dir=args.mas_dir,
    )

    results = await parallel_rollout(
        prompts=[args.prompt],
        reward_provider=build_reward_provider(),
        config=pipe_config,
        backend=backend,
        n_samples_per_prompt=args.n_samples,
        max_concurrent=args.n_samples,
    )

    output = [asdict(r) for r in results]
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Collected {len(results)} episodes → {args.output}")
    for r in results:
        print(f"  episode={r.trajectory.episode_id[:8]}  reward={r.final_reward}  agents={list(r.trajectory.agent_trajectories.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
```

上面的代码是示例参考，请你据此编写实际文件，确保能够正确运行。同时创建 `examples/trajectory/README.md`，用简短的几行说明前置条件和运行方式。

### 第六步：运行测试

```bash
cd /home/cxb/OrchRL
python -m pytest tests/trajectory/ -q --tb=short
```

**验收标准**：无失败。如有因 import 路径未改完导致的失败，修复后重跑。

### 第七步：提交

使用两个原子提交：

**提交 1：核心代码 + 测试**

```bash
git add trajectory/ tests/
git commit -m "feat: add trajectory engine for multi-agent trajectory collection

Core modules: ModelMonitor (HTTP proxy with strategy-pattern backend),
VLLMBackend, MASLauncher (subprocess lifecycle), TrajectoryCollector,
RewardWorker, AgentPipe (top-level orchestrator), parallel_rollout.

Migrated from MATE-reboot with package rename (mate.trajectory → trajectory).
All 56+ unit tests included and passing."
```

**提交 2：示例脚本**

```bash
git add examples/trajectory/
git commit -m "docs: add trajectory engine usage example for DrMAS Search MAS"
```

### 第八步：最终验证

```bash
# 测试仍通过
python -m pytest tests/trajectory/ -q --tb=short

# 工作树干净
git status

# 提交历史
git log --oneline -5
```

## 不要做的事

- 不要推送到远端（`git push`）—— 等用户确认后再推
- 不要复制 `test_orchrl_integration.py`（它依赖本地路径）
- 不要复制 `scripts/run_real_validation.py` 或 `tests/scripts/`
- 不要复制 `docs/`、`artifacts/`、`skills/` 等本地开发文件
- 不要修改 `examples/mas_app/` 下已有的任何文件
- 不要创建 `pyproject.toml` 或 `setup.py`（OrchRL 目前无包管理，保持一致）
- 不要在代码中留下 `mate` 相关的包名引用
