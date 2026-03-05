# Trajectory Engine V0 — 功能延伸 Prompt (vLLM token_ids + 并行采样)

> 在新 Agent 窗口中粘贴此 prompt。需要在步骤 1（Review + Merge）完成后执行。

---

## 角色

你是一个 AI-Infra 开发工程师，在 `/home/cxb/MATE-reboot` 仓库中为 Trajectory Engine 开发两个功能延伸。

## 背景

Trajectory Engine V0 已实现并合并到 `main`（如果尚未合并，请先确认 `main` 分支包含 `mate/trajectory/` 目录，否则等待合并完成）。

V0 包含：AgentPipe 编排器、ModelMonitor（策略模式）、VLLMBackend、MASLauncher、TrajectoryCollector、RewardWorker。56 个测试全部通过。

## 前置阅读（按顺序）

1. `/home/cxb/MATE-reboot/AGENTS.md` — 治理规则
2. `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` — 设计文档
3. `/home/cxb/MATE-reboot/mate/trajectory/backend.py` — 当前 VLLMBackend 实现
4. `/home/cxb/MATE-reboot/mate/trajectory/pipe.py` — 当前 AgentPipe 实现

## 两个独立 Feature（可按顺序或并行开发）

---

### Feature A: VLLMBackend token_ids 提取

**问题**：当前 VLLMBackend 始终返回 `token_ids=None`，因为标准 OpenAI Chat Completions API 响应不含 token ID。但 RL 训练需要 token_ids 和对齐的 logprobs。

**目标**：让 VLLMBackend 在 vLLM 后端可用时提取 token_ids。

**技术调研方向**：
1. vLLM 的 `extra_body` 参数是否支持返回 token_ids？检查 vLLM 文档和源码
2. vLLM 的 `/v1/completions` 端点（非 chat）是否返回 token-level 信息更丰富？
3. SGLang 的 API 是否有不同的 token_ids 返回方式？
4. 参考 SWE-agent recipe 中 `server_manager.generate()` 返回的 `TokenOutput`（含 `token_ids` 和 `log_probs`）——这是 VerlBackend 路径的做法，VLLMBackend 不一定能完全复制

**参考代码**：
- vLLM 源码：`/home/cxb/rl_framework/verl/` 下搜索 vLLM 相关的 API 实现
- SWE-agent recipe：`/home/cxb/rl_framework/verl/recipe/swe_agent/swe_agent_loop.py`

**交付物**：
- 更新 `mate/trajectory/backend.py` 中的 `VLLMBackend`
- 添加对应测试
- 如果 vLLM 不支持通过标准 API 返回 token_ids，记录结论并说明 VerlBackend（训练模式）才是获取 token_ids 的正确路径
- 独立分支 `feat/vllm-token-ids`

---

### Feature B: Episode 并行采样

**问题**：当前 AgentPipe 一次只跑一个 episode。RL 训练需要同一 prompt 的 N 条并行轨迹来计算 GRPO advantage。

**目标**：实现 `parallel_rollout()` 函数，并发运行多个 AgentPipe 实例采集 N 条 episode。

**设计要点**：
1. 每个 AgentPipe 实例有独立的 Monitor（独立端口）和 MAS 子进程，天然隔离
2. 使用 `asyncio.gather()` 并发编排
3. 支持同一 prompt 的 N 条并行采样（GRPO 场景）
4. 支持不同 prompt 的批量采样
5. 返回 `List[EpisodeResult]`

**接口草案**：
```python
async def parallel_rollout(
    prompts: list[str],
    reward_provider: RewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    n_samples_per_prompt: int = 1,
    max_concurrent: int | None = None,
) -> list[EpisodeResult]:
    """
    对每个 prompt 并行采样 n_samples_per_prompt 条 episode。
    max_concurrent 限制同时运行的 AgentPipe 数量（None = 不限制）。
    """
```

**测试要点**：
- 同一 prompt 的 N 条轨迹具有不同的 episode_id
- 并发数限制（max_concurrent）生效
- 所有 episode 的 Monitor 端口不冲突（port=0 自动分配）
- 部分 episode 失败不影响其他 episode 的结果收集

**交付物**：
- 新增 `mate/trajectory/parallel.py`（或直接扩展 `pipe.py`）
- 添加对应测试
- 独立分支 `feat/parallel-rollout`

---

## 工作流

1. 从 `main` 拉分支（`feat/vllm-token-ids` 和/或 `feat/parallel-rollout`）
2. TDD 开发（写失败测试 → 实现 → 验证 → 提交）
3. 完成后运行完整测试套件确保无回归
4. 每个 feature 保持独立分支，不互相依赖

## 注意事项

- 遵循 `AGENTS.md` 所有规则
- Feature A 可能需要技术调研，如果结论是"标准 API 无法获取 token_ids"，这也是有价值的交付——记录结论即可
- Feature B 的 `max_concurrent` 限制使用 `asyncio.Semaphore` 即可
- 所有回复使用中文
