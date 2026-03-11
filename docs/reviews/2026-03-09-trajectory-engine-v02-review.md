# Trajectory Engine V0.2 实施验收报告

> 日期：2026-03-09
> 审核方式：brainstorming 会话启动 code-reviewer subagent，对照实施计划逐 Task 验收
> 对照文档：`docs/plans/2026-03-09-trajectory-engine-v02-impl-plan.md`（8 个 Task）
> 设计文档：`docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`

---

## 结论：PASS

8/8 个计划 Task 全部通过。代码质量高，测试覆盖充分，向后兼容性完整。

---

## 量化指标

| 指标 | V0 基线 | V0.2 | 变化 |
|------|---------|------|------|
| 测试数 | 66 passed | 86 passed, 1 skipped | +20 |
| 核心代码行数 | ~798 | ~1080 | +~280 |
| 测试代码行数 | — | +~430 | 超出计划预估（~180）|
| 版本 | 0.1.0 | 0.2.0 | — |
| `parallel_rollout` 接口 | baseline | 未修改 | 无回归 |

---

## Task 逐项验收

### Task 1: 新增数据结构 — PASS

**对照**：`mate/trajectory/datatypes.py`

- `BranchResult` dataclass：包含 `episode_result`, `branch_turn`, `branch_agent_role`, `parent_episode_id` 四个字段，与计划一致
- `TreeEpisodeResult` dataclass：包含 `pilot_result`, `branch_results`, `prompt`, `tree_metadata`，与计划一致
- `EpisodeResult` 新增 `status`（默认 `"success"`）和 `failure_info`（默认 `None`），带默认值不破坏现有代码
- 测试覆盖：`test_branch_result_fields`, `test_tree_episode_result_fields`, `test_episode_result_status_default`

### Task 2: ReplayCache — PASS

**对照**：`mate/trajectory/replay_cache.py`

- `from_buffer` 按 timestamp 排序后切片截断，比计划中的 `enumerate + break` 更简洁
- `lookup(agent_role, turn_index, messages=None)` 增加了可选的 `messages` 参数做 hash 校验——超出计划的改进，实现了设计审核 WARNING-3 中的 `messages_hash` 规范（`sha256 + json.dumps(sort_keys=True)`）
- `__len__` 支持
- 测试覆盖：5 个测试，包含 hash 校验场景和截断场景

### Task 3: Monitor 支持 ReplayCache — PASS

**对照**：`mate/trajectory/monitor.py`

- `replay_cache=None` 默认参数，向后兼容
- 缓存命中时不调用 backend，直接返回缓存响应
- buffer 记录标记 `metadata["replayed"] = True`
- 缓存命中时同时传递 messages 做 hash 校验，命中但 hash 不匹配时回退到 backend（对应设计审核 BLOCKER-1 的降级语义）
- 测试覆盖：`test_monitor_uses_replay_cache`、`test_monitor_falls_back_to_backend_when_replay_messages_do_not_match`

### Task 4: AgentPipe 优雅降级 — PASS

**对照**：`mate/trajectory/pipe.py`

- `allow_partial: bool = False` 默认参数，向后兼容
- `allow_partial=True` 且 MAS 非零退出时：构建部分轨迹，返回 `status="failed"` 的 `EpisodeResult`
- `allow_partial=False` 时：保持原有 `raise RuntimeError` 行为
- `finally` 块正确处理 partial result 存在时的异常优先级
- 测试覆盖：`test_pipe_returns_partial_result_on_mas_failure`, `test_pipe_returns_partial_result_when_stop_fails`, 原有 `test_agent_pipe_run_raises_on_nonzero_exit_code`

### Task 5: tree_rollout 函数 — PASS

**对照**：`mate/trajectory/tree.py`、`mate/trajectory/pipe.py`

- `tree_rollout` 是 `async def` 函数，不是类——遵循设计审核 WARNING-1 的 YAGNI 修正
- 签名匹配：`prompt, reward_provider, config, backend, k_branches=3, max_concurrent_branches=None`
- 执行流程：
  1. Pilot run → `pilot_pipe.last_buffer()` 获取 buffer
  2. Pilot 失败或 buffer 为空时返回空树
  3. 按 timestamp 排序 buffer，每个 record 作为 branch point
  4. 每个 branch point × k_branches 创建异步 branch run
  5. 只收录 `status == "success"` 的 branch
  6. `asyncio.Semaphore` 控制并发
- `AgentPipe` 新增 `last_buffer()` 方法，返回 `copy.deepcopy(self._last_buffer)`，防御性编程
- `AgentPipe` 新增 `replay_cache` 构造参数，透传给 Monitor
- 测试覆盖：6 个测试，覆盖基本结构、pilot 失败、branch 失败、k_branches 参数、并发限制、replay_cache 透传

**实现改进**：计划中 branch runs 按 branch point 分组（内部循环 K 次），实际实现扁平化为独立 task。Semaphore 粒度更细，并发控制更精确。

### Task 6: Package 导出与版本更新 — PASS

**对照**：`mate/trajectory/__init__.py`、`pyproject.toml`

- `__init__.py` 导出 `BranchResult`, `TreeEpisodeResult`, `ReplayCache`, `tree_rollout`
- `__all__` 列表完整，按字母排序
- `pyproject.toml` 版本号 `"0.2.0"`

验证命令：`python -c "from mate.trajectory import tree_rollout, TreeEpisodeResult, BranchResult, ReplayCache; print('OK')"` → OK

### Task 7: 集成测试 — PASS

**对照**：`tests/trajectory/test_tree_integration.py`

- 使用真实 Python 子进程作为 mock MAS（通过 `urllib.request` 调用 Monitor HTTP 接口）
- 使用 `CountingBackend` 追踪 backend 调用次数
- 验证要点：
  - Pilot 成功（status, final_reward, 每个 agent 的 turn 数）
  - Branch 数量正确（2 turn × k=1 → 2 个 branch）
  - `parent_episode_id` 正确指向 pilot
  - Backend 调用计数正确（区分 fresh 调用和 replay）
  - Replayed 标记验证：被缓存命中的 turn 标记 `replayed=True`，response 与 pilot 一致
  - Branch_turn=0 的分支无 replay（cache 为空）

### Task 8: 更新设计文档 — PASS

**对照**：`docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`

已融入设计审核的所有修正：
- §2.1：工具调用确定性假设 + 缓存未命中降级语义
- §2.2：简化为 `tree_rollout` 函数
- §2.3：`messages_hash` 规范 + uid 构建规则 + 删除 `shared_prefix_turns`
- §3.2：失败模式矩阵（5 种场景）
- §4.1：固定 LLM 调用数对照实验

---

## 设计审核修正落实情况

| 审核项 | 级别 | 处理 | 验证 |
|--------|------|------|------|
| BLOCKER-1: 工具调用非确定性 | BLOCKER | 限定 PoC 为确定性场景 + messages_hash 校验降级 | ReplayCache.lookup 支持 messages 参数；Monitor 命中但 hash 不匹配时回退到 backend |
| BLOCKER-2: uid 赋值规则 | BLOCKER | 设计文档中给出完整伪代码 | 文档已更新 |
| WARNING-1: BranchCoordinator YAGNI | WARNING | 简化为 tree_rollout 函数 | 无 BranchCoordinator/BranchStrategy 类 |
| WARNING-2: shared_prefix_turns 冗余 | WARNING | 删除 | BranchResult 中无此字段 |
| WARNING-3: messages_hash 未规范 | WARNING | sha256 + json.dumps(sort_keys=True) | ReplayCache 中实现并有测试覆盖 |

---

## 额外质量检查

| 检查项 | 结果 |
|--------|------|
| 无 `BranchCoordinator`/`BranchStrategy` 类 | PASS |
| 无 `shared_prefix_turns` 字段 | PASS |
| `parallel_rollout` 完全未修改 | PASS |
| 代码风格一致（`from __future__ import annotations`, dataclass, async/await, type hints） | PASS |
| 全量回归测试 | PASS（86 passed, 1 skipped） |

---

## 改进建议（非阻塞，供后续参考）

1. **ReplayCache 复用**：同一 branch point 的 K 个 branch run 可共享同一 ReplayCache 实例，避免重复创建。当前 cache 创建开销很小（排序+切片+dict），但 k_branches 较大时可优化。

2. **非 success branch 日志**：`tree.py` 中 `status != "success"` 的 branch 被静默丢弃（异常路径有 warning 日志，但非异常的失败状态无日志）。建议加 debug 级别日志便于排查。

3. **长上下文保护**：当前通过 vLLM 原生 400 错误 + `allow_partial` 优雅降级间接处理。如需在 Monitor 层精确拦截（不依赖 vLLM 报错），可作为后续优化项。
