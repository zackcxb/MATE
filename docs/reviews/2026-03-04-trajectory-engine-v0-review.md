# Trajectory Engine V0 — 实施审查报告

> 日期：2026-03-04
> 审查范围：`feat/trajectory-engine-v0` 分支全部实现（25 commits, 56 tests）
> 审查依据：`docs/plans/2026-03-04-trajectory-engine-v0-design.md`（设计文档）、`docs/plans/2026-03-04-trajectory-engine-v0-impl-plan.md`（实施计划，11 Tasks）

---

## 1. 结论

**V0 全部 11 个 Task 已完成，56 个测试全部通过（5.15s），实现质量超出计划预期。** 执行 Agent 在忠实实现计划的基础上，针对并发安全、进程管理、数据校验做了多处有价值的 hardening。

---

## 2. 逐模块审查

### 2.1 数据结构（`datatypes.py`）

| 检查项 | 结论 |
|--------|------|
| 7 个 dataclass 是否与设计文档一致 | 完全匹配 |
| `EpisodeResult.rewards` 是否支持 `float \| List[float]` | 支持，兼容 episode-level 和 per-turn reward |
| metadata 字段（episode_id, agent_role, turn_index, timestamp）是否完整 | 完整 |

测试覆盖：5 个测试，覆盖创建、None 值、分组、per-turn reward。

### 2.2 InferenceBackend + VLLMBackend（`backend.py`）

| 检查项 | 结论 |
|--------|------|
| InferenceBackend 是否为抽象类 | 是，`ABC` + `@abstractmethod` |
| VLLMBackend 是否始终注入 `logprobs=True` | 是 |
| VLLMBackend 是否支持 `actual_model` 覆盖 | 是 |

**执行 Agent 增强点**：
- 增加了 `BACKEND_URL_OVERRIDE_KEY` 支持 per-role 后端路由（Monitor 按 agent_role 注入不同后端 URL）
- 增加了 finite logprob 过滤（拒绝 NaN、Inf、bool、非数字），防止脏数据流入训练
- 增加了畸形响应防御性解析（choices 为空、content 为 None 等边界情况）

测试覆盖：7 个测试，包含正常转发、logprobs 注入、model 覆盖、per-role URL、畸形响应、非 finite 过滤。

### 2.3 ModelMonitor（`monitor.py`）

| 检查项 | 结论 |
|--------|------|
| 是否通过 `model` 字段识别 agent role | 是 |
| 是否使用策略模式（`await backend.generate()`）| 是，无队列/anti-call 残留 |
| 未知 agent 是否返回 400 | 是 |
| 后端异常是否返回 502 | 是 |
| turn_index 是否按 agent_role 独立自增 | 是 |

**执行 Agent 增强点**：
- `threading.Lock` 保护 buffer 写入（防止并行 agent 并发请求竞态）
- `_buffer_generation` 机制：`clear_buffer()` 递增 generation 号，in-flight 的响应如果对应旧 generation 则不写入 buffer，避免跨 episode 数据串扰
- turn_index 在 `backend.generate()` 调用**之前**分配，保证按请求到达顺序编号

测试覆盖：11 个测试，包含路由、buffer 采集、turn_index 递增、错误处理、并发到达顺序、per-role 注入、clear_buffer 对 in-flight 请求的隔离。

### 2.4 MASLauncher（`launcher.py`）

| 检查项 | 结论 |
|--------|------|
| `prepare_config` 是否正确替换 base_url | 是 |
| `prepare_config` 是否为每个 agent 注入 model = role 名 | 是 |
| 是否保留其他配置字段不变 | 是 |
| 超时是否正确杀死进程 | 是 |

**执行 Agent 设计变更**：
- 从计划中的 `asyncio.create_subprocess_shell` 改为同步 `subprocess.Popen` + `start_new_session=True`
- 超时后通过 `os.killpg()` + `SIGKILL` 杀死整个进程组（确保 MAS 的子进程也被清理）
- 在 `AgentPipe` 中通过 `asyncio.to_thread()` 桥接同步调用

这个变更是合理的——进程组杀死在 asyncio 子进程中不容易可靠实现。

测试覆盖：8 个测试，包含配置准备、正常启动/等待、超时杀死、进程组子进程清理、临时文件错误恢复。

### 2.5 TrajectoryCollector（`collector.py`）

| 检查项 | 结论 |
|--------|------|
| 是否按 agent_role 分组 | 是 |
| 每组是否按 turn_index 排序 | 是 |
| 是否正确转换 InteractionRecord → TurnData | 是 |
| 空 buffer 是否正常处理 | 是 |

实现简洁，与计划完全一致。

测试覆盖：4 个测试，覆盖分组、排序、空 buffer、字段保留。

### 2.6 RewardWorker（`reward.py`）

| 检查项 | 结论 |
|--------|------|
| RewardProvider 是否为 Protocol 类型 | 是 |
| FunctionRewardProvider 是否正确包装函数 | 是 |
| 是否支持 per-turn reward（`List[float]`）| 是 |

**执行 Agent 增强点**：
- 严格 payload 验证：返回值必须是 dict，必须包含 `agent_rewards`（dict）和 `final_reward`
- `agent_rewards` 值只接受 `float`、`int` 或 `list[float|int]`，拒绝 bool
- 所有数值必须 finite（NaN、Inf、-Inf 全部拒绝）
- Provider 异常包装为带上下文的 `RuntimeError`

测试覆盖：13 个测试，包含正常调用、per-turn reward、缺失字段、类型错误、NaN/Inf/-Inf 拒绝（参数化测试）、provider 异常包装。

### 2.7 AgentPipe（`pipe.py`）

| 检查项 | 结论 |
|--------|------|
| 编排流程是否与设计文档第 7 节一致 | 一致 |
| 是否在 finally 块中清理资源 | 是（monitor.stop + launcher.cleanup） |
| 非零退出码是否正确处理 | 是，抛出 RuntimeError |

**执行 Agent 增强点**：
- `asyncio.to_thread()` 卸载同步阻塞操作（launcher.wait、reward 计算）
- `finally` 块中的错误优先级处理：主错误 > stop 错误 > cleanup 错误，确保不会丢失原始异常

测试覆盖：4 个测试，包含端到端（含 tiny MAS 脚本）、非零退出码、reward 计算异步卸载验证、cleanup 异常处理。

### 2.8 OrchRL 集成测试

使用 `ScriptedBackend`（返回预设脚本响应的 mock 后端），验证 AgentPipe 启动 OrchRL Search MAS 子进程 → Monitor 拦截 → 轨迹采集 → reward 计算的完整链路。

带 `@pytest.mark.skipif` 条件：OrchRL 不可用时自动跳过。测试通过。

---

## 3. 与设计文档 / 实施计划的偏差分析

| 项目 | 计划 | 实际 | 评估 |
|------|------|------|------|
| Monitor 推理模式 | 策略模式 `await backend.generate()` | 完全一致 | 匹配 |
| MASLauncher 子进程 | `asyncio.create_subprocess_shell` | `subprocess.Popen` + `asyncio.to_thread()` | 合理改进 |
| Buffer 并发保护 | 未提及 | `threading.Lock` + `_buffer_generation` | 必要增强 |
| Logprob 数值校验 | 未提及 | finite 过滤（NaN/Inf/bool 拒绝） | 防御性增强 |
| Reward payload 校验 | 简单 `dict.get()` | 严格类型 + finite 校验 | 防御性增强 |
| Per-role backend URL | 未提及 | `BACKEND_URL_OVERRIDE_KEY` 注入 | 为多模型场景预留 |

**所有偏差均为改进性质，无功能缺失或 spec 违反。**

---

## 4. 测试质量评估

### 4.1 覆盖统计

| 模块 | 测试数 | 特色测试 |
|------|--------|---------|
| datatypes | 5 | per-turn reward 类型兼容 |
| backend | 7 | 畸形响应解析、非 finite logprob 过滤 |
| monitor | 11 | 并发到达顺序、clear_buffer 对 in-flight 隔离 |
| launcher | 8 | 进程组杀死（子进程的子进程）、临时文件错误恢复 |
| collector | 4 | 标准覆盖 |
| reward | 13 | NaN/Inf/-Inf 参数化测试（3×3 组合） |
| pipe | 4 | 端到端含 tiny MAS 脚本、asyncio.to_thread 并发验证 |
| orchrl 集成 | 1 | 完整链路（含 skipif） |
| **合计** | **56** | |

### 4.2 测试设计亮点

- **Monitor 并发测试**：验证多个 agent 同时发请求时 turn_index 按到达顺序分配
- **进程组测试**：MAS 启动子进程后超时，验证整棵进程树都被 kill
- **Buffer generation 测试**：`clear_buffer()` 后 in-flight 的响应不污染新 buffer
- **Reward 参数化测试**：`@pytest.mark.parametrize` 覆盖 NaN/Inf/-Inf × scalar/list/final_reward 组合
- **AgentPipe 端到端**：用 Python 写的 tiny MAS 脚本（通过 `urllib` 调用 Monitor），验证完整编排

---

## 5. 已知局限（非问题，属于 V0 范围外）

| 局限 | 说明 | 计划解决阶段 |
|------|------|-------------|
| VLLMBackend 始终返回 `token_ids=None` | 标准 OpenAI API 不含 token ID | Feature A（vLLM token_ids 提取） |
| 单 episode 串行执行 | 无 `parallel_rollout()` | Feature B（并行采样） |
| 未对接真实 vLLM/SGLang | 集成测试用 mock backend | 后续真实环境验证 |
| 未实现 VerlBackend | 训练模式需要 `AsyncLLMServerManager` 集成 | 训练侧对接阶段 |

---

## 6. 下一步

1. **Code Review + Merge**：两阶段 review 后合并 `feat/trajectory-engine-v0` 到 `main`
2. **Feature A**：VLLMBackend token_ids 提取（独立分支 `feat/vllm-token-ids`）
3. **Feature B**：Episode 并行采样（独立分支 `feat/parallel-rollout`）
4. **训练侧对接**：与同事对齐 `EpisodeResult` 格式，实现 `VerlBackend`
