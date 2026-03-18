# MATE-reboot 项目上下文

> 最后更新：2026-03-18（V0.3 MATE runtime contract implemented and locally verified）

## 项目定位

MATE-reboot 是多智能体轨迹采集引擎（Agent Trajectory Engine）的开发仓库。最终集成目标是 OrchRL 框架（`/home/cxb/OrchRL`）。

## 当前阶段

**V0.2 已收口；V0.3 MATE-side implementation 已完成并通过本地验证。** 当前阶段的核心议题是：

1. 维持 `VerlBackend` + canonical renderer + hard validator 的 runtime contract 稳定性
2. 保持 exporter 优先消费 recorded token truth，而不是退回 training-side re-render
3. 继续将 drift 研究限定在 diagnostics/artifact 层，而不是扩张 runtime hard gate
4. 保持 `rollout_routed_experts` 为 optional capability-gated 路径
5. 维持 tree rollout 的显式 branch semantics，避免回退到 `global_turn_index`

## 当前冻结事实

1. MATE 采集侧已具备 `tree_rollout`、`ReplayCache`、验证脚本和可视化支持。
2. OrchRL 当前通过 adapter + `VLLMBackend` 集成 MATE，不使用 `AsyncLLMServerManager` 直连路径。
3. 当前 MATE 已能记录 `prompt_ids`，但它仍是 side-channel 观测数据，不是实际生成输入。
4. `swe_agent` 提供了更严格的参考路径：先生成 `prompt_ids`，再 direct `generate(prompt_ids=...)`，并做 replay/alignment 校验。
5. Slime 提供了另一类参考路径：存储完整序列 `tokens`，训练侧直接消费，不做重新 tokenize。
6. V0.2 tree 语义已经被 OrchRL 消费，当前没有新的 MATE blocker 级证据要求继续修改 `tree_rollout`、`TreeEpisodeResult` 或 `ReplayCache`。
7. OrchRL 侧训练实现由同事负责；本仓当前职责是定义和验证稳定的 MATE 契约与研究结论。

## 团队分工

| 成员 | 负责模块 | 仓库 |
|------|---------|------|
| 本仓库开发者 | 轨迹采集模块（AgentPipe） | MATE-reboot |
| 同事 | 训练侧调度 + 主入口（Verl main PPO 多 agent 扩展） | OrchRL |

## 冻结设计决策

| 阶段 | 决策 | 结论 | 参考文档 |
|------|------|------|------|
| V0 | Monitor 推理模式 | 策略模式（`InferenceBackend` 接口） | `docs/plans/2026-03-04-trajectory-engine-v0-design.md` |
| V0 | Agent 身份识别 | `model` 字段即 agent role | 同上 |
| V0 | MAS 启动模式 | 进程模式（subprocess） | 同上 |
| V0 | 轨迹输出粒度 | `Dict[agent_id, List[TurnData]]` | 同上 |
| V0.2 | 树状分支实现 | 重放式分支（pilot run + `ReplayCache` + branch runs），MAS 零侵入 | `docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md` |
| V0.2 | 接口演进 | 新增 `tree_rollout`，保留 `parallel_rollout` 不变 | 同上 |
| V0.2 | 训练侧消费 | OrchRL adapter 层（`mate_rollout_adapter` + `mate_dataproto_adapter`） | OrchRL `orchrl/trainer/` |

## 当前开放问题

| 议题 | 当前状态 |
|------|------|
| `VerlBackend` | 已在 MATE 实现 direct `prompt_ids -> generate(...)` runtime path，并接入 canonical validator |
| token drifting | 已实现 renderer fingerprint、diagnostic artifact hook、token-truth exporter boundary |
| `DataProto` 输出 | 仍为 optional exporter 边界；本仓未引入 native `DataProto` 输出 |
| `rollout_routed_experts` | 已实现 optional capability-gated 透传/采集/回放 |
| tree branch contract | 已实现显式 `replayed` / `branch_phase` 语义，替代 `global_turn_index` |

## 关键依赖

| 依赖 | 路径 | 用途 |
|------|------|------|
| OrchRL 主仓 | `/home/cxb/OrchRL/` | 训练侧适配层与真实集成目标 |
| OrchRL Search MAS | `/home/cxb/OrchRL/examples/mas_app/search/` | V0/V0.2 端到端验证场景 |
| Verl 参考实现 | `/home/cxb/rl_framework/verl/` | `swe_agent` 的 direct token generation、proxy、alignment 参考 |
| Slime 参考分析 | `docs/ref/slime-tokenization-drift-analysis.md` | full-sequence tokens 合同、零 re-tokenize 训练路径参考 |
| Verl 子模块 | `third_party/verl` | RL 框架基线 |
| 团队算法参考 | `/home/cxb/multi-agent/docs/multi-agent-rl.pdf` | MARL/BGRPO 相关算法参考 |

## 当前 Canonical Docs

### 当前阶段必读

1. `AGENTS.md`
2. `docs/plans/2026-03-17-bgrpo-v03-runtime-contract-design.md`
3. `docs/plans/2026-03-17-bgrpo-v03-implementation-plan.md`
4. `docs/ref/slime-tokenization-drift-analysis.md`
5. `docs/prompts/2026-03-15-v03-brainstorming-handoff.md`
6. `skills/document-entry-hygiene.md`

### 冻结历史设计

1. `docs/plans/2026-03-04-trajectory-engine-v0-design.md`
2. `docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`
3. `docs/plans/2026-03-11-trajectory-public-api-boundary.md`

### 历史证据与收口记录

1. `docs/retros/2026-03-09-trajectory-engine-real-validation.md`
2. `docs/retros/2026-03-11-orchrl-tree-adapter-sync.md`
3. `docs/retros/2026-03-14-orchrl-tree-smoke-server-sync.md`
4. `docs/retros/2026-03-15-orchrl-tree-smoke-upstreaming-wrapup.md`

## 当前待办

1. 将本次 V0.3 MATE-side runtime contract 改动同步给 OrchRL 训练侧同事，保持边界清晰。
2. 基于新 exporter boundary 评估 OrchRL 侧 zero re-tokenize 消费接入。
3. 继续做 hypothesis 级 token-drift diagnostics，不把研究性检查升级成 hard gate。
4. 视训练侧需要决定是否追加 optional `DataProto` exporter，而非 native output。
