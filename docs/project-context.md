# MATE-reboot 项目上下文

> 最后更新：2026-03-09（V0.2 设计完成）

## 项目定位

MATE-reboot 是多智能体轨迹采集引擎（Agent Trajectory Engine）的开发仓库。最终集成目标是 OrchRL 框架（`/home/cxb/OrchRL`）。

## 当前阶段

**V0.2 设计已完成，进入实施阶段。** V0 已合入 OrchRL 主仓，训练侧已完成端到端联调。V0.2 核心目标：重放式树状分支采样的可行性与收益验证。

- 合并提交：`bcb5b25`（`merge: trajectory engine v0 implementation (56 tests passing)`）
- 回归验证：`pytest tests/trajectory tests/scripts/test_run_real_validation.py -q` → `65 passed`
- 真实环境验证（2026-03-09，real 模式）：
  - 记录：`docs/retros/2026-03-09-trajectory-engine-real-validation.md`
  - 环境：vLLM（`http://127.0.0.1:8000`）+ OrchRL Search MAS + 检索服务（`http://127.0.0.1:18080/retrieve`）
  - 在线模型：`/data1/models/Qwen/Qwen3-4B-Instruct-2507`
  - GPU 约束：`CUDA_VISIBLE_DEVICES=0,1,2,3`（规避 GPU5 故障）
  - smoke 产物：`artifacts/trajectory_validation_real_smoke_fixed.json`
  - exact-match 产物：`artifacts/trajectory_validation_exact_match.json`
  - 多样本产物：`artifacts/trajectory_validation_real_fixed.json`
  - 已确认：
    - 成功 episode 上 `token_ids` 不为 None
    - 成功 episode 上 `logprobs` 与 `token_ids` 长度一致
    - 成功 episode 的 `episode_id` 全局唯一
    - 成功 episode 具备 `verifier/searcher/answerer` 完整 turn 数据
    - reward 解析 bug 已修复，受控 exact-match 样本可稳定得到 `final_reward=1.0`
  - 已定位限制：
    - prompt `when is the next deadpool movie being released?` 在 `max_turns=4` 时稳定触发 vLLM `400 -> 502`，导致 MAS exit code 1
    - 该问题在 `max_turns<=3` 时不复现，属于长上下文失败，不是随机环境抖动

## 团队分工

| 成员 | 负责模块 | 仓库 |
|------|---------|------|
| 本仓库开发者 | 轨迹采集模块（AgentPipe） | MATE-reboot |
| 同事 | 训练侧调度 + 主入口（Verl main PPO 多 agent 扩展） | OrchRL |

## 关键设计决策

### V0 决策（已冻结）

| 决策 | 结论 | 文档 |
|------|------|------|
| Monitor 推理模式 | 策略模式（InferenceBackend 接口） | `docs/plans/2026-03-04-trajectory-engine-v0-design.md` |
| Agent 身份识别 | `model` 字段即 agent role | 同上 |
| MAS 启动模式 | 进程模式（subprocess） | 同上 |
| 轨迹输出粒度 | `Dict[agent_id, List[TurnData]]` | 同上 |
| V0 验证场景 | DrMAS Search（OrchRL 抽取版） | 同上 |

### V0.2 决策（2026-03-09 brainstorming 确认）

| 决策 | 结论 | 文档 |
|------|------|------|
| 开发仓库 | 继续在 MATE-reboot，里程碑后同步到 OrchRL | `docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md` |
| 树状分支实现 | 重放式分支（pilot run + ReplayCache + branch runs），MAS 零侵入 | 同上 |
| 架构简化 | 无 BranchCoordinator/BranchStrategy 策略模式，直接用 `tree_rollout` 函数 | 设计审核修正（YAGNI） |
| 接口演进 | 扩展模式——新增 `tree_rollout`，保留 `parallel_rollout` 不变 | 同上 |
| 训练侧消费 | OrchRL adapter 层（`mate_rollout_adapter` + `mate_dataproto_adapter`），PYTHONPATH 导入 | OrchRL `orchrl/trainer/` |

## 依赖关系

| 依赖 | 路径 | 用途 |
|------|------|------|
| OrchRL 主仓（含训练代码） | `/home/cxb/OrchRL/` | 训练侧适配层（`orchrl/trainer/mate_*`）+ trajectory 副本 |
| OrchRL Search MAS | `/home/cxb/OrchRL/examples/mas_app/search/` | V0/V0.2 端到端验证的 MAS 应用 |
| Verl | `third_party/verl`（git submodule） | RL 框架基线 |
| 团队架构 PPT | `/home/cxb/multi-agent/docs/multi-agent-rl.pdf` | MARL GRPO 分组适配算法（V0.2 核心算法对标） |

### V0.2 技术参考

| 参考 | 来源 | 层级 | 用途 |
|------|------|------|------|
| AREAL-DTA | [arXiv:2602.00482](https://arxiv.org/abs/2602.00482) | 训练侧 | DFS 单路径物化前缀树（8.31x 训练加速） |
| Forge Prefix Tree Merging | MiniMax | 训练侧 | 全量物化 + Magi Attention（40x） |
| Tree Training | [arXiv:2511.00413](https://arxiv.org/abs/2511.00413) | 训练侧 | Gradient Restoration + Tree Packing（6.2x） |
| ROME/IPA rollback | [arXiv:2512.24873](https://arxiv.org/abs/2512.24873) | 采集侧参考 | 选择性分支策略（V0.2 后优化方向） |

## 文档索引

### V0

| 文档 | 类型 | 说明 |
|------|------|------|
| `docs/plans/2026-03-02-marl-grpo-v0-directions.md` | 方向设计 | V0 方向评估 |
| `docs/plans/2026-03-04-trajectory-engine-v0-design.md` | 架构设计 | V0 详细设计（已冻结） |
| `docs/plans/2026-03-04-trajectory-engine-v0-impl-plan.md` | 实施计划 | V0 实施细节 |
| `docs/plans/2026-03-05-training-integration-spec.md` | 对接规格 | 训练侧联调接口文档 |
| `docs/retros/2026-03-09-trajectory-engine-real-validation.md` | 验证记录 | 真实环境验证证据、异常样本与根因分析 |

### V0.2

| 文档 | 类型 | 说明 |
|------|------|------|
| `docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md` | 设计方向 | V0.2 核心设计、技术参考、职责边界 |
| `docs/plans/2026-03-09-trajectory-engine-v02-impl-plan.md` | 实施计划 | 9 个 Task、TDD 步骤、依赖关系 |
| `docs/prompts/2026-03-09-v02-brainstorming.md` | 会话 prompt | V0.2 brainstorming 启动模板 |
| `docs/prompts/2026-03-09-v02-implementation.md` | 会话 prompt | V0.2 实施会话启动模板 |

## 待办

### V0（已完成）

- [x] 制定实施计划
- [x] 实现 AgentPipe 核心模块
- [x] DrMAS Search 端到端验证
- [x] vLLM token_ids 提取链路验证（Monitor/Backend/序列化）
- [x] Episode 并行采样（并发 rollout 编排与稳定性验证）
- [x] 真实环境验证脚本与可视化工具落地
- [x] 编写训练侧对接规格文档
- [x] Real vLLM + 检索服务端到端验证（2026-03-09）
- [x] 与训练侧进行联调（训练侧已通过 adapter 方式完成端到端联调）
- [x] 整理代码向 OrchRL 仓库提交 PR（已合入 OrchRL main `1664eb5`）

### V0.2（进行中）

- [x] V0.2 设计：确定开发仓库（MATE-reboot）、特性范围、技术参考调研
- [x] V0.2 设计审核（APPROVE WITH CHANGES，已融入实施计划）
- [x] V0.2 实施计划编写（9 个 Task）
- [ ] Task 1: 数据结构（BranchResult, TreeEpisodeResult, EpisodeResult.status）
- [ ] Task 2: ReplayCache
- [ ] Task 3: Monitor replay 支持
- [ ] Task 4: AgentPipe 优雅降级
- [ ] Task 5: tree_rollout 函数（核心）
- [ ] Task 6: Package 导出 + 版本更新（0.1.0 → 0.2.0）
- [ ] Task 7: 集成测试
- [ ] Task 8: 设计文档更新（融入审核修正）
- [ ] 同步到 OrchRL + 训练侧 adapter 适配

