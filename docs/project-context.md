# MATE-reboot 项目上下文

> 最后更新：2026-03-09

## 项目定位

MATE-reboot 是多智能体轨迹采集引擎（Agent Trajectory Engine）的开发仓库。最终集成目标是 OrchRL 框架（`/home/cxb/OrchRL`）。

## 当前阶段

**V0 已完成并合入 OrchRL 主仓（`1664eb5`）。训练侧同事已通过适配方式完成端到端联调。准备进入 V0.2 设计阶段。**

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

| 决策 | 结论 | 文档 |
|------|------|------|
| Monitor 推理模式 | 策略模式（InferenceBackend 接口） | `docs/plans/2026-03-04-trajectory-engine-v0-design.md` |
| Agent 身份识别 | `model` 字段即 agent role | 同上 |
| MAS 启动模式 | 进程模式（subprocess） | 同上 |
| 轨迹输出粒度 | `Dict[agent_id, List[TurnData]]` | 同上 |
| V0 验证场景 | DrMAS Search（OrchRL 抽取版） | 同上 |

## 依赖关系

| 依赖 | 路径 | 用途 |
|------|------|------|
| OrchRL Search MAS | `/home/cxb/OrchRL/examples/mas_app/search/` | V0 端到端验证的 MAS 应用 |
| Verl | `third_party/verl`（git submodule） | RL 框架基线，VerlBackend 集成参考 |
| 团队架构 PPT | `/home/cxb/multi-agent/docs/multi-agent-rl.pdf` | 整体架构对齐参考 |

## 文档索引

| 文档 | 类型 | 说明 |
|------|------|------|
| `docs/plans/2026-03-02-marl-grpo-v0-directions.md` | 方向设计 | V0 方向评估 |
| `docs/plans/2026-03-04-trajectory-engine-v0-design.md` | 架构设计 | V0 详细设计（已冻结） |
| `docs/plans/2026-03-04-trajectory-engine-v0-impl-plan.md` | 实施计划 | V0 实施细节 |
| `docs/plans/2026-03-05-training-integration-spec.md` | 对接规格 | 训练侧联调接口文档 |
| `docs/retros/2026-03-09-trajectory-engine-real-validation.md` | 验证记录 | 真实环境验证证据、异常样本与根因分析 |

## 待办

- [x] 制定实施计划
- [x] 实现 AgentPipe 核心模块
- [x] DrMAS Search 端到端验证
- [x] vLLM token_ids 提取链路验证（Monitor/Backend/序列化）
- [x] Episode 并行采样（并发 rollout 编排与稳定性验证）
- [x] 真实环境验证脚本与可视化工具落地
- [x] 编写训练侧对接规格文档
- [x] Real vLLM + 检索服务端到端验证（2026-03-09，成功 episode 的五项验证要点已确认）
- [ ] 处理长上下文样本在 `max_turns=4` 下触发的 vLLM `400/502` 失败
- [x] 与训练侧进行联调（训练侧同事已通过适配方式带上本仓库代码完成端到端联调）
- [x] 整理代码向 OrchRL 仓库提交 PR（已合入 OrchRL main `1664eb5`，包名 `mate.trajectory` → `trajectory`）
- [ ] V0.2 设计：确定开发仓库（MATE-reboot vs OrchRL 直接开发）、下一阶段特性范围

