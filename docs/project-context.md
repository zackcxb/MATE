# MATE-reboot 项目上下文

> 最后更新：2026-03-05

## 项目定位

MATE-reboot 是多智能体轨迹采集引擎（Agent Trajectory Engine）的开发仓库。最终集成目标是 OrchRL 框架（`/home/cxb/OrchRL`）。

## 当前阶段

**V0 实现完成 + 真实环境验证完成，待训练侧联调。**

- 合并提交：`bcb5b25`（`merge: trajectory engine v0 implementation (56 tests passing)`）
- 回归验证：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/trajectory -v -p pytest_asyncio.plugin -p pytest_timeout --timeout=120` → `60 passed`
- 端到端验证：
  - 脚本：`scripts/run_real_validation.py`（并行 rollout + JSON 输出）
  - 可视化：`scripts/visualize_trajectories.py`（终端 + HTML 报告）
  - 最新产物：`artifacts/real_validation_realmode_full.json`、`artifacts/real_validation_realmode_full.html`
  - 最新模式：`real`（vLLM 实际推理，`n_prompts=5, n_samples=2, max_concurrent=2`）
  - 结果摘要：`episode=10`、`token_ids_none_turns=0`、`token/logprobs mismatch=0`、`episode_id` 全局唯一
  - 注：本机检索服务未启用，本轮使用 `search.provider=disabled` 配置做真实推理链路验证

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

## 待办

- [x] 制定实施计划
- [x] 实现 AgentPipe 核心模块
- [x] DrMAS Search 端到端验证
- [x] vLLM token_ids 提取链路验证（Monitor/Backend/序列化）
- [x] Episode 并行采样（并发 rollout 编排与稳定性验证）
- [x] 真实环境验证脚本与可视化工具落地
- [x] 编写训练侧对接规格文档
- [ ] 在无临时环境补丁条件下复跑 real vLLM + 检索服务端到端验证
- [ ] 与训练侧进行联调（VerlBackend + 训练主入口对接）
