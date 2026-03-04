# MATE-reboot 项目上下文

> 最后更新：2026-03-04

## 项目定位

MATE-reboot 是多智能体轨迹采集引擎（Agent Trajectory Engine）的开发仓库。最终集成目标是 OrchRL 框架（`/home/cxb/OrchRL`）。

## 当前阶段

**设计完成，待进入实施。**

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
| Verl | `/home/cxb/rl_framework/verl/` | RL 框架基线，VerlBackend 集成参考 |
| 团队架构 PPT | `/home/cxb/multi-agent/docs/multi-agent-rl.pdf` | 整体架构对齐参考 |

## 待办

- [ ] 制定实施计划
- [ ] 实现 AgentPipe 核心模块
- [ ] DrMAS Search 端到端验证
