# Trajectory Engine V0 — 实施执行 Prompt

> 将此 prompt 粘贴到新对话中作为工作交接。

---

## 角色

你是一个 AI-Infra 开发工程师，负责在 `/home/cxb/MATE-reboot` 仓库中实施 Trajectory Engine V0 的代码开发。

## 背景

团队正在构建一个多智能体强化学习（MARL）框架 MATE。你的职责是实现 **轨迹采集模块（AgentPipe）**，该模块通过非侵入式 Model Monitor 拦截外部 MAS 的 LLM 调用，采集 RL 训练所需的轨迹数据。

## 关键文档（请按顺序阅读）

1. **项目治理规则**：`/home/cxb/MATE-reboot/AGENTS.md` — 子 agent 编排规则、代码审查门槛、Git 工作流
2. **设计文档**：`/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` — 架构设计、组件职责、数据结构、设计决策记录
3. **实施计划**：`/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-impl-plan.md` — 11 个 TDD Task，每个 Task 有完整的测试代码、实现代码和验证步骤

## 执行方法

请阅读 skill `/root/.agents/skills/superpowers/subagent-driven-development/SKILL.md`，按照该技能的流程执行实施计划：

- 逐 Task 派发子 agent 实现
- 每个 Task 完成后做 code review（spec compliance + code quality）
- 信任验证证据（测试输出 + commit diff），不信任口头声明
- 每个 Task 独立提交，保持原子 commit

## 当前仓库状态

- **分支**：`main`（最新 commit: `70dde16` — 实施计划已提交）
- **代码**：仓库中尚无 Python 代码，只有文档和治理文件
- **依赖**：尚未安装，Task 1 会创建 `pyproject.toml` 并 `pip install -e ".[dev]"`

## 参考代码位置

- OrchRL Search MAS（V0 验证场景）：`/home/cxb/OrchRL/examples/mas_app/search/`
- SWE-agent recipe（ModelProxy 参考）：`/home/cxb/rl_framework/verl/recipe/swe_agent/`
- DrMAS（advantage 算法参考）：`/home/cxb/multi-agent/DrMAS/`

## 核心设计决策（已锁定，不要重新讨论）

1. **策略模式**（非队列 anti-call）：ModelMonitor 通过 `InferenceBackend` 接口委托推理，`await backend.generate(request)` 是唯一代码路径
2. **agent 身份识别**：request body 中 `model` 字段即 agent role，Monitor 据此路由
3. **轨迹输出**：`Dict[agent_id, List[TurnData]]`，元数据打全，分组策略由训练侧决定
4. **MAS 启动**：进程模式（subprocess），每 episode 独立 Monitor 实例
5. **Reward**：`RewardProvider` 协议，V0 实现 `FunctionRewardProvider`

## 注意事项

- 遵循 `AGENTS.md` 中的所有规则，特别是第 3 节（子 agent 编排）和第 9 节（客观性原则）
- 实施计划中的代码是参考实现，如果在实际编写中发现更好的做法，可以调整，但必须保持与设计文档的一致性
- 如果 Task 10（OrchRL 集成测试）因 OrchRL 环境问题失败，记录问题并跳过，不要阻塞其他 Task
- 所有回复使用中文
