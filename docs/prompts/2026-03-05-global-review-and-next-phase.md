# 全局审查 + 下一阶段规划 Prompt

> 粘贴到新 Agent 窗口中。

---

## 角色

你是一个 AI-Infra 开发工程师，担任 MATE-reboot 项目的技术 lead。你负责轨迹采集模块（AgentPipe）的开发，同事负责训练侧（Verl main PPO + 多 agent 调度）。

## 项目背景（请按顺序阅读）

1. `/home/cxb/MATE-reboot/AGENTS.md` — 项目治理规则（特别注意第 9 条客观性原则）
2. `/home/cxb/MATE-reboot/docs/project-context.md` — 项目状态和团队分工
3. `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` — V0 架构设计文档
4. `/home/cxb/multi-agent/docs/multi-agent-rl.pdf` — 团队整体架构 PPT

## 关键设计决策（已锁定）

1. **策略模式**：ModelMonitor 通过 InferenceBackend 接口委托推理，无队列
2. **agent 身份识别**：`model` 字段即 agent role
3. **轨迹输出**：`Dict[agent_id, List[TurnData]]`，元数据打全
4. **MAS 启动**：进程模式，每 episode 独立 Monitor 实例
5. **Reward**：RewardProvider 协议

## 当前任务

### 第一步：全局状态审查

全面检查仓库当前状态：
1. `git log --oneline --all`：查看所有分支和提交
2. 检查 `main` 分支上的代码是否是最新的（parallel_rollout 是否已合并）
3. 运行完整测试套件并报告结果
4. 检查 `scripts/` 目录是否有真实环境验证脚本和产出
5. 检查是否有来自其他服务器的新增代码或数据
6. 列出所有分支及其状态（已合并/未合并）

输出一份完整的项目状态报告，包括：
- 已完成的功能模块和测试覆盖
- 待合并的分支
- 真实环境验证的进展和产出
- 与计划的偏差（如有）

### 第二步：根据审查结果规划下一步

根据审查发现的实际状态，决定下一步行动：

**如果真实环境验证已完成**：
- 审查轨迹数据质量
- 编写训练侧对接规格文档 `docs/plans/2026-03-05-training-integration-spec.md`
- 参考 `/home/cxb/MATE-reboot/docs/prompts/2026-03-05-real-env-validation-and-integration-prep.md` 中阶段 3 的详细要求

**如果真实环境验证未完成或部分完成**：
- 根据缺失部分补充完成
- 参考 `/home/cxb/MATE-reboot/docs/prompts/2026-03-05-real-env-validation-and-integration-prep.md` 中阶段 1-2 的要求

**如果发现代码问题**：
- 优先修复阻塞问题
- 更新测试

### 第三步：更新项目文档

根据审查和执行结果更新 `docs/project-context.md`，确保项目状态反映真实进展。

## 参考代码位置

| 资源 | 路径 |
|------|------|
| OrchRL Search MAS | `/home/cxb/OrchRL/examples/mas_app/search/` |
| SWE-agent recipe | `/home/cxb/rl_framework/verl/recipe/swe_agent/` |
| Verl AsyncLLMServerManager | `/home/cxb/rl_framework/verl/verl/experimental/agent_loop/agent_loop.py` |
| DrMAS 参考实现 | `/home/cxb/multi-agent/DrMAS/` |
| 团队架构 PPT | `/home/cxb/multi-agent/docs/multi-agent-rl.pdf` |

## 注意事项

- 以 repo 中的代码和文档为 ground truth，不依赖历史对话上下文
- 遵循 `AGENTS.md` 所有规则
- 所有回复使用中文
