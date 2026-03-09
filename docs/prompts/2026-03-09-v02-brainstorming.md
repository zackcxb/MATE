# V0.2 方向讨论 + 开发仓库决策 Prompt

> 粘贴到新 Agent 窗口中。

---

## 角色

你是一个 AI-Infra 开发工程师，担任 MATE-reboot 项目的技术 lead。你负责轨迹采集模块（AgentPipe）的开发，同事负责训练侧（Verl main PPO + 多 agent 调度）。

## 项目背景（请按顺序阅读）

1. `/home/cxb/MATE-reboot/AGENTS.md` — 项目治理规则（特别注意第 9 条客观性原则）
2. `/home/cxb/MATE-reboot/docs/project-context.md` — 项目状态和团队分工
3. `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` — V0 架构设计（已冻结，注意附录 B 的演进路径）

## V0 已完成里程碑摘要

- **核心模块**：ModelMonitor（策略模式 HTTP 代理）、VLLMBackend、MASLauncher（subprocess）、TrajectoryCollector、RewardWorker、AgentPipe（编排器）、parallel_rollout
- **代码规模**：798 行核心代码 / 9 文件，60 个单元测试通过
- **真实验证**：vLLM + OrchRL Search MAS + 检索服务端到端跑通，token_ids/logprobs 采集链路完整
- **上库状态**：已合入 OrchRL 主仓（`1664eb5`），包名 `trajectory`
- **训练联调**：训练侧同事已通过适配方式完成端到端联调
- **已知限制**：长上下文样本在 `max_turns=4` 时触发 vLLM 400/502（已定位根因：`max_model_len=4096` 与长链路 team context 冲突）
- **launcher bug 修复**：已修复 per-agent `llm.base_url` 覆盖绕过 Monitor 的问题（已回移到 MATE-reboot）

## 当前待讨论议题

### 议题 1：开发仓库选择

V0 的开发流程是：MATE-reboot 开发 → 验证 → 提取到 OrchRL。现在需要决定 V0.2 的开发策略：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **继续在 MATE-reboot 开发，再合入 OrchRL** | 开发独立性强；不影响 OrchRL 主仓稳定性；可保留完整开发文档和 artifacts | 多了一步同步操作；两个仓库代码可能 diverge |
| **直接在 OrchRL 上开发** | 消除同步负担；代码只有一份；与同事的训练侧代码在同一仓库 | OrchRL 目前无包管理结构（无 pyproject.toml）；开发文档/临时产物会污染主仓；reviewer 看到更多噪声 |

用户对此犹豫，请客观分析利弊，给出明确建议。

### 议题 2：V0.2 特性范围

V0 设计文档附录 B 的演进路径：

| 阶段 | 功能 |
|------|------|
| V0.2 | 树状分支采样 + Branch Coordinator + Trie 存储 |
| V1 | 轨迹级训推异步 + Relay Worker + 长尾迁移 |
| V1+ | Reward Model + Multi-LoRA + 服务模式 MAS adapter |

需要讨论：
1. V0.2 的范围是否仍然合适？是否有新的优先级变化？
2. 训练侧已完成端到端联调这一事实，是否改变了 V0.2 的优先级排序？
3. 长上下文失败问题是否应该在 V0.2 中解决，还是独立处理？
4. 是否有 V0 中发现的架构不足需要在 V0.2 中修正？

### 议题 3：与训练侧的协作模式

训练侧同事已通过"适配方式"集成了 V0 代码。需要讨论：
1. 这种适配方式的长期可维护性如何？
2. V0.2 的新特性（如树状分支采样）是否会影响训练侧接口？
3. 是否需要建立更正式的接口版本管理？

## 参考代码位置

| 资源 | 路径 |
|------|------|
| OrchRL 主仓（含已合入的 trajectory） | `/home/cxb/OrchRL/` |
| MATE-reboot 开发仓库 | `/home/cxb/MATE-reboot/` |
| OrchRL Search MAS | `/home/cxb/OrchRL/examples/mas_app/search/` |
| SWE-agent recipe（Verl 参考） | 在 `third_party/verl` 子模块中搜索 |
| V0 设计文档（含演进路径） | `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` |
| 训练侧对接规格 | `/home/cxb/MATE-reboot/docs/plans/2026-03-05-training-integration-spec.md` |
| 真实环境验证记录 | `/home/cxb/MATE-reboot/docs/retros/2026-03-09-trajectory-engine-real-validation.md` |

## 工作方式

- 这是一个 brainstorming 阶段，以分析和讨论为主，不执行代码改动
- 以 repo 中的代码和文档为 ground truth，不依赖历史对话上下文
- 遵循 `AGENTS.md` 所有规则（特别是第 9 条客观性原则）
- 对每个议题给出明确的利弊分析和建议，不回避取舍
- 使用 Superpowers 或其他合适的 skills
- 所有回复使用中文
- 讨论结束后，根据结论输出一份 V0.2 的设计方向文档草案（放在 `docs/plans/` 下）
