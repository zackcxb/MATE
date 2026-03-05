# Trajectory Engine V0 — Code Review + Merge Prompt

> 在执行 Agent 窗口中粘贴此 prompt（该窗口已有实现上下文）。

---

## 任务

对 `feat/trajectory-engine-v0` 分支进行 code review，处理发现的问题后合并到 `main`，然后更新项目文档。

## 上下文

你（或前一位 Agent）刚刚完成了 Trajectory Engine V0 的实现，代码在 `feat/trajectory-engine-v0` 分支上，56 个测试全部通过。现在需要做以下三件事：

### 第一步：两阶段 Code Review

请阅读 `/home/cxb/MATE-reboot/AGENTS.md` 第 3.5 条，这是一个 meaningful change（新公共接口、跨模块、多个生产文件），需要两阶段 review。

**阶段 1 — Spec Compliance Review**

对照设计文档 `docs/plans/2026-03-04-trajectory-engine-v0-design.md`，逐项检查：

1. 策略模式是否正确实现（InferenceBackend 接口 → VLLMBackend，无队列/anti-call 残留）
2. ModelMonitor 是否通过 `model` 字段识别 agent role 并路由
3. 轨迹输出是否为 `Dict[agent_id, List[TurnData]]` 格式
4. AgentPipe 编排流程是否与设计文档第 7 节一致
5. RewardWorker 是否实现了 RewardProvider 协议
6. 数据结构中元数据字段（episode_id, agent_role, turn_index, timestamp）是否完整

**阶段 2 — Code Quality / Risk Review**

1. 错误处理：Monitor 后端异常是否返回 502？MAS 进程非零退出是否正确处理？
2. 资源清理：Monitor stop、临时文件 cleanup 是否在 finally 块中？
3. 线程安全：Monitor buffer 的并发访问是否有保护？
4. 测试覆盖：是否有边界情况遗漏（空 buffer、unknown agent、超时等）？
5. 代码风格：是否有冗余注释、过度注释或缺少必要注释？

对于每个阶段，输出通过/不通过的逐项结论。如有不通过项，修复后重新验证。

### 第二步：Merge 到 main

所有 review 项通过后：

```bash
cd /home/cxb/MATE-reboot
git checkout main
git merge feat/trajectory-engine-v0 --no-ff -m "merge: trajectory engine v0 implementation (56 tests passing)"
```

验证 merge 后测试仍通过：`python -m pytest tests/ -v`

### 第三步：更新项目文档

更新 `docs/project-context.md`：
- 当前阶段改为"V0 实现完成，已合并"
- 待办列表中已完成的项目打勾
- 新增下一阶段待办（vLLM token_ids 提取、Episode 并行采样）

## 核心设计决策（已锁定，review 时以此为准）

1. **策略模式**（非队列 anti-call）：`await backend.generate(request)` 是唯一代码路径
2. **agent 身份识别**：request body 中 `model` 字段即 agent role
3. **轨迹输出**：`Dict[agent_id, List[TurnData]]`，元数据打全
4. **MAS 启动**：进程模式（subprocess），每 episode 独立 Monitor 实例
5. **Reward**：`RewardProvider` 协议，V0 实现 `FunctionRewardProvider`

## 注意事项

- 遵循 `AGENTS.md` 所有规则
- 执行 Agent 在实施过程中做的有意设计变更（sync Popen、线程安全、严格校验）属于改进，不视为 spec 偏离
- 所有回复使用中文
