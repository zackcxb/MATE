# OrchRL Tree Adapter Execution Prompt

你在 `/home/cxb/OrchRL` 仓库中工作，目标是把 MATE-reboot V0.2 的树状分支采样接入 OrchRL 训练侧。

## 执行场地

请不要直接在当前 `/home/cxb/OrchRL` 工作区上开发。优先使用新窗口 + 新 worktree，原因是：

1. 这是跨仓集成任务，需要独立上下文和验证闭环
2. OrchRL 主工作区可能已有未提交改动
3. 本任务预期会改 trainer / adapter / test 多处文件，单独工作区更易审查和回滚

推荐启动方式：

```bash
cd /home/cxb/OrchRL
git worktree add ~/.config/superpowers/worktrees/OrchRL/mate-tree-adapter -b feat/mate-tree-adapter main
cd ~/.config/superpowers/worktrees/OrchRL/mate-tree-adapter
```

在新窗口中进入该目录后，再开始执行本 prompt。

## 目标

完成 OrchRL 侧对 `tree_rollout` / `TreeEpisodeResult` 的消费闭环，不改 MATE-reboot 核心采集库语义。

## 已知事实

1. MATE-reboot 侧 V0.2 已完成，`main` 上验证通过。
2. 当前 OrchRL 训练侧实际使用的是 `mate_rollout_adapter.py` + `mate_dataproto_adapter.py`。
3. 当前适配路径只支持 `parallel_rollout`，不支持 `tree_rollout`。
4. 当前 batch adapter 使用平铺 uid：`f"{prompt_group_id}:{agent_idx}"`，没有树分支语义。
5. 当前 OrchRL 集成不使用 `AsyncLLMServerManager` 直连 MATE，而是通过 server address + `VLLMBackend` 路径。
6. `docs/plans/2026-03-05-training-integration-spec.md` 是历史直连设想，不是这次实现的依据。
7. MATE 当前采用“软收紧”API 策略：保持导出不变，但 OrchRL 应只依赖稳定外部 API。

## 必读文件

- `/home/cxb/MATE-reboot/AGENTS.md`
- `/home/cxb/MATE-reboot/docs/plans/2026-03-11-trajectory-public-api-boundary.md`
- `/home/cxb/OrchRL/orchrl/trainer/mate_rollout_adapter.py`
- `/home/cxb/OrchRL/orchrl/trainer/mate_dataproto_adapter.py`
- `/home/cxb/OrchRL/orchrl/trainer/multi_agents_ppo_trainer.py`
- `/home/cxb/MATE-reboot/docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`
- `/home/cxb/MATE-reboot/docs/project-context.md`
- `/home/cxb/MATE-reboot/mate/trajectory/tree.py`
- `/home/cxb/MATE-reboot/mate/trajectory/datatypes.py`

## 执行约束

1. 把 `/home/cxb/MATE-reboot/AGENTS.md` 视为本次跨仓协作约束来源，尤其注意：
   - meaningful change 需要验证证据
   - 不要含糊地建议“另开窗口”，若需要切场地必须给出具体 handoff
   - 结论必须基于技术事实，不迎合既有偏好
2. 仅依赖 MATE 稳定外部 API：
   - `parallel_rollout`
   - `tree_rollout`
   - `AgentPipeConfig`
   - `ModelMappingEntry`
   - `VLLMBackend`
   - `FunctionRewardProvider`
   - `EpisodeResult` / `TreeEpisodeResult` / `BranchResult` / `TurnData` / `EpisodeTrajectory`
3. 不要让 OrchRL 新增对 `AgentPipe`、`ModelMonitor`、`ReplayCache`、`InferenceBackend` 等 MATE 内部/暂不承诺接口的依赖。

## 实施要求

1. 在 OrchRL trainer 配置中增加 rollout mode，至少支持：
   - `parallel`
   - `tree`
2. 当 mode=`tree` 时：
   - `mate_rollout_adapter.py` 调用 `tree_rollout`
   - 返回值可被训练侧后续 batch adapter 消费
3. 新增 tree-aware dataproto adapter：
   - 消费 `TreeEpisodeResult`
   - 跳过 replayed prefix turn
   - 为 branch point 生成 branch-aware uid
   - 保持 pilot turn 与现有 uid 兼容
4. 不要修改 MATE-reboot 核心采集逻辑，除非发现明确的 OrchRL 接入 blocker，并提供证据

## 推荐实现顺序

### Task 1: 配置与 rollout adapter 路径++

- 在 OrchRL 配置层引入 `rollout_mode`
- 更新 `mate_rollout_adapter.py`，支持 `parallel_rollout` / `tree_rollout` 分支
- 为新模式补单元测试或最小集成测试

### Task 2: Tree batch adapter

- 新增 `tree_episodes_to_policy_batches` 或等价函数
- 明确 replay turn 过滤规则
- 明确 branch-aware uid 规则
- 为 pilot / branch / replay skip 各写至少一个测试

### Task 3: Trainer 接线

- 更新 `multi_agents_ppo_trainer.py`
- 根据 rollout mode 选择普通或 tree-aware adapter
- 确保错误信息足够清晰

### Task 4: 验证

- 跑 OrchRL 侧相关测试
- 至少做一次 mock/tree smoke 验证
- 输出证据：命令、通过数量、关键日志

## 建议工作流

1. 先在 OrchRL worktree 内阅读上述文件，确认当前 `parallel_rollout -> dataproto adapter -> trainer` 的数据流。
2. 先设计 tree 模式的最小闭环，再开始实现：
   - rollout mode 配置入口
   - `tree_rollout` 返回结构如何进入 dataproto adapter
   - replay prefix skip 和 branch uid 规则
3. 优先在 OrchRL 侧解决适配，不把训练侧问题转嫁回 MATE。
4. 如果发现确实需要 MATE 调整，先形成 blocker 说明：
   - 具体卡点
   - 复现证据
   - 为什么不能在 OrchRL adapter 层消化

## 交付格式

最终请汇报：

1. 改了哪些文件
2. 使用的工作目录 / 分支 / worktree 是什么
3. rollout mode 如何配置
4. tree uid 规则是什么
5. replay turn 如何处理
6. 跑了哪些测试，结果是什么
7. 剩余风险和后续建议
