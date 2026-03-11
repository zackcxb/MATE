# OrchRL Tree Adapter Execution Prompt

你在 `/home/cxb/OrchRL` 仓库中工作，目标是把 MATE-reboot V0.2 的树状分支采样接入 OrchRL 训练侧。

## 目标

完成 OrchRL 侧对 `tree_rollout` / `TreeEpisodeResult` 的消费闭环，不改 MATE-reboot 核心采集库语义。

## 已知事实

1. MATE-reboot 侧 V0.2 已完成，`main` 上验证通过。
2. 当前 OrchRL 训练侧实际使用的是 `mate_rollout_adapter.py` + `mate_dataproto_adapter.py`。
3. 当前适配路径只支持 `parallel_rollout`，不支持 `tree_rollout`。
4. 当前 batch adapter 使用平铺 uid：`f"{prompt_group_id}:{agent_idx}"`，没有树分支语义。
5. 当前 OrchRL 集成不使用 `AsyncLLMServerManager` 直连 MATE，而是通过 server address + `VLLMBackend` 路径。

## 必读文件

- `/home/cxb/OrchRL/orchrl/trainer/mate_rollout_adapter.py`
- `/home/cxb/OrchRL/orchrl/trainer/mate_dataproto_adapter.py`
- `/home/cxb/OrchRL/orchrl/trainer/multi_agents_ppo_trainer.py`
- `/home/cxb/MATE-reboot/docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`
- `/home/cxb/MATE-reboot/docs/project-context.md`
- `/home/cxb/MATE-reboot/mate/trajectory/tree.py`
- `/home/cxb/MATE-reboot/mate/trajectory/datatypes.py`

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
4. 不要实现 `VerlBackend`
5. 不要修改 MATE-reboot 核心采集逻辑，除非发现明确的 OrchRL 接入 blocker，并提供证据

## 推荐实现顺序

### Task 1: 配置与 rollout adapter 路径

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

## 交付格式

最终请汇报：

1. 改了哪些文件
2. rollout mode 如何配置
3. tree uid 规则是什么
4. replay turn 如何处理
5. 跑了哪些测试，结果是什么
6. 剩余风险和后续建议
