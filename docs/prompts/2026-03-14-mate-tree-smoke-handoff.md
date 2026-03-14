# MATE Tree Smoke Handoff Prompt

你在新的服务器上作为 `MATE-reboot` 侧 Agent 工作。目标不是继续做抽象层重构，而是接住已经完成的 OrchRL tree adapter 代码闭环，支持真实 tree smoke 联调，并在出现明确证据时处理 MATE blocker。

## 角色

你是 MATE-reboot 仓库的技术 lead / 支撑 Agent。

你的职责：

1. 维护 MATE 侧上下文和事实基线
2. 在 OrchRL Agent 跑真实 tree smoke 时，判断问题是否属于 MATE blocker
3. 仅在有明确复现证据时修改 MATE 代码或文档
4. 若无新的 MATE blocker，不要把时间消耗在新的结构性整理上

## 当前结论

截至 2026-03-14：

1. MATE 采集侧 V0.2 已完成，`tree_rollout` / `ReplayCache` / 验证脚本 / 可视化均已落地
2. OrchRL 侧 tree adapter 代码闭环已在独立 worktree 完成，并通过本地测试
3. 真实 tree smoke 还没有完全跑通
4. 当前已知 smoke 最深推进位置是：
   - trainer 初始化成功
   - async vLLM rollout server 初始化成功
   - 训练主循环进入 collect
   - collect 进入 `tree_rollout -> pilot_pipe.run(...)`
5. 当前 blocker 已下沉到 Search MAS 子进程 / 当前服务器硬件环境，不是已确认的 MATE public API 缺口

## 当前 MATE 基线

- 仓库：`/home/cxb/MATE-reboot`
- 当前参考提交：`3f3af57`
- 2026-03-14 本地验证：

```bash
cd /home/cxb/MATE-reboot
python -m pytest tests/trajectory tests/scripts -q
```

结果：

- `90 passed, 1 skipped`

## 必读文件

1. `/home/cxb/MATE-reboot/AGENTS.md`
2. `/home/cxb/MATE-reboot/docs/project-context.md`
3. `/home/cxb/MATE-reboot/docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`
4. `/home/cxb/MATE-reboot/docs/plans/2026-03-11-trajectory-public-api-boundary.md`
5. `/home/cxb/MATE-reboot/docs/retros/2026-03-09-trajectory-engine-real-validation.md`
6. `/home/cxb/MATE-reboot/docs/retros/2026-03-11-orchrl-tree-adapter-sync.md`
7. `/home/cxb/MATE-reboot/docs/plans/2026-03-13-tokenization-drift-analysis.md`

## 与 OrchRL Agent 的边界

默认分工：

- OrchRL Agent：
  - 负责真实 tree smoke 执行
  - 负责 OrchRL worktree / trainer / adapter / runtime / train job 侧问题
- MATE Agent：
  - 负责 MATE 基线确认
  - 负责判断 smoke 失败是否暴露了新的 MATE blocker
  - 负责在必要时最小化修补 MATE 并补验证

不要做的事：

1. 不要在没有证据的情况下主动重写 MATE 采集逻辑
2. 不要因为 smoke 没跑通就默认把问题归咎到 MATE
3. 不要让 OrchRL 训练侧问题反向污染 MATE public API

## 已知关键事实

1. OrchRL 当前真实集成路径仍是 adapter + `VLLMBackend`，不是 `AsyncLLMServerManager` 直连
2. `docs/plans/2026-03-05-training-integration-spec.md` 中的 `VerlBackend` 是历史直连设想，不是当前近端实现目标
3. OrchRL tree adapter 仍依赖按 `timestamp` 重建 branch 内全局顺序；这是技术债，但不是当前 smoke blocker
4. 如果后续确实需要减轻训练侧推断复杂度，MATE 最有价值的潜在增强点是显式暴露稳定 global turn order

## 新服务器上的第一轮动作

1. 确认 `/home/cxb/MATE-reboot` 工作区可用并阅读上述文档
2. 跑一遍 MATE 基线测试，确认新服务器环境没有额外回归
3. 与 OrchRL Agent 对齐它当前使用的：
   - worktree 路径
   - smoke 命令
   - vLLM / retrieval / GPU 约束
   - 最新错误日志
4. 对 smoke 失败进行归因：
   - OrchRL trainer/runtime 问题
   - Search MAS 子进程问题
   - vLLM / retrieval / 环境问题
   - MATE 采集契约问题

## 遇到 smoke 失败时的处理原则

只有在同时满足以下条件时，才在 MATE 侧改代码：

1. 有清晰报错链路落到 MATE
2. 能构造最小复现或至少拿到稳定日志证据
3. 问题不能在 OrchRL adapter 层或运行环境层解决

如果满足以上条件，优先做：

1. 最小修补
2. 对应测试
3. 文档更新

## 参考 smoke 背景

旧服务器上，为了推进 smoke，OrchRL 侧曾使用过如下条件：

```bash
env -u SEARCH_MAS_LLM_BASE_URL -u OPENAI_BASE_URL \
  SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://127.0.0.1:8010/retrieve \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=3,4,6 \
  VLLM_USE_V1=1 \
  CONFIG_NAME=search_mas_tree_real_smoke \
  LOG_PATH=logs/search_mas_tree_real_smoke.log \
  bash scripts/run_search_mas_train_e2e.sh
```

这些 GPU 约束来自旧服务器坏卡现状。新服务器应根据实际 GPU 拓扑重选，不要机械照搬。

## 本轮输出要求

请最终汇报：

1. 新服务器上的 MATE 基线是否通过
2. 当前 smoke 最深推进到了哪里
3. 失败是否属于 MATE blocker
4. 如果做了 MATE 改动，改了哪些文件、为什么改、跑了哪些验证
5. 如果没有做 MATE 改动，下一步建议 OrchRL Agent 或用户做什么
