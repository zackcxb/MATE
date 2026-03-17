# V0.3 PR 审查 + 设计文档重整 — Agent 交接 Prompt

> 日期：2026-03-17
> 目标：审查 OrchRL 两个待合入 PR，评估 BGRPO 实现正确性，重整 V0.3 设计与计划文档

## 角色

你是 MATE 项目和 OrchRL 项目的技术 lead / Master Agent。当前有两个外部 PR 即将合入 OrchRL main，其中一个包含了同事的 BGRPO（即 JD algorithm）实现。你需要：

1. 审查 PR 中的 BGRPO 实现流程，判断其正确性
2. 基于 PR 的实际实现，重整已有的 V0.3 设计文档和实现计划

## 两个 PR 概览

### PR #9: Remove self-contained verl code, only depend on imported verl
- 作者：wangtiance
- 分支：`verl_dependency`
- 状态：DRAFT
- 改动：删除 verl 文件夹（-83078 行），适配新版 verl 依赖
- 主要影响文件（非 verl/ 的）：`orchrl/trainer/multi_agents_ppo_trainer.py`, `orchrl/trainer/train.py`, `orchrl/verl/` 下的适配层, 脚本
- **审查重点**：检查 verl 导入路径变更是否影响 MATE/OrchRL 集成层（`mate_dataproto_adapter.py`, `mate_rollout_adapter.py`）

### PR #10: add JD algorithm with tree sampling
- 作者：zehanzhu
- 分支：来自 fork `zehanzhu:main`
- 状态：OPEN
- 改动：+477 -69，8 个文件
- **这是本次审查的核心 PR**

PR #10 改动的文件：
1. `orchrl/trainer/mate_dataproto_adapter.py` — **新增 `tree_episodes_to_decision_point_batches` 函数**
2. `orchrl/trainer/multi_agents_ppo_trainer.py` — 切换 tree mode 到新函数 + group_id 机制
3. `orchrl/config/search/search_mas_nosearch_external.yaml` — tree mode 配置
4. `orchrl/config/search/search_mas_nosearch_external_5step_4x4_conservative.yaml` — 配置调整
5. `orchrl/utils/importlib_metadata_compat.py` — 新增 importlib 兼容层
6. `orchrl/__init__.py` — 注册 compat patch
7. `scripts/run_search_mas_train_e2e.sh` — 默认配置和 GPU 调整
8. `tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py` — 新增 3 个测试

## 核心审查任务

### 任务 1：分析 PR #10 的 BGRPO 实现

PR #10 新增了一个独立的 batch 构建路径 `tree_episodes_to_decision_point_batches`（与现有 `tree_episodes_to_policy_batches` 并存，未修改后者）。需要重点审查：

#### 1a. UID / group_id 分组方案

PR #10 引入了 `group_id` 概念（与 `uid` 分离）：
- `group_id = f"{prompt_group_id}:turn{global_turn_index}:agent{agent_idx}"` — 用于 GRPO advantage 分组
- `uid = f"{group_id}:{source}:{episode_id}"` — 保持全局唯一

**我们之前的判断**：现有代码的 uid 分配方式是错误的——pilot turns 和同分支点的 branch turns 使用不同 uid 格式（`pg:0` vs `pg:0:b2`），永远无法落入同一 GRPO group。

**需要验证**：
- PR #10 的 `group_id` 方案是否正确解决了这个问题？
- Pilot turn 和同分支点的 branch turns 是否共享相同的 `group_id`？
- `_update_parameters` 中的 uid 临时替换为 group_id 的机制是否正确？（替换 → compute_advantage → 恢复原 uid）

#### 1b. 分支点 turn 选择逻辑

PR 新增了 `_select_branch_decision_turn` 和 `_iter_global_turns` 函数。需要验证：
- branch 中的哪个 turn 被选为"分支点 action"？
- 选择逻辑是否与我们的"做法 C"一致？（每个 branch 只贡献分支点 turn，continuation turns 不进入 batch）
- `global_turn_index` 的推断方式（仍然使用 timestamp 排序的 `_turn_global_positions`）

#### 1c. Reward 处理

PR 使用 `_resolve_episode_final_reward(episode.final_reward)` 而非 per-role reward。需要判断：
- 对于 BGRPO，使用 final_reward（episode 级别）还是 per-role reward 更合理？
- 这与我们之前设计中的 `credit_assignment` 逻辑有什么区别？

#### 1d. 与 compute_advantage 的集成

PR 在 `_update_parameters` 中：
1. 调用 `_resolve_external_mas_group_keys` 获取 group_id
2. 临时将 `batch.non_tensor_batch["uid"]` 替换为 `group_keys`
3. 调用 `compute_advantage`（verl 的 GRPO 实现，按 uid 分组）
4. 恢复原始 uid

需要验证这个"替换-计算-恢复"模式是否安全（特别是异常路径）。

### 任务 2：评估 PR 对我们 V0.3 计划的影响

我们已经有一份 V0.3 设计文档和实现计划（见下方必读文件）。PR #10 的实现与我们的计划存在差异。需要回答：

1. **PR #10 实现了我们计划中的哪些部分？哪些没有？**
   - 我们计划中的 UID 修复 → PR 用了不同方案（group_id）
   - 我们计划中的 skip_turn_predicate 做法 C → PR 用了 `_select_branch_decision_turn`
   - 我们计划中的 prompt_ids / global_turn_index MATE 侧改动 → PR 未涉及
   - 我们计划中的 token-drift 防护 → PR 未涉及

2. **PR #10 的方案与我们的方案哪个更好？为什么？**
   - group_id 分离方案 vs 我们的 uid 格式统一方案
   - `_select_branch_decision_turn` vs 我们的 `skip_turn_predicate`
   - final_reward vs per-role credit_assignment

3. **如果接受 PR #10，V0.3 还需要做什么？**
   - MATE 侧改动（prompt_ids, global_turn_index）是否仍然需要？
   - OrchRL 侧还有哪些 gap？

### 任务 3：重整设计文档和计划文档

基于审查结论，更新：
- `/home/cxb/MATE-reboot/docs/plans/2026-03-16-bgrpo-v03-design.md`
- `/home/cxb/MATE-reboot/docs/plans/2026-03-16-bgrpo-v03-impl-plan.md`

更新原则：
- 如果 PR #10 的方案更好或可接受，调整我们的计划以适配
- 如果 PR #10 有正确性问题，在文档中记录问题并给出修复建议
- 保留 MATE 侧改动中不受 PR 影响的部分（如 prompt_ids, global_turn_index）
- 如果存在冲突，以 BGRPO 的正确实现为优先

## 必读文件

### MATE-reboot 仓（`/home/cxb/MATE-reboot/`）
1. `AGENTS.md` — agent 行为规范
2. `docs/plans/2026-03-16-bgrpo-v03-design.md` — V0.3 设计文档
3. `docs/plans/2026-03-16-bgrpo-v03-impl-plan.md` — V0.3 实现计划
4. `docs/plans/2026-03-13-tokenization-drift-analysis.md` — token-drift 分析
5. `mate/trajectory/datatypes.py` — MATE 当前数据结构

### OrchRL 仓（`/home/cxb/OrchRL/`）
6. `orchrl/trainer/mate_dataproto_adapter.py` — **当前 main 版本**（审查基线）
7. `orchrl/trainer/multi_agents_ppo_trainer.py` — 当前 main 版本

### PR 内容
8. PR #10 diff: `gh pr diff 10`（核心审查对象）
9. PR #9 diff: `gh pr diff 9`（verl 适配，关注非 verl/ 文件）

### BGRPO 算法参考
10. `/home/cxb/multi-agent/tmp/pdfs/marl-04.png` — BGRPO 算法伪代码图

## 工作方式要求

1. **只做审查和文档更新，不写实现代码**
2. 遵循 `AGENTS.md` 的客观性原则：如果 PR 方案更好，直接说；如果我们的方案更好，也直接说
3. 先完成任务 1（审查），再做任务 2（影响评估），最后做任务 3（文档重整）
4. 审查结论需要有具体的代码引用（文件:行号）
5. 如果发现 PR #10 有 bug 或正确性问题，明确列出并给出修复建议

## 预期产出

1. PR #10 BGRPO 实现审查报告（内联在对话中）
2. 更新后的 `docs/plans/2026-03-16-bgrpo-v03-design.md`
3. 更新后的 `docs/plans/2026-03-16-bgrpo-v03-impl-plan.md`
4. Git commit(s)
