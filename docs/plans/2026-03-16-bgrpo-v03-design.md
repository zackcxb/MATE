# V0.3 Design (Rev B): 基于 PR #10 的 BGRPO 方案重整

> Status: **Reviewed after PR #10/#9, pending implementation**  
> Date: 2026-03-17  
> Scope: OrchRL BGRPO 分组正确性 + MATE 数据契约补齐（token-drift / global_turn_index）

## 1. 结论摘要

本次以 OrchRL 外部 PR 为基线完成复审后，V0.3 的核心结论如下：

1. **PR #10 的 `group_id` 分组思路是正确方向**，可接受为主线实现。
2. **PR #10 已实现“做法 C”（每个 branch 只保留分支点 turn）**，但仍依赖 timestamp 推断全局 turn 顺序，稳定性不足。
3. **PR #10 未覆盖 token-drift 防护（`prompt_ids`）与采集侧 `global_turn_index` 契约增强**，这两项仍是 V0.3 必做。
4. **PR #9 对 MATE/OrchRL 适配层的核心影响主要是导入路径迁移**（`trajectory` -> `orchrl.agent_trajectory_engine`），但其 `orchrl/__init__.py` 存在语法错误，需先修复后才能安全合入。

## 2. PR 审查结果（代码级）

### 2.1 PR #10（核心）

#### 2.1.1 UID / group_id 分组

实现位置：
- `orchrl/trainer/mate_dataproto_adapter.py:266-283`（`group_id` 与 `uid` 生成）
- `orchrl/trainer/multi_agents_ppo_trainer.py:498-516`（`uid -> group_id` 临时替换并恢复）
- `orchrl/trainer/multi_agents_ppo_trainer.py:983-1007`（group key 解析）

结论：
- `group_id = "{prompt_group_id}:turn{global_turn_index}:agent{agent_idx}"` 能让 pilot 与同分支点 branch 落入同组。
- `uid = "{group_id}:{source}:{episode_id}"` 保持样本唯一，语义清晰。
- `_update_parameters` 的 “替换 -> `compute_advantage` -> 恢复” 使用 `try/finally`，异常路径可恢复，模式安全。

> 设计决策：V0.3 采用“`uid`（唯一标识）与 `group_id`（GRPO 分组键）分离”的模式，替代此前“统一 uid 格式兼做分组”的方案。

#### 2.1.2 分支点 turn 选择（做法 C）

实现位置：
- `orchrl/trainer/mate_dataproto_adapter.py:96-155`（`tree_episodes_to_decision_point_batches`）
- `orchrl/trainer/mate_dataproto_adapter.py:311-323`（`_select_branch_decision_turn`）

结论：
- 每个 branch 只选一个 decision-point turn；continuation turns 不进入 batch。
- 与 V0.3 既定“做法 C”一致（分支点参与 loss，延续轨迹只用于回报评估）。

注意：
- turn 全局顺序仍由 `_turn_global_positions` 基于 timestamp 推断（`mate_dataproto_adapter.py:290-299`），非采集侧显式 `global_turn_index`。

#### 2.1.3 Reward 处理

实现位置：
- `orchrl/trainer/mate_dataproto_adapter.py:265`（`_resolve_episode_final_reward(...)`）
- `orchrl/trainer/mate_dataproto_adapter.py:331-334`

结论：
- Tree/BGRPO 路径统一使用 `episode.final_reward`，不再走 per-role + `credit_assignment`。
- 对 BGRPO（基于分叉后完整 rollout outcome）是合理默认；更贴近“同决策点多候选动作比较最终结果”的算法语义。

边界：
- 若未来奖励体系显式需要 role-specific credit，需增加可配置策略（默认 final_reward，选配 per-role）。

#### 2.1.4 已识别风险与缺口

1. **多 sample 混组风险（需修）**  
   `group_id` 未包含 `sample_idx`（`mate_dataproto_adapter.py:259,266`），当 `n_samples_per_prompt > 1` 时可能把不同 pilot tree 混到同一组。  
   当前 PR 配置把 `train_sample_num` 调为 1（`orchrl/config/search/search_mas_nosearch_external.yaml`），但这是配置规避，不是代码层防护。

2. **分支点缺失时静默降级（需加校验）**  
   `_select_branch_decision_turn` 找不到匹配时返回 `None` 或 fallback 首个 turn（`mate_dataproto_adapter.py:317-323`），当前为静默行为，可能导致 group size 异常而不易察觉。

3. **仍有 timestamp 重建依赖（需由 MATE 合同消除）**  
   `_turn_global_positions` 继续依赖 timestamp（`mate_dataproto_adapter.py:290-299`），尚未接入采集侧显式 `global_turn_index`。

4. **token-drift 仍未处理（V0.3 blocker）**  
   prompt 仍由训练侧 `messages -> tokenize` 重建（`mate_dataproto_adapter.py:263,337-346`），未优先消费 rollout 侧 `prompt_ids`。

### 2.2 PR #9（依赖迁移）

审查重点文件：
- `orchrl/trainer/mate_dataproto_adapter.py:8`
- `orchrl/trainer/mate_rollout_adapter.py:11-18`
- `orchrl/trainer/train.py:53`

结论：
- 适配层主变更是导入路径迁移（`trajectory` -> `orchrl.agent_trajectory_engine`），只要该包随仓发布，MATE/OrchRL 集成层逻辑本身不受影响。
- 发现一个合入阻断问题：`orchrl/__init__.py:55` 有 f-string 引号冲突，存在 `SyntaxError` 风险，需先修复再合入。

## 3. V0.3 设计重整（相对旧版）

### 3.1 保留并采纳

1. Tree 模式只训练分支点 action（做法 C）。
2. 在 advantage 计算时按分组键而非样本唯一键分组。
3. Tree/BGRPO 默认使用 episode-level outcome（`final_reward`）。

### 3.2 方案替换

1. **旧方案（弃用）**：通过统一 uid 格式同时承担“唯一标识+分组语义”。
2. **新方案（采用）**：
   - `uid`：仅用于样本唯一标识；
   - `group_id`：仅用于 GRPO 分组；
   - `_update_parameters` 中临时将 `uid` 替换为 `group_id` 供 `compute_advantage` 使用。

### 3.3 必须补齐（PR #10 未覆盖）

1. MATE 合同新增 `global_turn_index`，OrchRL 优先消费该字段（无则 fallback timestamp）。
2. MATE 合同新增 `prompt_ids`，OrchRL 优先消费（无则 fallback tokenize）。
3. Tree group key 需纳入 sample 维度，防止跨 sample 混组。
4. 增加 decision-point 分组完整性校验（最少样本数、期望 K、异常日志）。

## 4. 接受 PR #10 后的目标架构

### 4.1 分组键定义（最终）

```text
group_id = {prompt_group_id}:sample{sample_idx}:turn{global_turn_index}:agent{agent_idx}
uid      = {group_id}:{source}:{episode_id}
```

说明：
- `sample_idx` 入 group key，避免多 sample 混组。
- `uid` 仍保留 `source/episode_id` 以保证全局唯一并支持审计。

### 4.2 turn 顺序来源（最终）

1. 首选 `turn.global_turn_index`（来自 MATE 采集侧，稳定）。
2. 缺失时 fallback `_turn_global_positions`（仅兼容旧数据）。

### 4.3 prompt token 来源（最终）

1. 首选 `turn.prompt_ids`（rollout 实际使用 token）。
2. 缺失时 fallback `_tokenize_messages(messages)`。

## 5. 验证标准（更新）

### Level 1（单测）

1. Pilot + branch 同 decision-point 共享 `group_id`，`uid` 各自唯一。
2. `n_samples_per_prompt=2` 时不同 sample 不会被分到同组。
3. 每个 branch 仅产出一个 decision-point 样本，continuation 不入 batch。
4. `compute_advantage` 期间替换为 `group_id`，退出后 `uid` 恢复（含异常路径）。

### Level 2（Tree smoke）

1. group size 统计与 `k_branches` 对齐（允许失败分支但需有告警）。
2. advantage 非全零（排除大量 singleton 退化）。
3. 无 NaN/Inf。

### Level 3（短跑）

1. 50-100 step 内 reward/advantage 分布无明显异常。
2. 抽样验证 `prompt_ids` 直通消费链路。
3. 抽样验证 `global_turn_index` 与 fallback 路径的一致性。

## 6. 向后兼容策略

1. 对旧轨迹（无 `prompt_ids` / `global_turn_index`）保持可训练：
   - `global_turn_index` fallback timestamp；
   - `prompt_ids` fallback tokenize。
2. 新增校验不应阻断旧数据训练，但需给出显式告警与统计。
