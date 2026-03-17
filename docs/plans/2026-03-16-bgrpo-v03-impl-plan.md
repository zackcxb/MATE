# V0.3 BGRPO Implementation Plan (Rev B)

> Goal: 在接受 PR #10 主体思路的前提下，补齐其 correctness gap，并完成 MATE 合同增强（`global_turn_index` + `prompt_ids`），使 BGRPO 在 tree 模式下稳定可用。

> Inputs reviewed on 2026-03-17:
> - OrchRL PR #10 (`add JD algorithm with tree sampling`)
> - OrchRL PR #9 (`Remove self-contained verl code...`)
> - `docs/plans/2026-03-16-bgrpo-v03-design.md`（Rev B）

---

## 0. 范围与原则

### In Scope

1. 接受 PR #10 的主干机制：`group_id` 分组 + decision-point batch。
2. 修复 PR #10 已识别风险：sample 混组、静默降级、分组完整性诊断。
3. 落地 MATE 合同增强：`global_turn_index`、`prompt_ids`。
4. OrchRL 适配层优先消费新字段，并保留 fallback。

### Out of Scope

1. 大规模训练策略优化（如 packing/吞吐调优）。
2. 非 BGRPO 必需的 reward 体系重构。

### 执行顺序

1. 先解决依赖 PR 的合入阻断。
2. 再补 OrchRL 侧 correctness patch。
3. 再落 MATE 合同增强与回传链路。
4. 最后进行端到端验证与短跑诊断。

---

## 1. Phase A: PR 基线落地与阻断修复

### A1. 处理 PR #9 合入阻断

**目标**：确保依赖迁移后仓库可导入、可启动。

- [ ] 修复 `orchrl/__init__.py` 的语法错误（f-string 引号冲突）。
- [ ] 验证 `trajectory -> orchrl.agent_trajectory_engine` 导入迁移在以下文件可运行：
  - `orchrl/trainer/mate_dataproto_adapter.py`
  - `orchrl/trainer/mate_rollout_adapter.py`
  - `orchrl/trainer/mate_reward_bridge.py`
- [ ] 验证 `train.py` 对新版 `verl` 导入路径可用：
  - `orchrl/trainer/train.py`

**验收**：
- `python -m py_compile` 覆盖上述关键文件通过。
- 最小化导入 smoke（仅 import）通过。

### A2. 合入 PR #10 主体能力

**目标**：以 PR #10 为 BGRPO 基线，保留其已验证可行的核心机制。

- [ ] 保留 `tree_episodes_to_decision_point_batches` 新路径。
- [ ] 保留 `_update_parameters` 中 `uid -> group_id -> restore` 流程。
- [ ] 保留 tree 模式使用 decision-point adapter 的路由切换。

**验收**：
- PR #10 新增单测（group_id 替换/恢复等）通过。

---

## 2. Phase B: OrchRL correctness gap 补齐（PR #10 后续补丁）

### B1. 修复 sample 维度混组

**问题**：当前 `group_id` 未包含 `sample_idx`，`n_samples_per_prompt > 1` 时会跨 sample 混组。

**改动文件**：
- `orchrl/trainer/mate_dataproto_adapter.py`

**任务**：
- [ ] 将 `sample_idx` 纳入 `group_id`。
- [ ] 同步更新调试字段/日志输出，确保可直接观察 sample 维度。

**验收**：
- [ ] 新增单测：同 `prompt_group_id` + 不同 `sample_idx` 不同组。
- [ ] `n_samples_per_prompt=2` 的构造数据组大小符合预期。

### B2. 分支点选择从“静默降级”改为“可观测校验”

**问题**：`_select_branch_decision_turn` 在异常匹配时 fallback/skip，可能悄然导致 singleton 组。

**改动文件**：
- `orchrl/trainer/mate_dataproto_adapter.py`

**任务**：
- [ ] 当 branch decision-turn 缺失或 role 不匹配时，记录结构化告警（包含 prompt_group_id/branch_turn/episode_id）。
- [ ] 提供严格模式开关（训练时可选择 fail-fast）。

**验收**：
- [ ] 新增单测：缺失匹配时触发告警或抛错（按模式）。

### B3. 增加 group 完整性诊断

**问题**：当前没有针对 decision-point group size 的硬性校验，异常时不易发现。

**改动文件**：
- `orchrl/trainer/multi_agents_ppo_trainer.py`

**任务**：
- [ ] 在 `_finalize_batch_for_update`/`_update_parameters` 前后记录 group size 分布。
- [ ] 增加最小组规模与异常比例指标（例如 singleton_ratio）。

**验收**：
- [ ] 训练日志中可直接看到 group_size histogram 与异常计数。
- [ ] 当大量 singleton 发生时有明确告警。

### B4. reward 语义策略化（默认 final_reward）

**问题**：PR #10 固定 `final_reward`，缺少显式配置声明。

**改动文件**：
- `orchrl/trainer/mate_dataproto_adapter.py`
- `orchrl/config/search/*.yaml`

**任务**：
- [ ] 保持默认 `final_reward`。
- [ ] 增加可选策略（例如 `tree_reward_source: final_reward|role_reward`），仅作为扩展点。

**验收**：
- [ ] 默认配置行为与 PR #10 一致。
- [ ] 切到 role_reward 时有单测覆盖。

---

## 3. Phase C: MATE 合同增强（V0.3 必做）

### C1. 数据结构扩展

**改动文件（MATE）**：
- `mate/trajectory/datatypes.py`

**任务**：
- [ ] `TurnData` 增加 `global_turn_index`、`prompt_ids`。
- [ ] `InteractionRecord` 增加 `global_turn_index`、`prompt_ids`。
- [ ] `ModelResponse` 增加 `prompt_ids`。

### C2. 采集链路贯通

**改动文件（MATE）**：
- `mate/trajectory/monitor.py`
- `mate/trajectory/backend.py`
- `mate/trajectory/collector.py`

**任务**：
- [ ] Monitor 维护全局 turn 计数并写入 record。
- [ ] VLLMBackend 生成并透传 prompt_ids（tokenizer 可用时）。
- [ ] Collector 透传新字段到 TurnData。

### C3. Tree/Replay 顺序对齐

**改动文件（MATE）**：
- `mate/trajectory/tree.py`
- `mate/trajectory/replay_cache.py`

**任务**：
- [ ] 排序主键从 timestamp 切换为 `global_turn_index`。
- [ ] 旧数据缺字段时保持 timestamp fallback。

### C4. OrchRL 消费新字段

**改动文件（OrchRL）**：
- `orchrl/trainer/mate_dataproto_adapter.py`

**任务**：
- [ ] `_iter_global_turns` 优先消费 `turn.global_turn_index`。
- [ ] `_tokenize_messages` 之前优先消费 `turn.prompt_ids`。
- [ ] 缺失时 fallback 保持兼容。

---

## 4. 测试与验证计划

### Level 1: 单元测试（无 GPU）

- [ ] `group_id` 规则测试：pilot/branch 同 decision-point 同组，`uid` 唯一。
- [ ] sample 隔离测试：`sample_idx` 不同不混组。
- [ ] decision-point 选择测试：continuation 不入 batch。
- [ ] `uid` 替换-恢复异常路径测试（`compute_advantage` 抛错时恢复）。
- [ ] MATE 新字段链路测试：monitor -> collector -> dataproto。

### Level 2: 集成 smoke（小规模）

- [ ] Tree mode 2-5 step 训练 smoke。
- [ ] 记录并检查 group_size histogram / singleton_ratio。
- [ ] 检查 advantage 非全零、loss 无 NaN/Inf。

### Level 3: 短跑诊断（8 卡，50-100 step）

- [ ] reward 曲线无明显异常发散。
- [ ] group 统计稳定（异常比例可解释）。
- [ ] 抽样比对 `prompt_ids` 与 fallback tokenize 差异。

---

## 5. 交付物与里程碑

### Milestone M1（PR 合入可用）

- PR #9 阻断修复完成。
- PR #10 主体能力可运行。

### Milestone M2（OrchRL correctness 补齐）

- sample 混组修复。
- 分支点校验与组完整性诊断上线。

### Milestone M3（V0.3 完成）

- MATE 合同增强完成并被 OrchRL 优先消费。
- Level 1/2/3 验证全部通过。

---

## 6. 风险清单

1. **依赖漂移风险**：PR #9 切到外部 `verl` 后 API 变更可能影响训练路径。  
   缓解：关键路径 smoke + 版本锁定。

2. **旧数据兼容风险**：历史轨迹无新字段。  
   缓解：保留 fallback，并在日志中标记 fallback 命中率。

3. **分组退化风险**：失败分支过多导致 singleton 组。  
   缓解：组规模监控 + 告警 + 训练前阈值检查。

4. **token-drift 残留风险**：`prompt_ids` 缺失时仍依赖 fallback tokenize。  
   缓解：推动 rollout 侧尽快稳定产出 `prompt_ids`，并在训练日志统计缺失比例。
