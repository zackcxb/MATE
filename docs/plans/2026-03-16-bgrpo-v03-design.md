# V0.3 Design: BGRPO 实现与 Token-drift 防护

> Status: **Design approved, pending implementation plan**
> Date: 2026-03-16
> Scope: MATE 采集侧契约增强 + OrchRL 训练侧 BGRPO 分组修复

## 1. 问题定义

V0.2 完成了 tree_rollout / ReplayCache / TreeEpisodeResult 的采集能力。OrchRL 已能消费 tree 轨迹。但两个问题阻碍 Branching-GRPO (BGRPO) 算法的正确实现：

1. **GRPO 分组 UID 错误**：OrchRL 当前 uid_factory 中，pilot turns 和同分支点的 branch turns 使用不同 uid 格式，永远无法落入同一 GRPO group，无法计算 comparative advantage
2. **Token-drift 风险**：MATE 不存储 prompt_ids，OrchRL 消费时用训练侧 tokenizer 重新 tokenize messages，与 vLLM 推理侧可能不一致

附带改进：
3. **缺少 global_turn_index**：当前依赖 timestamp 推断 episode 内 turn 全局顺序，MAS 并发场景下不可靠

## 2. 设计决策

### 2.1 V0.3 范围

| 方向 | 决策 | 理由 |
|------|------|------|
| BGRPO 实现 | **IN** | 下一阶段核心目标 |
| Token-drift 防护 | **IN** | BGRPO 正确性的前置条件 |
| Global turn order | **IN** | 采集侧提供比训练侧推断更可靠 |
| 采集侧保持不动 | OUT | 上述问题必须在采集侧解决 |
| Vendored 同步工具化 | OUT | 手工同步仍可控 |
| 训练效率接口（prefix packing 等） | OUT | 非 BGRPO blocker |

### 2.2 实现路径

**MATE-first**：先在 MATE 侧补足数据契约（prompt_ids + global_turn_index），再在 OrchRL 侧修 UID 分组并接入 BGRPO。

理由：
- MATE 改动是纯数据结构扩展，不改变 tree_rollout 编排逻辑，向后兼容
- prompt_ids 和 global_turn_index 不仅服务 BGRPO，也解决已识别的 token-drift 和 timestamp 脆弱性
- OrchRL 侧 UID 修复依赖对 tree 结构的正确理解，MATE 提供可靠元数据比让 OrchRL 自行推断更稳健

### 2.3 延续 turn 处理

每个 branch 的 turn 分三段：

| 段 | 条件 | 用途 | 进入训练 batch |
|---|------|------|---------------|
| Replayed prefix | `global_turn_index < branch_turn` | 重放前缀 | 否 |
| **分支点 action** | `global_turn_index == branch_turn` | **计算 loss** | **是** |
| Continuation | `global_turn_index > branch_turn` | 推进 episode 获取 reward | 否 |

采用**做法 C**：每个 branch 只贡献 1 条训练记录（分支点 turn），continuation turns 不进入 batch。

理由：延续 turn 的上下文已与 pilot 分叉，没有上下文匹配的比较对象，计算 advantage 在理论上不成立。

## 3. MATE 侧数据契约变更

### 3.1 TurnData 扩展

```python
@dataclass
class TurnData:
    agent_role: str
    turn_index: int                    # 保持：agent 内局部序号
    global_turn_index: int             # 新增：episode 内全局执行顺序
    messages: list[dict[str, Any]]     # 保持
    prompt_ids: list[int] | None       # 新增：推理侧实际使用的 prompt token IDs
    response_text: str
    token_ids: list[int] | None
    logprobs: list[float] | None
    finish_reason: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
```

| 新字段 | 类型 | 分配者 | 向后兼容 |
|--------|------|--------|---------|
| `global_turn_index` | `int` | Monitor（按请求到达顺序递增） | 默认值兼容旧数据 |
| `prompt_ids` | `list[int] \| None` | VLLMBackend（有 tokenizer 时） | None 允许 fallback |

### 3.2 InteractionRecord 同步扩展

与 TurnData 对应，新增 `global_turn_index` 和 `prompt_ids` 字段。

### 3.3 ModelResponse 扩展

新增 `prompt_ids: list[int] | None`，由 VLLMBackend.generate() 填充。

## 4. Token-drift 防护链路

数据流逐层传递 prompt_ids：

```
VLLMBackend.generate()
  ├─ tokenizer 可用 → apply_chat_template() → prompt_ids
  └─ tokenizer 不可用 → prompt_ids = None
    ↓
Monitor._handle_...()
  → InteractionRecord(prompt_ids=response.prompt_ids)
    ↓
Collector._to_turn_data()
  → TurnData(prompt_ids=record.prompt_ids)
    ↓
OrchRL mate_dataproto_adapter
  → turn.prompt_ids is not None ? 直接使用 : fallback 重新 tokenize
```

VLLMBackend 仍把 messages 原样发给 vLLM（推理路径不变）。本地 tokenize 仅为存储 prompt_ids。vLLM 底层使用相同 HuggingFace tokenizer，同一 model_path 下产出一致。

## 5. OrchRL 侧 GRPO 分组修复

### 5.1 当前问题

```python
# pilot: uid = "pg:0"
uid_factory=lambda *, prompt_group_id, agent_idx, **_: f"{prompt_group_id}:{agent_idx}"

# branch: uid = "pg:0:b2"
uid_factory=lambda ...: _tree_uid(prompt_group_id, agent_idx, branch_turn, ...)
```

pilot turn (uid=`pg:0`) 和同分支点 branch turn (uid=`pg:0:b2`) 永远不在同一 GRPO group。

### 5.2 修复后 UID 方案

```python
# Pilot turns
def pilot_uid(*, prompt_group_id, agent_idx, global_turn_index, **_):
    return f"{prompt_group_id}:{agent_idx}:bp{global_turn_index}"

# Branch turns
def branch_uid(*, prompt_group_id, agent_idx, branch_turn, global_turn_index, **_):
    if global_turn_index == branch_turn:
        # 分支点 turn → 与 pilot 同组
        return f"{prompt_group_id}:{agent_idx}:bp{branch_turn}"
    else:
        # 延续 turn → 唯一 uid（做法 C: 不进入 batch）
        return f"{prompt_group_id}:{agent_idx}:bp{branch_turn}:c{global_turn_index}"
```

### 5.3 skip_turn_predicate 修改

```python
# 当前：只跳过 replayed prefix
skip_turn_predicate=lambda turn, *, global_turn_index, **_: global_turn_index < branch.branch_turn

# 修复后：只保留分支点 turn（做法 C）
skip_turn_predicate=lambda turn, *, global_turn_index, **_: global_turn_index != branch.branch_turn
```

### 5.4 GRPO 分组效果

SearchMAS, K=3, pilot 有 4 个 turns：

| Group uid | 成员 | Size |
|-----------|------|------|
| `pg:0:bp0` | pilot_turn_0, branch_0a_turn_0, branch_0b_turn_0 | 3 |
| `pg:1:bp1` | pilot_turn_1, branch_1a_turn_1, branch_1b_turn_1 | 3 |
| `pg:0:bp2` | pilot_turn_2, branch_2a_turn_2, branch_2b_turn_2 | 3 |
| `pg:2:bp3` | pilot_turn_3, branch_3a_turn_3, branch_3b_turn_3 | 3 |

每个 GRPO group 恰好 K 个样本，干净对称。Continuation turns 不进入 batch。

## 6. 实现工作流

### 6.1 Phase 1: MATE V0.3

| 顺序 | 文件 | 改动 | 验证 |
|------|------|------|------|
| 1 | `datatypes.py` | ModelResponse / InteractionRecord / TurnData 加 prompt_ids, global_turn_index | 现有测试回归 |
| 2 | `monitor.py` | 新增 `_global_turn_counter`，填入 record | 新增 test: 递增性、跨 agent 唯一 |
| 3 | `backend.py` | VLLMBackend 有 tokenizer 时生成 prompt_ids | 新增 test: 有/无 tokenizer 两条路径 |
| 4 | `collector.py` | `_to_turn_data` 透传新字段 | 现有测试扩展 |
| 5 | `tree.py`（可选） | `_sorted_buffer` 改用 global_turn_index 排序 | 现有 tree 测试回归 |

**不改**：tree_rollout 编排逻辑、ReplayCache、parallel_rollout、AgentPipe.run()。

完成标准：`python -m pytest tests/trajectory tests/scripts -q` 全量通过。

### 6.2 Phase 2: OrchRL BGRPO

| 顺序 | 文件 | 改动 |
|------|------|------|
| 1 | vendored `trajectory/` | 同步 V0.3 新字段（仅改相对导入） |
| 2 | `mate_dataproto_adapter.py` | UID 方案修复 + skip_turn_predicate 做法 C + prompt_ids 优先消费 |
| 3 | `mate_rollout_adapter.py`（可能） | 确保 VLLMBackend 有 tokenizer |

完成标准：现有 OrchRL tree 测试回归 + 新增 GRPO 分组正确性测试 + Level 3a 8 卡短跑诊断通过。

## 7. 验证策略

### Level 1 — 单元验证（无 GPU）

构造 mock TreeEpisodeResult（K=3, 4 个 pilot turns），验证：
- 每个 GRPO group 恰好 K 个样本
- Continuation turns 不在 batch 中
- Pilot uid 和 branch uid 在分支点处相同

### Level 2 — SearchMAS tree smoke（需 vLLM + 检索服务）

用现有 SearchMAS tree smoke 跑 2 个 training step，验证：
- GRPO advantage 非全零（group 内有方差）
- Loss 数值合理（非 NaN/Inf）
- prompt_ids 在 TurnData 中非 None（tokenizer 可用时）

### Level 3a — 8 卡短跑诊断（V0.3 完成标准）

SearchMAS, BGRPO, 50-100 training steps，8 卡环境。诊断指标：

1. **Reward 趋势**：episode reward 均值是否有上升趋势（或至少不是完全平坦/发散）
2. **Advantage 分布**：GRPO group 内的 advantage 是否有方差（全零说明分组有问题）
3. **分支点 loss 贡献**：只有分支点 turn 贡献了梯度（做法 C 正确性）
4. **prompt_ids 一致性**：抽样对比 MATE 存储的 prompt_ids 与 OrchRL fallback tokenize 的结果

通过标准：上述 4 项均无异常。不要求 reward 必须上升（50-100 steps 可能不够），但要求**无明显 bug 信号**（reward 发散、advantage 全零、loss NaN）。

### Level 3b — 完整对比实验（后续，非 V0.3 scope）

Tree mode (BGRPO) vs Parallel mode (标准 GRPO) 在相同 LLM 调用预算下的 reward 对比，多 seed。

## 8. 向后兼容

所有新字段都是 Optional 或有默认值：
- `prompt_ids: list[int] | None = None`
- `global_turn_index: int` 需要默认值以兼容旧数据

OrchRL 侧对两个字段均有 fallback：
- prompt_ids 为 None 时回退到重新 tokenize
- global_turn_index 缺失时回退到 timestamp 排序

旧版 MATE 产出的数据仍可被消费。

## 9. 改动边界

| 改动 | MATE | OrchRL |
|------|------|--------|
| datatypes 扩展 | 是 | vendored 同步 |
| monitor 计数器 | 是 | — |
| backend prompt_ids | 是 | — |
| collector 透传 | 是 | — |
| UID 分组修复 | — | 是 |
| skip_turn_predicate | — | 是 |
| prompt_ids 消费 | — | 是 |
| tree_rollout 编排 | **不改** | — |
| ReplayCache | **不改** | — |
| parallel_rollout | **不改** | — |
