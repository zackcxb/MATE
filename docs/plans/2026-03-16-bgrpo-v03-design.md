# V0.3 Design (Reset): Token-Drift First

> Status: **Brainstorming reset**  
> Date: 2026-03-17  
> Scope: **MATE only**（本迭代）

## 1. 迭代重置决策

2026-03-17 决策：将 V0.3 状态回退到 brainstorming 前置阶段，重排优先级。

本迭代只做两件事：

1. **集中解决 MATE 侧 token drifting 风险**（首要目标）
2. **重新评估 `global_turn_index` 的必要性**（先评估、后决策）

明确不做：

1. OrchRL 训练侧改动（由同事负责）
2. BGRPO 训练侧分组策略落地
3. 与 OrchRL 侧实现细节绑定的设计冻结

## 2. 问题定义（本迭代）

### 2.1 Token drifting

当前训练侧经常通过 `messages -> tokenizer.apply_chat_template()` 重建 prompt token。若推理侧与训练侧 tokenizer/config/template 不完全一致，会出现：

1. rollout 实际使用的 prompt token 序列与训练还原序列不一致
2. policy 梯度对应的“输入条件”发生偏移
3. 收敛信号噪声增大且难以定位

### 2.2 `global_turn_index` 必要性（待验证）

我们已有“timestamp 重建 turn 顺序”的机制，但是否必须在 MATE 合同层强制引入 `global_turn_index`，目前不预设结论。本迭代目标是拿到证据后再冻结决策。

## 3. 设计范围（MATE）

### 3.1 必做：prompt_ids 合同链路

在 MATE 内建立可回放的 prompt token 事实链路：

1. `ModelResponse.prompt_ids`（可选）
2. `InteractionRecord.prompt_ids`（可选）
3. `TurnData.prompt_ids`（可选）

原则：

1. 推理侧可获得 tokenizer 时，写入 `prompt_ids`
2. 获取失败时允许 `None`，不阻断主流程
3. 先保证“可记录、可透传、可验证”，不在本迭代耦合 OrchRL 消费实现

### 3.2 暂不冻结：global_turn_index 合同

本迭代只做必要性评估，不默认进入必做实现。

评估输出仅允许两种：

1. **Promote**：证据显示 timestamp 重建在目标场景不可靠，下一迭代将 `global_turn_index` 升级为合同字段
2. **Defer**：证据显示现有机制在目标场景可接受，保留为后续优化项

## 4. prompt_ids 目标链路

```text
VLLMBackend.generate()
  -> ModelResponse.prompt_ids
ModelMonitor
  -> InteractionRecord.prompt_ids
TrajectoryCollector
  -> TurnData.prompt_ids
EpisodeTrajectory / EpisodeResult
  -> artifacts / 序列化产物可见
```

约束：

1. 不改变现有 rollout 编排行为
2. 不修改 reward 语义
3. 仅做数据合同增强与观测能力增强

## 5. global_turn_index 评估设计

### 5.1 评估问题

1. 在 MAS 并发请求下，timestamp 是否会导致可观测的顺序歧义？
2. 歧义是否会影响 replay 分支点选择与轨迹一致性？
3. 该影响是否达到“必须合同化”的阈值？

### 5.2 证据与判据

建议使用可重复实验，输出：

1. 歧义频次：同一 episode 中 timestamp 排序不稳定比例
2. 影响范围：是否改变 replay/branch 行为
3. 工程代价：引入合同字段后的实现与兼容成本

决策阈值（建议）：

1. 若出现可复现且影响分支行为的顺序歧义 -> Promote
2. 若仅有理论风险但无实证影响 -> Defer

## 6. 交付物（本迭代）

1. MATE token-drift 解决方案设计与实现计划（本文件 + impl plan）
2. prompt_ids 合同链路落地（代码由后续实现任务完成）
3. `global_turn_index` 必要性评估结论（Promote 或 Defer）

## 7. 非目标与边界

1. 本文档不定义 OrchRL 训练侧细节
2. 本文档不冻结 BGRPO 分组实现方案
3. 本文档不处理外部 PR 的合入策略与冲突
