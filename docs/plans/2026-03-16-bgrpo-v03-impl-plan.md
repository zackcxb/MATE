# V0.3 Implementation Plan (Reset): MATE Token-Drift First

> Status: **Brainstorming reset -> execution-ready plan**  
> Date: 2026-03-17  
> Scope: **MATE only**

## 1. 目标与边界

### 1.1 本迭代目标

1. 在 MATE 侧落地 `prompt_ids` 数据合同，降低 token drifting 风险。
2. 产出 `global_turn_index` 必要性的证据化结论（Promote/Defer）。

### 1.2 本迭代不做

1. OrchRL 侧实现与配置改动（由同事负责）。
2. BGRPO 分组逻辑改造。
3. 跨仓联动重构。

## 2. 执行策略

按“先合同、再验证、后决策”的顺序执行。

1. **Phase A**：`prompt_ids` 合同链路实现（MATE 内闭环）。
2. **Phase B**：token-drift 验证与可观测性增强。
3. **Phase C**：`global_turn_index` 必要性评估并给出结论。

## 3. Phase A — prompt_ids 合同链路

### A1. 数据结构扩展

改动文件：

1. `mate/trajectory/datatypes.py`

任务：

- [ ] `ModelResponse` 增加 `prompt_ids: list[int] | None = None`
- [ ] `InteractionRecord` 增加 `prompt_ids: list[int] | None = None`
- [ ] `TurnData` 增加 `prompt_ids: list[int] | None = None`

验收：

- [ ] 相关类型构造与序列化路径测试通过

### A2. backend 生成 prompt_ids

改动文件：

1. `mate/trajectory/backend.py`

任务：

- [ ] tokenizer 可用时，基于与推理一致的 chat template 生成 `prompt_ids`
- [ ] tokenizer 不可用时返回 `None`
- [ ] 不改变现有请求 payload 与生成路径

验收：

- [ ] 有/无 tokenizer 两条路径均有单测

### A3. monitor / collector 透传

改动文件：

1. `mate/trajectory/monitor.py`
2. `mate/trajectory/collector.py`

任务：

- [ ] Monitor 将 `ModelResponse.prompt_ids` 写入 `InteractionRecord`
- [ ] Collector 将 `InteractionRecord.prompt_ids` 写入 `TurnData`

验收：

- [ ] 端到端单测可在最终 `TurnData` 看到 `prompt_ids`

## 4. Phase B — token-drift 验证

### B1. 增加一致性对照检查（MATE 侧）

目标：提供可审计证据，而非仅靠主观判断。

任务：

- [ ] 在测试/诊断脚本中加入抽样对照：`stored_prompt_ids` vs `re-tokenized_prompt_ids`
- [ ] 记录 mismatch rate 与典型样本

产物建议：

1. `artifacts/*token_drift*.json`
2. 回归日志中的 mismatch 摘要

### B2. 回归验证

命令基线：

```bash
cd /home/cxb/MATE-reboot
python -m pytest tests/trajectory tests/scripts -q
```

验收：

- [ ] 全量通过
- [ ] token-drift 对照产物生成成功

## 5. Phase C — global_turn_index 必要性评估

### C1. 评估实验

任务：

- [ ] 构造并发 turn 采样场景，统计 timestamp 排序歧义是否可复现
- [ ] 判断歧义是否实际影响 replay/branch 语义

### C2. 决策输出

必须输出二选一结论：

1. **Promote**：下一迭代将 `global_turn_index` 升级为合同字段并实施
2. **Defer**：当前保持 timestamp 方案，记录风险和触发条件

交付位置：

1. `docs/plans/2026-03-16-bgrpo-v03-design.md`（结论回填）
2. `docs/project-context.md`（阶段状态更新）

## 6. 测试清单（本迭代）

### 必测

1. datatypes 新字段兼容性测试
2. backend prompt_ids 生成测试（有/无 tokenizer）
3. monitor/collector prompt_ids 透传测试
4. token-drift 对照诊断测试

### 可选增强

1. 真实环境 smoke 抽样验证 prompt_ids 完整率

## 7. 风险与缓解

1. **tokenizer 可用性不稳定**  
   缓解：允许 `prompt_ids=None` fallback，不阻断主链路。

2. **本地重算仍可能与推理端微差异**  
   缓解：明确对照统计，不把“理论一致”当成结论。

3. **global_turn_index 评估结论拖延**  
   缓解：将 C2 作为本迭代强制交付项，必须给 Promote/Defer。

## 8. 里程碑

### M1: prompt_ids 合同落地

- datatypes/backend/monitor/collector 完成
- 对应单测通过

### M2: token-drift 证据产出

- mismatch 统计与样例产物齐全
- 回归通过

### M3: global_turn_index 决策闭环

- 输出 Promote 或 Defer
- 文档状态更新完成
