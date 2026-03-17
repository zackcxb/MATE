# V0.3 Pre-Plan Reset: Brainstorming Agenda For VerlBackend And Drift Research

> Status: **Pre-plan only; do not execute as implementation checklist yet**
> Date: 2026-03-17  
> Scope: **Prepare next-window brainstorming and decision checkpoints**

## 1. Why This Is A Pre-Plan

当前 V0.3 还没有经过新一轮设计确认，因此本文件不再把自己表述为“execution-ready implementation plan”。

本文件的作用是：

1. 明确下一窗口 brainstorming 应该完成哪些决策
2. 把后续可能进入实现的工作拆成可讨论的工作包
3. 避免在问题尚未定义清楚时直接进入编码

本文件默认承接 `docs/plans/2026-03-16-bgrpo-v03-design.md` 的 handoff 设计，因此这里不再重复展开问题定义与方案背景，只保留决策 gate、工作包和退出条件。

## 2. Decision Gates

在进入实现前，必须完成以下 gate：

### Gate 1: `VerlBackend` Positioning

必须明确：

1. `VerlBackend` 是新增 backend、恢复旧设计，还是升级为主契约路径
2. 它与 `VLLMBackend` 的职责分工是什么
3. `messages` 与 `prompt_ids` 的输入边界是什么

### Gate 2: Token-Drift Research Spec

必须明确：

1. 要研究哪些 drift 风险
2. 需要记录哪些 runtime artifacts
3. 成功标准是什么
4. 哪些风险属于必须修复，哪些仅记录为限制

### Gate 3: Output Contract Choice

必须明确：

1. 是否提供 `DataProto` 输出选项
2. 若提供，是 primary output、optional exporter，还是 sidecar adapter
3. `rollout_routed_expert` 是否纳入 V0.3 合同

### Gate 4: Turn Ordering Decision

必须明确：

1. `global_turn_index` 是核心合同字段还是次级诊断字段
2. 它与 replay/alignment/branch 语义的真实依赖关系

## 3. Recommended Brainstorming Work Packages

### WP1. Establish The Architecture Baseline

输入材料：

1. `docs/plans/2026-03-04-trajectory-engine-v0-design.md`
2. `mate/trajectory/backend.py`
3. `mate/trajectory/monitor.py`
4. `/home/cxb/rl_framework/verl/recipe/swe_agent/model_proxy.py`
5. `/home/cxb/rl_framework/verl/recipe/swe_agent/swe_agent_loop.py`
6. `/home/cxb/rl_framework/verl/recipe/swe_agent/trajectory.py`
7. `docs/ref/slime-tokenization-drift-analysis.md`

需要回答：

1. 现有 backend abstraction 是否足以承载 direct token backend
2. replay/alignment 能力缺哪些最小部件
3. 哪些逻辑应抽为 shared renderer/validator

产出：

1. `VerlBackend` 候选结构图
2. 与当前 `VLLMBackend` 的差异表
3. 必做/可延后项清单

### WP2. Define The Token-Drift Research Program

需要回答：

1. 除 prompt re-render 外，还要研究哪些 drift source
2. 如何构造可重复实验与对照组
3. 如何让研究结果直接服务设计决策，而不是停留在现象收集

建议对照组：

1. 当前 side-channel `prompt_ids` 采集路径
2. 未来 `VerlBackend` direct token 路径
3. replay 场景
4. 多 agent / tool call / branch 场景
5. 不同 tokenizer/template/version 组合
6. Slime 的 full-sequence `tokens` 合同路径

产出：

1. drift taxonomy
2. experiment matrix
3. diagnostics artifact schema
4. success/failure criteria

### WP3. Evaluate Output Contracts

需要回答：

1. MATE 输出自有 dataclass 的优势是否仍成立
2. `DataProto` 输出是否值得作为可选项进入 V0.3
3. `rollout_routed_expert` 字段是否应在 V0.3 就进入合同

产出：

1. 契约选项对比表
2. 依赖/耦合成本分析
3. 推荐结论

### WP4. Reassess `global_turn_index`

需要回答：

1. 它在 direct token backend 设计里扮演什么角色
2. 没有它时，是否仍能稳定完成 replay / ordering / branch consumer 需求
3. 如果保留，应该是硬合同还是诊断字段

产出：

1. Promote / Defer / Optional 三选一建议
2. 依赖该字段的下游场景列表

## 4. Candidate Implementation Packages After Design Approval

只有在新设计批准后，才允许进入以下实现包拆分。

### Package A. `VerlBackend` Runtime Path

可能涉及：

1. backend interface 扩展
2. tokenizer/render 责任边界重构
3. `AsyncLLMServerManager.generate(prompt_ids=...)` 接入
4. direct token response capture

### Package B. Alignment And Replay Validation

可能涉及：

1. prompt replay validation
2. assistant replay validation
3. prefix delta / trailing template token 处理
4. failure reason vocabulary

### Package C. Drift Diagnostics

可能涉及：

1. drift artifact 输出
2. mismatch 汇总
3. regression test / smoke test / research script

### Package D. Output Contract Extensions

可能涉及：

1. `DataProto` exporter 或 native output path
2. `rollout_routed_expert` metadata plumbing
3. schema / compatibility tests

### Package E. Turn Ordering Contract

可能涉及：

1. `global_turn_index` 字段设计
2. backward compatibility
3. ordering validator/tests

## 5. Constraints For The Next Agent Window

1. 先完成 design review 和 option comparison，不要直接写实现。
2. 对 `DataProto` 和 `rollout_routed_expert` 只做客观评估，不要因为“以后可能有用”就默认纳入。
3. 如果 `VerlBackend` 方案与当前 backend abstraction 有冲突，优先保留抽象边界的清晰性，而不是局部拼补。
4. 如果 token-drift 研究表明风险主要来自非 prompt 环节，必须如实调整问题定义，不能强行把结论收束到 `prompt_ids`。

## 6. Exit Criteria For Brainstorming

新窗口 brainstorming 只有在以下结果齐备时才算结束：

1. 设计方向已在 A/B/C 候选方案中做出明确推荐
2. token-drift 研究范围、证据形式和 success criteria 已冻结
3. `DataProto` 与 `rollout_routed_expert` 的处理建议已明确
4. `global_turn_index` 的优先级与地位已明确
5. 新的实现计划值得编写
