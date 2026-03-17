# V0.3 Design Reset: VerlBackend + Token-Drift Research

> Status: **Brainstorming handoff**
> Date: 2026-03-17  
> Scope: **MATE first, OrchRL integration deferred to follow-up coordination**

## 1. Iteration Reset

2026-03-17 的新决策是：V0.3 不再以“仅补 side-channel `prompt_ids` 合同”为目标，而是提升为一个更严格的架构与研究议题。

本轮 brainstorming 需要围绕以下主线展开：

1. **在 MATE 中恢复并落地 `VerlBackend` 方案**，使生成链路具备真正的 token-in/token-out 能力
2. **把 token drifting 升级为专项研究问题**，不仅解决当前已知风险，还要系统排查其他 drifting 风险
3. **评估输出契约是否要向 `DataProto` 靠拢**，以及是否要为 MoE 模型增加 `rollout_routed_expert` 字段
4. **重新评估 `global_turn_index` 的必要性**，但它不再是唯一核心问题

这意味着 V0.3 当前处于“重新定义问题与方案边界”的状态，而不是执行态实现计划。

## 2. Current Facts

### 2.1 已经确认的事实

1. 当前 MATE 主路径仍是 OpenAI-style `messages -> /v1/chat/completions`，并非 direct `prompt_ids` generate。
2. 已落地的 `prompt_ids` 只是采集侧 side-channel 记录，不是实际生成输入。
3. `swe_agent` 参考实现采用 `prompt_ids` 作为真实生成输入，并包含严格 replay/alignment 检查。
4. 2026-03-04 的 V0 设计文档中已经提出 `VerlBackend` 概念：在 monitor backend 内调用 `AsyncLLMServerManager.generate(prompt_ids=...)`。
5. OrchRL 侧实现与训练流程改造本轮不在本仓直接落地，但 MATE 需要给出稳定契约与研究结论，供后续对接。

### 2.2 当前 gap

当前实现虽然能记录 `prompt_ids`，但仍存在以下不足：

1. 生成时真正送入模型的 prompt token 与采集时记录的 prompt token 不是同一份契约对象。
2. 缺乏对 prompt replay、assistant replay、template trailing token、tool message 归一化等风险的系统验证。
3. 输出契约仍偏向 MATE 自有 dataclass，尚未评估直接输出 `Verl DataProto` 的可行性与收益。
4. 尚未为 MoE 训推一致性预留 rollout 侧 expert routing 观测字段。

## 3. Problem Statement

V0.3 要回答的不是“是否记录 `prompt_ids`”这么窄的问题，而是以下四个更本质的问题。

### 3.1 Prompt Contract Problem

MATE 是否应该把 `prompt_ids` 提升为一等输入/输出契约，而不是继续依赖 `messages` 文本协议与事后重渲染。

### 3.2 Token Drift Problem

在多智能体 rollout 场景里，除了已知的“训练侧重渲染 prompt token”之外，是否还存在以下 drifting 风险：

1. chat template 差异
2. tokenizer 版本/配置差异
3. stop token / special token 处理差异
4. decode -> re-encode 引入的边界变化
5. replay 场景中 prompt prefix 与 assistant suffix 对齐漂移
6. tool call / multimessage 归一化造成的渲染差异
7. MoE 路由信息缺失导致的 rollout/training 条件不一致

### 3.3 Output Contract Problem

MATE 最终对外输出是否应继续以自有 dataclass 为主，还是应该评估提供 `Verl DataProto` 输出路径，从而减少 OrchRL 侧 adapter 成本与契约漂移。

### 3.4 Turn Ordering Problem

`global_turn_index` 是否仍是必要合同字段，需要结合 direct token backend、replay 语义与下游消费方式重新判断，而不是孤立讨论。

## 4. Candidate Directions

### 4.1 Direction A: 保守增强当前 VLLMBackend 路线

做法：继续保留 OpenAI chat completion 主路径，只增强 `prompt_ids` 记录、诊断与校验。

优点：

1. 改动最小
2. 对现有调用方兼容性最好
3. 便于快速产出 token-drift 证据

缺点：

1. 不能从根上消除“生成输入”和“记录输入”分离的问题
2. 很难证明不存在 drift，只能证明“当前样本下未观察到明显 drift”
3. 与 `swe_agent`/Verl 的 token-in 主路径仍有结构性差距

### 4.2 Direction B: 在 MATE 中恢复 `VerlBackend`，形成双 backend 架构

做法：保留 `VLLMBackend` 作为 OpenAI-style/测试路径，同时新增或恢复 `VerlBackend`，让训练/研究场景走 `prompt_ids -> AsyncLLMServerManager.generate()`。

优点：

1. 能把 token-in/token-out 变成真实运行路径，而非事后采样推断
2. 与 `swe_agent` 和 v0 设计思路一致
3. 保留现有 monitor/backend 抽象，不需要回退整体架构
4. 便于做 A/B 研究：比较 side-channel 路径与 direct token 路径的 drift 表现

缺点：

1. 需要明确 backend 选择、tokenizer 责任边界和错误语义
2. 需要补 replay/alignment 机制，否则 direct token 路径仍不够严谨
3. 会把输出契约和下游消费问题一并暴露出来

### 4.3 Direction C: 直接重构为 `prompt_ids` first 的统一 backend/contract

做法：把 `ModelRequest` 主契约改为可直接携带 `prompt_ids`，`messages` 退为高层调试/回放辅助信息。

优点：

1. 从契约层面最干净
2. 能系统承接 `DataProto`、alignment、MoE metadata 等扩展需求
3. 长期最接近训练框架真实需要

缺点：

1. 这不是局部增强，而是一次接口级重构
2. 需要更强的兼容层设计
3. 超出本轮“先完成 brainstorming 与范围收敛”的合理复杂度

### 4.4 Recommendation

本轮 brainstorming 的建议默认方向是 **Direction B**。

理由：

1. 它满足当前“必须实现 `VerlBackend`”的需求。
2. 它在架构上延续了 V0 已经验证过的 backend abstraction，不必为了研究问题重做整个 monitor 层。
3. 它允许把 `DataProto` 输出、`rollout_routed_expert` 字段、`global_turn_index`、strict replay 等议题拆成独立设计项，而不是一次性绑死在大重构里。
4. 它提供了最有价值的研究对照面：同一个 MATE 外壳下，比较 `VLLMBackend` 和 `VerlBackend` 的 drift 风险与契约稳定性。

## 5. Proposed Brainstorming Scope

### 5.1 Architecture Track

需要冻结以下问题：

1. `VerlBackend` 的输入契约是 `messages` 渲染后生成 `prompt_ids`，还是显式接收 `prompt_ids`，或同时支持两者
2. tokenizer/chat template 的所有权属于 MATE backend、外部调用方，还是共享 renderer 组件
3. strict replay/alignment 校验要放在 backend 层、monitor 层，还是独立 validator 组件
4. `VLLMBackend` 与 `VerlBackend` 的行为差异是否需要统一成可比较的观测指标

### 5.2 Research Track

需要设计一个专门的 token-drift 研究任务，至少覆盖：

1. drift taxonomy：列出所有怀疑路径
2. instrumentation：记录何处使用了哪份 token 序列
3. comparison matrix：`stored_prompt_ids`、runtime prompt ids、re-rendered ids、replay ids 的逐项比对
4. diagnostics artifacts：可供复现和审阅的 JSON/报告产物
5. acceptance threshold：怎样才算“问题已解决”或“风险可接受”

### 5.3 Output Contract Track

需要评估以下两个扩展是否值得并入 V0.3：

1. **DataProto 输出选项**
   - 目标：减少 OrchRL adapter/重打包逻辑
   - 关注点：MATE 是否应直接依赖 verl 数据结构，还是提供可选 adapter/exporter
2. **`rollout_routed_expert` 字段**
   - 目标：为 MoE 模型提供 rollout 时实际 expert routing 观测，缓解训推不一致
   - 关注点：字段来源、粒度、存储成本、与训练侧消费方的耦合边界

### 5.4 Secondary Question: `global_turn_index`

本轮仍要评估 `global_turn_index`，但应放在更完整的上下文中：

1. 对 direct token backend 是否仍然必要
2. 对 replay / branch 对齐是否带来实质收益
3. 是否能通过更强的记录与 validator 替代显式合同字段

## 6. Non-Goals For This Handoff

本次 handoff 不要求在当前窗口完成以下事项：

1. 直接实现 `VerlBackend`
2. 直接改 OrchRL trainer 或 adapter
3. 直接冻结 `DataProto`/MoE 字段最终 schema
4. 直接承诺 `global_turn_index` Promote 或 Defer

当前窗口的职责是：更新文档、把问题定义重置正确，并为新的 brainstorming 会话准备高质量启动上下文。

## 7. Expected Outputs Of The Next Brainstorming Session

新的 Agent 窗口需要产出：

1. 一份新的 V0.3 设计结论，明确 `VerlBackend` 在当前阶段的角色和边界
2. 一份 token-drift 研究设计，包含 taxonomy、实验矩阵、证据产物和判定标准
3. 对 `DataProto` 输出选项的客观评估
4. 对 `rollout_routed_expert` 字段的必要性、来源与边界评估
5. 对 `global_turn_index` 的重新排序后的客观判断
6. 在设计批准后，再写新的实现计划
