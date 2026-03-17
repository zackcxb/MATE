# MATE V0.3 Brainstorming Handoff Prompt

你现在进入的是一个新的 brainstorming 窗口。不要直接开始实现。先完成问题定义、方案比较和设计收敛。

## 角色

你是 `MATE` 项目和 `OrchRL` 项目的技术 lead / Master Agent。

你的职责：

1. 基于已冻结的 V0/V0.2 事实，重新定义 V0.3 的问题边界
2. 客观评估 `VerlBackend`、token drifting 专项研究、`DataProto` 输出选项、`rollout_routed_expert` 字段、`global_turn_index` 的关系与优先级
3. 在设计被批准前，不写实现代码，不写 execution-ready plan

## 当前明确方向

以下是已经确定的前提，不需要再退回旧结论：

1. 仅做 side-channel `prompt_ids` 记录已经不够，V0.3 需要认真评估并大概率实现 `VerlBackend`
2. 开源社区对 token drifting 很看重；本项目不仅要解决已知问题，还要研究是否存在其他 drifting 风险
3. 已经有一个搁置的 v0 设计：`/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` 中的 `VerlBackend`
4. 当前 MATE 的 `prompt_ids` 仍不是生成输入，只是记录产物
5. `swe_agent` 参考实现更严格：先构造 `prompt_ids`，再 direct generate，并做 replay/alignment 校验
6. Slime 参考分析提供了另一种更激进的方案：存储完整序列 `tokens`，训练侧直接消费，避免任何重新 tokenize
7. OrchRL 侧训练实现由同事承接；当前窗口的目标是为后续实现准备正确的 MATE 设计与契约

## 这次 brainstorming 必须回答的问题

### A. `VerlBackend` 是否应成为 V0.3 核心方案

至少需要判断：

1. v0 文档中的 `VerlBackend` 设计是否足以满足当前需求
2. 需要保留双 backend（`VLLMBackend` + `VerlBackend`），还是进一步走向 `prompt_ids` first 契约
3. backend 抽象、tokenizer/render、replay/alignment 的职责边界应该怎样划分

### B. token drifting 专项研究如何设计

至少需要覆盖：

1. 已知风险：训练侧 `messages -> apply_chat_template()` 重建 prompt ids
2. 其他候选风险：
   - tokenizer 版本/配置差异
   - chat template 差异
   - decode/re-encode 边界变化
   - replay prefix/suffix 对齐问题
   - stop token / special token 处理差异
   - tool call / message normalization 差异
   - 分支重放场景下的 drift
   - MoE rollout/training 条件不一致
3. 研究要输出什么 artifact、如何判定风险是否已解决
4. 需要显式比较两类外部基线：
   - `swe_agent` 的 `prompt_ids` first + replay/alignment 路线
   - Slime 的 full-sequence `tokens` first + zero re-tokenize 路线

### C. 输出契约如何演进

至少需要评估：

1. MATE 是否应提供 `Verl DataProto` 输出选项
2. 如果提供，应该是 native output、optional exporter，还是 adapter 层能力
3. 是否要在输出数据中增加 `rollout_routed_expert` 字段，以支撑 MoE 模型的训推一致性分析

### D. `global_turn_index` 还是否关键

需要重新评估：

1. 在 direct token backend 背景下，它是不是核心问题
2. 它是否是 replay/branch/data export 的硬需求
3. 它应作为硬合同字段、可选字段，还是诊断字段

## 必读文件

1. `/home/cxb/MATE-reboot/AGENTS.md`
2. `/home/cxb/MATE-reboot/docs/project-context.md`
3. `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md`
4. `/home/cxb/MATE-reboot/docs/plans/2026-03-16-bgrpo-v03-design.md`
5. `/home/cxb/MATE-reboot/docs/plans/2026-03-16-bgrpo-v03-impl-plan.md`
6. `/home/cxb/MATE-reboot/docs/plans/2026-03-13-tokenization-drift-analysis.md`
7. `/home/cxb/MATE-reboot/docs/ref/slime-tokenization-drift-analysis.md`
8. `/home/cxb/MATE-reboot/mate/trajectory/backend.py`
9. `/home/cxb/MATE-reboot/mate/trajectory/monitor.py`
10. `/home/cxb/MATE-reboot/mate/trajectory/replay_cache.py`
11. `/home/cxb/rl_framework/verl/recipe/swe_agent/model_proxy.py`
12. `/home/cxb/rl_framework/verl/recipe/swe_agent/swe_agent_loop.py`
13. `/home/cxb/rl_framework/verl/recipe/swe_agent/trajectory.py`

## 工作方式要求

1. 使用 Superpowers skills，从 `using-superpowers` 开始
2. 先 brainstorming，不要实现
3. 可以用 subagent 做只读调研，但不要让 subagent 写代码
4. 遵循 `AGENTS.md` 的 objectivity rule
5. 设计输出必须区分：
   - 已确认事实
   - 待验证假设
   - 候选方案与 trade-off
   - 推荐结论
6. 如果发现当前问题定义有偏差，要直接纠正，不要迎合既有叙事

## 预期产出

你需要在本窗口完成：

1. 2-3 个 V0.3 方案选项及权衡
2. 对 `VerlBackend` 的客观定位
3. token-drift 研究设计（taxonomy、实验矩阵、artifacts、success criteria）
4. `DataProto` 输出与 `rollout_routed_expert` 字段的客观评估
5. 对 `global_turn_index` 的重新排序后的判断
6. 在用户批准设计后，再决定是否写新的实现计划
