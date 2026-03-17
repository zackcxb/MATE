# MATE-reboot 项目上下文

> 最后更新：2026-03-17（V0.3 进入新一轮 brainstorming handoff：VerlBackend + token-drift research）

## 项目定位

MATE-reboot 是多智能体轨迹采集引擎（Agent Trajectory Engine）的开发仓库。最终集成目标是 OrchRL 框架（`/home/cxb/OrchRL`）。

## 当前阶段

**V0.2 已收口；V0.3 已再次重置为 brainstorming handoff。** 当前阶段不再把问题定义为“仅做 MATE 侧 `prompt_ids` side-channel 合同”，而是升级为更严格的架构与研究议题：恢复/实现 `VerlBackend`、系统研究 token drifting 风险、并评估输出契约向 `DataProto` 和 MoE rollout metadata 扩展的必要性。

- 当前状态：
  - MATE-reboot 采集侧已具备 `tree_rollout`、`ReplayCache`、V0.2 验证脚本和可视化支持
  - OrchRL 训练侧当前仍通过 adapter + `VLLMBackend` 集成 MATE，不使用 `AsyncLLMServerManager` 直连路径
  - OrchRL 本地 `main` 已包含 tree smoke 收口提交：`1c21f20`（tree rollout smoke 支持）和 `f3d9a48`（example 收拢）
  - OrchRL 真实 Search MAS tree smoke 已再次验证 clean exit `0`，证据为 `step 0 started`、`training/global_step:0.000`、`step 1 started`、`Cleanup completed`
  - OrchRL trainer 已接受 “partial policy batches 是合法轨迹结果” 这一现实语义；当前策略是仅更新当步有 batch 的 policy，并记录 skipped policy metrics
  - OrchRL vendored `trajectory/` 与 MATE V0.2 tree 语义保持一致；当前唯一代码差异是 `trajectory/replay_cache.py` 为 vendoring 所需的相对导入改写，不构成语义分叉
  - 截至 2026-03-15，未发现要求 MATE 再修改 `tree_rollout`、`TreeEpisodeResult`、`ReplayCache` 或核心采集逻辑的 blocker 级新证据
  - 当前已落地 `prompt_ids` 合同链路用于采集与观测，但它仍是 side-channel 记录，不是 direct token generation 路径
  - 当前已确认 `/home/cxb/rl_framework/verl/recipe/swe_agent` 的方案在 prompt 处理上更严格：先生成 `prompt_ids`，再直接 `generate(prompt_ids=...)`，并做 replay/alignment 校验
  - 当前 V0.3 的首要任务是开启新的 brainstorming，客观决定 `VerlBackend`、token-drift 研究、`DataProto` 输出选项和 `rollout_routed_expert` 字段的范围与优先级
- V0 真实环境验证（2026-03-09，real 模式）：
  - 记录：`docs/retros/2026-03-09-trajectory-engine-real-validation.md`
  - 环境：vLLM（`http://127.0.0.1:8000`）+ OrchRL Search MAS + 检索服务（`http://127.0.0.1:18080/retrieve`）
  - 在线模型：`/data1/models/Qwen/Qwen3-4B-Instruct-2507`
  - GPU 约束：`CUDA_VISIBLE_DEVICES=0,1,2,3`（规避 GPU5 故障）
  - smoke 产物：`artifacts/trajectory_validation_real_smoke_fixed.json`
  - exact-match 产物：`artifacts/trajectory_validation_exact_match.json`
  - 多样本产物：`artifacts/trajectory_validation_real_fixed.json`
  - 已确认：
    - 成功 episode 上 `token_ids` 不为 None
    - 成功 episode 上 `logprobs` 与 `token_ids` 长度一致
    - 成功 episode 的 `episode_id` 全局唯一
    - 成功 episode 具备 `verifier/searcher/answerer` 完整 turn 数据
    - reward 解析 bug 已修复，受控 exact-match 样本可稳定得到 `final_reward=1.0`
  - 已定位限制：
    - prompt `when is the next deadpool movie being released?` 在 `max_turns=4` 时稳定触发 vLLM `400 -> 502`，导致 MAS exit code 1
    - 该问题在 `max_turns<=3` 时不复现，属于长上下文失败，不是随机环境抖动

## 团队分工

| 成员 | 负责模块 | 仓库 |
|------|---------|------|
| 本仓库开发者 | 轨迹采集模块（AgentPipe） | MATE-reboot |
| 同事 | 训练侧调度 + 主入口（Verl main PPO 多 agent 扩展） | OrchRL |

## 关键设计决策

### V0 决策（已冻结）

| 决策 | 结论 | 文档 |
|------|------|------|
| Monitor 推理模式 | 策略模式（InferenceBackend 接口） | `docs/plans/2026-03-04-trajectory-engine-v0-design.md` |
| Agent 身份识别 | `model` 字段即 agent role | 同上 |
| MAS 启动模式 | 进程模式（subprocess） | 同上 |
| 轨迹输出粒度 | `Dict[agent_id, List[TurnData]]` | 同上 |
| V0 验证场景 | DrMAS Search（OrchRL 抽取版） | 同上 |

### V0.2 决策（2026-03-09 brainstorming 确认）

| 决策 | 结论 | 文档 |
|------|------|------|
| 开发仓库 | 继续在 MATE-reboot，里程碑后同步到 OrchRL | `docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md` |
| 树状分支实现 | 重放式分支（pilot run + ReplayCache + branch runs），MAS 零侵入 | 同上 |
| 架构简化 | 无 BranchCoordinator/BranchStrategy 策略模式，直接用 `tree_rollout` 函数 | 设计审核修正（YAGNI） |
| 接口演进 | 扩展模式——新增 `tree_rollout`，保留 `parallel_rollout` 不变 | 同上 |
| 训练侧消费 | OrchRL adapter 层（`mate_rollout_adapter` + `mate_dataproto_adapter`），PYTHONPATH 导入 | OrchRL `orchrl/trainer/` |

### V0.3 当前开放问题（待 brainstorming）

| 议题 | 当前状态 |
|------|------|
| `VerlBackend` | 需要重新评估并大概率恢复为当前阶段核心候选方案 |
| token drifting | 已确认需要专项研究，不仅限于 prompt re-render 风险 |
| `DataProto` 输出 | 作为候选输出契约，需要客观评估收益与耦合成本 |
| `rollout_routed_expert` | 作为 MoE 一致性候选字段，需要评估来源、粒度与必要性 |
| `global_turn_index` | 仍待评估，但优先级下调为完整设计中的一个子问题 |

## 依赖关系

| 依赖 | 路径 | 用途 |
|------|------|------|
| OrchRL 主仓（含训练代码） | `/home/cxb/OrchRL/` | 训练侧适配层（`orchrl/trainer/mate_*`）+ trajectory 副本 |
| OrchRL Search MAS | `/home/cxb/OrchRL/examples/mas_app/search/` | V0/V0.2 端到端验证的 MAS 应用 |
| Verl 参考实现 | `/home/cxb/rl_framework/verl/` | `swe_agent` 的 direct token generation、proxy、alignment 参考 |
| Slime 参考分析 | `docs/ref/slime-tokenization-drift-analysis.md` | full-sequence tokens 合同、零 re-tokenize 训练路径参考 |
| Verl 子模块 | `third_party/verl`（git submodule） | RL 框架基线 |
| 团队架构 PPT | `/home/cxb/multi-agent/docs/multi-agent-rl.pdf` | MARL/BGRPO 相关算法参考 |

### V0.2 技术参考

| 参考 | 来源 | 层级 | 用途 |
|------|------|------|------|
| AREAL-DTA | [arXiv:2602.00482](https://arxiv.org/abs/2602.00482) | 训练侧 | DFS 单路径物化前缀树（8.31x 训练加速） |
| Forge Prefix Tree Merging | MiniMax | 训练侧 | 全量物化 + Magi Attention（40x） |
| Tree Training | [arXiv:2511.00413](https://arxiv.org/abs/2511.00413) | 训练侧 | Gradient Restoration + Tree Packing（6.2x） |
| ROME/IPA rollback | [arXiv:2512.24873](https://arxiv.org/abs/2512.24873) | 采集侧参考 | 选择性分支策略（V0.2 后优化方向） |

## 文档索引

### V0

| 文档 | 类型 | 说明 |
|------|------|------|
| `docs/plans/2026-03-02-marl-grpo-v0-directions.md` | 方向设计 | V0 方向评估 |
| `docs/plans/2026-03-04-trajectory-engine-v0-design.md` | 架构设计 | V0 详细设计（已冻结，含早期 `VerlBackend` 设想） |
| `docs/plans/2026-03-04-trajectory-engine-v0-impl-plan.md` | 实施计划 | V0 实施细节 |
| `docs/plans/2026-03-05-training-integration-spec.md` | 对接规格（历史） | V0 阶段直连 `VerlBackend` 设想；当前 OrchRL 实际路径是 adapter + `VLLMBackend` |
| `docs/retros/2026-03-09-trajectory-engine-real-validation.md` | 验证记录 | 真实环境验证证据、异常样本与根因分析 |

### V0.2

| 文档 | 类型 | 说明 |
|------|------|------|
| `docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md` | 设计方向 | V0.2 核心设计、技术参考、职责边界 |
| `docs/plans/2026-03-09-trajectory-engine-v02-impl-plan.md` | 实施计划 | 9 个 Task、TDD 步骤、依赖关系 |
| `docs/plans/2026-03-11-trajectory-public-api-boundary.md` | API 边界 | V0.2 阶段稳定外部 API 与内部/暂不承诺接口划分 |
| `docs/plans/2026-03-13-tokenization-drift-analysis.md` | 技术分析 | 推理侧与训练侧 tokenization drift 风险分析 |
| `docs/ref/slime-tokenization-drift-analysis.md` | 参考分析 | Slime 如何通过完整序列 tokens 合同避免 tokenization drift |
| `docs/retros/2026-03-11-orchrl-tree-adapter-sync.md` | 联调进展 | OrchRL tree adapter 接入状态、本地测试结果、smoke 阻塞点 |
| `docs/retros/2026-03-14-orchrl-tree-smoke-server-sync.md` | 联调进展 | 新服务器真实 smoke clean exit 证据、partial policy batch 结论 |
| `docs/retros/2026-03-15-orchrl-tree-smoke-upstreaming-wrapup.md` | 收口记录 | OrchRL `main` 上库形态、保留项与 MATE 审查重点 |
| `docs/prompts/2026-03-09-v02-brainstorming.md` | 会话 prompt | V0.2 brainstorming 启动模板 |
| `docs/prompts/2026-03-09-v02-implementation.md` | 会话 prompt | V0.2 实施会话启动模板 |
| `docs/prompts/2026-03-09-v02-master-agent.md` | 会话 prompt | V0.2 Master Agent 启动模板（统筹+验证） |
| `docs/prompts/2026-03-14-mate-tree-smoke-handoff.md` | 会话 prompt | 新服务器 MATE Agent 交接 prompt（真实 tree smoke 支持） |
| `docs/prompts/2026-03-15-v03-brainstorming-handoff.md` | 会话 prompt | V0.3 brainstorming 启动模板（VerlBackend + token-drift research） |
| `scripts/USAGE.md` | 用法文档 | 三个验证脚本的参数、输出格式、工作流 |

## 待办

### V0（已完成）

- [x] 制定实施计划
- [x] 实现 AgentPipe 核心模块
- [x] DrMAS Search 端到端验证
- [x] vLLM token_ids 提取链路验证（Monitor/Backend/序列化）
- [x] Episode 并行采样（并发 rollout 编排与稳定性验证）
- [x] 真实环境验证脚本与可视化工具落地
- [x] 编写训练侧对接规格文档
- [x] Real vLLM + 检索服务端到端验证（2026-03-09）
- [x] 与训练侧进行联调（训练侧已通过 adapter 方式完成端到端联调）
- [x] 整理代码向 OrchRL 仓库提交 PR（已合入 OrchRL main `1664eb5`）

### V0.2（实现完成 + 收口完成）

- [x] V0.2 设计：确定开发仓库（MATE-reboot）、特性范围、技术参考调研
- [x] V0.2 设计审核（APPROVE WITH CHANGES，已融入实施计划）
- [x] V0.2 实施计划编写（9 个 Task）
- [x] Task 1: 数据结构（BranchResult, TreeEpisodeResult, EpisodeResult.status）
- [x] Task 2: ReplayCache
- [x] Task 3: Monitor replay 支持
- [x] Task 4: AgentPipe 优雅降级
- [x] Task 5: tree_rollout 函数（核心）
- [x] Task 6: Package 导出 + 版本更新（0.1.0 → 0.2.0）
- [x] Task 7: 集成测试
- [x] Task 8: 端到端验证脚本 `scripts/run_tree_validation.py`（三轮递进验证 + 对比模式）
- [x] Task 9: 轨迹可视化支持 V0.2 schema（`scripts/visualize_trajectories.py` 更新）
- [x] 在新服务器完成 OrchRL tree 模式真实 smoke，并确认当前没有新的 MATE blocker 级修补需求

### V0.3（brainstorming handoff）

- [ ] 在新窗口完成 `VerlBackend` 方案 brainstorming，并明确与 `VLLMBackend` 的职责边界
- [ ] 制定 token-drift 研究设计：taxonomy、实验矩阵、artifact schema、success criteria
- [ ] 客观评估 `DataProto` 输出选项的价值与耦合代价
- [ ] 客观评估 `rollout_routed_expert` 字段的必要性、来源和粒度
- [ ] 重新评估 `global_turn_index` 的角色与优先级
- [ ] 在设计批准后，再编写新的实现计划

#### V0.2 真实环境验证（2026-03-11，real 模式）

- 环境：vLLM（`http://127.0.0.1:8000`）+ OrchRL Search MAS + 检索服务（`http://127.0.0.1:8010/retrieve`）
- 在线模型：`/data1/models/Qwen/Qwen3-4B-Instruct-2507`（Qwen3-4B）
- 验证脚本：`scripts/run_tree_validation.py`（三轮递进：Smoke → Replay → Multi-prompt）
- 产物：`artifacts/tree_validation_diag.json` + `artifacts/tree_validation_diag_trajectories.json`（1.5MB）

**验证结果：**

| 轮次 | 结果 | 说明 |
|------|------|------|
| Round 1 (Smoke) | PASS | 树结构完整：pilot + 12 branches，所有角色齐全 |
| Round 2 (Replay) | PASS | 220/220 replay 标记全部正确 |
| Round 3 (Multi-prompt) | 1/2 PASS | deadpool prompt 失败（V0 已知 long-context 问题，非回归） |

**Trajectory 合理性（238 turns, 39 episodes）：**

| 指标 | 值 | 判定 |
|------|------|------|
| token_ids 完整性 | 0/238 缺失 | PASS |
| token/logprobs 一致性 | 0 不匹配 | PASS |
| Replay 标记正确性 | 220/220 正确 | PASS |
| Reward 多样性 | 均值 0.385, σ=0.493, 38.5% success | 有多样性 |
| 前缀共享率 | ~25%（turn 层面复用 40.9%） | 有效 |
| 分支多样性 | 1-2/6 分支点产生不同响应 | 合理（temperature 控制） |

**发现的环境兼容问题（已修复）：**

1. OrchRL `agent.py:29` f-string 含反斜杠，Python < 3.12 报 SyntaxError → 已修复为临时变量
2. `SEARCH_MAS_LLM_BASE_URL` 环境变量会覆盖 AgentPipe monitor URL → 验证脚本在 real 模式下主动 `os.environ.pop`

#### OrchRL tree smoke 收口复核（2026-03-15）

- 审查基线：`/home/cxb/OrchRL` 本地 `main`
- 关键提交：
  - `1c21f20 feat: add MATE tree rollout smoke support`
  - `f3d9a48 refactor: colocate search MAS smoke runner with example`
- 训练侧定向回归：

```bash
cd /home/cxb/OrchRL
PYTHONPATH=.:./verl pytest \
  tests/orchrl/trainer/test_mate_config.py \
  tests/orchrl/trainer/test_mate_rollout_adapter.py \
  tests/orchrl/trainer/test_mate_dataproto_adapter.py \
  tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py \
  tests/orchrl/examples/test_search_mas_tree_run.py \
  tests/orchrl/config/test_search_mas_tree_real_smoke_config.py \
  tests/trajectory -q
```

- 结果：`79 passed, 2 warnings`
- 当前判断：
  - OrchRL 对 MATE tree public API 的使用仍限制在稳定外部接口：`parallel_rollout` / `tree_rollout`、`AgentPipeConfig`、`ModelMappingEntry`、`VLLMBackend`、结果 datatypes
  - “partial policy batches” 当前被作为合法 Search MAS tree 轨迹结果处理，不再被错误视为 trainer 异常
  - OrchRL `trajectory/` vendored 快照没有发现 MATE 语义分叉；唯一差异为相对导入适配
  - 当前没有证据支持继续把 V0.2 问题定义为 MATE 侧 blocker
