# V0.2 全局审核与下一步规划 — 工作交接

> 用途：新 agent 会话启动 prompt。目标是全局代码审核 + 精简 + 下一步行动规划。

## 角色

你是 MATE-reboot 项目的技术 lead（Master Agent）。你负责V0.2 阶段的整体设计、规划、统筹和工作发布。请阅读以下材料了解项目的整体信息，然后使用合适的superpowers skills，借助subagents完成审查和精简任务。之后为用户提供项目现状分析和下一步规划。

## 前置阅读

1. `/home/cxb/MATE-reboot/AGENTS.md` — 项目治理规则
2. `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` — V0 架构设计（理解模块职责）
3. `/home/cxb/MATE-reboot/docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md` - V0.2 设计方向 
4. `/home/cxb/MATE-reboot/docs/project-context.md` — 项目状态和团队分工

## 项目状态

MATE-reboot 是多智能体轨迹采集引擎，当前 V0.2 已完成并合入 main：

- **当前分支**: `main`（commit `9419ad8`）
- **测试**: 87 passed（`pytest tests/trajectory tests/scripts/test_run_real_validation.py -q`）
- **真实环境验证**: 通过（vLLM + OrchRL Search MAS，238 turns / 39 episodes）

## 仓库结构

```
MATE-reboot/
├── mate/trajectory/         # 核心库（1057 行）
│   ├── datatypes.py    (87)   # EpisodeResult, TurnData, BranchResult, TreeEpisodeResult
│   ├── monitor.py     (159)   # ModelMonitor — 拦截 LLM 请求、记录 token/logprobs
│   ├── pipe.py        (142)   # AgentPipe — 子进程 MAS 编排 + Monitor 代理
│   ├── backend.py     (142)   # InferenceBackend — vLLM 推理后端策略
│   ├── launcher.py    (132)   # MAS 进程启动器
│   ├── tree.py        (113)   # tree_rollout — 重放式树状分支采样
│   ├── reward.py       (79)   # FunctionRewardProvider
│   ├── replay_cache.py (66)   # ReplayCache — pilot 响应缓存
│   ├── parallel.py     (57)   # parallel_rollout — Best-of-N 并行采样
│   ├── collector.py    (35)   # TrajectoryCollector 高层接口
│   └── __init__.py     (45)   # 公开 API 导出
├── tests/                    # 测试（3075 行，87 tests）
│   ├── trajectory/           # 核心库测试（12 文件）
│   └── scripts/              # 脚本测试（1 文件）
├── scripts/                  # 验证脚本（2773 行）
│   ├── run_real_validation.py   (965)  # V0.1 并行验证
│   ├── run_tree_validation.py   (884)  # V0.2 树状分支验证
│   ├── visualize_trajectories.py (924) # 轨迹可视化（终端 + HTML）
│   └── USAGE.md                        # 脚本用法文档
├── docs/                     # 文档
│   ├── project-context.md              # 项目上下文（关键参考）
│   ├── plans/                          # 设计文档
│   ├── prompts/                        # 会话 prompt
│   ├── retros/                         # 回顾记录
│   └── reviews/                        # 审核记录
├── AGENTS.md                 # Agent 执行规则
├── pyproject.toml            # 包定义（mate-trajectory 0.2.0）
└── third_party/verl          # Git submodule
```

## 代码量概览

| 区域 | 行数 | 文件数 | 说明 |
|------|------|--------|------|
| mate/trajectory/ | 1,057 | 11 | 核心库，精简 |
| tests/ | 3,075 | 14 | 测试覆盖完整 |
| scripts/ | 2,773 | 3 | **占总代码量 47%，可能有精简空间** |
| 总计 | 6,905 | 28 | |

## 审核关注点

### 1. scripts/ 占比过大

三个脚本共 2773 行，几乎等于核心库 + 测试的一半。主要原因：
- `run_real_validation.py` 和 `run_tree_validation.py` 有大量重复的 helper 函数
  - V0.2 脚本通过 `from scripts.run_real_validation import ...` 复用了约 15 个函数
  - 但 V0.1 脚本本身还有很多内联逻辑（环境探测、mock 构建、报告生成）
- `visualize_trajectories.py` 包含 V0.1 + V0.2 两套完整渲染（terminal + HTML × 2 schema）

**可能的精简方向**：
- 提取公共 helper 到 `scripts/_helpers.py`
- visualize 的 HTML 模板是否可以抽出为独立模板文件
- 评估 scripts 中是否有可以挪入 mate/ 核心库的逻辑

### 2. 核心库健康度

核心库 1057 行、11 个文件，每个模块职责清晰。需审核：
- 模块间依赖是否合理
- 公开 API 是否最小化（`__init__.py` 导出）
- 是否有死代码或过度抽象

### 3. 测试覆盖

87 tests 通过。需审核：
- 测试与核心库的比例 3:1，是否有冗余测试
- 是否存在测试耦合（一个模块变动导致大量测试修改）

### 4. 文档一致性

多个 docs 文件跨 V0/V0.2 多次更新，检查：
- project-context.md 与实际代码状态是否一致
- 是否有过时的计划文档可以归档

### 5. OrchRL适配

当前MATE-reboot中未对Verl后端进行适配，由OrchRL侧调用，需分析：
- MATE-reboot侧是否仍需实现规划中的VerlBackend
- OrchRL侧的适配是否合理

## 下一步待规划

| 方向 | 说明 | 优先级 |
|------|------|--------|
| 同步到 OrchRL | V0.2 代码同步 + 训练侧 adapter 适配 | project-context.md 中唯一未完成项 |
| 分支策略优化 | 选择性分支（非每个 turn 都分支）、ROME/IPA 参考 | V0.3 方向 |
| 性能基线 | 树状 vs 并行的采集效率对比、前缀共享实际加速比 | 需要训练侧配合 |

## 真实环境验证关键发现

1. **树状分支机制正常工作**：replay 标记 220/220 正确，前缀共享 ~25%
2. **deadpool prompt 长上下文失败**：已知限制，max_turns=4 时触发 vLLM 400 错误


## 参考代码位置

| 资源 | 路径 |
|------|------|
| OrchRL 主仓（含已合入的 trajectory） | `/home/cxb/OrchRL/` |
| MATE-reboot 开发仓库 | `/home/cxb/MATE-reboot/` |
| OrchRL Search MAS | `/home/cxb/OrchRL/examples/mas_app/search/` |
| V0 设计文档（含演进路径） | `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md` |
| 训练侧对接规格 | `/home/cxb/MATE-reboot/docs/plans/2026-03-05-training-integration-spec.md` |
| 真实环境验证记录 | `/home/cxb/MATE-reboot/docs/retros/2026-03-09-trajectory-engine-real-validation.md` |
- Agent 执行规则：`AGENTS.md`
- 脚本用法：`scripts/USAGE.md`
- V0.2 设计方向：`docs/plans/2026-03-09-trajectory-engine-v02-design-direction.md`
- V0.2 实施计划：`docs/plans/2026-03-09-trajectory-engine-v02-impl-plan.md`
