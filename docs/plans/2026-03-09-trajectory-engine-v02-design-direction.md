# Trajectory Engine V0.2 设计方向文档

> 日期：2026-03-09
> 状态：方向草案（待详细设计）
> 前序文档：`2026-03-04-trajectory-engine-v0-design.md`（V0 设计，已冻结）
> 讨论上下文：V0 已完成并合入 OrchRL（`1664eb5`），训练侧已完成端到端联调

---

## 1. V0.2 目标

验证**树状分支采样**在多智能体 GRPO 训练中的可行性与收益，同时提升采集管线的稳定性。

核心算法对标：团队架构文档中的 MARL GRPO 分组适配算法——先生成主路径，在每个 agent 节点处做 K 次分组替换采样，确保 GRPO 公平比较的前提成立。

### 1.1 V0.2 范围

| 包含 | 不包含 |
|------|--------|
| 重放式分支采样（PilotReplayBranching） | Trie 存储（采集层不是存储瓶颈） |
| tree_rollout 函数（分支编排） | 训练侧前缀合并（AREAL-DTA/Forge 风格，训练侧负责） |
| TreeEpisodeResult（树结构元数据输出） | 训推异步（V1 特性） |
| 优雅降级（partial result 返回） | Reward Model（V1+ 特性） |
| 收益对比实验（vs Best-of-N） | Multi-LoRA 采集侧支持（训练侧已独立实现） |

### 1.2 验证目标

1. 树状分支采样在 DrMAS Search 上可以正确运行并输出 TreeEpisodeResult
2. TreeEpisodeResult 的树索引元数据可被训练侧正确解析、用于 GRPO 分组
3. 对比实验：树状分支 vs Best-of-N 在 reward 分布方差、前缀共享率上有可量化差异

---

## 2. 核心特性设计方向

### 2.1 重放式分支采样（PilotReplayBranching）

#### 算法

对标 MARL GRPO 分组适配算法（团队架构文档 marl-04）：

```
1. 运行一条完整的主路径（pilot run）：P → V0 → S0 → V1 → A0
2. 在主路径的每个 agent turn 处，做 K 次分组替换采样（branch runs）
3. 每条 branch run 重放主路径中分支点之前的所有 turn（使用缓存响应），
   在分支点处重新采样，然后继续执行到 episode 结束
4. 输出：1 条主路径 + T×K 条分支轨迹，组织为树结构
```

#### 机制

- **ReplayCache**：当前实现存储 pilot episode 的全部 LLM 交互记录，按 `(agent_role, turn_index)` 索引，并用 `messages_hash` 校验上下文是否漂移
- **确定性假设**：Replay 仅缓存 LLM 响应；工具调用默认仍真实执行，因此“可重放”只对 LLM token 级输出成立，不保证外部工具副作用、时序或环境状态完全确定
- **分支 Monitor 行为**：当请求命中 ReplayCache 且 `messages_hash` 一致时，直接返回缓存响应（不调用 Backend）；cache miss 或 messages 不匹配时，会 fallback 到 Backend 正常推理，这应视为 degraded replay 语义，而不是“静默正确 replay”
- **MAS 零侵入**：每条分支仍是一个完整的 MAS 子进程，MAS 代码无需任何改动

#### 已知限制与后续优化方向

| 限制 | 后续优化方向 | 参考 |
|------|------------|------|
| 每条分支仍需完整 MAS 进程启动 | 探索 MAS 状态序列化或 filesystem checkpoint | ConTree, LangGraph persistent checkpointing |
| 工具调用在重放时仍真实执行 | 可选优化：工具调用缓存层 | — |
| 当前在所有 turn 均匀分支 | 选择性分支：只在高不确定性的关键决策点分支 | TreeRL EPTree, ROME IPA rollback |

### 2.2 tree_rollout 函数

编排 pilot run 和 branch runs 的执行。V0.2 当前直接采用函数实现，避免在 PoC 阶段提前抽象；当 V1 真正引入选择性分支时，再根据实际复杂度决定是否拆分编排层。

```python
async def tree_rollout(
    prompt: str,
    reward_provider: RewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    k_branches: int = 3,
    max_concurrent_branches: int | None = None,
) -> TreeEpisodeResult:
    # 1. 执行 pilot run，收集 ReplayCache
    # 2. 在主路径的每个 agent turn 处分叉（均匀分支）
    # 3. 编排所有 branch runs（可并行，受 max_concurrent_branches 限制）
    # 4. 组装 TreeEpisodeResult
    ...
```

V0.2 硬编码均匀分支逻辑——在主路径的每个 agent turn 处分叉，每个点 K 条分支。K 作为函数参数传入。

### 2.3 TreeEpisodeResult（树结构元数据输出）

**设计原则**：平坦存储 + 树索引。每条分支独立存储完整轨迹，但附带树结构元数据，供训练侧：
1. 重建树拓扑，用于 GRPO 分组
2. 识别共享前缀，用于 AREAL-DTA / Forge / Tree Training 风格的前缀合并

```python
@dataclass
class BranchResult:
    episode_result: EpisodeResult
    branch_turn: int                # 分叉点在主路径中的全局 turn 位置，不是 agent 内局部 turn_index
    branch_agent_role: str          # 分叉点对应的 agent role
    parent_episode_id: str          # 主路径的 episode_id

@dataclass
class TreeEpisodeResult:
    pilot_result: EpisodeResult
    branch_results: list[BranchResult]
    prompt: str
    tree_metadata: dict[str, Any]   # 树级统计：分支数、深度、前缀共享率等
```

当前实现对齐点：
- `tree_rollout` 当前只收录 `status == "success"` 的 branch；`failed` / `partial` branch 不进入 `branch_results`
- `AgentPipe.allow_partial` 在失败路径会返回带 `status="failed"` 的 `EpisodeResult`；这些结果可用于日志或诊断，但当前不进入训练 batch

#### messages_hash 规范

推荐规范：

```python
messages_hash = sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
```

用途：防止分支 replay 时因上下文漂移而误命中 `ReplayCache`。即便 `(agent_role, turn_index)` 相同，只要 messages 序列不一致，也必须走 degraded replay 的 backend fallback 路径。

#### uid 构建规则（与训练侧对齐）

当前训练侧 uid 构建：`f"{prompt_group_id}:{agent_idx}"`

V0.2 树状分支下的 uid 扩展提案：

```
for turn in pilot_result.turns:
    uid = f"{prompt_group_id}:{turn.agent_idx}"
    emit_to_training_batch(turn, uid)

for branch in branch_results:
    for turn in branch.episode_result.turns:
        if turn.global_turn_index < branch.branch_turn:
            # replayed turn：与主路径完全共享，只用于树结构诊断，不进入训练 batch
            continue
        if turn.global_turn_index == branch.branch_turn:
            uid = f"{prompt_group_id}:{turn.agent_idx}:b{branch.branch_turn}"
        else:
            uid = f"{prompt_group_id}:{turn.agent_idx}:b{branch.branch_turn}:t{turn.global_turn_index}"
        emit_to_training_batch(turn, uid)
```

分组语义：
- pilot turn 保持 V0 uid，不改变现有训练侧解析逻辑
- branch replayed turn 不进入训练 batch，避免把缓存重放片段误当作新的采样样本
- 同一 `prompt_group_id:agent_idx:b{branch_turn}` 下的 K 条分支点记录属于同一 GRPO 分组；它们共享完全相同的上下文前缀，只有分支点处的 action token 不同

---

## 3. 稳定性改进

### 3.1 长上下文处理

**问题**：V0 验证中发现 `max_turns=4` 的长上下文 prompt 触发 vLLM 400/502（`max_model_len=4096` 与累积 team context 冲突）。树状分支会产生更多长轨迹分支，加剧此问题。

方案：**依赖 vLLM 原生 400 错误 + 优雅降级（§3.2）**。Monitor 的 502 响应路径（`_handle_chat_completions` 的 except 分支）已能传递 vLLM 错误信息，AgentPipe 的 `allow_partial` 确保错误不会导致整条 episode 数据丢失。

### 3.2 优雅降级

当 MAS 进程非正常退出时（如 vLLM 错误导致），需要避免整条 episode 数据完全丢弃。

当前实现 / 预期语义区分：
- **当前实现**：`AgentPipe.allow_partial` 会返回带已采集 turn 数据的 `EpisodeResult(status="failed")`
- **预期语义**：后续如需细分失败阶段，可再扩展为更显式的 `partial` / `failed` 词汇，但不影响 `tree_rollout` 当前“仅收录 success branch”的行为

```python
@dataclass
class EpisodeResult:
    # ... 现有字段 ...
    status: str = "success"         # 当前实现至少包含 "success" | "failed"；"partial" 为预期扩展位
    failure_info: dict | None = None  # 失败原因、失败 turn 等
```

树状分支失败模式矩阵：

| 场景 | 当前实现 | 预期语义 / 训练侧处理 |
|------|----------|----------------------|
| pilot 失败 | 不应继续启动 branch；返回形式以当前 `tree_rollout` 失败路径为准 | 视为整棵树失败，不产生可训练 branch 样本 |
| 单个 branch 异常退出 | 局部失败，不应拖垮其他 branch；仅 `status == "success"` 的 branch 被收录 | 保留其他成功 branch，失败 branch 只进入日志/诊断 |
| branch `partial` / `failed` | `AgentPipe.allow_partial` 会返回 `status="failed"` 的 `EpisodeResult`；`tree_rollout` 当前跳过这些 branch | 不进入 `TreeEpisodeResult.branch_results`，也不进入训练 batch |
| ReplayCache miss / messages_hash 不匹配 | fallback 到 backend 正常推理 | 标记为 degraded replay 语义；可继续采样，但不再视为“纯重放” |
| 长上下文 backend error | backend 返回 400/502，Monitor 透传错误；若发生在 branch，通常被降级为失败 branch 并跳过 | 若发生在 pilot，应终止树；若发生在 branch，仅损失该 branch，不污染其他成功样本 |

### 3.3 Monitor 资源回收

确保 MAS 进程异常退出时，Monitor 的 aiohttp server 和端口正确释放。V0 的 `AgentPipe.run()` 中已有 `finally: await self.monitor.stop()`，需要验证异常路径下的行为并补充测试。

---

## 4. 收益验证实验

### 4.1 对比实验设计

| 维度 | Best-of-N（V0 baseline） | 树状分支（V0.2） |
|------|-------------------------|-----------------|
| 采样方式 | N 条独立 episode | 1 条主路径 + 逐 turn K 分支 |
| 总 episode 数 | 相同（控制变量） | 相同 |
| 指标 | reward 均值/方差、agent-wise reward 分布、episode 多样性 | 同左 + **前缀共享率**、**GRPO 分组方差** |

前缀共享率 = 树中共享前缀的总 token 数 / 所有分支的总 token 数。该指标直接衡量训练侧使用 AREAL-DTA/Forge 前缀合并时能节省多少 forward 计算。

为避免树状分支与 Best-of-N 的比较失真，除“固定总 episode 数”外，还应增加一组“固定真实 LLM 调用数”的对照实验：
- 统计口径只计算真实 backend LLM 调用，不把 ReplayCache 命中计入调用数
- `parallel_rollout` 与 `tree_rollout` 在相同 LLM 调用预算下比较 reward、样本效率和多样性，避免树状分支因缓存重放而在表面 episode 数上占优
- 如需进一步严格控制，可单独记录工具调用次数，避免外部工具成本差异干扰结论

### 4.2 训练侧 mock 验证

用 `TreeEpisodeResult` 输出构建训练 batch，验证：
1. uid 分组能正确匹配 GRPO 算法的公平比较要求
2. 训练侧 `episodes_to_policy_batches` 的扩展版本能正确处理树结构
3. 共享前缀标记可被解析为 AREAL-DTA 的前缀树输入

---

## 5. 开发与协作

### 5.1 开发仓库

继续在 MATE-reboot 开发，里程碑完成后同步到 OrchRL。

同步约定：
- MATE-reboot 包名 `mate.trajectory`，OrchRL 包名 `trajectory`
- 同步脚本处理包名映射
- OrchRL 中的 trajectory 副本为只读快照

### 5.2 接口演进策略

**扩展模式**：新增 `tree_rollout`，保留 `parallel_rollout` 不变。

```python
# V0 接口（保持不变）
from trajectory import parallel_rollout

# V0.2 新增接口
from trajectory import tree_rollout, TreeEpisodeResult, BranchResult, ReplayCache
```

训练侧适配：
- `MateRolloutAdapter` 增加 `tree_rollout` 调用路径（通过配置选择）
- `mate_dataproto_adapter` 增加 `tree_episodes_to_policy_batches` 函数

### 5.3 版本标记

MATE-reboot `pyproject.toml` 中标记版本：V0 → `0.1.0`，V0.2 → `0.2.0`。

---

## 6. 技术参考

### 6.1 采集侧（直接相关）

| 参考 | 来源 | 对 V0.2 的启发 |
|------|------|---------------|
| MARL GRPO 分组适配算法 | 团队架构文档 `multi-agent-rl.pdf` | 核心算法：主路径 + 逐 turn 分组替换采样 |
| ROME/IPA rollback | [Let It Flow (arXiv:2512.24873)](https://arxiv.org/abs/2512.24873) | 选择性分支：只在关键决策点做 rollback |
| TreeRL EPTree | [arXiv:2506.11902](https://arxiv.org/abs/2506.11902) | 基于后验不确定性的分支策略 |

### 6.2 训练侧（供训练侧同事参考，采集侧输出匹配的数据格式）

| 参考 | 来源 | 用途 |
|------|------|------|
| AREAL-DTA | [arXiv:2602.00482](https://arxiv.org/abs/2602.00482) | DFS 单路径物化前缀树，显存友好的训练加速（8.31x） |
| Forge Prefix Tree Merging | MiniMax | 全量物化 + Magi Attention（40x 训练加速） |
| Tree Training | [arXiv:2511.00413](https://arxiv.org/abs/2511.00413) | Gradient Restoration + Tree Packing（6.2x） |
| Prompt Trees | Scaled Cognition | 单次 forward pass 编码整棵树（70x+） |

### 6.3 职责边界

- **采集侧**：生成树状轨迹，输出平坦存储 + 树索引元数据
- **训练侧**：从树索引重建前缀树，选择 AREAL-DTA/Forge/Tree Training 中的方案做 forward 前缀合并

---

## 7. 与 V0 路线图的对照

| V0 路线图中的 V0.2 定义 | 实际 V0.2 方向 | 变化原因 |
|------------------------|---------------|---------|
| 树状分支采样 | ✅ 保留（重放式实现） | — |
| 分支编排层 | ✅ 收敛为 `tree_rollout` 函数 | PoC 阶段先验证语义与收益，避免过早抽象 |
| Trie 存储 | ❌ 移除 | 采集层不是存储瓶颈；前缀去重价值在训练侧 forward pass |
| （未列出）优雅降级 | ✅ 新增 | 提高训练管线可靠性；覆盖长上下文等所有 vLLM 错误场景 |
| （未列出）收益验证实验 | ✅ 新增 | V0.2 定位为 PoC，需要量化收益 |

---

## 附录 A：训练侧现有适配层架构

基于 OrchRL `366090e` 分析（2026-03-09 同步后）：

```
orchrl/trainer/
├── multi_agents_ppo_trainer.py   # 训练主循环，调用 _collect_mate_step_batches
├── mate_rollout_adapter.py       # 封装 parallel_rollout，管理 VLLMBackend + AgentPipeConfig
├── mate_dataproto_adapter.py     # EpisodeResult → verl DataProto 转换
│   └── uid 构建: f"{prompt_group_id}:{agent_idx}"
│   └── credit_assignment: "all_turns" | "last_turn"
├── mate_reward_bridge.py         # 动态加载 reward provider
└── mate_prompt_loader.py         # 从 parquet/jsonl 加载 prompt
```

V0.2 需要训练侧变更的文件：
- `mate_rollout_adapter.py`：增加 `tree_rollout` 调用路径
- `mate_dataproto_adapter.py`：增加 `tree_episodes_to_policy_batches`，扩展 uid 规则

## 附录 B：后续演进路径（更新）

| 阶段 | 功能 |
|------|------|
| V0（已完成） | AgentPipe 全链路 + DrMAS Search 验证 |
| **V0.2（本设计）** | **重放式树状分支采样 + 收益验证** |
| V1 | 轨迹级训推异步 + Relay Worker + 选择性分支策略 |
| V1+ | Reward Model + 服务模式 MAS adapter + MAS 状态序列化（零开销分支） |
