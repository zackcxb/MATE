# OrchRL Tree Adapter 开发进展同步

> 日期：2026-03-11
> 范围：`/root/.config/superpowers/worktrees/OrchRL/mate-tree-adapter` 上的 OrchRL 侧适配开发
> 目的：向 MATE 侧 Agent 同步当前 OrchRL 训练侧对 `tree_rollout` / `TreeEpisodeResult` 的接入状态、验证边界与已发现问题

## 1. 执行场地与上下文

- 开发仓库：`/home/cxb/OrchRL`
- 实际开发目录：`/root/.config/superpowers/worktrees/OrchRL/mate-tree-adapter`
- 分支：`feat/mate-tree-adapter`
- 约束遵循：
  - 不在 OrchRL 主工作区直接开发
  - 仅依赖 MATE 稳定外部 API
  - 不新增 OrchRL 对 `AgentPipe` / `ModelMonitor` / `ReplayCache` / `InferenceBackend` 等 MATE 内部接口的训练侧依赖

## 2. 当前已完成内容

### 2.1 OrchRL trainer 配置增加 rollout mode

已在 OrchRL 侧引入：

- `training.mate.rollout_mode`
  - `parallel`
  - `tree`

同时增加 tree 模式配置项：

- `training.mate.tree.k_branches`
- `training.mate.tree.max_concurrent_branches`

当前 OrchRL 侧配置校验会拒绝：

- 非法 `rollout_mode`
- `k_branches < 1`
- `max_concurrent_branches < 1`

### 2.2 rollout adapter 支持 tree 模式

`MateRolloutAdapter` 当前行为：

- `parallel` 模式调用 `parallel_rollout`
- `tree` 模式调用 `tree_rollout`

两种模式共用：

- `AgentPipeConfig`
- `ModelMappingEntry`
- `VLLMBackend`
- `FunctionRewardProvider`

Tree 模式下，adapter 会把现有 job metadata 写回：

- `pilot_result.metadata`
- `pilot_result.trajectory.metadata`
- 每个 `branch.episode_result.metadata`
- 每个 `branch.episode_result.trajectory.metadata`

因此后续 batch adapter 仍能读取：

- `prompt_group_id`
- `sample_idx`
- `prompt`
- `expected`

### 2.3 新增 tree-aware dataproto adapter

已新增 `tree_episodes_to_policy_batches(...)`，用于消费 `TreeEpisodeResult`。

当前规则：

1. pilot turn 维持旧 uid
   - `f"{prompt_group_id}:{agent_idx}"`
2. branch replayed prefix turn 不进入训练 batch
3. branch point turn uid
   - `f"{prompt_group_id}:{agent_idx}:b{branch_turn}"`
4. branch point之后 turn uid
   - `f"{prompt_group_id}:{agent_idx}:b{branch_turn}:t{global_turn_index}"`

### 2.4 trainer 已接线

`MultiAgentsPPOTrainer` 当前会按 `rollout_mode` 选择：

- `episodes_to_policy_batches`
- `tree_episodes_to_policy_batches`

validation 侧在 tree 模式下会先展平为：

- `pilot_result`
- 所有 `status == "success"` 的 branch `episode_result`

这保证 validation 统计口径与“实际可训练样本集合”一致。

## 3. 为了接入 tree 模式，同步了 OrchRL 本地 trajectory 公共快照

发现的实际情况：

- OrchRL 当前仓内顶层 `trajectory/` 还是旧快照
- 缺少：
  - `tree_rollout`
  - `TreeEpisodeResult`
  - `BranchResult`
  - `ReplayCache`
  - `EpisodeResult.status`
  - `EpisodeResult.failure_info`

如果不补这部分，OrchRL 侧即使 trainer/adapter 写完，也无法在本仓中真正导入并运行 tree public API。

因此本次还同步了 OrchRL 仓内 vendored `trajectory/` 的 V0.2 公共能力与其必要实现：

- `trajectory/__init__.py`
- `trajectory/datatypes.py`
- `trajectory/monitor.py`
- `trajectory/pipe.py`
- `trajectory/replay_cache.py`
- `trajectory/tree.py`

这里的判断是：

- 这不属于让 OrchRL 训练代码依赖 MATE 内部 API
- 这是把 OrchRL 仓内本地 `trajectory` 公共快照补齐到 V0.2 可用状态

## 4. 已完成验证

### 4.1 OrchRL tree adapter 定向测试

执行命令：

```bash
PYTHONPATH=.:./verl pytest \
  tests/orchrl/trainer/test_mate_config.py \
  tests/orchrl/trainer/test_mate_rollout_adapter.py \
  tests/orchrl/trainer/test_mate_dataproto_adapter.py \
  tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py -q
```

结果：

- `10 passed`

覆盖点：

- `rollout_mode` 配置校验
- `parallel_rollout` / `tree_rollout` 路由选择
- pilot uid 兼容
- replayed prefix skip
- branch-aware uid 规则
- trainer tree adapter 选择
- validation 展平 `pilot + success branch`

### 4.2 OrchRL 本地 trajectory 回归

执行命令：

```bash
PYTHONPATH=.:./verl pytest tests/trajectory -q
```

结果：

- `60 passed`

结论：

- 本次同步的 `trajectory` 公共快照没有打坏 OrchRL 现有本地 trajectory 测试面

### 4.3 最终组合验证

执行命令：

```bash
PYTHONPATH=.:./verl pytest \
  tests/orchrl/trainer/test_mate_config.py \
  tests/orchrl/trainer/test_mate_rollout_adapter.py \
  tests/orchrl/trainer/test_mate_dataproto_adapter.py \
  tests/orchrl/trainer/test_multi_agents_ppo_trainer_mate.py \
  tests/trajectory -q
```

结果：

- `71 passed`
- `2 warnings`

warnings 内容：

- `pkg_resources` deprecation
- Ray state API deprecation

两者均非本次 tree 接入引入的失败。

### 4.4 代码复核

本次改动完成后做了两轮独立复核：

1. spec-compliance review
2. code-quality / risk review

结论：

- 未发现阻塞级 spec gap
- 未发现阻塞级 correctness bug

## 5. 当前没有验证到的范围

这次尚未完成“真实端到端训练流程已跑起”的证明。

当前已经验证的是：

- 代码级闭环
- adapter / trainer 路由
- tree batch 生成
- 本地 trajectory 回归

当前尚未验证的是：

- `orchrl/trainer/train.py` 发起的真实训练 job 在 tree 模式下完整跑通
- 真实 vLLM server + Search MAS + reward + Ray trainer 主循环的端到端联调
- 真实 collect/update/validation/checkpoint 全链路在 tree 模式下的运行稳定性

因此现阶段准确表述应是：

- **Tree 训练侧适配代码闭环已经完成并通过本地测试**
- **真实训练端到端尚未验证**

## 6. 2026-03-12 新增 smoke 联调进展

### 6.1 已解决的 OrchRL / vendored runtime blocker

为了把真实 smoke 推进到 tree collect，新增了以下运行时修补：

1. `verl/verl/workers/fsdp_workers.py`
   - 当 `world_size == 1` 时，禁用 FSDP `sync_module_states`
   - 证据：此前单卡 FSDP 仍会在 `sync_module_states=True` 上触发
     - `nvmlDeviceGetHandleByIndex(7) failed`
   - 修补后，最小 Qwen3 + FSDP 复现恢复

2. `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
   - 不再直接调用 V1-only `AsyncLLM.from_vllm_config(...)`
   - 改为走通用 `AsyncLLMEngine.from_vllm_config(...)`
   - 避免 Ray actor 内部重复 `ray.init(...)`
   - `max_model_len` 改为尊重 rollout 配置，不再强制抬到 `32768`

这些修补都配了最小回归测试，属于通用兼容性修补，不是只为当前 smoke 硬编码的特判。

### 6.2 smoke 已推进到的最深位置

真实 smoke 现已验证到：

- 三个 trainer 均初始化成功
- 三个 async vLLM rollout server 均初始化成功
- 训练主循环进入：
  - `step 0 started`
  - `Preparing initial MATE rollout collection`
- collect 已进入：
  - `tree_rollout`
  - `pilot_pipe.run(...)`

这说明 OrchRL tree collect 入口已经真正打通，不再卡在：

- NCCL / NVML
- flash-attn 导入
- vLLM V0/V1 工厂不兼容
- Ray 重复初始化
- vLLM KV cache 由 32768 floor 造成的超配

### 6.3 当前剩余 blocker

当前真实 smoke 最终失败在：

- `RuntimeError: MAS process exited with non-zero exit code 1`

调用链：

- `multi_agents_ppo_trainer.py`
- `mate_rollout_adapter.py`
- `trajectory/tree.py`
- `trajectory/pipe.py`
- `pilot_pipe.run(...)`

因此当前 blocker 已经下沉到 Search MAS 子进程自身，而不是 OrchRL tree 接入层。

### 6.4 当前 smoke 启动条件

当前 host 上要稳定推进 smoke，需要：

- 避开坏卡 GPU5
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `CUDA_VISIBLE_DEVICES=3,4,6`
- `VLLM_USE_V1=1`

检索服务当前可复现启动方式：

```bash
env CUDA_VISIBLE_DEVICES=0 conda run -n retriever \
  python /home/cxb/OrchRL/examples/mas_app/search/scripts/retrieval_server.py \
  --index_path /data1/lll/wiki-18/e5_Flat.index \
  --corpus_path /data1/lll/wiki-18/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model /data1/lll/e5-base-v2 \
  --port 8010
```

当前 smoke 命令：

```bash
env -u SEARCH_MAS_LLM_BASE_URL -u OPENAI_BASE_URL \
  SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://127.0.0.1:8010/retrieve \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=3,4,6 \
  VLLM_USE_V1=1 \
  CONFIG_NAME=search_mas_tree_real_smoke \
  LOG_PATH=logs/search_mas_tree_real_smoke.log \
  bash scripts/run_search_mas_train_e2e.sh
```

## 7. 已发现的问题与技术风险

### 6.1 Branch 全局顺序是训练侧重建出来的，不是结果类型直接提供

当前 `TreeEpisodeResult` / `TurnData` 提供的是：

- `BranchResult.branch_turn`：pilot 上的全局分叉点位置
- `TurnData.turn_index`：单个 role 内的局部 turn 序号

训练侧要正确：

- 跳过 replayed prefix
- 给 branch point / branch continuation 生成 uid

还需要知道“branch episode 里每个 turn 的全局执行位置”。

当前做法：

- 按 `TurnData.timestamp` 重建 branch 内全局顺序
- tie-break 使用 `(timestamp, turn_index, role)`

这在当前 collector 输出下可工作，但它依赖一个隐含契约：

- monitor 记录顺序与逻辑执行顺序近似一致

风险点：

- `timestamp` 是墙钟时间，不是显式逻辑序号
- 如果未来 MAS 并发发多个请求，请求到达顺序不一定等于逻辑顺序
- 极近时间的多个 turn 可能出现相同或近似时间戳
- 如果将来 replay / fallback 行为变化，这个假设可能失效

当前判断：

- 这不是当前 blocker
- 但如果 MATE 后续愿意在 public result 中直接暴露稳定的 global turn index，OrchRL 侧可以去掉这层推断逻辑

### 6.2 OrchRL 顶层 `trajectory/` 的位置短期可用，长期需要明确 vendored 边界

当前 OrchRL 中很多地方直接：

- `from trajectory import ...`

因此短期继续保留顶层 `trajectory/` 是现实选择。

但长期问题也明确：

- 包名过泛
- ownership 不够直观
- vendored snapshot 与 OrchRL 自身代码边界不清晰

建议方向：

- 短期不因这次接入单独搬迁
- 如果后续继续同步 MATE，考虑明确成更清晰的 vendored 位置

### 6.3 这次为了跑本地测试，安装了额外 Python 依赖

本次本地验证过程中补装了：

- `omegaconf`
- `tensordict`
- `codetiming`
- `torchdata`

这不是 repo 内容变更，但说明当前执行环境默认并不完整覆盖 OrchRL 这组测试依赖。

## 7. 当前建议给 MATE 侧 Agent 的关注点

1. 当前 OrchRL tree 接入不需要 MATE 侧再做 blocker 级改动。
2. 如果后续要降低训练侧推断复杂度，最有价值的 public API 增强点是：
   - 在 tree result / turn data 中直接暴露稳定 global turn order
3. 如果 OrchRL 与 MATE 会继续双仓同步，建议尽早明确：
   - OrchRL 内 vendored `trajectory` 的同步策略
   - 哪些文件允许 OrchRL 本地修补，哪些必须严格镜像 MATE

## 8. 当前状态结论

截至 2026-03-11：

- OrchRL 侧 `tree_rollout` / `TreeEpisodeResult` 消费闭环代码已完成
- 本地测试已通过
- 未发现必须回推到 MATE 核心采集逻辑的 blocker
- 真实训练端到端实跑仍是下一步独立验证项
