# OrchRL Tree Smoke 新服务器联调同步

> 日期：2026-03-14
> 范围：`/home/cxb/OrchRL` 分支 `feat/mate-tree-adapter`
> 目的：向 MATE 侧 Agent 同步 OrchRL tree-mode 真实 smoke 在新服务器上的推进结果、环境修复和最终可复现结论

## 1. 结论先行

这次联调已经把 OrchRL tree-mode 真实 smoke 从“启动即失败”推进到最终跑通：

- 三个 PPO trainer 初始化成功
- 三个 async vLLM rollout server 初始化成功
- 训练主循环进入：
  - `step 0 started`
  - `Preparing initial MATE rollout collection (using base model)`
  - `step:1 - ... training/global_step:0.000`
  - `step 1 started`
- 进程 clean exit `0`

也就是说：

- OrchRL tree collect 入口已经在这台新服务器上真实打通
- OrchRL tree-mode collect/update/下一步训练循环已经在这台服务器上端到端跑通
- 当前已不存在需要 MATE 侧立即处理的新 blocker

## 2. 当前 smoke 配置与运行方式

### 2.1 最终推进到最深位置时使用的配置

- 仓库：`/home/cxb/OrchRL`
- branch：`feat/mate-tree-adapter`
- config：`orchrl/config/search/search_mas_tree_real_smoke.yaml`
- template：`orchrl/config/search/templates/search_mas_tree_real_smoke_template.yaml`
- prompt：`orchrl/config/search/data/search_mas_tree_real_smoke_prompts.jsonl`

### 2.2 当前有效的 smoke 形态

为了推进真实 smoke，当前采用的是保守配置：

- `specialization=full`
- 3 agents 各自独立 policy
- 模型降级为 `Qwen3-0.6B`
- `training.max_prompt_length=2048`
- `training.max_response_length=1024`
- `actor.ppo_mini_batch_size=1`
- `actor.ppo_micro_batch_size=1`
- `actor.ppo_micro_batch_size_per_gpu=1`
- `actor.ppo_max_token_len_per_gpu=4096`
- `rollout.gpu_memory_utilization=0.6`
- `rollout.max_model_len=4096`
- `rollout.enforce_eager=true`
- `application.max_turns=4` 仍保持不变

原因：

- 4B + 3 独立 policy 在本机 vLLM 初始化阶段直接撞上显存门槛
- 用户要求优先保留 `full specialization`
- 这组保守参数已经足够让 `0.6B + full specialization + max_turns=4` 跑通真实 smoke

### 2.3 真实 smoke 命令

当前 OrchRL 侧新增了一个受控 smoke runner：

```bash
PYTHONPATH=.:./verl python3 -m orchrl.testing.tree_real_smoke \
  --retrieval-service-url http://127.0.0.1:8010/retrieve \
  --cuda-visible-devices 0,1,2 \
  --log-path logs/search_mas_tree_real_smoke.log
```

## 3. 本次已解决的环境问题

### 3.1 旧 worktree 绝对路径已失效

原 smoke config / runbook 中仍残留：

- `/root/.config/superpowers/worktrees/OrchRL/mate-tree-adapter/...`

当前已统一改成以 `/home/cxb/OrchRL` 为根的 repo-local 路径。

### 3.2 retrieval server 依赖与路径已核实

本机实际存在的 retrieval 资产：

- index：`/data1/lll/datasets/wiki-18/e5_Flat.index`
- corpus：`/data1/lll/datasets/wiki-18/wiki-18.jsonl`
- model：`/data1/lll/models/e5-base-v2`

当前 shell 中没有可直接调用的 `conda`，因此这次直接使用当前 Python 环境并补装：

- `faiss-cpu`

其余依赖（`torch/transformers/datasets/fastapi/uvicorn/pydantic/numpy`）当前 Python 已具备。

### 3.3 Triton / vLLM 链接问题已定位

在 0.6B + full specialization 推进到 `step 0` 后，曾命中过：

- `/usr/bin/ld: cannot find -lcuda`

这是 Triton/Inductor 在采样阶段编译 CUDA helper 时的链接问题，不是 MATE 逻辑问题。

当前结论：

- 需要给 smoke 进程注入 `LIBRARY_PATH`
- 可用前缀：
  - `/usr/local/cuda/compat`
  - `/usr/local/cuda/targets/x86_64-linux/lib/stubs`

注意：

- 不能把 `/usr/local/cuda/compat` 注入 `LD_LIBRARY_PATH`
- 那会触发 CUDA 检测走到错误的 forward-compat 路径，后续 FSDP 会错误退化到 `HCCL` backend

因此当前 OrchRL 侧 runner 只注入：

- `LIBRARY_PATH`

而不改：

- `LD_LIBRARY_PATH`

## 4. 当前最终成功证据

来自 `logs/search_mas_tree_real_smoke.log` 的关键证据：

- `step 0 started`
- `Preparing initial MATE rollout collection (using base model)`
- `step:1 - ... training/global_step:0.000`
- `step 1 started`
- `MATE rollout produced partial policy batches; missing=['searcher_model'], available=['verifier_model', 'answerer_model']. Skipping updates for missing policies in this step.`
- `Cleanup completed`

来自真实运行会话的额外证据：

- OrchRL smoke 进程返回 `exit code 0`
- 同时跑过的回归集返回：
  - `79 passed`

同时可见：

- 三个 trainer 全部初始化成功
- 三个 async vLLM rollout server 全部初始化成功

这说明这次的真实 blocker 已经不再是：

- tree adapter 接线
- tree public API 导入
- trainer 初始化
- async vLLM server 初始化
- retrieval server 未启动

## 5. 这次成功证明了什么，没证明什么

这次成功证明：

- MATE tree public API 当前语义足够支撑 OrchRL tree adapter 跑真实训练
- `max_turns=4` 在这次保守配置下没有先撞上 MATE 侧已知 long-context blocker
- 之前看到的 `MAS process exited with non-zero exit code 1` 并不是稳定必现的 MATE blocker，至少在本次最终配置下已不再阻塞

这次没有证明：

- 更大模型（例如 4B）在同样 topology 下也稳定
- 更激进的长度/显存配置也稳定

## 6. 对 MATE 侧的含义

当前没有证据表明还需要 MATE 侧再修改：

- `tree_rollout`
- `ReplayCache`
- `TreeEpisodeResult`
- MATE 核心采集逻辑
- vendored `trajectory/`

更准确地说：

- 当前 smoke 已跑通，说明 MATE tree API 语义没有挡住 OrchRL 的真实 smoke
- 这次真正需要修的是 OrchRL 训练侧的资源配置与 trainer 对“部分 policy batch”场景的错误假设

因此这次同步给 MATE 侧最重要的信息是：

- **MATE tree collect 公共 API 在 OrchRL 侧已经完成真实 smoke 验证**
- **这次没有发现必须回推到 MATE 核心逻辑的新 blocker**

## 7. 推荐的下一步尝试顺序

如果后续要继续放宽 smoke 或提高模型规格，推荐顺序是：

1. 先保持当前成功配置作为基线，不要丢
2. 优先尝试回升模型规模或长度预算时单次只放宽一个维度
3. 若 4B 或更长上下文再次失败，再回看是否与 MATE 侧已知 long-context 问题重合
4. 如果只是为了保底跑通后续回归，`prompt specialization` 仍然是比改 MATE 核心逻辑更合理的降级路径

## 8. 本次 OrchRL 侧新增的最小支撑

为了让 smoke 在新服务器可复现，OrchRL 侧当前新增了：

- `orchrl/testing/tree_real_smoke.py`
- `tests/orchrl/testing/test_tree_real_smoke.py`

用途：

- 解析 smoke config
- 校验 repo-local 路径
- 清理会覆盖 Search MAS base URL 的环境变量
- 注入必要的 `LIBRARY_PATH`
- 以一致方式启动真实 smoke

这些改动都在 OrchRL 侧，未要求 MATE 侧配合改代码。

## 9. 当前建议给 MATE 侧 Agent 的结论

可以直接据此更新认知：

1. OrchRL tree smoke 在新服务器上已经完整跑通，真实训练进程 clean exit `0`。
2. 当前没有证据要求 MATE 再改 tree public API、`trajectory/` 快照语义或采集核心逻辑。
3. 这次必须修的一个真实问题在 OrchRL trainer：Search MAS trace 可能合法地产生 partial policy batches，不能假设每一步都覆盖所有 policy。
4. 后续如果再遇到 `MAS exit code 1`，先区分：
   - 是否是 OrchRL 资源/长度配置退化
   - 是否才是 MATE 侧已知 long-context 问题再次出现
