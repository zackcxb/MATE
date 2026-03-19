# MATE Codebase Cheatsheet

> 面向 demo 讲解和快速代码定位的参考文档。

## 目录总览

```
mate/trajectory/
├── datatypes.py      数据结构定义（所有 dataclass）
├── backend.py        推理后端抽象（VLLMBackend / VerlBackend）
├── renderer.py       Canonical prompt 渲染（messages → token IDs）
├── validator.py      Runtime 硬约束检查
├── monitor.py        ModelMonitor — Gateway 核心，HTTP 拦截层
├── launcher.py       MASLauncher — 外部 MAS 进程管理
├── pipe.py           AgentPipe — 单次 episode 编排入口
├── parallel.py       parallel_rollout — 并发多 episode 采集
├── tree.py           tree_rollout — 分支采样 + 显式 branch 语义
├── replay_cache.py   ReplayCache — 分支重放前缀缓存
├── collector.py      TrajectoryCollector — buffer → 结构化轨迹
├── reward.py         RewardProvider / RewardWorker — 奖励计算
├── diagnostics.py    build_drift_artifact — Token drift 诊断
├── exporters.py      export_tokenized_turn — 训练侧消费出口
├── display.py        format_episode / format_tree — Demo 可视化
└── __init__.py       公开 API 汇总导出

scripts/
├── demo_search_mas.py        Demo runner（vLLM + OrchRL search MAS）
└── run_real_validation.py    真实环境批量验证
```

---

## 核心数据结构 — `datatypes.py`

```python
@dataclass
class ModelMappingEntry:
    actual_model: str | None   # 转发给后端时的实际 model 名
    backend_url:  str | None   # 可选：覆盖全局 backend URL（per-agent 路由到不同 vLLM）

@dataclass
class ModelRequest:
    request_id:           str
    agent_role:           str                # 从 HTTP body["model"] 字段识别
    messages:             list[dict]
    generation_params:    dict
    prompt_ids:           list[int] | None   # Canonical render 结果（V0.3 新增）
    render_fingerprint:   dict               # Tokenizer/template 快照
    sampling_fingerprint: dict

@dataclass
class ModelResponse:
    content:          str
    token_ids:        list[int] | None       # Response token IDs（训练 loss 直接需要）
    logprobs:         list[float] | None     # Per-token log probs（importance sampling）
    finish_reason:    str
    prompt_ids:       list[int] | None       # Runtime canonical prompt token IDs
    routed_experts:   list | None            # MoE 路由专家（optional，capability-gated）
    runtime_metadata: dict

@dataclass
class InteractionRecord:
    # Monitor buffer 中的原始记录，一个 LLM 调用对应一条
    agent_role:        str
    turn_index:        int       # 该 agent 的第几次调用（0-based）
    timestamp:         float
    messages:          list[dict]
    generation_params: dict
    response_text:     str
    token_ids:         list[int] | None
    logprobs:          list[float] | None
    finish_reason:     str
    episode_id:        str
    prompt_ids:        list[int] | None
    metadata:          dict

@dataclass
class TurnData:
    # TrajectoryCollector 将 InteractionRecord 转成 TurnData，面向下游的干净结构
    agent_role:     str
    turn_index:     int
    messages:       list[dict]
    response_text:  str
    token_ids:      list[int] | None
    logprobs:       list[float] | None
    finish_reason:  str
    timestamp:      float
    prompt_ids:     list[int] | None
    replayed:       bool                   # 是否为 tree branch 的重放前缀 turn
    branch_phase:   str | None             # "replay_prefix" / "branch_point" / "post_branch"
    routed_experts: list | None
    metadata:       dict

@dataclass
class EpisodeTrajectory:
    episode_id:         str
    agent_trajectories: dict[str, list[TurnData]]   # key = agent role
    metadata:           dict

@dataclass
class EpisodeResult:
    trajectory:   EpisodeTrajectory
    rewards:      dict[str, float | list[float]]    # per-agent reward
    final_reward: float | None
    metadata:     dict
    status:       str                               # "success" | "failed"
    failure_info: dict | None

@dataclass
class BranchResult:
    episode_result:    EpisodeResult
    branch_turn:       int      # 全局时序位置（在哪一步分叉）
    branch_agent_role: str      # 哪个 agent 触发了分叉
    parent_episode_id: str

@dataclass
class TreeEpisodeResult:
    pilot_result:   EpisodeResult
    branch_results: list[BranchResult]
    prompt:         str
    tree_metadata:  dict    # n_branch_points, k_branches, total_branches_collected
```

---

## 推理后端 — `backend.py`

```python
class InferenceBackend(ABC):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...

class VLLMBackend(InferenceBackend):
    # 走 HTTP POST /v1/chat/completions，兼容任何 OpenAI API 兼容服务
    VLLMBackend(backend_url, actual_model=None, timeout=120.0, tokenizer=None, renderer=None)

    # 带本地 tokenizer 的便捷构造（本地 render canonical prompt_ids）
    VLLMBackend.with_tokenizer(backend_url, model_path, actual_model=None, timeout=120.0)

    async def generate(request) -> ModelResponse
    # 1. 转发 messages 到 vLLM，收取 token_ids + logprobs
    # 2. 若有 renderer 且 request.prompt_ids 为 None，则 render 补全 prompt_ids
    # 3. 支持 generation_params["_backend_url"] 覆盖 per-request 路由
    # 4. 支持 routed_experts 透传（MoE）

class VerlBackend(InferenceBackend):
    # 走 direct prompt_ids → server_manager.generate()，跳过 HTTP chat completions
    # 对接 VERL AsyncLLMServerManager 的直接路径
    VerlBackend(server_manager, tokenizer=None, decoder=None)

    async def generate(request) -> ModelResponse
    # 1. 要求 request.prompt_ids 非 None（否则直接报错）
    # 2. 调用 server_manager.generate(prompt_ids=..., sampling_params=...)
    # 3. 返回 token_ids + logprobs + finish_reason + routed_experts
```

---

## Canonical 渲染 — `renderer.py`

```python
class ChatRenderer:
    # messages → prompt_ids 的唯一权威路径（V0.3 token contract 基石）
    ChatRenderer(tokenizer, model_name=None)
    ChatRenderer.from_tokenizer(tokenizer, model_name=None)

    def render(messages, *, add_generation_prompt: bool) -> tuple[list[int], dict]
    # 返回 (prompt_ids, render_fingerprint)
    # render_fingerprint 记录 model_name、add_generation_prompt、tokenizer_class
```

---

## Runtime 验证 — `validator.py`

```python
def validate_runtime_request(request: ModelRequest) -> None
# 硬约束：prompt_ids 不为 None（仅在 canonical token 路径上调用）

def validate_runtime_response(response: ModelResponse) -> None
# 硬约束：token_ids 非空，且 len(token_ids) == len(logprobs)
```

---

## Gateway 核心 — `monitor.py`

```python
class ModelMonitor:
    # HTTP server，暴露 /v1/chat/completions
    # MAS 的所有 LLM 请求发往此处，被透明拦截采集
    ModelMonitor(backend, model_mapping, episode_id=None, replay_cache=None, renderer=None)

    async def start(host, port=0) -> int     # 返回实际绑定端口（port=0 自动分配）
    async def stop() -> None
    def get_buffer() -> list[InteractionRecord]
    def clear_buffer() -> None

    # 每次 LLM 请求的内部处理流程：
    # 1. 从 body["model"] 识别 agent_role → 查 model_mapping 路由
    # 2. 若有 renderer：render messages → prompt_ids + fingerprint
    # 3. 若有 replay_cache：先查缓存（tree branch 重放前缀用）
    # 4. cache miss：调用 backend.generate()
    # 5. 若在 canonical 路径：validate_runtime_response
    # 6. 记录 InteractionRecord 入 buffer
    # 7. 返回干净的 OpenAI 格式 response 给 MAS（不含 RL 元数据）
```

---

## MAS 进程管理 — `launcher.py`

```python
class MASLauncher:
    MASLauncher(work_dir=None)

    def prepare_config(config_template, monitor_url, agent_roles) -> Path
    # 1. 将 config_template["llm"]["base_url"] 替换为 Monitor URL
    # 2. 对每个 agent role 设置 agents[role]["model"] = role
    # 3. 写入临时 YAML 文件，返回路径
    # ↑ 这是 MAS 零侵入的实现机制：仅修改配置，MAS 代码不感知

    def launch(command, env_vars=None) -> subprocess.Popen
    # shell=True，在 work_dir 下运行，new session（便于整组 kill）

    def wait(process, timeout=None) -> int   # 超时则 kill 进程组
    def cleanup() -> None                    # 删除临时 YAML 文件
```

---

## Episode 编排 — `pipe.py`

```python
@dataclass
class AgentPipeConfig:
    mas_command_template: str           # 含 {config_path} 和 {prompt} 占位符
    config_template:      dict          # MAS 基础配置（Launcher 会注入 monitor_url）
    model_mapping:        dict[str, ModelMappingEntry]
    timeout:              float = 300.0
    monitor_host:         str = "127.0.0.1"
    monitor_port:         int = 0       # 0 = 自动分配
    mas_work_dir:         str | Path | None
    renderer:             ChatRenderer | None   # 提供时启用 canonical token contract

class AgentPipe:
    AgentPipe(config, backend, replay_cache=None)

    async def run(prompt, reward_provider, allow_partial=False) -> EpisodeResult
    # 完整 episode 生命周期：
    # 1. 启动 ModelMonitor
    # 2. MASLauncher.prepare_config（注入 monitor_url）
    # 3. MASLauncher.launch（拉起 MAS 子进程）
    # 4. MASLauncher.wait（等待 MAS 完成）
    # 5. monitor.get_buffer()
    # 6. 若有 renderer：_validate_canonical_buffer（硬约束检查）
    # 7. TrajectoryCollector.build（buffer → EpisodeTrajectory）
    # 8. RewardWorker.compute（计算 reward）
    # 9. 返回 EpisodeResult
    # allow_partial=True：MAS 非零退出时也返回已采集的 partial 结果

    def last_buffer() -> list[InteractionRecord]   # 上一次 run 的原始 buffer（tree 使用）
```

---

## 并发采集 — `parallel.py`

```python
async def parallel_rollout(
    prompts, reward_provider, config, backend,
    n_samples_per_prompt=1, max_concurrent=None
) -> list[EpisodeResult]
# 对每个 prompt 启动 n_samples_per_prompt 个 AgentPipe
# max_concurrent 用 Semaphore 控制并发上限
# 失败的 episode 被丢弃并 warning，不中断其他任务
```

---

## 树状分支采样 — `tree.py`

```python
async def tree_rollout(
    prompt, reward_provider, config, backend,
    k_branches=3, max_concurrent_branches=None
) -> TreeEpisodeResult
# 1. 跑一次完整 pilot episode
# 2. 以 pilot_buffer 的每个 turn 作为分叉点
# 3. 每个分叉点：创建 ReplayCache（截断到该 turn 之前），跑 k_branches 个 branch episode
# 4. branch 完成后调用 _annotate_branch_result 打 branch semantics
# 5. 返回 TreeEpisodeResult（pilot + 所有成功 branches）

def _annotate_branch_result(result, branch_turn: int) -> None
# 按时序对所有 turns 打标：
#   idx < branch_turn  → replayed=True,  branch_phase="replay_prefix"
#   idx == branch_turn → replayed=False, branch_phase="branch_point"
#   idx > branch_turn  → replayed=False, branch_phase="post_branch"
```

---

## 分支前缀缓存 — `replay_cache.py`

```python
class ReplayCache:
    # 存储 pilot episode 的 response，供 branch episode 直接回放（不重新生成）
    # key = (agent_role, turn_index)，附带 messages_hash 做完整性校验

    ReplayCache.from_buffer(buffer, branch_at_global_position=None) -> ReplayCache
    # 从 pilot buffer 构建，只保留 branch 点之前的 turns（排他）

    def lookup(agent_role, turn_index, messages=None) -> ModelResponse | None
    # messages 非 None 时比对 messages_hash，不一致则 cache miss（防止上下文错位）
    # Monitor 在 cache hit 时直接返回缓存 response，不调用 backend
```

---

## 轨迹组装 — `collector.py`

```python
class TrajectoryCollector:
    def build(buffer: list[InteractionRecord], episode_id: str) -> EpisodeTrajectory
    # 将 Monitor buffer 转成结构化轨迹：
    # 1. 按 agent_role 分组
    # 2. 每组按 turn_index 排序
    # 3. InteractionRecord → TurnData（从 metadata 提取 replayed、branch_phase 等字段）
```

---

## 奖励计算 — `reward.py`

```python
class RewardProvider(Protocol):
    def compute(trajectory: EpisodeTrajectory) -> dict
    # 返回格式：{"agent_rewards": {role: float|list}, "final_reward": float|None}

class FunctionRewardProvider:
    # 用任意 callable 包装成 RewardProvider
    FunctionRewardProvider(func: Callable[[EpisodeTrajectory], dict])

class RewardWorker:
    def compute(trajectory, provider) -> EpisodeResult
    # 调用 provider.compute，做类型/格式校验，返回 EpisodeResult
```

---

## Token Drift 诊断 — `diagnostics.py`

```python
def build_drift_artifact(
    *, messages, runtime_prompt_ids, rerendered_prompt_ids,
    response_ids, response_logprobs, render_fingerprint, sampling_fingerprint=None
) -> dict
# 构建诊断快照，含 mismatch: bool（runtime vs rerendered 是否一致）
# 只记录，不阻塞 runtime；挂在 InteractionRecord.metadata["drift_artifact"] 中
```

---

## 训练侧出口 — `exporters.py`

```python
def export_tokenized_turn(turn: TurnData) -> dict
# 将 TurnData 转成扁平 training-ready dict，明确 token-truth 优先策略：
#   - prompt_ids 为 None → 直接报错（拒绝 re-tokenize fallback）
#   - token_ids 为 None → 直接报错
# 输出字段：agent_role, turn_index, prompt_ids, response_ids,
#           response_logprobs, replayed, branch_phase, routed_experts
```

---

## Demo 可视化 — `display.py`

```python
def format_episode(result: EpisodeResult, *, show_mapping=False) -> str
# 完整展示：Episode Overview + 每个 turn 的 Training-Critical Fields
# show_mapping=True 追加 MATE → VERL 字段映射对照表

def format_tree(result: TreeEpisodeResult, *, expand_pilot=True, show_mapping=False) -> str
# Tree rollout 展示：Overview（pilot + branch 列表）+ 可选 pilot detail

def format_training_mapping() -> str
# 独立的 MATE → VERL 字段映射对照表

# 细粒度 API（可单独调用）：
def format_episode_overview(result: EpisodeResult) -> str
def format_turn_detail(turn: TurnData, index: int, total: int) -> str
def format_tree_overview(result: TreeEpisodeResult) -> str
```

---

## Demo 脚本 — `scripts/demo_search_mas.py`

```bash
# 基本用法
python scripts/demo_search_mas.py \
    --question "What is the capital of France?" \
    --model /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
    --show-mapping

# 保存轨迹 JSON
python scripts/demo_search_mas.py \
    --question "..." \
    --output artifacts/demo_episode.json

# Tree rollout 模式
python scripts/demo_search_mas.py \
    --question "..." --tree --k-branches 2
```

| 参数 | 说明 |
|------|------|
| `--question` | 问题（必填）|
| `--vllm-url` | vLLM 地址，默认 `http://127.0.0.1:8000` |
| `--model` | 模型路径（本地路径时启用 canonical `prompt_ids`，效果最佳）|
| `--config` | 使用已有 MAS config 文件 |
| `--show-mapping` | 展示 MATE → VERL 字段映射表 |
| `--tree` | 启用 tree rollout 模式 |
| `--k-branches` | tree 模式下每个分叉点的分支数，默认 2 |
| `--output` | 保存完整 JSON 轨迹到文件 |

---

## 完整数据流

```
                   ┌─────────────────────────────────────────────────┐
  prompt ──────────▶              AgentPipe.run()                    │
                   │                                                 │
                   │  MASLauncher                                    │
                   │    prepare_config  → 注入 monitor_url 到 YAML   │
                   │    launch          → 启动 MAS 子进程             │
                   │                                                 │
                   │  ModelMonitor  (HTTP /v1/chat/completions)      │
                   │    ① 识别 agent_role（body["model"]）            │
                   │    ② ChatRenderer → prompt_ids + fingerprint    │
                   │    ③ ReplayCache.lookup（tree branch 重放）      │
                   │    ④ backend.generate(ModelRequest)             │
                   │    ⑤ validate_runtime_response（硬约束）         │
                   │    ⑥ 记录 InteractionRecord 入 buffer           │
                   │    ⑦ 返回干净 response 给 MAS                   │
                   │                                                 │
                   │  MASLauncher.wait()  → 等待子进程退出            │
                   │                                                 │
                   │  TrajectoryCollector.build(buffer)              │
                   │    InteractionRecord → TurnData（按 role 分组）  │
                   │                                                 │
                   │  RewardWorker.compute(trajectory)               │
                   └────────────────────────┬────────────────────────┘
                                            │
                                     EpisodeResult
                                     ├─ trajectory
                                     │   └─ agent_trajectories
                                     │       ├─ verifier: [TurnData, ...]
                                     │       ├─ searcher: [TurnData, ...]
                                     │       └─ answerer: [TurnData, ...]
                                     ├─ rewards: {role: float}
                                     └─ final_reward: float
                                            │
                              ┌─────────────▼──────────────┐
                              │   export_tokenized_turn()   │
                              │   mate_dataproto_adapter    │  ← OrchRL 侧适配层
                              └─────────────┬──────────────┘
                                            │
                                      VERL DataProto
                                      ├─ prompts       (prompt_ids, left-padded)
                                      ├─ responses     (token_ids, right-padded)
                                      ├─ response_mask
                                      └─ non_tensors:
                                          agent_name, reward, uid,
                                          turn_idx, env_idx, episode_id
```

---

## MATE → VERL 字段映射

| MATE 字段 | VERL 训练侧消费 | 用途 |
|-----------|---------------|------|
| `prompt_ids` | `DataProto.prompts`（无需 re-tokenize）| PPO/GRPO prompt 输入 |
| `token_ids` | `DataProto.responses` | response token 序列 |
| `logprobs` | `DataProto.old_log_probs` | importance sampling ratio |
| `agent_role` | per-policy batch 分组 | 多 agent 路由 |
| `turn_index` | per-turn record 对齐 | 多轮 credit assignment |
| `finish_reason` | 截断/完成标志 | 样本质量控制 |
| `replayed` | tree branch skip predicate | 跳过重放前缀，避免重复训练 |
| `branch_phase` | `branch_point` / `post_branch` | 结构化 credit assignment 分组 |
