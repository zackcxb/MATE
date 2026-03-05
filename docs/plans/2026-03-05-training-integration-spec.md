# 训练侧对接规格文档

> 日期：2026-03-05
> 状态：待训练侧确认
> 前序文档：`2026-03-04-trajectory-engine-v0-design.md`
> 验证产物：`artifacts/real_validation_realmode_full.json`

---

## 1. 数据接口规格

### 1.1 核心数据结构

AgentPipe 输出 `EpisodeResult`，训练侧从中读取所有所需数据：

```python
@dataclass
class EpisodeResult:
    trajectory: EpisodeTrajectory
    rewards: dict[str, float | list[float]]   # agent_role → reward(s)
    final_reward: float | None
    metadata: dict[str, Any]

@dataclass
class EpisodeTrajectory:
    episode_id: str                            # 全局唯一 hex UUID
    agent_trajectories: dict[str, list[TurnData]]  # agent_role → turns
    metadata: dict[str, Any]

@dataclass
class TurnData:
    agent_role: str           # "verifier" / "searcher" / "answerer"
    turn_index: int           # 该 role 内从 0 自增
    messages: list[dict]      # 该轮完整 prompt messages（OpenAI 格式）
    response_text: str        # 模型生成的文本
    token_ids: list[int] | None     # 生成 token 的 ID 列表
    logprobs: list[float] | None    # 对应 token 的 log probability
    finish_reason: str        # "stop" / "length"
    timestamp: float          # UNIX timestamp
    metadata: dict[str, Any]  # 包含 episode_id, agent_role, turn_index, timestamp
```

### 1.2 字段说明

| 字段 | 类型 | 训练侧用途 |
|------|------|-----------|
| `token_ids` | `list[int]` | 训练 batch 的 response token 序列 |
| `logprobs` | `list[float]` | 用于 PPO/GRPO 的 old log_prob（behavior policy） |
| `messages` | `list[dict]` | 训练 batch 的 prompt 部分（apply_chat_template 后可得 prompt_ids） |
| `agent_role` | `str` | 用于 group_by_agent_id 分组、agent-wise advantage normalization |
| `turn_index` | `int` | 用于 multi-turn 序列拼接顺序 |
| `rewards[agent_role]` | `float` 或 `list[float]` | episode-level 或 per-turn reward |
| `final_reward` | `float \| None` | episode 总体 reward（如需全局信号） |

### 1.3 数据保证

以下属性已通过真实环境验证确认（10 episodes, 90 turns, vLLM real mode）：

1. **token_ids 非空**：所有 turn 的 `token_ids` 均不为 None
2. **长度一致**：`len(token_ids) == len(logprobs)` 对所有 turn 成立
3. **episode_id 唯一**：每个 episode 具有全局唯一 ID
4. **角色完整**：DrMAS 场景下 verifier/searcher/answerer 三角色均有数据
5. **turn 有序**：同一 agent_role 内 turn_index 单调递增

### 1.4 真实数据样本（脱敏）

以下为一个完整 episode 的结构示例（token/logprobs 已截断）：

```json
{
  "trajectory": {
    "episode_id": "93b15649d4454856b890c29d938a12bf",
    "agent_trajectories": {
      "verifier": [
        {
          "agent_role": "verifier",
          "turn_index": 0,
          "messages": [
            {"role": "user", "content": "# Task Introduction\nwho got the first nobel prize in physics?\n..."}
          ],
          "response_text": "<verify>no</verify>",
          "token_ids": [27, 12446, 29, 2533, 5765, 29, 6427, 151645],
          "logprobs": [-3.57e-07, -2.38e-06, -1.01e-05, -0.000149, -5.96e-06, -0.281, -0.0039, -2.86e-05],
          "finish_reason": "stop",
          "timestamp": 1772703387.388,
          "metadata": {"episode_id": "93b1...", "agent_role": "verifier", "turn_index": 0}
        }
      ],
      "searcher": [
        {
          "agent_role": "searcher",
          "turn_index": 0,
          "messages": [{"role": "user", "content": "..."}],
          "response_text": "<search_query>who got the first Nobel Prize in Physics</search_query>...",
          "token_ids": [27, 2095, 9498, 29, ...],
          "logprobs": [-3.57e-07, -1.70e-05, -0.00107, ...],
          "finish_reason": "stop",
          "timestamp": 1772703387.526
        }
      ],
      "answerer": [
        {
          "agent_role": "answerer",
          "turn_index": 0,
          "messages": [{"role": "user", "content": "..."}],
          "response_text": "Based on the information provided...\n<answer>Wilhelm Conrad Röntgen</answer>",
          "token_ids": [29317, 389, 279, ...],
          "logprobs": [-0.0098, -0.0156, -0.0635, ...],
          "finish_reason": "stop",
          "timestamp": 1772703391.234
        }
      ]
    },
    "metadata": {}
  },
  "rewards": {
    "verifier": 0.0,
    "searcher": 0.0,
    "answerer": 0.0
  },
  "final_reward": 0.0,
  "metadata": {"exit_code": 0}
}
```

---

## 2. VerlBackend 实现规格

### 2.1 接口契约

```python
from mate.trajectory import InferenceBackend, ModelRequest, ModelResponse

class VerlBackend(InferenceBackend):
    """训练模式后端，对接 Verl AsyncLLMServerManager。"""

    def __init__(self, server_manager: AsyncLLMServerManager, tokenizer):
        self.server_manager = server_manager
        self.tokenizer = tokenizer

    async def generate(self, request: ModelRequest) -> ModelResponse:
        # 1. messages → prompt_ids
        prompt_ids = self.tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
        )

        # 2. 准备 sampling_params
        sampling_params = {
            "temperature": request.generation_params.get("temperature", 0.6),
            "top_p": request.generation_params.get("top_p", 0.95),
            "max_tokens": request.generation_params.get("max_tokens", 1536),
            "logprobs": True,
        }

        # 3. 调用 server_manager.generate
        output: TokenOutput = await self.server_manager.generate(
            request_id=request.request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

        # 4. 组装 ModelResponse
        return ModelResponse(
            content=self.tokenizer.decode(output.token_ids, skip_special_tokens=True),
            token_ids=output.token_ids,
            logprobs=output.log_probs,
            finish_reason=output.stop_reason or "stop",
        )
```

### 2.2 Token-in-token-out 保障

```
Monitor.handle_request(http_json)
  → 解析 model → agent_role
  → 构造 ModelRequest(messages, generation_params)
  → VerlBackend.generate(request)
    → tokenizer.apply_chat_template(messages) → prompt_ids: list[int]
    → server_manager.generate(prompt_ids=prompt_ids) → TokenOutput
    → ModelResponse(token_ids=output.token_ids, logprobs=output.log_probs)
  → Monitor 记录 InteractionRecord(token_ids, logprobs)
  → 返回 OpenAI 格式 HTTP 响应给 MAS agent
```

全程无 detokenize → retokenize 环节。`token_ids` 和 `logprobs` 直接来自 Verl 推理引擎，训练侧可直接使用。

### 2.3 Verl 接口参考

VerlBackend 依赖 Verl 的以下接口（来自 `third_party/verl/verl/experimental/agent_loop/agent_loop.py`）：

```python
class AsyncLLMServerManager:
    def __init__(self, config, server_handles, max_cache_size=10000): ...

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput: ...

class TokenOutput(BaseModel):
    token_ids: list[int]
    log_probs: Optional[list[float]] = None
    stop_reason: Optional[str] = None
    num_preempted: Optional[int] = None
```

---

## 3. 需要训练侧确认的问题

| # | 问题 | 上下文 |
|---|------|--------|
| Q1 | `AsyncLLMServerManager` 实例如何传给 AgentPipe？ | 当前设计是构造 `VerlBackend(server_manager, tokenizer)` 后注入 `AgentPipe(config, backend=verl_backend)`。训练主循环是否方便获取 server_manager 引用？ |
| Q2 | 多模型场景下，每个 agent_role 对应不同的 server_manager，还是同一个 manager 管理多个模型？ | V0 DrMAS 三个 agent 共享同一模型，但架构支持 per-role 不同模型。如需多模型，VerlBackend 需扩展为 per-role backend 路由。 |
| Q3 | 训练侧需要从 TurnData 中取哪些字段构建训练 batch？ | 采集侧已提供：`token_ids`（response）, `logprobs`（old policy log_prob）, `messages`（可 tokenize 为 prompt_ids）。是否还需要其他字段？ |
| Q4 | DrMAS `group_by_agent_id` 所需的 uid 格式是什么？ | 当前 `episode_id` 为 hex UUID。PPT 第 7 页提到 "E1A1B1" 格式——此 uid 应由训练侧按 `episode_id + agent_role + turn_index` 拼接生成，还是采集侧预生成？ |
| Q5 | Reward 粒度：直接用 `rewards[agent_role]`（float），还是需要 token-level reward？ | 当前 `rewards` 支持 `float`（episode-level）和 `list[float]`（per-turn）。GRPO 通常使用 episode-level reward，是否需要更细粒度？ |
| Q6 | `parallel_rollout()` 的调用方式：训练主循环直接 `await parallel_rollout(...)`，还是通过 Ray actor 封装？ | 当前 `parallel_rollout` 是纯 asyncio 函数。如需 Ray 调度，需加一层 actor wrapper。 |

---

## 4. 联调计划

### Phase 1: 单 episode 数据流通验证

**目标**：确认 VerlBackend → AgentPipe → EpisodeResult 数据链路完整。

```python
# 训练侧代码
from mate.trajectory import AgentPipe, AgentPipeConfig, ModelMappingEntry

backend = VerlBackend(server_manager=verl_sm, tokenizer=tokenizer)
config = AgentPipeConfig(
    mas_command_template="python -m search_mas.scripts.run_search_mas --config {config_path} --question {prompt}",
    config_template=yaml.safe_load(open("configs/search_mas_example.yaml")),
    model_mapping={
        "verifier": ModelMappingEntry(actual_model=None),
        "searcher": ModelMappingEntry(actual_model=None),
        "answerer": ModelMappingEntry(actual_model=None),
    },
    mas_work_dir="/home/cxb/OrchRL/examples/mas_app/search",
)
pipe = AgentPipe(config, backend=backend)
result = await pipe.run(prompt="who got the first nobel prize in physics?", reward_provider=reward_fn)

# 验证点
assert all(t.token_ids is not None for turns in result.trajectory.agent_trajectories.values() for t in turns)
assert result.final_reward is not None
```

**验收标准**：
- `token_ids` 全部非空，与 logprobs 等长
- `finish_reason` 合理（"stop" 或 "length"）
- reward 计算正确

### Phase 2: N episode 并行 + agent-wise advantage

**目标**：验证 `parallel_rollout` 产出的 batch 能正确输入 GRPO 训练。

```python
from mate.trajectory import parallel_rollout

results = await parallel_rollout(
    prompts=prompts[:8],
    reward_provider=reward_fn,
    config=config,
    backend=backend,
    n_samples_per_prompt=4,
    max_concurrent=4,
)

# 训练侧按 agent_role 分组
for role in ["verifier", "searcher", "answerer"]:
    role_rewards = [r.rewards[role] for r in results]
    advantage = normalize(role_rewards)  # agent-wise normalization
    role_token_ids = [
        turn.token_ids
        for r in results
        for turn in r.trajectory.agent_trajectories[role]
    ]
    # 构建训练 batch...
```

**验收标准**：
- 32 个 episode（8 prompts × 4 samples）全部成功
- agent-wise advantage 计算无 NaN/Inf
- 训练 batch shape 正确

### Phase 3: 完整训练循环

**目标**：验证端到端训练循环，包含权重同步。

流程：
1. Verl 初始化 → 创建 `AsyncLLMServerManager`
2. 构建 `VerlBackend` → 注入 `AgentPipe`
3. `parallel_rollout()` 采集 N 条轨迹
4. 训练侧构建 batch → PPO/GRPO 更新
5. 权重同步回推理引擎
6. 重复 3-5

**验收标准**：
- 训练 loss 正常下降（至少不发散）
- 权重同步后新一轮采集的 logprobs 与旧轮不同
- 无内存泄漏（Monitor/Launcher 资源正确释放）

---

## 5. API 快速参考

### 5.1 安装

```bash
cd /home/cxb/MATE-reboot && pip install -e .
```

### 5.2 核心 import

```python
from mate.trajectory import (
    AgentPipe,
    AgentPipeConfig,
    ModelMappingEntry,
    InferenceBackend,
    ModelRequest,
    ModelResponse,
    EpisodeResult,
    EpisodeTrajectory,
    TurnData,
    FunctionRewardProvider,
    parallel_rollout,
)
```

### 5.3 RewardProvider 协议

```python
class RewardProvider(Protocol):
    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]:
        """
        返回值必须包含：
        - "agent_rewards": dict[str, float | list[float]]  # 每个 agent_role 的 reward
        - "final_reward": float | None                      # episode 总体 reward
        """
        ...
```

### 5.4 配置示例（DrMAS Search）

```python
config = AgentPipeConfig(
    mas_command_template=(
        "python -m search_mas.scripts.run_search_mas "
        "--config {config_path} --question {prompt}"
    ),
    config_template={
        "llm": {"base_url": "http://placeholder", "model": "placeholder"},
        "agents": {
            "verifier": {"temperature": 0.2},
            "searcher": {"temperature": 0.6},
            "answerer": {"temperature": 0.4},
        },
        "search": {"provider": "disabled"},
    },
    model_mapping={
        "verifier": ModelMappingEntry(actual_model="Qwen3-4B-Instruct-2507"),
        "searcher": ModelMappingEntry(actual_model="Qwen3-4B-Instruct-2507"),
        "answerer": ModelMappingEntry(actual_model="Qwen3-4B-Instruct-2507"),
    },
    timeout=300.0,
    mas_work_dir="/home/cxb/OrchRL/examples/mas_app/search",
)
```
