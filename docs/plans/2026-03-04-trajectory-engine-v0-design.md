# Agent Trajectory Engine V0 设计文档

> 日期：2026-03-04
> 状态：设计确认
> 范围：轨迹采集模块（AgentPipe），不含训练侧调度
> 前序文档：`2026-03-02-marl-grpo-v0-directions.md`
> 团队架构参考：`/home/cxb/multi-agent/docs/multi-agent-rl.pdf`（第 1-3、7 页）

---

## 1. V0 目标与边界

### 1.1 交付目标

构建 AgentPipe 轨迹采集模块，实现非侵入式多智能体轨迹采集，以 DrMAS Search 场景（OrchRL 抽取版）为首个端到端验证目标。

### 1.2 V0 范围

| 包含 | 不包含 |
|------|--------|
| ModelMonitor（策略模式 + VLLMBackend） | 训练侧调度（同事负责） |
| MASLauncher（进程模式） | 树状分支采样（V0.2+） |
| TrajectoryCollector（按 agent 分组输出） | Trie 前缀树存储（V0.2+） |
| RewardWorker（函数式 reward） | Reward Model（V1+） |
| Episode 并行采样 | 轨迹级训推异步（V1+） |
| OrchRL Search MAS 端到端验证 | Multi-LoRA（V1+） |

### 1.3 验证场景

DrMAS Search（OrchRL `examples/mas_app/search/`）：

- 三个 Agent：Verifier（路由）、Searcher、Answerer
- 编排流程：Verifier 判断信息充分性 → 不充分则 Search → 充分则 Answer
- 含工具调用（HTTP 检索服务）
- 所有 Agent 通过 OpenAI 兼容接口调用 LLM

### 1.4 关键设计决策记录

| 决策项 | 结论 | 理由 |
|--------|------|------|
| Agent-模型关系 | Role-specific（每个 agent 角色映射到独立模型后端） | 必须从一开始支持多模型，测试可共享模型 |
| Agent 身份识别 | `model` 字段即 agent role | MAS 零代码改动，只改配置 |
| 轨迹输出粒度 | `Dict[agent_id, List[TurnData]]` | 训练侧按需 group_by，采集侧不承担分组策略 |
| MAS 启动模式 | 进程模式（subprocess） | 天然 episode 隔离，服务模式后续可加 adapter |
| Monitor 部署拓扑 | 每 episode 独立实例 | 与进程模式配合，天然隔离 |
| Monitor 推理模式 | 策略模式（InferenceBackend 接口），V0 实现 VLLMBackend，训练时替换为 VerlBackend | 单一代码路径，更简单的错误处理，天然支持并行 agent 并发请求 |

---

## 2. 整体架构

```
AgentPipe (顶层编排，对标 Verl AgentLoop)
│
│  run(prompt, reward_func) → EpisodeResult
│
├── ModelMonitor      (数据采集层)
├── MASLauncher       (MAS 生命周期管理)
├── RewardWorker      (奖励计算)
└── TrajectoryCollector (数据组装)
```

### 组件职责边界

| 组件 | 输入 | 输出 | 职责 | 不做什么 |
|------|------|------|------|---------|
| ModelMonitor | MAS 的 HTTP 请求 | InteractionRecord 列表 | 拦截 LLM 调用、委托给 InferenceBackend 推理、采集原始数据 | 不做数据组装、不算 reward |
| MASLauncher | MAS 配置模板 + 启动命令 | 子进程退出码 | 准备配置、启动/等待 MAS 进程 | 不关心轨迹数据 |
| RewardWorker | 组装后的轨迹 + reward provider | 带 reward 的轨迹 | 调用用户定义的 reward 函数/模型 | 不做 advantage 计算 |
| TrajectoryCollector | Monitor 的原始 buffer | EpisodeTrajectory | 按 agent_id 分组、组装结构化数据 | 不做分组策略 |
| AgentPipe | 用户调用参数 | EpisodeResult | 编排以上四个组件的执行顺序 | 不实现具体的采集/计算逻辑 |

### 一次 Episode 的执行流程

```
1. AgentPipe 初始化 Monitor（绑定端口，注入 InferenceBackend）
   - 测试模式：注入 VLLMBackend（直接转发到 vLLM 服务）
   - 训练模式：注入 VerlBackend（调用 AsyncLLMServerManager.generate）
2. MASLauncher 准备配置（base_url → Monitor 地址，model → agent role 名）
3. MASLauncher 拉起 MAS 子进程
4. MAS 执行中：
   Agent 发 LLM 请求 → Monitor HTTP Server 接收
   → 解析 agent_role → await backend.generate(request)
   → 提取 token_ids, logprobs → 记入 buffer
   → 返回干净的 OpenAI 格式响应给 Agent
5. MAS 进程结束
6. TrajectoryCollector 从 buffer 组装结构化轨迹
7. RewardWorker 计算奖励
8. AgentPipe 返回 EpisodeResult
```

---

## 3. ModelMonitor 详细设计

### 3.1 核心机制：策略模式（InferenceBackend）

Monitor 通过 `InferenceBackend` 抽象接口将推理委托给可替换的后端实现。HTTP handler 收到请求后直接 `await backend.generate(request)`，无队列、无间接通信。不同场景通过注入不同的 Backend 实现来切换行为。

```
MAS Agent
  │ POST /v1/chat/completions
  ▼
ModelMonitor HTTP Server
  │ 解析 model → agent_role
  │ 查 model_mapping 获取路由信息
  │ await backend.generate(request)
  │   ├── VLLMBackend: HTTP 转发到 vLLM 服务
  │   └── VerlBackend: 调用 AsyncLLMServerManager.generate()
  │ 提取 token_ids, logprobs → 记入 buffer
  │ 返回干净的 OpenAI 格式响应给 MAS
  ▼
MAS Agent 收到响应，继续执行
```

#### 设计决策说明

曾考虑的替代方案是 anti-call 队列模式（参考 SWE-agent recipe 的 ModelProxy）：HTTP handler 将请求放入 `asyncio.Queue`，由外部消费者拉取处理。经客观评估后放弃，原因：

1. **策略模式以更低复杂度实现了同样的"统一代码路径"**——Monitor 内部始终是 `await backend.generate()`，无需管理队列、事件、消费者生命周期
2. **队列引入了不必要的故障模式**——消费者静默崩溃会导致请求永久挂起（死锁），策略模式下后端异常直接在 handler 的 try/except 中捕获
3. **顺序及并发请求场景均无需队列**——单 episode 内 LLM 调用为顺序或有限并发（aiohttp 天然处理），队列的核心价值（速率解耦、缓冲、批处理）不适用
4. **Verl 的 `AsyncLLMServerManager.generate()` 是标准 async 方法**——天然适配策略模式的 `await backend.generate()` 调用，无需队列适配

### 3.2 InferenceBackend 接口

```python
class InferenceBackend(ABC):
    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """执行推理，返回包含 token_ids/logprobs 的完整响应。"""
        ...

class VLLMBackend(InferenceBackend):
    """测试/离线模式：HTTP 转发到独立 vLLM/SGLang 服务。"""
    def __init__(self, backend_url: str, actual_model: str | None = None, timeout: float = 120.0):
        ...

    async def generate(self, request: ModelRequest) -> ModelResponse:
        # 构造转发请求，注入 logprobs=True
        # HTTP POST 到 backend_url/v1/chat/completions
        # 提取 content, token_ids, logprobs, finish_reason
        ...

class VerlBackend(InferenceBackend):
    """训练模式：调用 Verl AsyncLLMServerManager，保障 token-in-token-out。"""
    def __init__(self, server_manager: AsyncLLMServerManager, tokenizer):
        ...

    async def generate(self, request: ModelRequest) -> ModelResponse:
        prompt_ids = self.tokenizer.apply_chat_template(request.messages)
        output = await self.server_manager.generate(
            request_id=str(uuid.uuid4()),
            prompt_ids=prompt_ids,
            sampling_params=request.generation_params,
        )
        return ModelResponse(
            content=self.tokenizer.decode(output.token_ids),
            token_ids=output.token_ids,
            logprobs=output.log_probs,
            finish_reason=output.stop_reason,
        )
```

### 3.3 ModelMonitor 接口

```python
class ModelMonitor:
    def __init__(self, backend: InferenceBackend, model_mapping: Dict[str, ModelMappingEntry]):
        """
        backend: 推理后端实现
        model_mapping: agent_role → ModelMappingEntry(actual_model, backend_config)
        """

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """启动 aiohttp 服务，返回实际端口。port=0 由 OS 分配。"""

    async def stop(self):
        """关闭 HTTP 服务。"""

    def get_buffer(self) -> List[InteractionRecord]:
        """返回已采集的全部交互记录。"""

    def clear_buffer(self):
        """清空 buffer。"""
```

Monitor 不再暴露 `get_request()` / `send_response()` 消费者接口——推理完全由注入的 Backend 处理。

### 3.4 Token-in-token-out 保障

- **训练模式（VerlBackend）**：在 Backend 内部完成 tokenize → `server_manager.generate(prompt_ids)` → 拿到原始 token_ids + logprobs，全程无 detokenize-retokenize 环节
- **测试模式（VLLMBackend）**：通过 OpenAI API 通信，需配置 vLLM 返回 token_ids（vLLM 扩展参数），主要用于流程验证
- **跨 Agent 上下文传递**：Agent 1 的 response 文本进入 Agent 2 的 prompt，属于 prompt 侧（非训练 token），不影响各 agent 自身 response 的 token-in-token-out

### 3.5 数据结构

```python
@dataclass
class ModelMappingEntry:
    actual_model: str | None    # 实际模型名（None 则保持原 model 字段）
    backend_url: str | None     # VLLMBackend 专用：该 role 对应的后端地址（None 则使用默认）

@dataclass
class ModelRequest:
    request_id: str
    agent_role: str             # 从 request body 的 model 字段解析
    messages: List[Dict]
    generation_params: Dict     # temperature, top_p, max_tokens 等

@dataclass
class ModelResponse:
    content: str
    token_ids: List[int] | None
    logprobs: List[float] | None
    finish_reason: str

@dataclass
class InteractionRecord:
    agent_role: str
    turn_index: int             # Monitor 内部按 agent_role 自增
    timestamp: float
    messages: List[Dict]
    generation_params: Dict
    response_text: str
    token_ids: List[int] | None
    logprobs: List[float] | None
    finish_reason: str
    episode_id: str
    metadata: Dict[str, Any]
```

---

## 4. MASLauncher 详细设计

### 4.1 接口

```python
class MASLauncher:
    def prepare_config(
        self,
        config_template: dict,
        monitor_url: str,
        model_mapping: Dict[str, str],
    ) -> Path:
        """
        修改配置模板：
        1. llm.base_url → monitor_url
        2. agents.<role>.model → role 名
        写出临时配置文件，返回路径。
        """

    async def launch(
        self,
        command: str,
        config_path: Path,
        prompt: str,
        env_vars: Dict[str, str] | None = None,
    ) -> asyncio.subprocess.Process:
        """启动 MAS 子进程。command 中 {config_path}, {prompt} 为占位符。"""

    async def wait(self, process, timeout: float | None = None) -> int:
        """等待子进程结束。超时则 kill，返回 exit_code。"""
```

### 4.2 配置准备示例（OrchRL Search MAS）

输入模板：
```yaml
llm:
  base_url: http://127.0.0.1:8000/v1
  model: Qwen3-4B-Instruct-2507
agents:
  verifier: { temperature: 0.2 }
  searcher: { temperature: 0.6 }
  answerer: { temperature: 0.4 }
```

处理后输出：
```yaml
llm:
  base_url: http://127.0.0.1:19521/v1    # Monitor 地址
  model: default
agents:
  verifier: { temperature: 0.2, model: verifier }
  searcher: { temperature: 0.6, model: searcher }
  answerer: { temperature: 0.4, model: answerer }
```

### 4.3 启动命令约定

```python
mas_config = {
    "command": "python -m search_mas.scripts.run_search_mas --config {config_path} --question {prompt}",
    "config_template_path": "configs/search_mas_example.yaml",
    "working_dir": "/home/cxb/OrchRL/examples/mas_app/search",
    "timeout": 300,
}
```

---

## 5. TrajectoryCollector 详细设计

### 5.1 输出数据结构

```python
@dataclass
class TurnData:
    agent_role: str
    turn_index: int
    messages: List[Dict]         # 该轮完整 prompt messages
    response_text: str
    token_ids: List[int] | None
    logprobs: List[float] | None
    finish_reason: str
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class EpisodeTrajectory:
    episode_id: str
    agent_trajectories: Dict[str, List[TurnData]]   # agent_role → turns
    metadata: Dict[str, Any]     # episode 级元数据
```

### 5.2 组装逻辑

```python
class TrajectoryCollector:
    def build(self, buffer: List[InteractionRecord], episode_id: str) -> EpisodeTrajectory:
        """
        1. 按 agent_role 分组
        2. 每组内按 turn_index 排序
        3. InteractionRecord → TurnData
        4. 组装 EpisodeTrajectory
        """
```

### 5.3 元数据策略

每个 TurnData 携带完整元数据（episode_id, agent_role, turn_index, timestamp），使训练侧能按任意维度构建分组视图（group_by agent_id / turn_index / episode_id），无需采集侧参与分组策略。自定义 key 视图（如 PPT 第 7 页的 "E1A1B1"）由训练侧按需实现。

---

## 6. RewardWorker 详细设计

### 6.1 接口

```python
class RewardProvider(Protocol):
    def compute(self, trajectory: EpisodeTrajectory) -> Dict[str, Any]: ...

class FunctionRewardProvider:
    """V0 实现：包装用户定义的 reward 函数。"""
    def __init__(self, func: Callable[[EpisodeTrajectory], Dict]):
        self.func = func
    def compute(self, trajectory: EpisodeTrajectory) -> Dict[str, Any]:
        return self.func(trajectory)

# 预留：ModelRewardProvider（V1+，调用 reward model 服务）

@dataclass
class EpisodeResult:
    trajectory: EpisodeTrajectory
    rewards: Dict[str, float | List[float]]   # agent_role → reward(s)
    final_reward: float | None
    metadata: Dict[str, Any]

class RewardWorker:
    def compute(
        self, trajectory: EpisodeTrajectory, provider: RewardProvider,
    ) -> EpisodeResult:
        """调用 provider，组装 EpisodeResult。"""
```

### 6.2 Reward 粒度兼容

- `rewards` 字段类型为 `Dict[str, float | List[float]]`
- V0（DrMAS）：每个 agent 一个 float（episode-level reward）
- 后续（per-turn reward）：每个 agent 一个 `List[float]`，与 turn 数对齐
- 无需改动数据结构即可支持分步 reward

### 6.3 reward_func 示例（DrMAS Search）

```python
def search_mas_reward(trajectory: EpisodeTrajectory) -> Dict:
    answer_turns = trajectory.agent_trajectories.get("answerer", [])
    if not answer_turns:
        return {"agent_rewards": {}, "final_reward": 0.0}
    predicted = extract_answer(answer_turns[-1].response_text)
    correct = is_search_answer_correct(predicted, expected_answer)
    final_reward = 1.0 if correct else 0.0
    agent_rewards = {role: final_reward for role in trajectory.agent_trajectories}
    return {"agent_rewards": agent_rewards, "final_reward": final_reward}
```

---

## 7. AgentPipe 编排

### 7.1 完整流程

```python
class AgentPipe:
    def __init__(self, config: AgentPipeConfig, backend: InferenceBackend):
        self.monitor = ModelMonitor(backend, config.model_mapping)
        self.launcher = MASLauncher()
        self.collector = TrajectoryCollector()
        self.reward_worker = RewardWorker()

    async def run(self, prompt: str, reward_provider: RewardProvider) -> EpisodeResult:
        episode_id = generate_episode_id()

        # 1. 启动 Monitor
        port = await self.monitor.start()
        monitor_url = f"http://127.0.0.1:{port}/v1"

        # 2. 准备 MAS 配置并启动
        config_path = self.launcher.prepare_config(
            self.config.mas_config_template, monitor_url, self.config.model_mapping
        )
        process = await self.launcher.launch(
            self.config.mas_command, config_path, prompt
        )

        # 3. 等待 MAS 完成
        exit_code = await self.launcher.wait(process, self.config.timeout)

        # 4. 组装轨迹
        trajectory = self.collector.build(self.monitor.get_buffer(), episode_id)

        # 5. 计算 reward
        result = self.reward_worker.compute(trajectory, reward_provider)

        # 6. 清理
        await self.monitor.stop()
        return result
```

### 7.2 Episode 并行采样

```python
async def parallel_rollout(prompts, reward_provider, config, backend, n_parallel):
    pipes = [AgentPipe(config, backend) for _ in range(n_parallel)]
    results = await asyncio.gather(*[
        pipe.run(prompt, reward_provider)
        for pipe, prompt in zip(pipes, prompts)
    ])
    return results   # List[EpisodeResult]
```

每个 AgentPipe 实例拥有独立的 Monitor（独立端口）和 MAS 进程，天然隔离。多个 Monitor 的并发推理请求由 InferenceBackend 处理——VLLMBackend 依赖 vLLM 的 continuous batching，VerlBackend 依赖 AsyncLLMServerManager 的负载均衡。

---

## 8. 与训练侧的对接约定

### 8.1 数据接口

轨迹采集模块输出 `EpisodeResult`，训练侧从中读取：

- `trajectory.agent_trajectories[agent_role]`：该 agent 所有 turn 的 `TurnData`
- `rewards[agent_role]`：该 agent 的 reward

训练侧负责：
- 按需构建分组视图（group_by agent_id / turn_index）
- advantage 计算（DrMAS agent-wise normalization 等）
- 构建训练 batch

### 8.2 训练集成

训练时注入 `VerlBackend`，直接利用 Verl 的 `AsyncLLMServerManager`：

```python
# 训练模式初始化
backend = VerlBackend(server_manager=verl_server_manager, tokenizer=tokenizer)
pipe = AgentPipe(config, backend=backend)
result = await pipe.run(prompt, reward_provider)
```

调用链路：
```
Monitor.handle_request()
  → VerlBackend.generate()
    → tokenizer.apply_chat_template(messages) → prompt_ids
    → server_manager.generate(prompt_ids=prompt_ids) → TokenOutput
    → ModelResponse(token_ids, logprobs, ...)
```

Token-in-token-out 在此链路中得到完全保障：VerlBackend 内部完成 tokenize，`server_manager.generate()` 以 token ID 为输入输出，全程无 detokenize-retokenize 环节。同时自动享受 AsyncLLMServerManager 的负载均衡和粘性会话（prefix caching）能力。

---

## 附录

### A. OrchRL Search MAS 适配清单

对 OrchRL 代码零改动，仅通过配置变更接入：

1. `llm.base_url` → Monitor 地址
2. `agents.<role>.model` → role 名（verifier / searcher / answerer）
3. MASLauncher 的 `prepare_config` 自动完成以上两步

### B. 后续演进路径

| 阶段 | 功能 |
|------|------|
| V0（本设计） | AgentPipe 全链路 + DrMAS Search 验证 |
| V0.2 | 树状分支采样 + Branch Coordinator + Trie 存储 |
| V1 | 轨迹级训推异步 + Relay Worker + 长尾迁移 |
| V1+ | Reward Model + Multi-LoRA + 服务模式 MAS adapter |
