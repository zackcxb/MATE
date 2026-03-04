# MARL GRPO 框架设计——外接 MAS + Agent Pipe + Model Monitor + Trie Trajectory Store

> 日期：2026-02-25
> 状态：设计阶段，基于前期 brainstorming
> 更新：2026-03-03 V0 范围重置（Verl-first trajectory engine）
> 前序文档：`2026-02-25-marl-framework-design-final.md`、`marl_grpo_brainstorm.md`
> 团队：6 人（均具备 Coding Agent 开发经验）

---

## 1. 工作重心

经过前期 brainstorming，当前工作重心收敛如下：

1. **以Verl为基线框架**，V0将Verl作为submodule，在其基础上进行开发。但是轨迹采集的能力不可与Verl强耦合，为后续支持Slime留下空间。
2. **优先支持 GRPO 类算法**，V0 主线为 episode 并行采样，树状分支采样在 V0.2+ 推进。
3. **外接 MAS + Agent Pipe + Model Monitor** 作为 rollout 层核心架构，在协议层面解耦轨迹收集与 MAS 编排。通过SWE agent recipe中已经验证的模式达成无侵入式轨迹采集。目标既能适配外部MAS框架（如LangGraph、AutoGen），也能适配自编写的采样拓扑（如DrMAS、PettingLLM）。

---

## 2. MARL GRPO 算法分类

以**采样策略**为分类轴：

### 范式一：Episode 并行采样 + 后处理式 credit assignment

| 方法 | 采样方式 | Credit Assignment |
|------|---------|-------------------|
| DrMAS | N 条完整 episode 并行 | Agent-wise normalization（按 agent_id 分组归一化 advantage） |
| GiGPO | N 条完整 episode 并行 | Anchor state grouping（跨 trajectory 匹配相似状态，step-level 对比） |
| MAGRPO | N 条完整 episode 并行 | Agent + turn-wise 分组（不做树状分支） |

**共同特点**：采样阶段不做分支，完整跑完 episode 后在 advantage 计算阶段做细粒度归因。

**框架需求**：高效的 episode 并行采样、可插拔的 advantage 后处理器。

### 范式二：树状/分支采样 + 结构化 credit assignment

| 方法 | 采样方式 | Credit Assignment |
|------|---------|-------------------|
| AtGRPO（StrongerMAS） | 按 agent 和 turn 分支，形成树状 trajectory | Turn-wise advantage，同前缀不同分支直接对比 |
| 朴素分叉变体 | 每个 agent/round 决策点 N 次分支 | 同上下文下分支做 group 对比 |

**共同特点**：采样阶段制造结构化对比组，advantage 计算天然具备 per-step 粒度。

**框架需求**：分支点管理、前缀共享、树状 trajectory 存储。

### 范式三：层级式解耦训练（V1+ 范围）

| 方法 | 采样方式 | Credit Assignment |
|------|---------|-------------------|
| M-GRPO | Main/sub-agent 分别采样 + trajectory alignment | 分层 advantage，通过 shared store 交换统计量 |
| MARFT（Flex-MG） | 异步/异构 agent，action 和 token 两种粒度 | 联合优化 |

**框架需求**：跨服务 trajectory 对齐、统计量共享、异步协调。

### 分类与框架设计的对应关系

```
MARL GRPO 算法族
├── 范式一：Episode 并行 + 后处理归因
│   → Rollout Protocol：标准 episode 级轨迹
│   → Advantage：可插拔后处理器（DrMAS norm / GiGPO anchor / agent+turn grouping）
├── 范式二：树状分支采样
│   → Rollout Protocol：树状轨迹 + branch_id 标注
│   → Trajectory Store：Trie 前缀共享（highlight 核心受益场景）
│   → Monitor：分支采样能力（highlight 核心实现点）
└── 范式三：层级解耦训练
    → Rollout Protocol：分层轨迹 + alignment metadata
    → V1+ 扩展（不在 V0 范围内）
```

---

## 3. 核心架构：外接 MAS + Agent Pipe + Model Monitor

### 3.1 设计原则

> **"不重新发明 MAS 编排轮子，专注做 MARL 训练的最优解。"**

所有 MAS 调度交给外部框架或者外部MAS脚本。`Agent Pipe` 定义 agent 管理流程（根据Config拉起MAS + 请求生命周期），`Model Monitor` 定义在链路上的 LLM 网关。我们的框架通过 Model Monitor 拦截所有 LLM 调用，**非侵入地**采集 RL 训练所需的全部数据。

### 3.2 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│                     用户配置空间                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │  Model   │ │  Reward  │ │ Training │ │  MAS Adapter   │  │
│  │  Mapping │ │  Config  │ │  Config  │ │  Config        │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬────────┘  │
│       └────────────┴────────────┴───────────────┘            │
│                            │                                  │
│  ┌─────────────────────────▼────────────────────────────────┐ │
│  │                    Rollout Layer                          │ │
│  │                                                          │ │
│  │  ┌──────────────┐     ┌────────────────┐   ┌──────────┐  │ │
│  │  │  外部 MAS    │ --> │ Agent Pipe     │-->| Model    │  │ │
│  │  │ (LangGraph / │     │ (agent 管理流程)|   │ Monitor  │  │ │
│  │  │  AutoGen /   │ <-- │                │<--| (链路网关)│  │ │
│  │  │  自定义)     │     └────────────────┘   └────┬─────┘  │ │
│  │  └──────────────┘                               │        │ │
│  │                                  请求/响应  ────┘        │ │
│  │                                                └──> 推理服务 │ │
│  │                                                     (vLLM/SGLang)│ │
│  └──────────────────────────────────────────────┼──────────┘ │
│                    Rollout Data Protocol         │            │
│  ┌───────────────────────────────────────────────▼──────────┐ │
│  │          Trie-based Trajectory Store（Highlight）         │ │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │ │
│  │  │ Context  │  │  Radix   │  │   Training Views    │   │ │
│  │  │  Store   │  │   Tree   │  │ · Episode / Branch  │   │ │
│  │  └──────────┘  └──────────┘  └─────────────────────┘   │ │
│  └──────────────────────────┬────────────────────────────── ┘ │
│                             │                                 │
│  ┌──────────────────────────▼────────────────────────────────┐ │
│  │                   RL Training Core                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐    │ │
│  │  │ Reward Engine│  │  Advantage   │  │   Policy    │    │ │
│  │  │（per-agent + │  │  Calculator  │  │  Optimizer  │    │ │
│  │  │  global）    │  │（可插拔）    │  │  （GRPO）   │    │ │
│  │  └──────────────┘  └──────────────┘  └─────────────┘    │ │
│  │                 权重同步 ───────────────> 推理服务         │ │
│  └──────────────────────────┬────────────────────────────── ┘ │
│  ┌──────────────────────────▼────────────────────────────────┐ │
│  │   Runtime Substrate（V0 固定 Verl，后续可替换）            │ │
│  └────────────────────────────────────────────────────────── ┘ │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 Agent Pipe + Agent-Aware Model Monitor

`Agent Pipe` 负责根据MAS参数拉起MAS服务，并确保所有 LLM 请求统一经过 `Model Monitor`。基于团队已验证的 **SWE-agent recipe Model Monitor** 模式，该方案增强为多智能体版本，承担三项职责（V0 仅交付职责一；职责二属于 V0.2+ 增强能力）。

#### 职责一：非侵入式轨迹采集

外部 MAS 的每个 agent 将 LLM 调用发往 monitor，monitor 透明地完成以下操作：

1. 从 request 的 `model` 字段识别 `agent_id`，并收集训练所需其他字段，查 `model_mapping` 路由到正确推理引擎
2. 从 response 中提取 `token_ids`、`logprobs`、`finish_reason`
3. 将采集数据组装成结构化轨迹输出后用于后续训练
4. 向 MAS 返回干净的 response（不含 RL 元数据）

#### 请求格式（标准 OpenAI Chat Completions API）

Monitor 接收标准 HTTP POST 到 `/v1/chat/completions`，请求体示例：

```json
{
  "model": "solver",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "temperature": 0.7,
  "max_tokens": 512,
  "tools": [...],
  "logprobs": false,
  "stream": false
}
```

####核心接口

RunMAS(prompt: str, mas_config: dict, reward_func: Callable, actor_manager: Dict) -> Future<Trajectory>

mas_config中包含了mas系统的类型，mas系统启动命令，以及mas系统所需要的配置模板。
Trajectory Engine需要修改mas系统中的llm后端url，替换为ModelMonitor的url，Trajectory Engine拉起MAS系统，传入prompt进行rollout；rollout过程中，ModelMonitor截获mas系统中的agent的llm请求，根据其中各个agent的后端llm的id，从actor_manager中的找出对应的待训练模型Actor后端调用（llm_server），收集完后调用reward_func进行奖励计算，返回的Trajectory的Future表示一条单独的轨迹数据，目前同步等待实现，后续支持异步后，可直接将生成的轨迹存入buffer。

#### 职责二：Monitor 层树状采样（V0.2+ 增强方向）

在 V0 中，我们先保留分支兼容协议与采样钩子，不把在线树状采样执行作为交付范围。树状采样仍是后续增强重点能力。

**工作原理**：

```
MAS 框架发送 1 次 request
        │
        ▼
┌─────────────────┐
│   Model Monitor   │
│                 │
│  1. 转发到推理引擎，生成 K 个 response（K 个 branch）
│  2. 选择 1 个返回给 MAS 框架（主路径继续执行）
│  3. 将 K 个分支全部记录到 Trajectory Store
│                 │
│  MAS 框架看到：一次正常的 request-response
│  Training Core 得到：一棵完整的分支树
└─────────────────┘
```

**实现模式（增强阶段）**：

| 模式 | 实现方式 | 权衡 |
|------|---------|------|
| Step-wise branching（原型） | Monitor 生成 K 个候选 response，返回 1 个，记录全部 | 实现简单；分支为"虚拟"trajectory，仅主路径被实际执行 |
| Replay-Fork（增强） | Monitor 发分支事件，Coordinator 拉起分支 run，重放到分叉点后继续到终局 | 终局 reward 保真度更高；实现与计算成本更高 |
| Checkpoint-based full-tree（远期） | 结合 MAS 原生 checkpoint，对每个分支实际执行完整流程 | 分支质量最高；框架耦合与复杂度最高 |


### 3.4 Rollout Data Protocol

连接 Monitor 与 外部MAS的标准化数据协议，需要能够支持不同类别的算法。必要时通过adapter的形式将
```
## 附录

### A. 本地参考资料

| 资料 | 路径 |
|------|------|
| Replay-Fork 详细设计（V0.2+ 参考） | `docs/plans/2026-03-01-replay-fork-tree-sampling-design.md` |
| GRPO brainstorm 记录 | `/home/cxb/multi-agent/docs/marl_grpo_brainstorm.md` |
| DrMAS 代码 + 分析报告 | `/home/cxb/multi-agent/DrMAS` |
| MARTI 代码 | `/home/cxb/multi-agent/MARTI` |
| PettingLLMs（AT-GRPO）代码 | `/home/cxb/multi-agent/PettingLLMs` |
| Verl 代码 | `/home/cxb/rl_framework/verl` |
| Slime 代码 | `/home/cxb/rl_framework/slime` |
| SWE-agent recipe（Model Monitor 参考） | `/home/cxb/rl_framework/verl/recipe/swe_agent/` |
| LangGraph 代码 | `/home/cxb/MAS/langgraph` |

### B. 关键技术参考

- **SWE-agent Model Monitor**：`verl/recipe/swe_agent/model_proxy/proxy_server.py` — Anti-Call 机制、asyncio.Queue 解耦请求/响应、logprob 透明采集
- **MARTI workflow 接口**：`marti/agent_workflows/workflow_wrapper.py` — 返回 trajectory + reward_matrix
- **AT-GRPO 分组策略**：`pettingllms/verl/ray_trainer.py` — agent-wise + turn-wise 双维度分组
- **SGLang RadixAttention**：前缀树 KV Cache 共享 — Trajectory Store 设计参考

