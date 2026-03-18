# Slime 如何解决 Tokenization Drift 问题

**Date:** 2026-03-17
**Scope:** 分析 `/home/cxb/rl_framework/slime` 代码库中对 tokenization drift 的处理方式

---

## 核心结论

文档描述的 tokenization drift 风险根源在于**训练侧重新 tokenize**。Slime 的根本策略是**完全避免这一步**——推理侧生成什么 token，训练侧就用什么 token，中间不经过任何重新 tokenize 的环节。

---

## 一、核心数据结构

### Sample（`slime/utils/types.py`）

```python
@dataclass
class Sample:
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)  # 完整序列（prompt + response）
    response: str = ""
    response_length: int = 0
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None   # 推理侧 logprobs
    teacher_log_probs: list[float] | None = None   # 蒸馏用
    multimodal_inputs: dict[str, Any] | None = None
    multimodal_train_inputs: dict[str, Any] | None = None
    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)
```

`sample.tokens` 存储推理侧生成的**完整 token 序列**（prompt + response），是解决 tokenization drift 的核心字段。

### RolloutBatch（`slime/utils/types.py`）

```python
RolloutBatch = dict[str, list[torch.Tensor] | list[int] | list[float] | list[str]]
```

主要字段：
- `tokens`：完整 token 序列列表
- `response_lengths`：每个样本的 response token 数
- `total_lengths`：每个样本的总序列长度
- `rewards`：标量奖励
- `loss_masks`：per-token loss mask
- `rollout_log_probs`：推理侧 logprobs（可选）

---

## 二、解决方案详解

### 2.1 存储完整 token 序列（解决 Prompt/Response Tokenization 差异）

推理侧在生成时直接存储完整的 token ids：

```python
# slime/rollout/sglang_rollout.py
prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
sample.tokens = prompt_ids  # 初始化为 prompt tokens

# 生成后追加 response tokens
new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
sample.tokens = sample.tokens + new_response_tokens
```

训练侧通过 `RolloutBatch["tokens"]` 直接消费，**不做任何重新 tokenize**：

```python
# slime/backends/megatron_utils/actor.py
rollout_data["tokens"] = [
    torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device())
    for t in rollout_data["tokens"]
]
```

**效果**：直接绕过文档中 2.1（Prompt Tokenization）、2.8（Response Tokenization）两类风险。

---

### 2.2 Chat Template 只应用一次（解决 Chat Template 版本差异）

Chat template 在**数据集加载阶段**就已经渲染为字符串，之后不再涉及 template：

```python
# slime/utils/data.py
if apply_chat_template:
    output_prompt = tokenizer.apply_chat_template(
        prompt,
        tools=tools,
        tokenize=False,          # 返回字符串，不是 token ids
        add_generation_prompt=True,
        **(apply_chat_template_kwargs or {}),
    )
```

推理侧拿到的是已渲染好的字符串，encode 一次后存入 `sample.tokens`。训练侧直接用存好的 token ids，**chat template 版本差异的问题根本不会出现**。

---

### 2.3 同一 Checkpoint 加载 Tokenizer（解决推理框架差异、初始化参数差异）

```python
# 推理侧 (slime/rollout/sglang_rollout.py)
self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

# 训练侧 (slime/backends/megatron_utils/actor.py)
self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
```

两侧都从同一个 `hf_checkpoint` 加载，参数一致。加之训练侧不重新 tokenize，即使存在细微差异也不会影响训练。

---

### 2.4 Logprobs 与 Token IDs 同源存储（解决 Logprobs 一致性问题）

Logprobs 在推理时与 token ids **同步从同一数据源取出**：

```python
# slime/rollout/sglang_rollout.py
# output_token_logprobs 的每个元素是 (log_prob, token_id) 对
new_response_tokens   = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]

sample.tokens          = sample.tokens + new_response_tokens
sample.rollout_log_probs += new_response_log_probs
```

token 和 logprob 来自同一个 `output_token_logprobs`，**天然一一对应，不存在长度不一致的问题**。

---

### 2.5 截断策略统一（解决截断策略差异）

截断在数据加载阶段通过 `filter_long_prompt` 统一处理，推理侧和训练侧看到的是同一份已处理的数据，不存在两侧截断策略不同的问题。

---

### 2.6 多模态支持（解决多模态 Tokenization 差异）

对于多模态输入，使用 processor 统一处理：

```python
# slime/rollout/sglang_rollout.py
if state.processor and sample.multimodal_inputs:
    processor_output = state.processor(text=sample.prompt, **processor_kwargs)
    prompt_ids = processor_output["input_ids"][0]
    sample.multimodal_train_inputs = {
        k: v for k, v in processor_output.items()
        if k not in ["input_ids", "attention_mask"]
    }
```

图像 token 的处理结果直接存入 `sample.multimodal_train_inputs`，训练侧直接使用，不重新处理。

---

## 三、数据流全景

```
数据加载阶段
  messages → apply_chat_template(tokenize=False) → prompt_str

推理阶段
  prompt_str → tokenizer.encode() → prompt_ids
  prompt_ids → sglang 生成 → (new_tokens, new_logprobs)
  sample.tokens = prompt_ids + new_tokens
  sample.rollout_log_probs = new_logprobs

打包阶段
  sample → RolloutBatch {
      "tokens": sample.tokens,           # 完整序列
      "rollout_log_probs": sample.rollout_log_probs,
      "response_lengths": ...,
      "loss_masks": ...,
  }

训练阶段
  RolloutBatch["tokens"] → torch.tensor → 直接用于 forward pass
  RolloutBatch["rollout_log_probs"] → 用于 PPO/GRPO loss 计算
  （无任何重新 tokenize）
```

---

## 四、一致性校验机制

### CI 测试中的 Logprobs 校验（`slime/backends/megatron_utils/data.py`）

```python
if args.ci_test and reduced_log_dict is not None:
    if rollout_id == 0:
        # 推理侧 logprobs 与训练侧重新计算的 logprobs 对比，误差 < 1e-8
        assert abs(
            reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]
        ) < 1e-8

    # Logprobs 合理性检查
    assert -0.5 < reduced_log_dict["rollout/log_probs"] < 0
    assert 0 < reduced_log_dict["rollout/entropy"] < 0.5
```

### Loss Mask 对齐（`slime/backends/megatron_utils/data.py`）

```python
for loss_mask, total_length, response_length in zip(
    batch["loss_masks"], batch["total_lengths"], batch["response_lengths"], strict=True
):
    prompt_length = total_length - response_length
    loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
```

---

## 五、与文档建议方案的对比

| 文档中的风险 | 文档建议方案 | Slime 实际方案 |
|---|---|---|
| Prompt Tokenization 差异 | 存储 `prompt_ids` | 存储完整 `sample.tokens`（更彻底） |
| Chat Template 版本差异 | 存储 `prompt_ids` | 数据加载时一次性渲染为字符串 |
| 推理框架 Tokenizer 差异 | 统一 Tokenizer 来源 | 同一 checkpoint + 不重新 tokenize |
| 特殊 Token 处理差异 | 显式配置 | 完全绕过（不重新 tokenize） |
| Tokenizer 初始化参数差异 | 参数显式传递 | 完全绕过（不重新 tokenize） |
| Response Tokenization | 强制存储 `token_ids` | 推理时直接存，无 `None` 情况 |
| Logprobs 一致性 | 长度校验 | 同源存储，天然对齐 |
| 截断策略差异 | 统一截断策略 | 数据加载阶段统一处理 |
| 多模态 Tokenization | 存储完整 `input_ids` | processor 结果直接存入 `multimodal_train_inputs` |

文档建议的"核心方案"是存储 `prompt_ids`，Slime 走得更彻底——**直接存整个序列的 ids**，从根本上消除了推理侧与训练侧之间所有 tokenization 相关的不一致风险。

---

## 六、关键文件索引

| 文件 | 作用 |
|---|---|
| `slime/utils/types.py` | `Sample`、`RolloutBatch` 数据结构定义 |
| `slime/utils/data.py` | 数据集加载、chat template 应用、`filter_long_prompt` |
| `slime/utils/processing_utils.py` | `load_tokenizer`、`load_processor` |
| `slime/rollout/sglang_rollout.py` | 推理生成、token ids 和 logprobs 存储 |
| `slime/ray/rollout.py` | `_convert_samples_to_train_data`，打包 RolloutBatch |
| `slime/backends/megatron_utils/actor.py` | 训练侧消费 RolloutBatch，logprobs CP 切分 |
| `slime/backends/megatron_utils/data.py` | DataIterator、loss mask 对齐、CI 校验 |
| `slime/backends/megatron_utils/loss.py` | 训练侧 logprobs 计算、PPO/GRPO loss |
