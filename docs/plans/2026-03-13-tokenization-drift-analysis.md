# Tokenization Drift 分析报告

**Status:** 技术分析文档

**Created:** 2026-03-13

**Scope:** RL 训练中推理侧与训练侧 tokenization 一致性风险分析

## 概述

在基于 LLM 的 RL 训练流程中，推理侧（Rollout）生成的轨迹数据需要在训练侧（Trainer）被消费。当推理侧和训练侧对同一文本内容产生不同的 tokenization 结果时，会导致 **Tokenization Drift** 问题。这种不一致性会直接影响训练效果，因为模型实际看到的输入与训练时计算的 loss 不匹配。

本文档系统性地分析可能导致 tokenization drift 的各类因素，并提供相应的缓解策略。

---

## 一、Tokenization Drift 的核心风险

### 1.1 问题描述

```
┌─────────────────────────────────────────────────────────────────┐
│  理想情况：推理侧与训练侧 tokenization 完全一致                   │
├─────────────────────────────────────────────────────────────────┤
│  推理侧: "你好" → [101, 2769, 3456]                              │
│  训练侧: "你好" → [101, 2769, 3456]  ✓ 一致                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Drift 情况：推理侧与训练侧 tokenization 不一致                   │
├─────────────────────────────────────────────────────────────────┤
│  推理侧: "你好" → [101, 2769, 3456]                              │
│  训练侧: "你好" → [101, 882, 3456]    ✗ 不一致                   │
│                                                                 │
│  后果：                                                          │
│  - PPO loss 计算基于错误的 input_ids                             │
│  - logprobs 与实际生成不匹配                                     │
│  - 模型学习到错误的 token 分布                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 影响范围

| 影响维度 | 具体表现 |
|---------|---------|
| **训练稳定性** | Loss 震荡、收敛困难 |
| **模型性能** | 学习到错误的 token 映射关系 |
| **数据一致性** | logprobs 与 token_ids 不匹配 |
| **可复现性** | 难以复现训练结果 |

---

## 二、Tokenization Drift 影响因素分析

### 2.1 Prompt Tokenization（Prompt IDs）

#### 问题描述

推理侧生成时使用的 prompt tokenization 与训练侧重新 tokenize 时产生差异。

#### 作用机制

```
推理侧流程:
  messages → tokenizer.apply_chat_template() → prompt_ids → 模型生成

训练侧流程（当前实现）:
  messages (存储) → tokenizer.apply_chat_template() → prompt_ids (重新计算)
  token_ids (存储) → response_ids
  
  input_ids = prompt_ids + response_ids
```

**风险点**：推理侧和训练侧可能使用不同的 tokenizer 实例或配置。

#### 当前代码参考

[mate_dataproto_adapter.py](file:///home/cxb/OrchRL/orchrl/trainer/mate_dataproto_adapter.py#L99-L108):

```python
def _tokenize_messages(tokenizer, messages, max_prompt_length: int) -> list[int]:
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return [int(token_id) for token_id in prompt_ids][-max_prompt_length:]
```

#### 缓解策略

| 策略 | 说明 | 优先级 |
|------|------|--------|
| **存储 prompt_ids** | 推理侧直接存储 tokenize 后的 prompt_ids，训练侧直接使用 | 高 |
| **混合模式** | 同时存储 messages 和 prompt_ids，训练侧优先使用 prompt_ids | 高 |
| **Tokenizer 校验** | 训练前校验推理侧和训练侧 tokenizer 版本一致性 | 中 |

---

### 2.2 Chat Template 版本差异

#### 问题描述

不同版本的 tokenizer 可能使用不同的 chat template，导致相同的 messages 产生不同的 token 序列。

#### 作用机制

```python
# Qwen2.5 chat template (示例)
"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

# Qwen3 chat template (可能变化)
"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
```

#### 风险场景

| 场景 | 风险程度 |
|------|---------|
| 推理服务使用旧版模型，训练侧使用新版 tokenizer | 高 |
| 推理服务使用 vLLM 内置 template，训练侧使用 HuggingFace template | 高 |
| 模型微调后更新了 chat template | 中 |

#### 实际案例

[naive_chat_scheduler.py](file:///home/cxb/OrchRL/verl/verl/schedulers/naive_chat_scheduler.py) 中针对 DeepSeek/Qwen 模型的 template 补丁：

```python
def patch_tokenizer_chat_template_fn(tokenizer):
    template_dir = os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(searchpath=template_dir))
    template = env.get_template('deepseek_qwen.jinja')
    
    def _apply_deepseek_chat_template(self, messages, **kwargs):
        context = {
            "messages": messages,
            "add_generation_prompt": kwargs.get("add_generation_prompt", False),
            "tokenize": kwargs.get("tokenize", False),
        }
        return template.render(**context)
    
    tokenizer.apply_chat_template = _apply_deepseek_chat_template.__get__(tokenizer, type(tokenizer))
```

这表明 chat template 确实存在需要手动补丁的情况。

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **存储 prompt_ids** | 完全绕过 chat template 重新应用 |
| **Template 版本锁定** | 在推理和训练配置中显式指定 template 版本 |
| **Template Hash 校验** | 训练前比对推理侧和训练侧的 template hash |

---

### 2.3 推理框架 Tokenizer 差异

#### 问题描述

不同推理框架（vLLM、TGI、sglang、TensorRT-LLM）使用的 tokenizer 实现可能有细微差异。

#### 作用机制

```
┌─────────────────────────────────────────────────────────────────┐
│  不同推理框架的 Tokenizer 实现                                   │
├─────────────────────────────────────────────────────────────────┤
│  HuggingFace Tokenizer (训练侧默认)                              │
│    - 基于 tokenizers 库 (Rust 实现)                              │
│    - 完整支持所有 tokenizer 类型                                 │
│                                                                 │
│  vLLM Tokenizer                                                 │
│    - 内部使用 HuggingFace tokenizer                             │
│    - 可能有初始化参数差异                                        │
│                                                                 │
│  TensorRT-LLM Tokenizer                                         │
│    - 可能使用自定义实现                                          │
│    - 对某些特殊 token 处理可能不同                               │
│                                                                 │
│  TGI (Text Generation Inference)                                │
│    - 使用 tokenizers 库                                         │
│    - 可能有预处理/后处理差异                                     │
└─────────────────────────────────────────────────────────────────┘
```

#### 风险场景

| 场景 | 风险点 |
|------|--------|
| vLLM 推理 + HuggingFace 训练 | 初始化参数差异（如 `use_fast`） |
| TensorRT-LLM 推理 + HuggingFace 训练 | 实现差异 |
| 多节点推理，不同节点使用不同框架 | 框架间差异 |

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **统一 Tokenizer 来源** | 推理和训练使用同一份 tokenizer 文件 |
| **存储 prompt_ids** | 完全避免重新 tokenize |
| **Tokenizer 版本校验** | 训练前检查 tokenizer 版本和配置 |

---

### 2.4 特殊 Token 处理差异

#### 问题描述

不同框架对特殊 token（BOS、EOS、PAD、UNK）的处理方式可能不同。

#### 作用机制

```python
# 不同框架对 BOS token 的处理
HuggingFace: 自动添加 BOS token (如果配置了 add_bos_token=True)
vLLM: 可能不添加 BOS token
TGI: 可能有不同的默认行为

# 不同框架对 PAD token 的处理
某些框架: pad_token_id = eos_token_id
其他框架: 需要显式设置
```

#### 当前代码参考

[tokenizer.py](file:///home/cxb/OrchRL/verl/verl/utils/tokenizer.py#L36-L61):

```python
def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)  # 设置 pad_token_id = eos_token_id
    return tokenizer
```

这表明训练侧会对 tokenizer 进行"修正"，如果推理侧没有做同样的修正，就会产生差异。

#### 风险场景

| Token 类型 | 风险点 |
|-----------|--------|
| **BOS Token** | 是否自动添加、添加位置 |
| **EOS Token** | 是否在生成末尾添加、token ID 值 |
| **PAD Token** | pad_token_id 的值、padding 位置 |
| **UNK Token** | 未知字符的处理方式 |

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **显式配置** | 在推理和训练配置中显式指定所有特殊 token 的处理方式 |
| **Token ID 校验** | 训练前校验推理侧和训练侧的特殊 token ID 是否一致 |
| **存储完整序列** | 存储 prompt_ids 时包含所有特殊 token |

---

### 2.5 Tokenizer 初始化参数差异

#### 问题描述

即使使用相同的 tokenizer 文件，不同的初始化参数也会导致 tokenization 结果不同。

#### 关键参数

| 参数 | 影响 | 风险程度 |
|------|------|---------|
| `use_fast` | 是否使用 Rust 实现的快速 tokenizer | 中 |
| `add_bos_token` | 是否添加 BOS token | 高 |
| `add_eos_token` | 是否添加 EOS token | 高 |
| `add_prefix_space` | 是否在文本前添加空格（如 GPT-2） | 高 |
| `trim_offsets` | 是否修剪偏移量 | 低 |
| `model_max_length` | 最大序列长度 | 中 |
| `padding_side` | padding 方向（left/right） | 中 |
| `truncation_side` | 截断方向（left/right） | 中 |

#### 作用机制

```python
# 推理侧可能使用
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# 训练侧可能使用
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# 对于某些 tokenizer，use_fast=True 和 use_fast=False 可能产生不同结果
```

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **参数显式传递** | 在推理和训练配置中显式指定所有 tokenizer 参数 |
| **配置文件共享** | 使用同一份 tokenizer 配置文件 |
| **参数校验** | 训练前校验推理侧和训练侧的 tokenizer 参数是否一致 |

---

### 2.6 Unicode 和编码处理差异

#### 问题描述

不同系统或框架对 Unicode 字符的处理方式可能不同，导致 tokenization 结果差异。

#### 作用机制

```
┌─────────────────────────────────────────────────────────────────┐
│  Unicode 处理差异示例                                            │
├─────────────────────────────────────────────────────────────────┤
│  原始文本: "Hello, 世界! 🌍"                                     │
│                                                                 │
│  处理方式 A (UTF-8):                                             │
│    "Hello, 世界! 🌍" → [15496, 11, 303, 102, 125, 0, 234, 128]  │
│                                                                 │
│  处理方式 B (规范化后):                                          │
│    "Hello, 世界! 🌍" → [15496, 11, 303, 102, 125, 0, 234, 128]  │
│    (如果进行了 NFC/NFD 规范化，可能产生不同结果)                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 风险场景

| 场景 | 风险点 |
|------|--------|
| 多语言文本 | 中文、日文、韩文等非 ASCII 字符 |
| Emoji 表情 | 不同系统对 emoji 的处理可能不同 |
| 特殊符号 | 零宽字符、不可见字符 |
| 编码转换 | 不同编码格式之间的转换 |

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **统一编码** | 确保所有文本使用 UTF-8 编码 |
| **Unicode 规范化** | 在 tokenization 前进行统一的 Unicode 规范化（如 NFC） |
| **存储 prompt_ids** | 完全避免编码处理差异 |

---

### 2.7 多模态 Tokenization 差异

#### 问题描述

对于多模态模型（如 LLaVA、Qwen-VL），图像 token 的处理方式可能在不同框架间存在差异。

#### 作用机制

```
┌─────────────────────────────────────────────────────────────────┐
│  多模态 Tokenization 示例                                        │
├─────────────────────────────────────────────────────────────────┤
│  输入: [图像] + "描述这张图片"                                    │
│                                                                 │
│  框架 A:                                                         │
│    [IMG_TOKEN_0, IMG_TOKEN_1, ..., IMG_TOKEN_N] + 文本 tokens   │
│                                                                 │
│  框架 B:                                                         │
│    [BOI] + [IMG_TOKEN_0, ..., IMG_TOKEN_N] + [EOI] + 文本 tokens│
└─────────────────────────────────────────────────────────────────┘
```

#### 风险场景

| 场景 | 风险点 |
|------|--------|
| 图像 token 数量 | 不同框架可能使用不同数量的 image tokens |
| 图像 token 位置 | 图像 token 在序列中的位置可能不同 |
| 特殊 token | BOI/EOI 等 token 的处理 |

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **存储完整 input_ids** | 包括图像 token 在内的完整序列 |
| **Processor 校验** | 确保推理和训练使用相同的 processor |

---

### 2.8 响应部分 Tokenization（Response IDs）

#### 问题描述

虽然推理侧通常会存储生成的 token_ids，但在某些情况下，响应部分也可能需要重新 tokenize。

#### 作用机制

```
┌─────────────────────────────────────────────────────────────────┐
│  响应 Tokenization 风险场景                                      │
├─────────────────────────────────────────────────────────────────┤
│  场景 1: token_ids 未存储                                        │
│    推理侧只存储 response_text，训练侧需要重新 tokenize            │
│    → 风险：与推理时的 token_ids 不一致                           │
│                                                                 │
│  场景 2: token_ids 部分存储                                      │
│    推理侧只存储部分 token_ids（如截断后）                         │
│    → 风险：与完整响应不匹配                                      │
│                                                                 │
│  场景 3: token_ids 格式转换                                      │
│    推理侧存储的是 tensor，训练侧需要 list                         │
│    → 风险：转换过程中可能丢失精度或信息                           │
└─────────────────────────────────────────────────────────────────┘
```

#### 当前代码参考

[datatypes.py](file:///home/cxb/OrchRL/trajectory/datatypes.py#L44-L54):

```python
@dataclass
class TurnData:
    agent_role: str
    turn_index: int
    messages: list[dict[str, Any]]
    response_text: str
    token_ids: list[int] | None  # 可能为 None
    logprobs: list[float] | None
    finish_reason: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
```

`token_ids` 字段是可选的，这意味着可能存在 `token_ids is None` 的情况。

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **强制存储 token_ids** | 推理侧必须存储完整的 token_ids |
| **校验 token_ids 完整性** | 训练前检查 token_ids 是否与 response_text 匹配 |
| **存储 token_ids hash** | 用于校验 token_ids 的完整性 |

---

### 2.9 Logprobs 与 Token IDs 一致性

#### 问题描述

logprobs 应该与 token_ids 一一对应，但在实际存储和传输过程中可能出现不一致。

#### 作用机制

```
┌─────────────────────────────────────────────────────────────────┐
│  Logprobs 一致性要求                                             │
├─────────────────────────────────────────────────────────────────┤
│  正确情况:                                                       │
│    token_ids:  [101, 2769, 3456, 102]                           │
│    logprobs:   [-0.1, -0.5, -0.2, -0.3]                         │
│    len(token_ids) == len(logprobs) ✓                            │
│                                                                 │
│  错误情况:                                                       │
│    token_ids:  [101, 2769, 3456, 102]                           │
│    logprobs:   [-0.1, -0.5, -0.2]                               │
│    len(token_ids) != len(logprobs) ✗                            │
│                                                                 │
│  后果:                                                           │
│    - PPO loss 计算错误                                          │
│    - advantage 估计偏差                                         │
└─────────────────────────────────────────────────────────────────┘
```

#### 当前验证

根据 V0.2 验证结果，已确认：
- 成功 episode 上 `token_ids` 不为 None
- 成功 episode 上 `logprobs` 与 `token_ids` 长度一致

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **长度校验** | 训练前检查 token_ids 和 logprobs 长度是否一致 |
| **数据清洗** | 过滤掉长度不一致的样本 |
| **存储时校验** | 推理侧存储时进行一致性检查 |

---

### 2.10 截断策略差异

#### 问题描述

当序列长度超过限制时，不同的截断策略会导致不同的 tokenization 结果。

#### 作用机制

```python
# 左截断（保留末尾）
truncated = token_ids[-max_length:]

# 右截断（保留开头）
truncated = token_ids[:max_length]

# 中间截断（保留开头和结尾）
truncated = token_ids[:max_length//2] + token_ids[-max_length//2:]
```

#### 风险场景

| 场景 | 风险点 |
|------|--------|
| Prompt 截断 | 推理侧和训练侧可能使用不同的截断策略 |
| Response 截断 | 可能截断重要的生成内容 |
| 多轮对话 | 历史上下文的截断方式可能不同 |

#### 当前代码参考

[mate_dataproto_adapter.py](file:///home/cxb/OrchRL/orchrl/trainer/mate_dataproto_adapter.py#L99-L117):

```python
def _tokenize_messages(tokenizer, messages, max_prompt_length: int) -> list[int]:
    # ...
    return [int(token_id) for token_id in prompt_ids][-max_prompt_length:]  # 左截断

def _normalize_response_ids(token_ids, max_response_length: int) -> list[int]:
    # ...
    return response_ids[:max_response_length]  # 右截断
```

#### 缓解策略

| 策略 | 说明 |
|------|------|
| **统一截断策略** | 推理侧和训练侧使用相同的截断策略 |
| **存储完整序列** | 存储完整的 token_ids，在训练时统一截断 |
| **截断元数据** | 记录截断信息，便于调试 |

---

## 三、影响因素汇总表

| 因素 | 风险等级 | 影响范围 | 缓解难度 |
|------|---------|---------|---------|
| Prompt Tokenization | 高 | 所有训练样本 | 低（存储 prompt_ids） |
| Chat Template 版本 | 高 | 所有对话模型 | 低（存储 prompt_ids） |
| 推理框架 Tokenizer | 中 | 跨框架场景 | 中 |
| 特殊 Token 处理 | 中 | 所有模型 | 中 |
| Tokenizer 初始化参数 | 中 | 所有模型 | 低 |
| Unicode 编码处理 | 低 | 多语言场景 | 中 |
| 多模态 Tokenization | 高 | 多模态模型 | 高 |
| Response Tokenization | 中 | token_ids 缺失场景 | 低 |
| Logprobs 一致性 | 高 | PPO 训练 | 低 |
| 截断策略 | 中 | 长序列场景 | 低 |

---

## 四、推荐缓解方案

### 4.1 核心方案：存储 prompt_ids

**优先级：高**

扩展 `TurnData` 数据结构，存储推理侧的 prompt token ids：

```python
@dataclass
class TurnData:
    agent_role: str
    turn_index: int
    messages: list[dict[str, Any]]      # 保留，用于调试和灵活性
    prompt_ids: list[int] | None        # 新增：推理侧的 prompt token ids
    response_text: str
    token_ids: list[int] | None         # 生成部分的 token ids
    logprobs: list[float] | None
    finish_reason: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
```

训练侧适配：

```python
def get_prompt_ids(turn, tokenizer, max_length):
    if turn.prompt_ids is not None:
        return turn.prompt_ids[-max_length:]
    return _tokenize_messages(tokenizer, turn.messages, max_length)
```

**优点**：
- 完全避免 prompt tokenization drift
- 保持向后兼容
- 保留调试能力

### 4.2 辅助方案：Tokenizer 一致性校验

**优先级：中**

训练前进行 tokenizer 一致性校验：

```python
def validate_tokenizer_consistency(inference_tokenizer, training_tokenizer):
    # 1. 版本校验
    assert inference_tokenizer.name_or_path == training_tokenizer.name_or_path
    
    # 2. 特殊 token 校验
    assert inference_tokenizer.bos_token_id == training_tokenizer.bos_token_id
    assert inference_tokenizer.eos_token_id == training_tokenizer.eos_token_id
    assert inference_tokenizer.pad_token_id == training_tokenizer.pad_token_id
    
    # 3. Chat template 校验
    test_messages = [{"role": "user", "content": "test"}]
    inf_result = inference_tokenizer.apply_chat_template(test_messages, tokenize=True)
    train_result = training_tokenizer.apply_chat_template(test_messages, tokenize=True)
    assert inf_result == train_result, "Chat template mismatch"
    
    # 4. 基础 tokenization 校验
    test_texts = ["Hello", "你好", "🎉", "Hello, 世界!"]
    for text in test_texts:
        assert inference_tokenizer.encode(text) == training_tokenizer.encode(text)
```

### 4.3 数据完整性校验

**优先级：中**

在数据加载时进行完整性校验：

```python
def validate_turn_data(turn: TurnData):
    # 1. token_ids 存在性检查
    if turn.token_ids is None:
        raise ValueError(f"Turn {turn.turn_index} has no token_ids")
    
    # 2. logprobs 一致性检查
    if turn.logprobs is not None:
        if len(turn.token_ids) != len(turn.logprobs):
            raise ValueError(f"Token/logprobs length mismatch: {len(turn.token_ids)} vs {len(turn.logprobs)}")
    
    # 3. prompt_ids 校验（如果存在）
    if turn.prompt_ids is not None:
        if len(turn.prompt_ids) == 0:
            raise ValueError(f"Empty prompt_ids in turn {turn.turn_index}")
```

---

## 五、实施建议

### 5.1 短期（V0.3）

1. **扩展 TurnData 结构**：添加 `prompt_ids` 字段
2. **修改推理侧**：在生成时存储 `prompt_ids`
3. **修改训练侧**：优先使用 `prompt_ids`，回退到重新 tokenize
4. **添加校验**：训练前进行 tokenizer 一致性校验

### 5.2 中期

1. **完善校验机制**：添加更全面的数据完整性校验
2. **监控告警**：添加 tokenization drift 监控
3. **文档完善**：更新 API 文档，明确 tokenization 要求

### 5.3 长期

1. **统一 Tokenizer 管理**：建立统一的 tokenizer 版本管理机制
2. **自动化测试**：添加 tokenization 一致性自动化测试
3. **跨框架兼容**：支持不同推理框架的 tokenizer 适配

---

## 六、参考文档

| 文档 | 说明 |
|------|------|
| [mate_dataproto_adapter.py](file:///home/cxb/OrchRL/orchrl/trainer/mate_dataproto_adapter.py) | 训练侧数据适配器 |
| [datatypes.py](file:///home/cxb/OrchRL/trajectory/datatypes.py) | 轨迹数据结构定义 |
| [tokenizer.py](file:///home/cxb/OrchRL/verl/verl/utils/tokenizer.py) | Tokenizer 工具函数 |
| [rl_dataset.py](file:///home/cxb/OrchRL/verl/verl/utils/dataset/rl_dataset.py) | RLHF 数据集处理 |
| [naive_chat_scheduler.py](file:///home/cxb/OrchRL/verl/verl/schedulers/naive_chat_scheduler.py) | Chat template 补丁 |
| [2026-03-11-trajectory-public-api-boundary.md](file:///home/cxb/MATE-reboot/docs/plans/2026-03-11-trajectory-public-api-boundary.md) | API 边界定义 |
