# Trajectory Engine V0 真实环境验证记录

> 日期：2026-03-09
> 范围：`scripts/run_real_validation.py` + OrchRL Search MAS + vLLM + 检索服务

## 环境结论

- vLLM 服务在线：`http://127.0.0.1:8000/v1/models`
- 检索服务在线：`http://127.0.0.1:18080/retrieve`
- 在线模型：`/data1/models/Qwen/Qwen3-4B-Instruct-2507`
- GPU 约束：由于服务器 GPU5 故障，验证命令统一使用 `CUDA_VISIBLE_DEVICES=0,1,2,3`

## 本次修复

1. `scripts/run_real_validation.py` 现在会优先按在线 vLLM 暴露的模型集合解析真实模型，不再直接信任失效的 YAML `llm.model`。
2. 检索服务健康检查默认地址从 `/health` 修正为 `/retrieve`。
3. `expected_answer="['...']"` 这类字符串化答案会先归一化，再进入 reward checker，避免 reward 被解析错误压成全 0。
4. 相对模型路径按 YAML 文件所在目录解析；多模型 vLLM 场景下不再静默切到“第一个模型”。

## 验证命令与产物

### Smoke 验证

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_real_validation.py \
  --config /home/cxb/OrchRL/examples/mas_app/search/configs/search_mas_example.yaml \
  --n-prompts 1 \
  --n-samples 2 \
  --max-concurrent 2 \
  --timeout 180 \
  --output artifacts/trajectory_validation_real_smoke_fixed.json
```

产物：`artifacts/trajectory_validation_real_smoke_fixed.json`

结果：

- `2/2` 条 episode 成功采集
- `reward success_rate=0%`（该轮回答均未命中 strict EM）
- `token_ids=None` turn 数：`0`
- `token_ids/logprobs` 长度不一致 turn 数：`0`
- `episode_id` 全局唯一：`true`
- reward 区间：`[0.0, 0.0]`

说明：

- 两条 episode 都包含 `verifier/searcher/answerer` 三个 agent 的完整 turn 数据。
- 该轮 reward 全 0 并不再指向解析 bug，而是 strict EM 与模型回答格式/内容不匹配；reward 链路是否正常由下方 exact-match 验证单独确认。

### 受控 exact-match 验证

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_real_validation.py \
  --config /home/cxb/OrchRL/examples/mas_app/search/configs/search_mas_example.yaml \
  --prompts-file artifacts/trajectory_validation_exact_match.parquet \
  --n-prompts 1 \
  --n-samples 1 \
  --max-concurrent 1 \
  --timeout 180 \
  --output artifacts/trajectory_validation_exact_match.json
```

结果：

- `1/1` 条 episode 成功采集
- `success_rate=100%`
- `final_reward=1.0`

说明：

- 这条验证专门用于区分两类问题：
  - reward 解析链路是否仍有 bug
  - 真实样本 reward 为 0 是否只是严格 EM 与模型回答格式不匹配
- 结论是 reward 解析链路已恢复正常；真实样本仍可能为 0，主要受 strict EM 和回答表述影响。

### 多样本验证

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_real_validation.py \
  --config /home/cxb/OrchRL/examples/mas_app/search/configs/search_mas_example.yaml \
  --n-prompts 3 \
  --n-samples 2 \
  --max-concurrent 4 \
  --timeout 180 \
  --output artifacts/trajectory_validation_real_fixed.json
```

产物：`artifacts/trajectory_validation_real_fixed.json`

结果：

- 计划 episode：`6`
- 成功采集：`4`
- 异常退出：`2`
- 成功采集的 4 条 episode 中：
  - `token_ids=None` turn 数：`0`
  - `token_ids/logprobs` 长度不一致 turn 数：`0`
  - `episode_id` 全局唯一：`true`
  - 三 agent turn 数据完整

## 异常退出分析

异常样本固定出现在 prompt：

```text
when is the next deadpool movie being released?
```

复现实验表明：

- `max_turns=1/2/3` 时，MAS 退出码为 `0`
- `max_turns=4` 时，MAS 稳定退出码为 `1`

stderr 关键证据：

```text
RuntimeError: LLM call failed after 3 retries: Error code: 502
...
{'error': "Client error '400 Bad Request' for url 'http://127.0.0.1:8000/v1/chat/completions'"}
```

结论：

- 这是可复现的长上下文失败，不是瞬时环境抖动。
- `Deadpool` 样本在 `max_turns=4` 时会累计过长 team context，最终在 Answerer 调用阶段触发 vLLM `400 Bad Request`，MAS 因上游 `502` 退出。
- 当前 vLLM 服务的 `max_model_len=4096`，与该样本的长链路提示拼接存在冲突。

## 当前可复用结论

1. Trajectory Engine V0 已在真实 vLLM + 真实检索服务 + 真实 OrchRL Search MAS 环境下跑通并行轨迹采集。
2. `token_ids` / `logprobs` 采集链路在成功 episode 上是完整且一致的。
3. reward 解析 bug 已修复，但 reward 是否为 1 仍受严格 EM 评估和模型回答格式影响。
4. 需要单独处理长上下文 prompt 的 `max_turns=4` 失败问题，候选方向：
   - 限制 Search Agent 输出冗长度
   - 压缩 team context
   - 降低 `max_turns`
   - 使用更大 context window 的推理服务
