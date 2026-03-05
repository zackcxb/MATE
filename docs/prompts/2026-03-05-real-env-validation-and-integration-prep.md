# 真实环境验证 + 训练侧对接准备 Prompt

> 粘贴到新 Agent 窗口中执行。

---

## 角色

你是一个 AI-Infra 开发工程师，在 `/home/cxb/MATE-reboot` 仓库中完成 Trajectory Engine 的真实环境验证和训练侧对接准备。

## 背景

Trajectory Engine V0 已完成全部开发，包含：
- AgentPipe 编排器、ModelMonitor（策略模式）、VLLMBackend（含 token_ids 提取）
- MASLauncher、TrajectoryCollector、RewardWorker
- `parallel_rollout()` 并行 episode 采样
- 60 个测试全部通过

当前分支状况：
- `main`：已包含 V0 核心实现
- `feat/parallel-rollout`：包含 parallel_rollout + vLLM token_ids 修复，待合并

## 前置阅读

1. `/home/cxb/MATE-reboot/AGENTS.md`
2. `/home/cxb/MATE-reboot/docs/plans/2026-03-04-trajectory-engine-v0-design.md`
3. `/home/cxb/MATE-reboot/mate/trajectory/parallel.py`
4. `/home/cxb/MATE-reboot/mate/trajectory/pipe.py`
5. OrchRL Search MAS: `/home/cxb/OrchRL/examples/mas_app/search/` — 特别是 `configs/search_mas_example.yaml`、`scripts/run_search_mas.py`

---

## 任务概览

共两个阶段，按顺序执行：

---

### 阶段 1: 真实环境端到端验证

**目标**：用实际 vLLM 推理服务 + OrchRL Search MAS + 真实检索工具，跑通 N 条并行 episode 轨迹采集。

#### 1.1 环境搭建检查清单

逐项检查以下资源是否可用（如果某项不可用，记录原因并跳过，不要阻塞后续工作）：

- [ ] **vLLM 推理服务**：检查是否有运行中的 vLLM 服务（尝试 `curl http://127.0.0.1:8000/v1/models`），如果没有需要启动一个
- [ ] **模型**：确认 `/data1/models/Qwen/Qwen3-4B-Instruct-2507` 或其他可用模型路径。
- [ ] **检索服务**：检查 OrchRL 的检索服务是否可用（`curl http://127.0.0.1:18080/health`），如果没有考虑使用 `search.provider: disabled` 跳过。
- [ ] **测试数据**：检查 `~/data/drmas_search_mas/test_sampled.parquet` 是否存在，如果没有运行 `prepare_drmas_search_data.py`

#### 1.2 编写验证脚本

在 `/home/cxb/MATE-reboot/scripts/` 下创建 `run_real_validation.py`：

```python
"""
真实环境端到端验证脚本。

使用 AgentPipe + parallel_rollout 对 DrMAS Search 任务进行轨迹采集。
输出采集到的轨迹数据（JSON 格式）供后续可视化和审查。
"""
```

脚本需要支持的参数：
- `--config`：OrchRL Search MAS 的 YAML 配置路径
- `--vllm-url`：vLLM 服务地址（默认 `http://127.0.0.1:8000`）
- `--model`：模型名称（默认从 config 中读取）
- `--prompts-file`：测试题目文件路径（parquet）
- `--n-prompts`：选取多少道题目（默认 5，用于快速验证）
- `--n-samples`：每道题采样几条 episode（默认 2）
- `--max-concurrent`：最大并发数（默认 2）
- `--output`：输出 JSON 文件路径

脚本核心逻辑：
1. 加载配置和测试数据
2. 构建 AgentPipeConfig（从 OrchRL 配置映射）
3. 构建 VLLMBackend
4. 构建 reward 函数（使用 OrchRL 的 `is_search_answer_correct`）
5. 调用 `parallel_rollout()`
6. 将 `List[EpisodeResult]` 序列化为 JSON 输出
7. 打印摘要统计（总 episode 数、成功率、平均 turn 数等）

#### 1.3 验证要点

运行脚本后检查：
1. `token_ids` 不为 None（VLLMBackend + `return_token_ids=true` 生效）
2. `logprobs` 长度与 `token_ids` 长度一致
3. 每条 episode 有不同的 episode_id
4. 三个 agent（verifier/searcher/answerer）的 turn 数据完整
5. reward 计算结果合理

如果 vLLM 或检索服务不可用，使用 mock 模式（ScriptedBackend + 禁用检索）完成流程验证，并在输出中标注"mock 模式"。

---

### 阶段 2: 轨迹数据可视化

**目标**：编写一个轻量可视化脚本，帮助审查采集数据质量。

在 `/home/cxb/MATE-reboot/scripts/` 下创建 `visualize_trajectories.py`：

输入：阶段 1 生成的 JSON 文件
输出：打印到终端或保存为简单的 HTML 报告

可视化维度：

1. **轨迹结构展示**：每条 episode 的 agent 调用时间线
   ```
   Episode abc123 (reward=1.0, 3 turns):
     [0] verifier → "Based on information..." (decision=no)
     [1] searcher → "Let me search..." (query="...")
     [2] verifier → "Now I have enough..." (decision=yes)
     [3] answerer → "The answer is..." (answer="42")
   ```

2. **数据完整性报告**：
   - 每个 turn 的 token_ids 长度、logprobs 长度、是否一致
   - 是否有 token_ids=None 的 turn（标红警告）
   - messages 上下文累积是否单调递增

3. **统计摘要**：
   - 总 episode 数、成功（reward=1）/失败比例
   - 每个 agent 的调用次数分布
   - 每个 agent 的平均生成 token 数
   - episode 平均 turn 数

---

## 更新项目文档

完成以上阶段后，更新 `docs/project-context.md`：
- 当前阶段改为"V0 实现完成 + 真实环境验证完成，待训练侧联调"
- 待办列表更新

## 注意事项

- 遵循 `AGENTS.md` 所有规则
- 如果真实环境不可用（无 vLLM / 无检索服务），先尝试安装相关软件。安装失败可以暂停向用户求助，或者建议用 mock 模式完成流程验证并明确标注
- 脚本注重实用性，不需要过度工程化
- 所有回复使用中文
