# MATE-reboot 验证脚本用法

## 概览

`scripts/` 目录包含三个脚本，构成一个分层验证流水线：

```
                       外部服务
           ┌───────────┴───────────┐
           │ vLLM (端口 8000)       │ 检索服务 (端口 8010)
           └───────────┬───────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │  run_real_validation.py (V0.1)          │
    │  parallel_rollout 端到端验证             │
    │  输出: schema v1.0 JSON                 │
    └──────┬────────────────────┬─────────────┘
           │                    │
    ┌──────▼──────────┐  ┌─────▼─────────────────────────┐
    │ visualize_      │  │ run_tree_validation.py (V0.2)  │
    │ trajectories.py │  │ tree_rollout 树状分支验证       │
    │ 终端/HTML 报告   │  │ 复用 V0.1 ~15 个基础函数       │
    │ (仅支持 v1.0)   │  │ 输出: schema v2.0 JSON         │
    └─────────────────┘  └───────────────────────────────┘
```

| 脚本 | 功能 | 验证目标 | 输出 |
|------|------|----------|------|
| `run_real_validation.py` | 并行采样 + 轨迹完整性 | Episode 结构、reward 合理性、token 一致性 | JSON (v1.0) |
| `run_tree_validation.py` | 树状分支 + 前缀重用 | 树结构、replay 标记、前缀共享率 | JSON (v2.0) |
| `visualize_trajectories.py` | 轨迹可视化审查 | 数据质量、上下文单调性 | 终端文本 / HTML |

---

## 前置条件

### 必需服务

| 服务 | 默认地址 | 启动方式 | 缺失时行为 |
|------|----------|----------|-----------|
| vLLM | `http://127.0.0.1:8000` | `bash ~/local_scripts/start_vllm.sh` | 自动回退 mock 模式 |
| 检索服务 | `http://127.0.0.1:8010/retrieve` | `bash ~/local_scripts/start_searchr1_retrieval.sh` | 自动回退 mock 模式 |

### 环境变量

验证脚本**不需要**手动设置 LLM 环境变量。

> **注意**：不要设置 `SEARCH_MAS_LLM_BASE_URL`！OrchRL MAS 的 config loader 会用此
> 变量覆盖 AgentPipe 的 monitor URL，导致 MAS 直接连 vLLM 而绕过 monitor 代理。
> 验证脚本会在 real 模式下主动清除该变量。

唯一可能需要的变量：

```bash
# 仅当检索服务不在默认端口时需要
export SEARCH_MAS_RETRIEVAL_SERVICE_URL=http://127.0.0.1:8010/retrieve
```

---

## 脚本 1: run_real_validation.py (V0.1)

### 用途

使用 `parallel_rollout` 采集多条独立 episode，验证端到端轨迹完整性。

### 用法

```bash
cd /home/cxb/MATE-reboot

# 最小用法（自动检测环境，缺服务时回退 mock）
python scripts/run_real_validation.py

# 完整参数
python scripts/run_real_validation.py \
  --config /path/to/search_mas_example.yaml \
  --vllm-url http://127.0.0.1:8000 \
  --model /data1/models/Qwen/Qwen3-4B-Instruct-2507 \
  --prompts-file /path/to/test_sampled.parquet \
  --n-prompts 5 \
  --n-samples 2 \
  --max-concurrent 2 \
  --timeout 120 \
  --search-health-url http://127.0.0.1:8010/retrieve \
  --output artifacts/trajectory_validation.json

# 强制 mock 模式（无需外部服务）
python scripts/run_real_validation.py --force-mock --n-prompts 2
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 内置默认配置 | OrchRL Search MAS YAML 配置路径 |
| `--vllm-url` | `http://127.0.0.1:8000` | vLLM 服务地址 |
| `--model` | 自动检测 | 模型名称（优先级：CLI > 配置文件 > vLLM 自动检测） |
| `--prompts-file` | 预定义候选路径 | 测试题目 parquet 文件 |
| `--n-prompts` | `5` | 抽样题目数 |
| `--n-samples` | `2` | 每题采样 episode 数 |
| `--max-concurrent` | `2` | 最大并发 episode 数 |
| `--timeout` | `120.0` | 单 episode 超时（秒） |
| `--search-health-url` | `http://127.0.0.1:18080/retrieve` | 检索服务健康检查 URL |
| `--output` | `artifacts/trajectory_validation.json` | 输出 JSON 路径 |
| `--force-mock` | `false` | 强制 mock 模式 |
| `--mas-work-dir` | 自动检测 | MAS 运行目录 |
| `--mas-command-template` | 自动检测 | MAS 命令模板（含 `{config_path}` 和 `{prompt}` 占位符） |

### 输出文件结构 (schema v1.0)

```
trajectory_validation.json
├── environment      # vLLM / 检索服务 / 模型 / prompt 诊断信息
├── summary
│   ├── total_episodes, success_rate, avg_turns
│   ├── agent_call_counts          # 各角色调用次数
│   ├── agent_avg_generated_tokens # 各角色平均生成 token 数
│   └── validation                 # episode_id 唯一性、token 一致性、reward 范围
└── episodes[]                     # 完整 episode 数据（含 trajectory）
```

---

## 脚本 2: run_tree_validation.py (V0.2)

### 用途

使用 `tree_rollout` 执行树状分支采样，通过三轮递进测试验证：
1. **Smoke** — 树结构完整性（k=2, 单并发）
2. **Replay** — 前缀重放标记正确性（k=2, 独立调用）
3. **Multi-prompt** — 多题目稳定性（完整 k, 并发）

### 用法

```bash
cd /home/cxb/MATE-reboot

# 最小用法
python scripts/run_tree_validation.py \
  --config /path/to/search_mas_example.yaml

# 推荐用法（3 题 × 3 分支 + 对比模式 + 保存轨迹）
python scripts/run_tree_validation.py \
  --config /path/to/search_mas_example.yaml \
  --n-prompts 3 \
  --k-branches 3 \
  --compare \
  --save-trajectories \
  --output artifacts/tree_validation.json

# 强制 mock 模式（快速冒烟测试）
python scripts/run_tree_validation.py --force-mock --n-prompts 1 --k-branches 2

# 完整参数
python scripts/run_tree_validation.py \
  --config /path/to/search_mas_example.yaml \
  --vllm-url http://127.0.0.1:8000 \
  --n-prompts 3 \
  --k-branches 3 \
  --max-concurrent-branches 4 \
  --timeout 180 \
  --search-health-url http://127.0.0.1:8010/retrieve \
  --compare --compare-n-samples 2 \
  --save-trajectories \
  --output artifacts/tree_validation.json
```

### 参数说明（仅列出 V0.2 新增参数，其余同 V0.1）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--k-branches` | `3` | 每个分支点的分支数 |
| `--max-concurrent-branches` | 不限 | tree_rollout 最大并发分支数 |
| `--compare` | `false` | 启用 parallel_rollout 基线对比 |
| `--compare-n-samples` | `2` | 对比模式每题采样数 |
| `--save-trajectories` | `false` | 保存完整轨迹到独立文件 |
| `--output` | `artifacts/tree_validation.json` | 输出 JSON 路径 |
| `--timeout` | `180.0` | 单 episode 超时（秒） |

### 输出文件

**验证报告** (`tree_validation.json`, schema v2.0)：

```
tree_validation.json
├── environment         # 同 V0.1
├── input               # 运行参数
├── rounds
│   ├── smoke           # Round 1: 结构检查
│   │   └── validations: { structure, prefix_sharing, reward_stats }
│   ├── replay          # Round 2: 重放标记检查
│   │   └── validations: { structure, replay_markers[], prefix_sharing, reward_stats }
│   └── multi_prompt    # Round 3: 多题目
│       ├── trees[]     # 每棵树的检查结果
│       └── aggregate   # 汇总指标
├── comparison          # parallel_rollout 对比（如启用）
└── summary             # 总体通过/失败
```

**轨迹文件** (`tree_validation_trajectories.json`, 仅 `--save-trajectories` 时生成)：

```
tree_validation_trajectories.json
├── smoke               # Round 1 完整 tree_payload
│   ├── pilot_result    #   pilot episode 全部 turn 数据
│   ├── branch_results  #   每条 branch 的 turn 数据
│   └── tree_metadata
├── replay              # Round 2 完整 tree_payload
└── multi_prompt[]      # Round 3 每棵树
    └── tree_payload    #   pilot + branches 完整数据
```

### 终端输出示例

```
=== V0.2 树状分支验证摘要 ===
模式: real 模式
Round 1 (Smoke): PASS
Round 2 (Replay): PASS
Round 3 (Multi-prompt): PASS
  成功树数: 3/3
  总 episode 数: 40 (pilot + branches)
  前缀共享率: 25.0%
  reward 均值: 0.35
对比模式: 启用
  对比 status: pass
  对比 reward 均值: 0.30
输出文件: artifacts/tree_validation.json
```

---

## 脚本 3: visualize_trajectories.py

### 用途

将 V0.1 或 V0.2 的 JSON 输出转换为人类可读的终端报告或 HTML 报告，用于数据质量审查。

> 已支持 V0.1 (schema v1.0) 和 V0.2 (schema v2.0)，自动检测输入格式。

### 用法

```bash
cd /home/cxb/MATE-reboot

# 终端报告
python scripts/visualize_trajectories.py \
  --input artifacts/trajectory_validation.json

# 限制显示 episode 数
python scripts/visualize_trajectories.py \
  --input artifacts/trajectory_validation.json \
  --limit 5

# 生成 HTML 报告
python scripts/visualize_trajectories.py \
  --input artifacts/trajectory_validation.json \
  --html artifacts/report.html
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | **必需** | V0.1 输出的 JSON 文件路径 |
| `--limit` | `10` | 终端显示的最大 episode 数 |
| `--html` | 无 | 生成 HTML 报告的输出路径 |

### 检查项

- **Token 一致性**：token_ids 是否为 None、token_ids 与 logprobs 长度是否匹配
- **上下文单调性**：每个 agent 的 messages 上下文是否单调递增
- **Agent 调用统计**：各角色调用次数、平均生成 token 数
- **Timeline 可视化**：提取 `<verify>`, `<search>`, `<answer>` XML 标签内容

---

## 典型工作流

### 1. 快速冒烟（无需外部服务）

```bash
# V0.1 mock
python scripts/run_real_validation.py --force-mock --n-prompts 2

# V0.2 mock
python scripts/run_tree_validation.py --force-mock --n-prompts 1 --k-branches 2
```

### 2. 完整端到端验证

```bash
# 确保服务已启动
bash ~/local_scripts/start_vllm.sh
bash ~/local_scripts/start_searchr1_retrieval.sh

# Step 1: V0.1 并行验证
python scripts/run_real_validation.py \
  --config ~/OrchRL/examples/mas_app/search/configs/search_mas_example.yaml \
  --n-prompts 5 --n-samples 2

# Step 2: 可视化 V0.1 结果
python scripts/visualize_trajectories.py \
  --input artifacts/trajectory_validation.json --html artifacts/v0_report.html

# Step 3: V0.2 树状分支验证
python scripts/run_tree_validation.py \
  --config ~/OrchRL/examples/mas_app/search/configs/search_mas_example.yaml \
  --n-prompts 3 --k-branches 3 --compare --save-trajectories
```

### 3. CI/CD 集成

```bash
# 退出码：0 = 全部通过，1 = 有失败
python scripts/run_tree_validation.py \
  --config $CONFIG_PATH \
  --n-prompts 3 --k-branches 3 --timeout 180 \
  --output artifacts/tree_validation.json

# 检查退出码
if [ $? -ne 0 ]; then
  echo "验证失败，请检查 artifacts/tree_validation.json"
  exit 1
fi
```

---

## 已知问题

| 问题 | 影响 | 状态 |
|------|------|------|
| deadpool prompt 在 max_turns=4 时触发 vLLM 长上下文 400 错误 | 个别 prompt 采样失败 | V0.1 已知，非回归 |
| OrchRL `agent.py` f-string 语法不兼容 Python < 3.12 | MAS 启动失败 | 已修复（需 Python 3.10+ 兼容写法） |
| `SEARCH_MAS_LLM_BASE_URL` 环境变量覆盖 monitor URL | real 模式 MAS 报 404 | 验证脚本已主动清除 |
