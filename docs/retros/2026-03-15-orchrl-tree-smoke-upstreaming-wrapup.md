# OrchRL Tree Smoke 上库收口同步

> 日期：2026-03-15
> 范围：`/home/cxb/OrchRL` 本地 `main`
> 目的：向 MATE 侧 Agent 同步 OrchRL tree smoke 在准备提主仓 PR 前的最终收口状态，说明当前上库形态、保留项与已明确排除项

## 1. 结论先行

OrchRL 侧 tree rollout smoke 目前已经完成从“分支内联调产物”到“可准备提 PR 的上库形态”的收口。

当前本地 `main` 头部提交：

- `1c21f20 feat: add MATE tree rollout smoke support`
- `f3d9a48 refactor: colocate search MAS smoke runner with example`

当前状态可以概括为：

- tree rollout 训练侧接入代码仍保持可运行
- 真实 Search MAS tree smoke 已再次验证可跑通
- smoke 入口已经从临时开发式结构收拢为 example 形态
- `docs/plans/**` 没有进入 `main`
- 未发现需要 MATE 侧新增 blocker 级修改的证据

## 2. 当前上库后的最终形态

### 2.1 smoke 入口不再放在临时开发路径

之前分支内为了联调方便，曾使用过偏开发态入口。

当前上库后，面向复现/运维的入口固定为：

- `examples/search_mas_tree/README.md`
- `examples/search_mas_tree/run.sh`
- `examples/search_mas_tree/run.py`

这次收口明确采用了与 `verl/examples` 更一致的组织方式：

- `examples/` 放可直接复现某一具体场景的入口与说明
- 不再把这个 smoke runner 放在顶层 `scripts/`

因此这次对 MATE 侧最重要的变化之一是：

- **Search MAS tree smoke 现在在 OrchRL 中被定位成 example / repro entrypoint，而不是 CI 用例，也不是通用功能脚本**

### 2.2 `docs/plans/**` 明确不上 `main`

用户要求非常明确：

- `docs/plans/**` 属于 Agent 开发时的临时工作资料
- 不应进入主线

当前本地 `main` 已按此要求收口：

- 没有保留此前 tree smoke 相关的 `docs/plans/**`

这意味着 MATE 侧如果需要理解当前最终形态，应以：

- OrchRL 仓内实际代码
- `examples/search_mas_tree/**`
- 真实 smoke 日志与测试

为准，而不是以先前的 Agent 设计/实现计划文档为准。

## 3. 当前保留了什么，删掉了什么

### 3.1 当前保留项

保留且准备上主仓的内容主要有三类：

1. tree rollout 训练接入与兼容修正
2. 保守但真实可跑的 smoke 配置
3. example + 轻量回归测试

对应的代表性文件：

- `orchrl/config/search/search_mas_tree_real_smoke.yaml`
- `orchrl/config/search/templates/search_mas_tree_real_smoke_template.yaml`
- `orchrl/config/search/data/search_mas_tree_real_smoke_prompts.jsonl`
- `orchrl/trainer/mate_config.py`
- `orchrl/trainer/mate_dataproto_adapter.py`
- `orchrl/trainer/mate_rollout_adapter.py`
- `orchrl/trainer/multi_agents_ppo_trainer.py`
- `examples/search_mas_tree/README.md`
- `examples/search_mas_tree/run.sh`
- `examples/search_mas_tree/run.py`

### 3.2 当前明确删掉/不保留项

这次收口明确排除了：

- `docs/plans/**`
- 原先临时放在 OrchRL 顶层 `scripts/search_mas_tree_smoke.py` 的位置

另外，当前 smoke 不被定位成：

- 常规 CI 流水线用例
- 强依赖统一环境的自动门禁

原因并没有变化：

- 需要真实 retrieval 服务
- 需要多卡资源
- 需要本机模型资产
- 需要较复杂的运行环境

## 4. 真实 smoke 当前的最终证据

本轮收口后，仍使用 `examples/search_mas_tree/run.sh` 对真实 smoke 做了再次验证。

当前 `logs/search_mas_tree_real_smoke.log` 中确认存在：

- `step 0 started`
- `training/global_step:0.000`
- `step 1 started`
- `Cleanup completed`

因此目前可以继续维持此前结论：

- OrchRL tree-mode 在 Search MAS 上已经真实走通 collect / update / 下一步训练循环 / clean exit

## 5. 当前测试面的收口判断

收口前专门做了一次“是否过度设计 / 过度测试”的额外审视。

结论是：

- 没有发现高严重度的过度设计信号
- smoke runner 保留一个 python 执行内核是合理的
- 保留两个 smoke 相关轻量测试也是合理的

但做了一处有价值的减法：

- `tests/orchrl/config/test_search_mas_tree_real_smoke_config.py`

它最开始更像“把所有调参值钉死”的测试，后面已收缩为只守住 smoke 的核心不变量，避免未来仅因资源调参微调就产生无意义 churn。

这点对 MATE 侧的意义是：

- OrchRL 当前不是在把临时联调参数永久冻结成不可调整的主线契约
- 当前保留的是“保守 smoke 不漂移”的约束，而不是“任何数值都绝对不能改”的约束

## 6. 对 MATE 侧最 relevant 的信息

截至目前，没有新增证据要求 MATE 再修改：

- `tree_rollout`
- `TreeEpisodeResult`
- `ReplayCache`
- MATE 核心采集逻辑

当前 OrchRL 侧仍然认为最关键的真实语义结论是：

- Search MAS trace 可能合法地产生 partial policy batches
- OrchRL trainer 不能再假设每一步都覆盖所有 policy

这是目前已经落实在 OrchRL 侧的兼容修正，不是新的 MATE blocker。

## 7. 建议 MATE 侧 Agent 这次审查重点看什么

这次已经不建议再把重点放在：

- “tree API 是否能被 OrchRL 接起来”
- “真实 smoke 是否完全跑不起来”

这些问题已经基本被证伪。

更适合 MATE 侧 Agent 这次审查的角度是：

1. 当前 OrchRL 对 tree public API 的消费语义是否与 MATE 预期一致
2. OrchRL 对 partial policy batches 的容忍是否与 MATE 侧真实 Search MAS 轨迹语义一致
3. 当前 vendored `trajectory/` 快照与 MATE 主仓语义是否仍保持一致，没有被 OrchRL 单边改歪
4. 当前 example / smoke 入口组织是否足够清晰，便于未来双仓协作定位问题

## 8. 当前可给 MATE 侧的简短结论

如果只同步一句话，可以直接说：

- **OrchRL tree smoke 已完成真实跑通与主线收口；当前没有发现要求 MATE 核心 tree 逻辑继续改动的新 blocker，后续更适合从 API 语义一致性和 vendored trajectory 对齐角度做审查。**
