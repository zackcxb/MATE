# 2026-03-09 验证修复收口 + 项目状态更新 Prompt

> 粘贴到 coding agent 窗口中执行。

---

## 角色

你是一位代码整理工程师，负责将 MATE-reboot 仓库工作树中的未提交变更收口为干净的原子提交。

## 前置阅读（按顺序）

1. `/home/cxb/MATE-reboot/AGENTS.md` — 项目治理规则（特别注意第 8 条 Git 规则和第 3.6 条验证证据要求）
2. `/home/cxb/MATE-reboot/docs/retros/2026-03-09-trajectory-engine-real-validation.md` — 本次真实环境验证的完整记录

## 背景

2026-03-09 的工作会话对真实环境验证脚本做了若干修复并完成了复跑，结果已写入 retro 文档和 `artifacts/` 产物。但这些变更停留在工作树，尚未整理为独立提交。目前 `main` 分支在 `d72abf8`，与 `origin/main` 同步。

**训练侧同事已初步开发完成，通过适配方式带上本仓库代码完成了端到端联调。** 当前工作重心已转向推理侧开发和上库（向 OrchRL 提交 PR）。

## 当前工作树状态

**已修改（未暂存）**：
- `scripts/run_real_validation.py` — 4 项修复：模型解析优先取 vLLM 在线模型集合、健康检查地址修正、expected_answer 字符串化列表归一化、相对模型路径按 config 目录解析
- `docs/project-context.md` — 更新到 2026-03-09 最新状态（验证进展、待办项、已知限制）
- `docs/prompts/2026-03-05-global-review-and-next-phase.md` — Prompt 模板简化

**未跟踪**：
- `tests/scripts/test_run_real_validation.py` — 验证脚本的 6 个单元测试
- `docs/retros/2026-03-09-trajectory-engine-real-validation.md` — 真实环境验证记录
- 7 个 `artifacts/trajectory_validation_*.json/parquet` 文件

## 任务

### 第一步：验证当前测试通过

```bash
cd /home/cxb/MATE-reboot
python -m pytest tests/ -q --tb=short
```

确认 66 passed，无失败。如有失败，先修复再继续。

### 第二步：更新 .gitignore

在 `/home/cxb/MATE-reboot/.gitignore` 末尾追加以下规则，排除被取代的旧验证产物和大型二进制文件：

```
# 验证产物：仅保留 *_fixed.json 和 exact_match.json，旧产物排除
artifacts/trajectory_validation_real.json
artifacts/trajectory_validation_real_smoke.json
artifacts/trajectory_validation_real_smoke_with_model.json
artifacts/*.parquet
```

### 第三步：更新 project-context.md

在 `/home/cxb/MATE-reboot/docs/project-context.md` 中做以下更新：

1. 在「当前阶段」顶部的 **粗体摘要** 改为：

```
**V0 实现完成；真实环境验证已在当前工作树复跑并定位 1 个长上下文失败样本；训练侧对接规格已写，但 `VerlBackend` 和训练主入口联调尚未开始。**
```

**注意**：训练侧同事已通过适配方式完成端到端联调。以上摘要中"联调尚未开始"是指本仓库的 `VerlBackend` 代码尚未落地，请将此句改为更准确的描述：

```
**V0 实现完成；真实环境验证通过（含 1 个已定位的长上下文失败样本）；训练侧同事已通过适配方式完成端到端联调；当前重心转向推理侧开发与上库。**
```

2. 确认「仓库状态」表格中的「待落盘工作」行改为：`2026-03-09 的验证修复、测试和产物已提交`（在第四步提交后生效）

3. 确认待办列表中：
   - `[ ] 将 2026-03-09 的验证修复、测试和证据整理为独立提交` → 改为 `[x]`
   - `[ ] 实现 VerlBackend 并与训练主入口完成联调` → 追加说明 `（训练侧已通过适配方式联调通过，本仓库 VerlBackend 待决定是否仍需独立实现）`
   - 新增待办：`[ ] 整理代码向 OrchRL 仓库提交 PR（不含本地临时文档和产物）`

### 第四步：整理并提交

分两个原子提交：

**提交 1：验证修复和测试**

```bash
git add scripts/run_real_validation.py
git add tests/scripts/test_run_real_validation.py
git add .gitignore
git commit -m "fix: real validation script fixes (model resolution, health check, reward normalization)

- Prefer live vLLM model list over potentially stale YAML config
- Fix retrieval service health check endpoint (/retrieve not /health)
- Normalize stringified expected_answer lists before reward comparison
- Resolve relative model paths from config file directory
- Add 6 unit tests for validation script"
```

**提交 2：文档和验证证据**

```bash
git add docs/project-context.md
git add docs/retros/2026-03-09-trajectory-engine-real-validation.md
git add docs/prompts/2026-03-05-global-review-and-next-phase.md
git add docs/prompts/2026-03-09-validation-fix-landing.md
git add artifacts/trajectory_validation_real_smoke_fixed.json
git add artifacts/trajectory_validation_real_fixed.json
git add artifacts/trajectory_validation_exact_match.json
git commit -m "docs: add 2026-03-09 real validation retro, update project context

- Real validation retro with evidence, root cause analysis for long-context failure
- Update project context: training side integrated via adapter, focus shifts to inference dev and upstreaming
- Include validated artifacts (*_fixed.json, exact_match.json)"
```

### 第五步：验证提交结果

```bash
# 确认测试仍通过
python -m pytest tests/ -q --tb=short

# 确认工作树干净（除了被 .gitignore 排除的旧产物）
git status

# 确认提交历史
git log --oneline -5
```

**验收标准**：
1. `pytest` 全部通过（≥66 passed）
2. `git status` 显示工作树干净或仅剩被 `.gitignore` 排除的文件
3. 两个新提交出现在 `git log` 中
4. `docs/project-context.md` 准确反映训练侧已完成适配联调的事实

## 不要做的事

- 不要修改 `mate/trajectory/` 下的任何源码
- 不要推送到远端（`git push`）—— 等用户确认后再推
- 不要创建新分支 —— 直接在 `main` 上提交
- 不要提交被 `.gitignore` 排除的旧产物文件
- 不要修改任何 `docs/plans/` 下的冻结设计文档
