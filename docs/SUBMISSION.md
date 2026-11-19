# 提交清单核对（对照手册 §6 全部七大项）

> 对照《AIAA4220 期末仿真项目手册》§6 提交清单逐项核对仓库交付物状态（以 **支线 2：FiLM 跨模态融合 `b2_film`** 为唯一主修改）。
> 状态图例：✅ = 已在仓库就绪并核验通过；⚠️ = 模板/素材已备齐，提交前需成员填写个人信息或制作最终演示文稿。

---

## 1. 手册 §6 提交物逐项核对表

| # | 手册 §6 规定提交物 | 状态 | 仓库对应文件与核验说明 |
|---|---|---|---|
| 1 | **PPTX 与 PDF 格式的汇报幻灯片** | ⚠️ 待录制/导出 | 核心图表（`reports/figures/`、`reports/figures/b2_film/`）、双基线与 B2 FiLM 定量表、失败与修复 GIF/案例已全部就绪；请按 `AIAA4220_Project3_Presentation_Template.pptx` 排版（15 分钟总时长，每位成员均须发言，展示含不确定性的 3-seed 结果） |
| 2 | **DOCX 与 PDF 格式的书面报告** | ✅ / 同步产出 | 正式报告源码与导出文件位于 `reports/`（`FINAL_REPORT_EN.md` / `FINAL_REPORT_CN.md` 及对应 `AIAA4220_Project3_Report_{EN,CN}.{docx,pdf}`），严格遵守**正文 ≤ 6 页（含图表）+ Appendix A ≤ 1 页（团队贡献与 AI 披露）**限制 |
| 3 | **代码仓库压缩包或可访问 URL** | ✅ | 完整仓库位于 `p3-babyai/`：冻结核心 `src/`、唯一主修改 `branches/b2_film/`、四条辅助/次指标支线 `branches/b{1,3,4,5}_*/` 及顶层一键脚本 `run_baseline.sh`、`run_film_main.sh` |
| 4 | **锁定的环境规格与完整 README 命令** | ✅ | `requirements-lock.txt`（36 包精确锁定）、`pyproject.toml`、`docs/environment.md`（硬件/单线程配置/耗时/commit）与 `README.md`（含安装、基线、**B2 FiLM 主修改**、评估、作图完整命令） |
| 5 | **原始结果文件（Raw result files）** | ✅ | 主修改 B2：`reports/b2_film_eval.json`、`reports/b2_film_s7n4.json`、`reports/b2_language_sensitivity.json`、`reports/b2_failure_compare.json`；基线：`reports/baseline_eval.json`、`reports/s7n4_transfer.json`、`reports/b3_per_episode_baseline.json`、`reports/failures/seed*.json`；次指标与辅助支线：`reports/b5_data_efficiency_eval.json`、`reports/b{1,3,4,5}_*.json` |
| 6 | **处理后的汇总表、作图脚本与配置文件** | ✅ | 汇总表：`reports/b2_film_summary.md`、`reports/figures/summary_table.md`、`reports/data_efficiency.csv`、`reports/mission_breakdown.csv`、`reports/b{1,3,4,5}_*_summary.md`；作图脚本：`src/make_figures.py`、`branches/b2_film/make_figures.py`；配置文件：每个 `runs/*/config.json` |
| 7 | **至少一段 30–60 秒演示视频** | ✅ | `videos/demo_all.gif`（197 帧 @ 4fps ≈ **49 秒**，涵盖成功绕障与典型失败模式，严格落在 30–60 秒区间） |
| 8 | **团队贡献说明与生成式 AI 使用披露** | ✅ / ⚠️ | 生成式 AI 披露：`docs/AI_USAGE.md`（✅ 已完整备妥）；团队贡献表见下方模板及书面报告 Appendix A（⚠️ 提交前填入组员姓名/学号与具体分工） |

---

## 2. 书面报告与汇报幻灯片结构对齐（主修改：B2 FiLM 跨模态融合）

与课程报告模板及手册 §4 / §5 顺序严格对齐：

1. **任务与研究问题**：`BabyAI-GoToLocalS6N3-v0` 部分可观测语言导航模仿学习；提出**跨模态特征线性调制（FiLM）**改善视觉-语言属性绑定的可检验假设。
2. **基线复现与双基线协议**：复现未修改 `BCNet`（原基线 `patience=5` 为 `0.747 ± 0.093`），诊断并剥离早停截断效应，建立严格按 `val_action_acc` 选模的 30-epoch 受控基线 **`BC30`（S6N3 `0.814 ± 0.045`，S7N4 `0.697 ± 0.056`）**。
3. **唯一主修改（B2 FiLM）**：零额外专家样本下，单层线性恒等起步 `(1+Δγ)⊙h1+Δβ` 调制（`film_nwd` 225,183 参数；`film_only_wd` 221,087 参数）及等参数对照 `concat_wide`（224,683 参数）。
4. **受控定量结果与次指标**：`film_nwd` S6N3 达 **`0.959 ± 0.006`（`+0.144`，3/3 正）**，S7N4 零样本迁移达 **`0.900 ± 0.021`（`+0.203`，3/3 正）**；数据效率次指标显示 **`film_step@20%`（900 ep）即达 `0.814`**（5× 样本压缩追平 100% `BC30`）。
5. **机制探针、失败分解与逐例轨迹分析**：颜色敏感度从 `0.065` 升至 **`0.268`（4.1×）**，表征失败从 `38.3` 降至 **`6.3`（−83.5%）**；5 例基线语言接地失败 100% 修复，并诚实剖析 `seed 4568` 背后视野盲区振荡及 `FiLM + DAgger`（`0.970`）正交互补机制。

---

## 3. 团队贡献说明表（提交前按实际分工填写）

| 成员姓名 / 学号 | 主要负责工作 | 可独立解释的完整模块 |
|---|---|---|
| （待填成员 1） | 环境锁定、基线复现与数据审计、双基线（`patience=5` vs `BC30`）协议核验 | 全项目实验协议、基线架构、评估指标与失败分类 |
| （待填成员 2） | 主修改 B2（FiLM 跨模态融合）实现、参数匹配与正则消融、反事实语言探针与个案轨迹核验 | 全项目实验协议、`FiLMBCNet` 数学形式、机制探针与跨房间迁移 |
| （待填成员 3） | 数据效率次指标（B5）与辅助对照支线（B1/B3/B4）整合、图表脚本、演示视频与报告撰写 | 全项目实验协议、样本效率曲线、正交消融结论与局限性分析 |

> 手册 §7 提示：每位团队成员均须理解并能回答关于**完整项目**（而非仅个人代码片段）的技术问题。

---

## 4. 提交前最终自检命令

```bash
# 1. 核验全部冻结核心代码、专家数据、基线模型与历史评估 JSON 零改动
git diff --stat -- src/ data/ runs/ videos/ reports/baseline_eval.json reports/b2_film_eval.json

# 2. 秒级只读核验主修改 B2 FiLM 全部核心指标与双基线对比
bash run_film_main.sh --eval-only
```
