"""Build AIAA4220 Project 3 presentation ON the official template
(Attention-Centered 3-Stage Mechanistic Narrative: Late Concat -> Channel FiLM -> Spatial & Relational Attention).

Run:  python3 scripts/make_plots.py && python3 build_ppt.py
Inputs : src/AIAA4220_Project3_Presentation_Template.pptx (official template, copied from 4090)
         assets/*.png|gif (generated from raw repository JSONs / rollouts), src/remote/json/*.json
Output : AIAA4220_Project3_Presentation_Attention.pptx (also synced to AIAA4220_Project3_Presentation.pptx)
"""
import copy, json, shutil
from pathlib import Path
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
A = ROOT / "assets"; J = ROOT / "src/remote/json"
TEMPLATE = ROOT / "src/AIAA4220_Project3_Presentation_Template.pptx"
OUT_ATTN = ROOT / "AIAA4220_Project3_Presentation_Attention.pptx"
OUT = ROOT / "AIAA4220_Project3_Presentation.pptx"

NAVY = RGBColor(0x11, 0x2B, 0x5C); BLUE = RGBColor(0x11, 0x4C, 0xB7); GOLD = RGBColor(0x82, 0x5D, 0x1C)
DARK = RGBColor(0x22, 0x22, 0x22); GREY = RGBColor(0x55, 0x5F, 0x6B); LIGHT = RGBColor(0xF2, 0xF4, 0xF8)
GREEN = RGBColor(0x0E, 0x96, 0x64); ORANGE = RGBColor(0xD9, 0x73, 0x0D); RED = RGBColor(0xC0, 0x39, 0x2B)
PURPLE = RGBColor(0x6D, 0x28, 0xD9); TEAL = RGBColor(0x0F, 0x76, 0x6E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF); PALEG = RGBColor(0xE3, 0xF4, 0xEC); PALEO = RGBColor(0xFB, 0xEB, 0xDA)
PALEB = RGBColor(0xE8, 0xF0, 0xFE); PALET = RGBColor(0xE6, 0xF4, 0xF1)
FONT = "Arial"

# ------------------------------------------------------------------ raw numbers
def load(f):
    return json.load(open(J / f))

b2 = load("b2_film_eval.json"); b2s7 = load("b2_film_s7n4.json")
base = load("baseline_eval.json"); s7 = load("s7n4_transfer.json")
ev2 = load("b2_attn_v2_eval.json"); f20_2 = load("b2_attn_v2_f020_eval.json")
pr2 = load("b2_attn_v2_probe.json"); fl2 = load("b2_attn_v2_failures.json")
fm = load("b2_attn_v2_feature_mechanics.json")
AGG, AGG7 = b2["aggregate"], b2s7["aggregate"]

def msd(v, nd=3):
    return f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=0):.{nd}f}"

def get_ev2(key, env_tag):
    if key in ("bc30", "film_nwd"):
        return ev2["references"][key][env_tag]
    if f"v1_{key}" in ev2["references"]:
        return ev2["references"][f"v1_{key}"][env_tag]
    return ev2["variants"][key][env_tag]

def get_fl2_agg(key):
    return fl2["references"]["aggregate"][key] if key in fl2["references"]["aggregate"] else fl2["aggregate"][key]

def get_fl2_case(key, seed_str):
    src = fl2["references"]["cases"][key] if key in fl2["references"]["cases"] else fl2["cases"][key]
    return src[seed_str]

def get_f20_s6(key):
    if key == "bc30":
        d = f20_2["references"]["bc_step_f020"]["S6N3"]
        return f"{d['success_mean']:.3f} ± {d['success_sd']:.3f}"
    if key == "film_nwd":
        d = f20_2["references"]["film_step_f020"]["S6N3"]
        return f"{d['success_mean']:.3f} ± {d['success_sd']:.3f}"
    if f"v1_{key}" in f20_2["variants"]:
        d = f20_2["variants"][f"v1_{key}"]["S6N3"]["closed_loop_success"]
        return f"{d['mean']:.3f} ± {d['sd']:.3f}"
    if f"v1_{key}" in f20_2["references"]:
        d = f20_2["references"][f"v1_{key}"]["S6N3"]["closed_loop_success"]
        return f"{d['mean']:.3f} ± {d['sd']:.3f}"
    if key in f20_2["variants"]:
        d = f20_2["variants"][key]["S6N3"]["closed_loop_success"]
        return f"{d['mean']:.3f} ± {d['sd']:.3f}"
    return "—"

p5 = [r for r in base["runs"] if "baseline_seed" in r["run"]]
p5_7 = [r["closed_loop_success"] for r in s7["runs"] if "baseline_seed" in r["run"]]

prs = Presentation(TEMPLATE)
SW, SH = prs.slide_width, prs.slide_height
tpl_slides = list(prs.slides)
TPL_CONTENT = tpl_slides[2]
LAYOUT = TPL_CONTENT.slide_layout
SECTION_LAYOUT = [l for l in prs.slide_layouts if l.name == "SECTION_HEADER"][0]

# ------------------------------------------------------------------ helpers
def _strip_placeholders(slide):
    for sh in list(slide.placeholders):
        sh._element.getparent().remove(sh._element)

def new_slide(title, title_size=23):
    s = prs.slides.add_slide(LAYOUT)
    _strip_placeholders(s)
    for sh in TPL_CONTENT.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 1:
            continue
        s.shapes._spTree.append(copy.deepcopy(sh._element))
    t = s.shapes.title
    bp = t.text_frame._txBody.find(qn("a:bodyPr"))
    for c in list(bp):
        bp.remove(c)
    t.text_frame.text = title
    r = t.text_frame.paragraphs[0].runs[0]
    r.font.size = Pt(title_size); r.font.color.rgb = NAVY; r.font.bold = True; r.font.name = FONT
    return s

def tb(slide, x, y, w, h, paras, size=17, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=4, font=FONT):
    box_ = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box_.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(spacing)
        runs = [(p, {})] if isinstance(p, str) else p
        for text, st in runs:
            r = para.add_run(); r.text = text
            f = r.font; f.name = font; f.size = Pt(st.get("size", size)); f.bold = st.get("bold", False)
            f.italic = st.get("italic", False); f.color.rgb = st.get("color", color)
    return box_

def box(slide, x, y, w, h, text, fill=LIGHT, line=GREY, size=14, color=DARK, bold=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE, lw=1.25):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line; sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = FONT
    return sh

def arrow(slide, x1, y1, x2, y2, color=GREY, w=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color; c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c

def pic(slide, path, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)

def table(slide, rows, x, y, w, col_w, size=12, header_fill=NAVY, row_h=0.28, bold_first_col=True, hl_rows=()):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gt.table
    tblPr = gt._element.graphic.graphicData.tbl.tblPr
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = t.cell(i, j); cell.text = ""
            cell.margin_left = cell.margin_right = Inches(0.04); cell.margin_top = cell.margin_bottom = Inches(0.015)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(rows[i][j]); r.font.name = FONT; r.font.size = Pt(size)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill; r.font.bold = True; r.font.color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = PALEB if i in hl_rows else (WHITE if i % 2 else LIGHT)
                r.font.color.rgb = DARK; r.font.bold = (j == 0 and bold_first_col) or i in hl_rows
    return gt

def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()

def footer(slide, text, y=5.08, w=7.95, size=10.5):
    tb(slide, 0.35, y, w, 0.3, [[(text, {"color": GREY, "size": size})]])


# ================================================================== SLIDE 1: title (0:15, A)
s1 = tpl_slides[0]
for sh in s1.placeholders:
    idx = sh.placeholder_format.idx
    if idx == 0:
        tf = sh.text_frame
        tf.text = "From Channel FiLM to Spatial & Relational Attention in BabyAI Imitation"
        r = tf.paragraphs[0].runs[0]; r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = NAVY
    elif idx == 1:
        tf = sh.text_frame
        bp = tf._txBody.find(qn("a:bodyPr"))
        for c in list(bp): bp.remove(c)
        for p in tf.paragraphs[1:]: p._p.getparent().remove(p._p)
        for r_ in list(tf.paragraphs[0].runs): r_._r.getparent().remove(r_._r)
        lines = [
            ("Track 2 · BabyAI GoToLocal · Mechanistic study: Late Concat → FiLM → Spatial & Relational Attention", 16.5, GREY, False),
            ("[Member 1 · ID]     [Member 2 · ID]     [Member 3 · ID]", 20, DARK, True),
            ("AIAA 4220 Project 3  ·  Branch: attention-mainline", 17, GREY, False),
        ]
        for i, (t_, sz, col, b_) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER; p.space_after = Pt(8)
            r = p.add_run(); r.text = t_; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b_
notes(s1, """
【讲者 A｜0:00–0:15｜15 秒】
大家好，我们是赛道 2 BabyAI 小组。我们围绕赛道手册提出的跨模态融合（FiLM vs Attention），不仅做了受控复现，更从表征机理出发，系统阐明了为什么空间竞争与关系注意力会在同等参数下本质超越通道调制 FiLM。下面先介绍任务与基线。
""")

# ================================================================== SLIDE 2: task & hypothesis (0:30, A)
s = new_slide("Task and testable mechanistic hypothesis")
pic(s, A / "p_task.png", 0.35, 1.12, h=3.95)
tb(s, 4.75, 1.15, 4.9, 3.8, [
    [("Task  ", {"bold": True, "color": BLUE}), ("BabyAI-GoToLocalS6N3-v0: face the named object within 36 steps", {})],
    [("Input  ", {"bold": True, "color": BLUE}), ("7×7×3 egocentric view + 5-token mission + heading → 7 actions", {})],
    [("Core Question  ", {"bold": True, "color": BLUE}), ("why does late concat fail, and why should spatial attention beat channel FiLM?", {})],
    [("Hypothesis  ", {"bold": True, "color": TEAL}), ("competitive spatial/relational attention beats BC30 by > 0.045 (3/3) AND surpasses param-matched FiLM on cross-scale S7N4/S8N6 via cross-cell competition", {"bold": True})],
], size=18, spacing=12)
footer(s, "Lit cells = agent's 7×7 egocentric view. Env: MiniGrid 3.2.0 / BabyAI [1, 5].", y=5.08, w=7.9)
notes(s, """
【讲者 A｜0:15–0:45｜30 秒】
环境是 BabyAI-GoToLocalS6N3：智能体看到 7×7×3 的局部符号视野、5 个词的指令和朝向，选 7 个离散动作，36 步内站到目标前并面朝它算成功。
比如左图任务是找紫色箱子，旁边却有同色的紫色钥匙。
我们的核心科学问题是：为什么晚期拼接会失效？同样把语言提前注入视觉前端，为什么跨网格竞争的空间注意力会在本质上优于全图广播的通道调制 FiLM？假设它在 S6N3 上 3/3 超过受控基线 0.045，并在跨尺度 S7N4/S8N6 显著超越等参数 FiLM。
""")

# ================================================================== SLIDE 3: exact baseline (1:00, A)
s = new_slide("Exact baseline: behavior cloning with late-concat BCNet")
r1, r2, r3, bh = 1.15, 1.92, 2.6, 0.58
row1 = [(0.35, 0.9, "obs\n7×7×3"), (1.45, 1.5, "Embed(11,8)×3\n→ 24×7×7"), (3.15, 1.42, "Conv 3×3 (32)\n+ ReLU"),
        (4.77, 1.42, "Conv 3×3 (32)\n+ ReLU"), (6.39, 0.8, "flatten\n1,568")]
for i, (x, w, t_) in enumerate(row1):
    box(s, x, r1, w, bh, t_, size=14)
    if i:
        px, pw, _ = row1[i - 1]; arrow(s, px + pw, r1 + bh / 2, x, r1 + bh / 2)
box(s, 0.35, r2, 0.9, bh, "mission\n5 tokens", size=13)
box(s, 1.45, r2, 3.82, bh, "Embed(14,32) · masked mean → ℓ (32)", size=14)
arrow(s, 1.25, r2 + bh / 2, 1.45, r2 + bh / 2)
box(s, 0.35, r3, 0.9, bh, "heading", size=14)
box(s, 1.45, r3, 1.57, bh, "one-hot → 4", size=14)
arrow(s, 1.25, r3 + bh / 2, 1.45, r3 + bh / 2)
cx = 7.38
box(s, cx, r1, 0.95, r3 + bh - r1, "late\nconcat\n1,604", fill=PALEO, line=ORANGE, size=14, color=ORANGE, lw=2)
arrow(s, 7.17, r1 + bh / 2, cx, r1 + bh / 2); arrow(s, 5.27, r2 + bh / 2, cx, r2 + bh / 2); arrow(s, 3.02, r3 + bh / 2, cx, r3 + bh / 2)
box(s, 8.52, r1, 1.13, 0.85, "Linear 128\n+ ReLU", size=14)
box(s, 8.52, r1 + 1.18, 1.13, 0.85, "Linear 7\n→ argmax", size=14)
arrow(s, cx + 0.95, r1 + 0.42, 8.52, r1 + 0.42); arrow(s, 9.085, r1 + 0.85, 9.085, r1 + 1.18)
tb(s, 2.75, r3 + 0.04, 4.55, 0.5, [[("vision is mission-blind (Δh2 = 0) until here →", {"color": ORANGE, "bold": True, "size": 14.5})]], align=PP_ALIGN.RIGHT)
tb(s, 0.35, 3.35, 9.3, 1.6, [
    [("Bottleneck  ", {"bold": True, "color": ORANGE}), ("Linear(1604→128) lacks multiplicative binding → color-swap flip rate 0.051, 38.3 repr. failures / 300", {})],
    [("Algorithm  ", {"bold": True, "color": BLUE}), ("BC on BabyAIBot demos · CE loss · Adam 1e-3 · 223,071 params · greedy argmax", {})],
    [("Setup  ", {"bold": True, "color": BLUE}), ("5,000 eps · ≤30 epochs · 3 seeds · PyTorch 2.15 1-thread CPU · MiniGrid 3.2.0", {})],
], size=17, spacing=6)
footer(s, "BabyAI [1, 5]. Disjoint seed range keeps 5,000 unique leak-free episodes (backup B1).")
notes(s, """
【讲者 A｜0:45–1:45｜60 秒】
冻结基线是行为克隆网络 BCNet，223,071 个参数：三通道符号图像经嵌入和两层 3×3 卷积展平为 1,568 维；指令词嵌入掩码平均为 32 维；朝向 4 维 one-hot。三者在最后橙色框才做晚期拼接，经单层 MLP 输出 7 个动作。
关键根因在于：视觉卷积直到展平前对指令完全盲视，换色词时 Conv2 特征变化 Δh2 恒为 0；单层加性线性投影难以构造“颜色 ∧ 类型”的乘性合取，导致换色词时动作仅翻转 5.1%，每 300 局出现 38.3 次表征绑定失败。
环境采用 PyTorch 2.15 单线程确定性执行，MiniGrid 3.2.0。种子划分细节在备份 B1。
""")

# ================================================================== SLIDE 4: baseline reproduction (1:00, A)
s = new_slide("Baseline reproduction evidence")
pic(s, A / "p_baseline.png", 0.35, 1.08, w=9.3)
tb(s, 0.35, 3.82, 9.3, 1.2, [
    [("Discrepancy  ", {"bold": True, "color": BLUE}), (f"prescribed patience=5: {msd([r['closed_loop_success'] for r in p5])}; seeds 1–2 stop at epochs 13/14 → controlled BC30 (30 epochs): {msd(AGG['bc30']['closed_loop_success']['values'])}", {})],
    [("Sanity  ", {"bold": True, "color": BLUE}), ("always-forward: 56.4% val acc but 0.060 Success; random walk 0.543 · S8N6 BC30 collapses to 0.490", {})],
], size=16.5, spacing=3)
footer(s, "3 init seeds × 300 unseen test seeds; bars = mean ± 1 SD (ddof=0), ○ □ △ = seeds 0/1/2. Raw: baseline_eval.json, b2_film_eval.json", size=10, w=8.05)
notes(s, """
【讲者 A｜1:45–2:45｜60 秒】
这是 3 个初始化种子、每种子 300 个未见测试种子的基线复现。
左图红虚线是手册原样的 patience=5 早停：种子 1 和 2 在第 13、14 个 epoch 就过早停止，成功率只有 0.747 ± 0.093。保持“按验证动作准确率选模”不变，跑满 30 个 epoch 得到更强、方差更小的受控基线 BC30：S6N3 0.814 ± 0.045，零样本 S7N4 0.697，而到 8×8 六物体 S8N6 崩盘至 0.490。
中间是合理性检查：永远前进有 56.4% 动作准确率但成功率仅 0.060，说明离线准确率不能替代闭环任务指标。下面由 B 深入剖析通道 FiLM 与空间/关系注意力的机理差异。
""")

# ================================================================== SLIDE 5: Mechanistic Core (1:00, B)
s = new_slide("Core principle: FiLM vs spatial & relational attention", title_size=22)
y1, y2, y3, bh = 1.12, 1.88, 2.66, 0.56
# Stage 1: Channel FiLM
box(s, 0.35, y1, 1.85, bh, "Stage 1 · Channel FiLM\n(225,183 p · +0.95%)", fill=PALEG, line=GREEN, color=GREEN, size=11, lw=1.8)
box(s, 2.34, y1, 3.03, bh, "H' = (1 + Δγ(ℓ)) ⊙ H + Δβ(ℓ)\nuniform across all 49 cells", fill=WHITE, line=GREEN, color=DARK, size=11.5, bold=False)
arrow(s, 2.20, y1 + bh / 2, 2.34, y1 + bh / 2, color=GREEN)
box(s, 5.52, y1, 4.13, bh, "Zero cell competition → h_mod T/maxD = 1.005 (S6) / 0.939 (S8)\nΔβ leaks background drift (0.787); O(N-1) distractor noise", fill=PALEO, line=ORANGE, color=DARK, size=10.8, bold=False)
arrow(s, 5.37, y1 + bh / 2, 5.52, y1 + bh / 2, color=ORANGE)

# Stage 2: Spatial-Softmax
box(s, 0.35, y2, 1.85, bh, "Stage 2 · Spatial-Softmax\n(225,183 p · exact tie!)", fill=PALEB, line=BLUE, color=BLUE, size=11, lw=2.0)
box(s, 2.34, y2, 3.03, bh, "α = softmax_49(⟨Wq ℓ, Wk H⟩ / √C)\nH' = (49 · α) ⊙ H", fill=WHITE, line=BLUE, color=DARK, size=11.5, bold=True)
arrow(s, 2.20, y2 + bh / 2, 2.34, y2 + bh / 2, color=BLUE)
box(s, 5.52, y2, 4.13, bh, "Global 49-cell competition → h_mod T/maxD = 7.669×!\nCrushes distractors (S8N6 +14.9%), but shrinks blocking box to 0.60×", fill=PALEB, line=BLUE, color=DARK, size=10.8, bold=False)
arrow(s, 5.37, y2 + bh / 2, 5.52, y2 + bh / 2, color=BLUE)

# Stage 3: v2 Attention Design Laws
box(s, 0.35, y3, 1.85, bh, "Stage 3 · v2 Attention\n(226k–230k p)", fill=PALET, line=TEAL, color=TEAL, size=11, lw=2.0)
box(s, 2.34, y3, 3.03, bh, "Map-Inject: [H; 49α] → Conv2(36→32)\nMH-Rel: + 49×49 ego self-attn (NoPE)", fill=WHITE, line=TEAL, color=DARK, size=11.2, bold=True)
arrow(s, 2.20, y3 + bh / 2, 2.34, y3 + bh / 2, color=TEAL)
box(s, 5.52, y3, 4.13, bh, "Resolves obstacle erasure & pos_emb overfitting:\nMap-Inject drift 0.569 (4501 3/3) · MH-Rel S6 0.972, repr 2.3/300", fill=PALET, line=TEAL, color=DARK, size=10.8, bold=False)
arrow(s, 5.37, y3 + bh / 2, 5.52, y3 + bh / 2, color=TEAL)

tb(s, 0.35, 3.38, 9.3, 1.55, [
    [("Controlled insertion  ", {"bold": True, "color": NAVY}), ("every module sits strictly between ReLU(Conv1) and Conv2; data, encoders, MLP head, optimizer & 30-epoch budget stay frozen", {})],
    [("Why competition matters  ", {"bold": True, "color": BLUE}), ("in softmax_49, conjunctive (color ∧ type) beats partial-match distractors exponentially exp(Δs), scaling cleanly from N=3 to N=6", {})],
], size=16.5, spacing=8)
footer(s, "Code: branches/b2_attn/attn_net.py, attn_v2_net.py, probe_feature_mechanics.py · Literature: Perez [3], Chaplot [8], Xu [9], Wang [10], Mott [11], Zambaldi [12].", size=10, w=8.8)
notes(s, """
【讲者 B｜2:45–3:45｜60 秒】
谢谢。所有的跨模态模块都严格插在 Conv1 和 Conv2 之间，其余结构、数据、优化器和 30 epoch 预算完全冻结。
为什么空间竞争注意力在本质上优于通道 FiLM？看三层演进：
第一层通道 FiLM，把同一组 Δγ 和 Δβ 广播到全部 49 个格子，网格之间零竞争。实测在融合层刚输出时，FiLM 的目标/最强干扰物能量比只有 1.005，到 S8N6 甚至跌破 1 到 0.939，且 Δβ 把背景空白格漂移推高到 0.787。
第二层 Spatial-Softmax，参数量与 FiLM 一模一样（225,183），但分母对 49 个格子做全局 softmax 归一化：同时匹配颜色和类型的目标以指数比值压垮部分匹配干扰物，融合层信噪比瞬间飙升到 7.669 倍！代价是纯乘性归一化也会把挡路箱子压暗到 0.60 倍。
第三层是我们的 v2 几何改进：Map-Inject 让原始特征 H 保持 1.0 倍零失真，直接把 4 张空间热图拼进 Conv2；MH-Relational 则叠加平滑自我中心坐标的 49×49 关系自注意力，让目标和障碍物做横向消息传递，彻底破解了“聚焦 vs 保真”的张力。
""")

# ================================================================== SLIDE 6: controlled protocol (1:00, B)
s = new_slide("Controlled experimental protocol & diagnostic probes")
yb, hb = 1.24, 0.5
segs = [(0.35, 3.3, "train [0:4000]", BLUE), (3.65, 1.25, "val [4000:4500]", GOLD), (4.9, 1.3, "test [4500:4800]", GREEN),
        (6.2, 1.3, "train [4800:5300]", BLUE)]
for xx, ww, lab, col in segs:
    box(s, xx, yb, ww, hb, lab, fill=col, line=WHITE, color=WHITE, size=12, shape=MSO_SHAPE.RECTANGLE)
box(s, 7.7, yb, 1.95, hb, "OOD S7N4 / S8N6\n[10000:10300]", fill=TEAL, line=WHITE, color=WHITE, size=11.5, shape=MSO_SHAPE.RECTANGLE)
tb(s, 0.35, yb + hb + 0.02, 9.3, 0.3, [[("4,500 train + 500 val = 5,000 expert eps", {"size": 13, "color": GREY}), ("   ·   300 unseen test seeds per env (S6N3, S7N4, S8N6) + 20% step-matched low-data", {"size": 12.5, "color": TEAL, "bold": True})]])
rows = [
    ("Budget & selection", ["5,000 expert eps · 30 epochs · wd=0 · max val action acc (test seeds never seen)"]),
    ("Architecture grid", ["14 controlled models (223k–231k params): BC30, FiLM, Chaplot GA, 5 v1 + 7 v2 Attention"]),
    ("Causal probes", ["(1) 183-state counterfactual swap  (2) 150-scene hidden SNR & background drift  (3) 300-ep failure taxonomy"]),
    ("Uncertainty", ["n = 3 init seeds · mean ± SD (ddof=0) · same-seed paired Δ vs BC30 & FiLM"]),
]
yy = 2.22
for k, v in rows:
    tb(s, 0.35, yy, 2.25, 0.55, [[(k, {"bold": True, "color": BLUE})]], size=17)
    tb(s, 2.6, yy, 7.05, 0.55, v, size=17, spacing=0)
    yy += 0.65
footer(s, "Frozen data/expert_5000.pkl & src/eval_closed_loop.py shared identically. FiLM snapshot archived on Git branch archive/film-mainline (v1.0-film-archive).", size=10, w=8.8)
notes(s, """
【讲者 B｜3:45–4:45｜60 秒】
实验协议完全受控。全部 14 个模型（从 223k 到 231k 参数）共用同一份冻结数据：4,500 条训练、500 条验证、300 个未见测试种子 4500 到 4799，以及零样本 7×7 四物体 S7N4 和更严苛的 8×8 六物体 S8N6（种子 10000 到 10299），再加步数对齐的 20% 低数据测试。
选模严格只看验证动作准确率。为了不只停留在刷分，我们配套了三套因果机制探针：183 态反事实换词探针、150 场景隐藏层信噪比与背景漂移探针，以及 300 局全量失败分类。所有指标均按 3 个独立初始化种子报告均值 ± 总体标准差 ddof=0 和同种子配对差。此前的 FiLM 版本已在 Git 封存为 archive/film-mainline。
""")

# ================================================================== SLIDE 7: Quantitative & Mechanistic Evidence (1:30, B)
s = new_slide("Quantitative & mechanistic proof: why Attention beats FiLM", title_size=22)
pic(s, A / "p_main_attn.png", 0.25, 1.02, w=4.62)
pic(s, A / "p_mechanics_slide.png", 5.05, 1.02, w=4.48)
ss_s6 = get_ev2("spatial_softmax_attn", "S6N3")["closed_loop_success"]
ss_s7 = get_ev2("spatial_softmax_attn", "S7N4")["closed_loop_success"]
ss_s8 = get_ev2("spatial_softmax_attn", "S8N6")["closed_loop_success"]
rel_s6 = get_ev2("mh_spatial_relational", "S6N3")["closed_loop_success"]
rel_p = ev2["variants"]["mh_spatial_relational"]["paired_vs_bc30"]["S6N3"]
ss_f20 = f20_2["variants"]["v1_spatial_softmax_attn"]["S6N3"]["closed_loop_success"]
tb(s, 0.35, 4.14, 9.3, 0.86, [
    [("Equal-param (225,183)  ", {"bold": True, "color": BLUE}),
     (f"Spatial-SS beats FiLM on S6N3 ({ss_s6['mean']:.3f} vs 0.959), S7N4 ({ss_s7['mean']:.3f} vs 0.900), S8N6 ({ss_s8['mean']:.3f} vs 0.738) & 20% ({ss_f20['mean']:.3f} vs 0.814)", {})],
    [("Mechanistic drivers  ", {"bold": True, "color": TEAL}),
     (f"MH-Rel hits top S6N3 {rel_s6['mean']:.3f} ± {rel_s6['sd']:.3f} (paired +{rel_p['delta_mean']:.3f}, 3/3) & SNR 2.43 · NoPE lifts Cross-MHA S8N6 0.747 → 0.809 (+6.2%)", {})],
], size=12.5, spacing=2)
footer(s, "Full 14-model table, feature-space probe & 20% curves in B2, B4–B6, B9. Raw: b2_attn_v2_{eval,f020_eval,feature_mechanics}.json", y=5.12, size=9.5, w=9.0)
notes(s, """
【讲者 B｜4:45–6:15｜90 秒】
看实验与机制双重证据。左图是三级架构演进在 S6N3、零样本 S7N4 和高拥挤 S8N6 上的闭环成功率。
首先看同等 225,183 参数的直接对决：FiLM 在 S6N3 到 S8N6 从 0.959 暴跌到 0.738；而 Spatial-Softmax 仅从 0.971 缓降到 0.887，在 S7N4 领先 FiLM 6.4 个点、在 S8N6 领先 14.9 个点！在 20% 低数据下，Spatial-Softmax 拿到 0.898，领先 FiLM 8.4 个点，仅用五分之一数据就反超了全量基线 BC30 的 0.814。
为什么会有这么大差距？看右图机制散点：横轴是 S8N6 换色词时的空间选择性，纵轴是 S8N6 闭环成功率。纯通道调制 FiLM 和 Chaplot 被锁死在左下角（选择性仅 1.90 倍，背景漂移 0.79）；一旦引入 49 格空间竞争，Spatial-Softmax 选择性翻倍至 3.59 倍，直接拉升到右上角 0.887！
同时看紫色虚线箭头：标准 Transformer Cross-MHA 原本受绝对位置编码拖累（背景漂移 0.823，全场最差），仅移除 pos_emb 得到 NoPE，S8N6 立刻回升 6.2 个点到 0.809！而叠加了关系自注意力的 MH-Rel 拿下全库最高 S6N3 0.972（相对 BC30 配对 +0.158，3/3 全胜）和最高部分匹配信噪比 2.43。下面由 C 结合视频剖析失败案例与边界。
""")

# ================================================================== SLIDE 8: Cases, Failure Mechanics & Limits (1:30 incl. 40 s video, C)
s = new_slide("Cases, Attention failure laws & hard limits (40 s clip)", title_size=22)
pic(s, A / "demo_bc30_vs_film.gif", 0.35, 1.05, w=4.75)
pic(s, A / "p_failshift.png", 5.38, 1.03, h=2.88)
tb(s, 0.35, 4.02, 9.3, 0.92, [
    [("Attention laws verified  ", {"bold": True, "color": TEAL}),
     ("4502 crowded keys: FiLM 0/3 → Spatial-SS & MH-Rel 3/3 · 4501 detour: v1 Softmax 1/3 → Map-Inject & MH-Rel 3/3 (attr. confusion 1.0/300)", {})],
    [("Hard boundary preserved  ", {"bold": True, "color": RED}),
     ("out-of-view L/R loops (seed 4568, 0/3 across single-frame models) → requires memory or DAgger, not stronger single-frame fusion", {})],
], size=12.5, spacing=3)
footer(s, "Video shows BC30 vs early modulation on unseen seeds 4501/4516/4560/4568; Attention head heatmaps & 4501 obstacle check in B6.", y=5.12, size=9.5, w=9.0)
notes(s, """
【讲者 C｜6:15–7:45｜90 秒，含 40 秒视频（进入本页即自动播放）】
谢谢。左边 40 秒视频展示未见测试种子上的典型行为对照：第 1 段是 4501 绕过挡路黄箱子，第 2 段 4516 紫色箱子对抗紫色钥匙（属性混淆），第 3 段 4560 绿色球对抗更近灰球（干扰物优先），第 4 段 4568 视野外背身目标陷入原地左右振荡。
结合右图四阶段失败拆解可以看到每一步改进背后的精准因果：
第一，从 BC30（55.7 次失败）到 FiLM（12.3 次），早期跨模态调制先消除了大部分表征错误，但 FiLM 仍在 4502 拥挤多钥匙局 0/3 失败；
第二，Spatial-Softmax 靠跨格竞争将总失败压到 8.7、4502 变为 3/3，但乘性掩码把 4501 挡路箱子压暗到 0.60 倍，导致绕障反降到 1/3；
第三，我们的 v2 改进彻底打通了这一矛盾：Map-Inject 零失真保留原始通道、MH-Relational 通过网格间关系消息传递，两者都将 4501 恢复到 3/3，MH-Rel 更将总失败压至全场最低 8.3，其中表征失败仅 2.3 次（属性混淆仅 1.0 例）且 4501、4502 双 3/3！
最后，所有单帧模型在视野外盲区 4568 上依然 0/3，这划清了视觉融合的物理边界：它属于部分可观测与状态覆盖问题，需要记忆或 DAgger 而非更强单帧融合。
""")

# ================================================================== SLIDE 9: Conclusion & Design Laws (0:15, C)
s = new_slide("Three Attention design laws, tradeoffs & questions", title_size=22)
tb(s, 0.35, 1.15, 9.3, 2.62, [
    [("1. Why Attention > FiLM  ", {"bold": True, "color": BLUE}),
     ("49-cell softmax lifts pre-Conv2 SNR 1.005 → 7.669× & selectivity 1.90 → 3.59×, curbing O(N-1) drift (S8N6 0.738 → 0.887; 20% data 0.814 → 0.898)", {})],
    [("2. Three design laws  ", {"bold": True, "color": TEAL}),
     ("(a) Translation-equivariant NoPE (+6.2% S8N6); (b) Uncorrupted map injection (drift 0.569, 4501 3/3); (c) Ego relational self-attn (S6 0.972, repr 2.3, 4501 & 4502 3/3)", {})],
    [("3. Honest tradeoff & limits  ", {"bold": True, "color": RED}),
     ("Single-head Softmax wins OOD (0.888) & 20% data (0.898); MH-Rel wins in-dist (0.972) & detours (3/3) · n=3 seeds; blind-spot 4568 needs memory/DAgger", {})],
], size=15.0, spacing=9)
box(s, 0.35, 3.92, 9.0, 0.42, "A: task & baseline   ·   B: FiLM vs Attention & feature mechanics   ·   C: failure laws & limits", fill=LIGHT, line=GREY, size=14.0, bold=False)
tb(s, 0.35, 4.46, 9.0, 0.62, [[("Thank you — questions?", {"bold": True, "color": NAVY, "size": 26})]], align=PP_ALIGN.CENTER)
notes(s, """
【讲者 C｜7:45–8:00｜15 秒】
总结三点：同等参数下空间竞争注意力借 49 格归一化全面超越通道 FiLM；我们提炼并验证了无位置过拟合 NoPE、无损热图注入和关系消息传递三条设计定律；同时诚实呈现单头 OOD 优势与多头绕障优势的帕累托权衡，以及视野外盲区边界。谢谢，欢迎提问！
""")

# ================================================================== BACKUP divider
s = prs.slides.add_slide(SECTION_LAYOUT)
s.shapes.title.text = "Backup slides (B1–B13)"
r = s.shapes.title.text_frame.paragraphs[0].runs[0]; r.font.color.rgb = NAVY; r.font.bold = True
for sh in TPL_CONTENT.shapes:
    if sh.is_placeholder and sh.placeholder_format.idx == 12:
        s.shapes._spTree.append(copy.deepcopy(sh._element))
tb(s, 0.5, 3.25, 8.6, 1.4, [
    "B1 config & seed split · B2 14-model master table · B3 FiLM capacity/wd ablation · B4 full v1+v2 Attention benchmark",
    "B5 feature-space SNR & drift · B6 4501 obstacle & Token-CoAttn heads · B7 counterfactual probe · B8 mission breakdown",
    "B9 data efficiency (10–100%) · B10 failure cases · B11 loss curves · B12 auxiliary branches · B13 references & AI use",
], size=14, color=GREY)
notes(s, "【备份页，不计入 8 分钟】仅用于 Q&A 深度追问。")

# ================================================================== B1 config
s = new_slide("B1 · Configuration, seed split and documented deviation", title_size=22)
rows = [["Item", "Setting (strictly identical across all 14 controlled models)"],
        ["Environment", "BabyAI-GoToLocalS6N3-v0 (H=36) · zero-shot S7N4 (H=49) & S8N6 (8×8, N=6, H=64)"],
        ["Expert data", "BabyAIBot, 5,000 successful eps = 4,500 train (16,534 steps) + 500 val (1,844 steps)"],
        ["Seed split", "train [0:4000] ∪ [4800:5300] · val [4000:4500] · test [4500:4800] · S7N4/S8N6 [10000:10300]"],
        ["Deviation", "cycling [0:4000] would duplicate 1,000 eps → disjoint [4800:5300]; assert_no_leak() passes"],
        ["Training", "Adam lr 1e-3, wd=0, batch 128, 30 epochs; select max val action acc (patience=5 also shown)"],
        ["Param budget", "BCNet 223,071 · FiLM 225,183 · Spatial-Softmax 225,183 · Map-Inject 226,335 · MH-Rel 229,572"],
        ["Software", "Python 3.12.14, PyTorch 2.15.0+cpu, MiniGrid 3.2.0, Gymnasium 1.4.0, NumPy 2.5.4"],
        ["Determinism", "CPU, OMP_NUM_THREADS=1, torch.set_num_threads(1), dual AMD EPYC 7453"],
        ["Git archive", "FiLM mainline sealed at archive/film-mainline (tag v1.0-film-archive); Attention on attention-mainline"]]
table(s, rows, 0.35, 1.12, 9.3, [1.5, 7.8], size=12.5, row_h=0.37)
notes(s, "【备份 B1｜A 负责回答】全部 14 个受控模型的共享超参、种子无泄漏验证与 Git 分支封存记录。")

# ================================================================== B2 master 14-model table
s = new_slide("B2 · Complete 14-model benchmark (3 seeds × 300 eps, ddof=0)", title_size=19)
m_order = [
    ("bc30", "BC30 (Concat)"),
    ("film_nwd", "FiLM-nwd (Channel)"),
    ("gated_attn_chaplot", "Chaplot GA (AAAI 18)"),
    ("spatial_attn_gate", "v1: Spatial-Gate"),
    ("spatial_softmax_attn", "v1: Spatial-Softmax"),
    ("cross_attn_mha", "v1: Cross-MHA (pos)"),
    ("cross_mha_nopos", "v2: Cross-MHA (NoPE)"),
    ("film_spatial_attn", "v1: FiLM+Spatial"),
    ("film_cross_mha", "v1: FiLM+MHA"),
    ("res_spatial_softmax", "v2: Res-Spatial-SS"),
    ("mh_res_spatial_softmax", "v2: MH-Res-SS"),
    ("attn_map_inject", "v2: Attn-Map-Inject"),
    ("token_coattn_spatial", "v2: Token-CoAttn"),
    ("mh_spatial_relational", "v2: MH-Spatial-Rel"),
]
trows = [["Method", "Params", "S6N3", "S7N4", "S8N6", "20% S6", "Fail (R+C)", "4501", "4502"]]
for k, nm in m_order:
    d6 = get_ev2(k, "S6N3"); d7 = get_ev2(k, "S7N4"); d8 = get_ev2(k, "S8N6")
    fa = get_fl2_agg(k)
    c4501 = get_fl2_case(k, "4501")["n_solved"]; c4502 = get_fl2_case(k, "4502")["n_solved"]
    trows.append([
        nm, f"{d6['n_params']:,}",
        f"{d6['closed_loop_success']['mean']:.3f}±{d6['closed_loop_success']['sd']:.3f}",
        f"{d7['closed_loop_success']['mean']:.3f}±{d7['closed_loop_success']['sd']:.3f}",
        f"{d8['closed_loop_success']['mean']:.3f}±{d8['closed_loop_success']['sd']:.3f}",
        get_f20_s6(k),
        f"{fa['n_failure']['mean']:.1f} ({fa['representation']['mean']:.1f}+{fa['state_coverage']['mean']:.1f})",
        f"{c4501}/3", f"{c4502}/3",
    ])
table(s, trows, 0.25, 1.05, 9.5, [1.82, 0.72, 1.15, 1.15, 1.15, 1.15, 1.28, 0.54, 0.54], size=10.2, row_h=0.255, hl_rows=(5, 10, 12, 14))
footer(s, "Highlighted: Spatial-Softmax (best S7N4/20%), Res-Spatial-SS (best S8N6 0.888), Map-Inject (4501 3/3), MH-Spatial-Rel (best S6N3 0.972, lowest repr 2.3).", y=5.06, size=9.8, w=9.2)
notes(s, "【备份 B2｜B 负责回答】全部 14 个受控模型在同分布、双跨尺度零样本、20% 低数据、失败拆分与 4501/4502 个案上的完整汇总表。")

# ================================================================== B3 FiLM capacity/wd ablation
s = new_slide("B3 · Capacity control & FiLM / weight-decay ablations", title_size=22)
pic(s, A / "p_ablation.png", 0.35, 1.08, w=9.3)
tb(s, 0.35, 4.6, 8.8, 0.6, [
    [("Capacity  ", {"bold": True, "color": BLUE}), ("Concat-Wide (224,683 p) = 0.718 vs FiLM-wd 0.946; FiLM-only (fewer params than BCNet) = 0.947 → gain is multiplicative structure, not param count", {})],
    [("Weight decay  ", {"bold": True, "color": BLUE}), ("wd=1e-4 hurts narrow concat (0.836 → 0.707), confirming wd=0 as fair controlled setting for all v1/v2 Attention runs", {})],
], size=13, spacing=1)
notes(s, "【备份 B3｜B 负责回答】证明增益来自早期跨模态乘性结构而非参数量；同时说明为何全部 Attention 与 FiLM 对照均统一采用 wd=0。")

# ================================================================== B4 Full v1+v2 Attention Benchmark
s = new_slide("B4 · Full v1 + v2 Attention benchmark (3 scales, 20%, failures)", title_size=19)
pic(s, A / "b2_attn_v2_benchmark.png", 1.55, 1.06, h=3.94)
footer(s, "Source: reports/b2_attn_v2_{eval,f020_eval,probe,failures}.json · 39 new v2 + f020 runs on 4090.", y=5.10, size=10, w=8.8)
notes(s, "【备份 B4｜B 负责回答】全部 12 个核心对比架构在全量三尺度、20% 步数对齐低数据、以及表征/覆盖失败拆解上的总览图。")

# ================================================================== B5 Feature-Space Mechanistic Proof
s = new_slide("B5 · Feature-space proof: hidden SNR, spatial selectivity & drift", title_size=20)
pic(s, A / "b2_attn_v2_feature_mechanics.png", 0.25, 1.12, w=9.5)
tb(s, 0.35, 4.04, 9.3, 0.90, [
    [("Key proof  ", {"bold": True, "color": BLUE}), ("FiLM h_mod T/maxD = 1.005 (S6) / 0.939 (S8) vs Spatial-Softmax 7.669 / 5.193 at identical 225,183 params", {})],
    [("Drift control  ", {"bold": True, "color": TEAL}), ("Removing pos_emb (NoPE) cuts Cross-MHA S8N6 drift 0.823 → 0.749 (+6.2% S8N6); Map-Inject achieves min drift 0.569", {})],
], size=12.8, spacing=2)
footer(s, "150 visible-target scenes × 3 seeds per env · Code: branches/b2_attn/probe_feature_mechanics.py · b2_attn_v2_feature_mechanics.json", y=5.12, size=9.8, w=9.0)
notes(s, "【备份 B5｜B 负责回答】直接量化了为什么 FiLM 在 N=6 拥挤环境下崩溃、为什么 Spatial-Softmax 与 MH-Rel 信噪比高出数倍，以及 NoPE 和 Map-Inject 压低背景漂移的直接证据。")

# ================================================================== B6 Diagnostic Mechanisms
s = new_slide("B6 · Attention mechanisms: 4501 obstacle, NoPE & CoAttn heads", title_size=19)
pic(s, A / "b2_attn_v2_mechanisms.png", 1.35, 1.06, h=3.94)
footer(s, "Token-CoAttn spontaneously allocates 0.69/0.73 weight to <color>+<type> in Heads 1–2 (target lock) while Head 3 tracks detour corridors.", y=5.10, size=10, w=9.0)
notes(s, "【备份 B6｜B 负责回答】(a) 验证 v1 Spatial-Softmax 把 4501 挡路黄箱子压暗至 0.60×；(b) 验证 NoPE 跨尺度曲线恢复；(c) 展示 Token-CoAttn 在无词级监督下自发分化出目标锁头与绕障走廊头。")

# ================================================================== B7 Counterfactual probe
s = new_slide("B7 · Counterfactual language probe (183 test states)", title_size=22)
pic(s, A / "p_probe.png", 0.35, 1.1, w=5.4)
tb(s, 5.95, 1.20, 3.7, 3.5, [
    [("Probe  ", {"bold": True, "color": BLUE}), ("fix (image, heading); swap mission to another object in the scene; count greedy action flips", {})],
    [("Channel vs Spatial  ", {"bold": True, "color": TEAL}), ("BC30 0.051 → Chaplot GA 0.225 → FiLM 0.268 → Spatial-Softmax 0.295 (type-swap 0.367)", {})],
], size=15.5, spacing=12)
footer(s, "Sources: b2_language_sensitivity.json, b2_attn_probe.json, b2_attn_v2_probe.json.", size=10.5)
notes(s, "【备份 B7｜B 负责回答】183 态反事实换词探针，Spatial-Softmax 颜色敏感度 0.295 与类型敏感度 0.367 均高于 FiLM（0.268 / 0.299）和 Chaplot GA（0.225 / 0.288）。")

# ================================================================== B8 Mission breakdown
s = new_slide("B8 · Mission-type breakdown (pooled 900 test episodes)", title_size=22)
pic(s, A / "p_mission.png", 0.6, 1.08, w=8.8)
tb(s, 0.35, 4.35, 9.3, 0.4, [[("BC30 → FiLM: 'the' (unique target, n=804) 0.797 → 0.954 · 'a' (n=96) 0.958 → 1.000 · weakest color blue 0.762 → 0.944", {})]], size=14.5)
footer(s, "Missions with <20 train demos not interpreted. Source: per_mission in b2_film_eval.json, baseline_eval.json", y=4.75, size=10.5)
notes(s, "【备份 B8｜A 负责回答】手册要求的 mission 类型分解。")

# ================================================================== B9 Data efficiency
s = new_slide("B9 · Step-matched data efficiency: 10 / 20 / 50 / 100% demos", title_size=22)
pic(s, A / "p_dataeff.png", 0.35, 1.06, w=6.15)
tb(s, 6.62, 1.18, 3.05, 3.6, [
    [("20% Spatial-SS  ", {"bold": True, "color": BLUE}), ("0.898 ± 0.009 S6N3 (0.854 S7N4) — beats 100% BC30 (0.814) with 5× less data!", {})],
    [("20% FiLM vs BC  ", {"bold": True, "color": GREEN}), ("FiLM 0.814 ± 0.023 vs BC 0.742 ± 0.026 (Res-SS 0.886, Cross-MHA 0.867)", {})],
    [("Why  ", {"bold": True, "color": TEAL}), ("bilinear spatial retrieval matches GoToLocal's 1-target geometry", {})],
], size=14.5, spacing=10)
footer(s, "Step-matched = 3,900 gradient updates per fraction. Raw: b5_data_eff_eval.json, b2_attn_v2_f020_eval.json", size=10.5)
notes(s, "【备份 B9｜C 负责回答】20% 低数据步数对齐下，Spatial-Softmax 达到 0.898，领先同参数 FiLM 整整 8.4 个点，且 S7N4 达 0.854（FiLM 为 0.750）。")

# ================================================================== B10 Failure cases
s = new_slide("B10 · Language-grounding failure cases (≥3 required)", title_size=22)
cases = json.load(open(A / "cases.json"))
info = {"4516": "attribute confusion: purple key vs box", "4565": "attribute confusion: yellow balls vs box",
        "4570": "attribute confusion: locks onto red ball", "4582": "attribute confusion: purple vs blue box",
        "4560": "distractor priority: nearer grey ball", "4568": "state coverage: target out of view, loop"}
pos = [(0.35, 1.12), (3.45, 1.12), (6.55, 1.12), (0.35, 2.98), (3.45, 2.98), (6.55, 2.98)]
for (x, y), sd in zip(pos, ["4516", "4565", "4570", "4582", "4560", "4568"]):
    c = cases[sd]; nfix = sum(f[0] for f in c["film"]); okst = [f[1] for f in c["film"] if f[0]]
    pic(s, A / f"case_{sd}_t0.png", x, y, w=1.3)
    film_txt = (f"FiLM {nfix}/3 ✓" if nfix else "FiLM 0/3 ✗") + (f" ({min(okst)}–{max(okst)} st.)" if okst and min(okst) != max(okst) else (f" ({okst[0]} st.)" if okst else ""))
    tb(s, x + 1.33, y - 0.04, 1.8, 1.75, [
        [(f"seed {sd}", {"bold": True, "color": NAVY, "size": 13})],
        [(c["mission"].replace("go to ", ""), {"size": 12, "italic": True})],
        [(info[sd], {"size": 11, "color": GREY})],
        [("BC30: timeout 36", {"size": 12, "color": RED, "bold": True})],
        [(film_txt, {"size": 12, "color": GREEN if nfix else RED, "bold": True})],
    ], spacing=1)
tb(s, 0.35, 4.60, 9.2, 0.65, [[("On crowded 4502 (purple key flanked by yellow keys): BC30 0/3, FiLM 0/3, Chaplot GA 0/3 → Spatial-Softmax, Res-SS, Token-CoAttn & MH-Rel ALL solve 3/3! On blocked 4501: Map-Inject & MH-Rel solve 3/3.", {"size": 11, "color": TEAL, "bold": True})]])
notes(s, "【备份 B10｜C 负责回答】5 个典型语言接地失败案例 + 1 个盲区案例，并补充说明拥挤 4502 与绕障 4501 上通道调制与空间/关系注意力的完整对比。")

# ================================================================== B11 Loss curves
s = new_slide("B11 · Training / validation loss curves (mean ± SD over 3 seeds)", title_size=22)
pic(s, A / "p_loss_backup.png", 0.35, 1.1, w=9.3)
be = {r["run"]: r["best_epoch"] for r in b2["runs"] + b2["baseline_runs"]}
tb(s, 0.35, 4.5, 8.8, 0.6, [[("Checkpoint = max val action acc. Best epochs: BC30 " +
    "/".join(str(be[f"runs/b3_bc30_seed{i}"]) for i in range(3)) + ", FiLM-nwd " +
    "/".join(str(be[f"runs/b2_film_nwd_seed{i}"]) for i in range(3)) + ", patience=5 26/8/9 · all v2 Attention runs val_acc 0.963–0.985.", {})]], size=14)
notes(s, "【备份 B11｜A 负责回答】训练/验证交叉熵损失曲线及选模规则说明。")

# ================================================================== B12 Auxiliary branches
s = new_slide("B12 · Auxiliary branches (isolated non-fusion controls)", title_size=22)
rows = [["Branch", "What changed", "S6N3 Success (n=3)", "Take-away"],
        ["B1 language encoder", "GRU / Transformer / pos-MLP, concat fusion", "0.819–0.868", "text encoder is not the bottleneck"],
        ["B4 curriculum", "easy→hard pacing, multi-scale envs (concat)", "0.790–0.857", "pacing does not fix binding"],
        ["B3 DAgger", "loop-unsticking queries (+8,526 labels)", "0.890 ± 0.015", "fixes state coverage, not grounding"],
        ["FiLM + DAgger", "B2 FiLM + B3 queries (+6,005 labels)", "0.970 ± 0.007", "exceeds 5,000-ep label budget"],
        ["B2 Attention v2", "Spatial-Softmax / MH-Spatial-Relational (same budget!)", "0.971–0.972", "matches FiLM+DAgger with zero extra labels"]]
table(s, rows, 0.35, 1.15, 9.3, [1.8, 3.3, 1.6, 2.6], size=12.5, row_h=0.5, hl_rows=(5,))
footer(s, "Sources: b1_langenc_eval.json, b4_curriculum_eval.json, b3_v2_eval_s6n3.json, b2_attn_v2_eval.json.", y=4.4, size=10.5)
notes(s, "【备份 B12｜C 负责回答】对比发现：在零额外专家查询的冻结预算下，空间/关系注意力（0.971–0.972）已经追平甚至微超消耗额外 6,005 条 DAgger 标签的 FiLM+DAgger（0.970）！")

# ================================================================== B13 References & AI use
s = new_slide("B13 · References and AI-use disclosure", title_size=22)
tb(s, 0.35, 1.06, 9.3, 2.95, [
    "[1] Chevalier-Boisvert et al., BabyAI (FiLM + AttGRU), ICLR 2019 (arXiv:1810.08272).",
    "[2] Perez et al., FiLM: Visual reasoning with a general conditioning layer, AAAI 2018 (arXiv:1709.07871).",
    "[3] Chaplot et al., Gated-Attention architectures for task-oriented language grounding, AAAI 2018.",
    "[4] Xu et al., Show, Attend and Tell (soft spatial attention), ICML 2015 · Wang et al., Residual Attention Network, CVPR 2017.",
    "[5] Mott et al., Attention Augmented Agents, NeurIPS 2019 · Zambaldi et al., Relational Deep RL, ICLR 2019.",
    "[6] Liu et al., CoordConv, NeurIPS 2018 · Lu et al., Co-Attention, NIPS 2016 · Kim et al., BAN, NeurIPS 2018.",
    "[7] Hashemi Dijujin et al., Inductive biases for zero-shot systematic generalization in language-informed RL, arXiv:2501.15270.",
    "[8] Farama Foundation, MiniGrid 3.2.0 / Gymnasium 1.4.0 · Ross et al., DAgger, AISTATS 2011.",
], size=12, spacing=2.5)
tb(s, 0.35, 4.02, 8.8, 1.0, [
    [("AI use (Handbook §7): ", {"bold": True, "color": BLUE}),
     ("Claude/Gemini IDE agents helped scaffold training/evaluation/plotting scripts, layout this deck and polish wording. "
      "No numbers, trajectories or citations were generated by AI: every value comes from runs/* and reports/*.json executed deterministically on our server.", {})],
], size=12)
notes(s, "【备份 B13｜C 负责回答】全部文献引用与生成式 AI 使用披露。")

# ================================================================== drop template slides 2..10
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst)[1:10]:
    prs.part.drop_rel(sldId.rId); sldIdLst.remove(sldId)
for k, sl in enumerate(prs.slides, 1):
    for fld in sl._element.iter(qn("a:fld")):
        if fld.get("type") == "slidenum":
            fld.find(qn("a:t")).text = str(k)
prs.save(OUT_ATTN)
shutil.copyfile(OUT_ATTN, OUT)
print("saved", OUT_ATTN, "and", OUT, "slides:", len(prs.slides))
