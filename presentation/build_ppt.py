"""Build AIAA4220 Project 3 presentation ON the official template
(100% Attention-Protagonist Narrative; Channel FiLM & Late Concat as Controlled Baselines).

Output : AIAA4220_Project3_Presentation_Attention.pptx (synced to AIAA4220_Project3_Presentation.pptx)
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

NAVY = RGBColor(0x11, 0x2B, 0x5C); BLUE = RGBColor(0x1D, 0x4E, 0xD8); GOLD = RGBColor(0x82, 0x5D, 0x1C)
DARK = RGBColor(0x22, 0x22, 0x22); GREY = RGBColor(0x55, 0x5F, 0x6B); LIGHT = RGBColor(0xF2, 0xF4, 0xF8)
SLATE = RGBColor(0x47, 0x55, 0x69); PALESLATE = RGBColor(0xEC, 0xF0, 0xF5)
GREEN = RGBColor(0x0E, 0x96, 0x64); ORANGE = RGBColor(0xD9, 0x73, 0x0D); RED = RGBColor(0xC0, 0x39, 0x2B)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF); PALEO = RGBColor(0xFB, 0xEB, 0xDA)
PALEB = RGBColor(0xE8, 0xF0, 0xFE); PALET = RGBColor(0xE6, 0xF4, 0xF1)
FONT = "Arial"


def load(f):
    return json.load(open(J / f))


b2 = load("b2_film_eval.json"); b2s7 = load("b2_film_s7n4.json")
base = load("baseline_eval.json"); s7 = load("s7n4_transfer.json")
ev2 = load("b2_attn_v2_eval.json"); f20_2 = load("b2_attn_v2_f020_eval.json")
pr2 = load("b2_attn_v2_probe.json"); fl2 = load("b2_attn_v2_failures.json")
fm = load("b2_attn_v2_feature_mechanics.json")
sc_attn = load("b2_attn_v2_scaling.json")
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

prs = Presentation(TEMPLATE)
SW, SH = prs.slide_width, prs.slide_height
tpl_slides = list(prs.slides)
TPL_CONTENT = tpl_slides[2]
LAYOUT = TPL_CONTENT.slide_layout
SECTION_LAYOUT = [l for l in prs.slide_layouts if l.name == "SECTION_HEADER"][0]


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


def footer(slide, text, y=5.08, w=8.8, size=10.2):
    tb(slide, 0.35, y, w, 0.3, [[(text, {"color": GREY, "size": size})]])


# ================================================================== SLIDE 1: title (0:15, A)
s1 = tpl_slides[0]
for sh in s1.placeholders:
    idx = sh.placeholder_format.idx
    if idx == 0:
        tf = sh.text_frame
        tf.text = "Spatial & Relational Attention for Grounded Imitation in BabyAI"
        r = tf.paragraphs[0].runs[0]; r.font.size = Pt(31); r.font.bold = True; r.font.color.rgb = NAVY
    elif idx == 1:
        tf = sh.text_frame
        bp = tf._txBody.find(qn("a:bodyPr"))
        for c in list(bp): bp.remove(c)
        for p in tf.paragraphs[1:]: p._p.getparent().remove(p._p)
        for r_ in list(tf.paragraphs[0].runs): r_._r.getparent().remove(r_._r)
        lines = [
            ("Track 2 · BabyAI GoToLocal · Cross-Cell Competition, Uncorrupted Map Injection & Relational Self-Attention", 16, GREY, False),
            ("[Member 1 · ID]     [Member 2 · ID]     [Member 3 · ID]", 20, DARK, True),
            ("AIAA 4220 Project 3 · Embodied AI", 17.5, GREY, False),
        ]
        for i, (t_, sz, col, b_) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER; p.space_after = Pt(8)
            r = p.add_run(); r.text = t_; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b_
notes(s1, """
【讲者 A｜0:00–0:15｜15 秒】
大家好，我们是赛道 2 BabyAI 小组。我们的核心主线是早期空间与关系注意力机制：我们将通过严格的数学机理与隐藏层实测证明，为什么跨网格竞争与关系自注意力能在同等参数预算下全面超越晚期拼接和通道调制 FiLM。下面先介绍任务与基线。
""")

# ================================================================== SLIDE 2: task & hypothesis (0:30, A)
s = new_slide("Task and testable mechanistic hypothesis")
pic(s, A / "p_task.png", 0.35, 1.12, h=3.88)
tb(s, 4.68, 1.14, 4.98, 3.82, [
    [("Task  ", {"bold": True, "color": BLUE}), ("BabyAI-GoToLocalS6N3-v0: face the named object within 36 steps", {})],
    [("Challenge  ", {"bold": True, "color": BLUE}), ("seed 4502 (left): purple key flanked by 2 yellow keys — both late concat & channel FiLM fail 0/3!", {})],
    [("Question  ", {"bold": True, "color": BLUE}), ("does cross-cell spatial & relational attention solve crowded grounding & scale transfer?", {})],
    [("Hypothesis  ", {"bold": True, "color": TEAL}), ("early spatial/relational attention beats BC30 by > 0.045 (3/3 seeds) AND outperforms param-matched channel FiLM on S7N4 / S8N6 & 20% data", {"bold": True})],
], size=17.2, spacing=10)
footer(s, "Lit cells = agent's 7×7 egocentric view. Env: BabyAI [1], MiniGrid 3.2.0 [8].", y=5.10, w=8.8)
notes(s, """
【讲者 A｜0:15–0:45｜30 秒】
环境是 BabyAI-GoToLocalS6N3：智能体看到 7×7×3 的局部符号视野、5 词指令和朝向，在 36 步内站到目标前并面朝它算成功。
请看左图经典难例 seed 4502：目标是角落的紫色钥匙，但面前挡着两把同类异色的黄色钥匙——无论晚期拼接基线还是全局通道调制 FiLM，在这一局都是 0/3 全军覆没！
因此我们的可证伪假说是：引入跨 49 格竞争的空间与关系注意力，不仅在 S6N3 上 3/3 超过受控基线 0.045，还能在零样本大地图 S7N4、S8N6 和 20% 低数据下显著击败等参数的通道调制 FiLM。
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
footer(s, "BabyAI [1], MiniGrid [8]. Disjoint seed range keeps 5,000 unique leak-free episodes (backup B1).")
notes(s, """
【讲者 A｜0:45–1:45｜60 秒】
冻结基线是行为克隆网络 BCNet，223,071 个参数：三通道符号图像经嵌入和两层 3×3 卷积展平为 1,568 维；指令掩码平均为 32 维；朝向 4 维 one-hot，最后在橙色框才做晚期拼接。
根本症结在于：两层视觉卷积对指令完全盲视，换色词时特征变化 Δh2 恒为 0；展平后的单层加性线性投影无法直接计算“颜色 ∧ 类型”的乘性合取，导致换色词动作翻转率仅 5.1%，每 300 局出现 38.3 次表征绑定失败。
环境采用 PyTorch 2.15 单线程确定性执行，MiniGrid 3.2.0。
""")

# ================================================================== SLIDE 4: baseline reproduction (1:00, A)
s = new_slide("Baseline reproduction evidence")
pic(s, A / "p_baseline.png", 0.35, 1.10, w=9.3)
tb(s, 0.35, 3.86, 9.3, 1.15, [
    [("Discrepancy  ", {"bold": True, "color": BLUE}), (f"prescribed patience=5: {msd([r['closed_loop_success'] for r in p5])}; seeds 1–2 stop at epochs 13/14 → controlled BC30 (30 epochs): {msd(AGG['bc30']['closed_loop_success']['values'])}", {})],
    [("Sanity  ", {"bold": True, "color": BLUE}), ("always-forward: 56.4% val acc but 0.060 Success; random walk 0.543 · S8N6 BC30 collapses to 0.490", {})],
], size=16.5, spacing=3)
footer(s, "3 init seeds × 300 unseen test seeds; bars = mean ± 1 SD (ddof=0), ○ □ △ = seeds 0/1/2. Raw: baseline_eval.json, b2_attn_v2_eval.json", size=10, w=8.8)
notes(s, """
【讲者 A｜1:45–2:45｜60 秒】
这是 3 个初始化种子、每种子 300 个未见测试种子的基线复现。
左图红虚线是手册原样的 patience=5 早停：种子 1 和 2 在验证平台期的第 13、14 个 epoch 过早停止，成功率只有 0.747 ± 0.093。保持“最大验证动作准确率选模”不变，跑满 30 个 epoch 得到更严格的受控基线 BC30：S6N3 0.814 ± 0.045，零样本 S7N4 0.697，而到 8×8 六物体 S8N6 崩盘至 0.490。
中间合理性检查说明离线准确率不等于闭环成功率。下面由 B 介绍我们的空间与关系注意力方法及原理。
""")

# ================================================================== SLIDE 5: Our Method & Core Principle (1:00, B)
s = new_slide("Our method: spatial & relational attention (vs channel FiLM)", title_size=21)
# Top pipeline strip showing exact insertion site
py, pbh = 1.14, 0.42
p_nodes = [
    (0.35, 1.15, "obs 7×7×3", LIGHT, GREY, DARK),
    (1.68, 1.65, "Embed + Conv1\n→ H (32×7×7)", LIGHT, GREY, DARK),
    (3.52, 2.55, "Early Attention Module\n(ℓ / tokens × 7×7 grid)", PALEB, BLUE, BLUE),
    (6.26, 1.48, "Conv2 (32)\n+ ReLU", LIGHT, GREY, DARK),
    (7.92, 1.73, "flatten + MLP\n→ 7 actions", LIGHT, GREY, DARK),
]
for i, (xx, ww, tt, fl, ln, tc) in enumerate(p_nodes):
    box(s, xx, py, ww, pbh, tt, fill=fl, line=ln, color=tc, size=10.5, lw=2.0 if i == 2 else 1.2)
    if i:
        px, pw = p_nodes[i - 1][0], p_nodes[i - 1][1]
        arrow(s, px + pw, py + pbh / 2, xx, py + pbh / 2, color=BLUE if i in (2, 3) else GREY, w=1.8)

# 3-row architectural & mathematical comparison
y1, y2, y3, bh = 1.68, 2.38, 3.08, 0.58
# Row 1: Muted Slate Control (Channel FiLM)
box(s, 0.35, y1, 1.92, bh, "Control · Channel FiLM\n(225,183 p · +0.95%)", fill=PALESLATE, line=SLATE, color=SLATE, size=10.8, lw=1.4)
box(s, 2.40, y1, 2.98, bh, "H' = (1 + Δγ(ℓ)) ⊙ H + Δβ(ℓ)\nuniform across all 49 cells", fill=WHITE, line=SLATE, color=SLATE, size=11.2, bold=False)
arrow(s, 2.27, y1 + bh / 2, 2.40, y1 + bh / 2, color=SLATE)
box(s, 5.51, y1, 4.14, bh, "No cross-cell competition → h_mod T/maxD = 1.005 / 0.939\nΔβ leaks 0.787 drift; O(N-1) distractor noise (4502: 0/3)", fill=PALESLATE, line=SLATE, color=DARK, size=10.6, bold=False)
arrow(s, 5.38, y1 + bh / 2, 5.51, y1 + bh / 2, color=SLATE)

# Row 2: Protagonist 1 (Spatial-Softmax Attention)
box(s, 0.35, y2, 1.92, bh, "Ours · Spatial-Softmax\n(225,183 p · exact tie!)", fill=PALEB, line=BLUE, color=BLUE, size=10.8, lw=2.2)
box(s, 2.40, y2, 2.98, bh, "α = softmax_49(⟨Wq ℓ, Wk H⟩ / √C)\nH' = (49 · α) ⊙ H", fill=WHITE, line=BLUE, color=BLUE, size=11.2, bold=True)
arrow(s, 2.27, y2 + bh / 2, 2.40, y2 + bh / 2, color=BLUE)
box(s, 5.51, y2, 4.14, bh, "49-cell winner-take-all → h_mod T/maxD = 7.669×!\nS8N6 +14.9%, 20% data +8.4% (4502: 3/3); shrinks box to 0.60×", fill=PALEB, line=BLUE, color=DARK, size=10.6, bold=False)
arrow(s, 5.38, y2 + bh / 2, 5.51, y2 + bh / 2, color=BLUE)

# Row 3: Protagonist 2 (v2 Geometric Attention Laws)
box(s, 0.35, y3, 1.92, bh, "Ours v2 · Geometric Laws\n(226k–230k p)", fill=PALET, line=TEAL, color=TEAL, size=10.8, lw=2.2)
box(s, 2.40, y3, 2.98, bh, "Map-Inject: [H; 49α] → Conv2(36→32)\nMH-Rel: + 49×49 ego self-attn (NoPE)", fill=WHITE, line=TEAL, color=TEAL, size=11.0, bold=True)
arrow(s, 2.27, y3 + bh / 2, 2.40, y3 + bh / 2, color=TEAL)
box(s, 5.51, y3, 4.14, bh, "Fixes obstacle erasure & pos_emb overfitting:\nMap-Inject drift 0.569 (4501 3/3) · MH-Rel S6 0.972, repr 2.3/300", fill=PALET, line=TEAL, color=DARK, size=10.6, bold=False)
arrow(s, 5.38, y3 + bh / 2, 5.51, y3 + bh / 2, color=TEAL)

tb(s, 0.35, 3.78, 9.3, 1.22, [
    [("Core mechanism  ", {"bold": True, "color": BLUE}), ("in softmax_49, conjunctive (color ∧ type) suppresses partial-match distractors exponentially by exp(Δs), scaling cleanly from N=3 to N=6", {})],
    [("Strict fairness  ", {"bold": True, "color": TEAL}), ("data, encoders, MLP head, Adam 1e-3, wd=0, 30 epochs & val-acc selection stay 100% frozen across all models", {})],
], size=15.5, spacing=5)
footer(s, "Code: branches/b2_attn/{attn_net,attn_v2_net,probe_feature_mechanics}.py · Literature: Perez [2], Chaplot [3], Xu & Wang [4], Mott & Zambaldi [5], Liu/Lu/Kim [6].", y=5.10, size=9.6, w=9.1)
notes(s, """
【讲者 B｜2:45–3:45｜60 秒】
谢谢。上方流程图展示了我们的受控插入位置：在第一层卷积 ReLU(Conv1) 与第二层卷积 Conv2 之间插入早期注意力模块，其余一切结构与超参完全冻结。
为什么我们的空间与关系注意力能全面击败等参数的通道调制对照 FiLM？看下方三行算子对比：
第一行灰框是等参数对照 Channel FiLM（225,183 参数）：它把同一组 Δγ、Δβ 无差别广播到全部 49 个格子，网格之间零竞争，实测融合层输出目标/最强干扰物信噪比仅 1.005（S8N6 跌至 0.939），且 Δβ 造成 0.787 背景漂移，4502 拥挤局 0/3 溃败。
第二行蓝框是我们的 Spatial-Softmax，参数量与 FiLM 逐位相等（225,183），但分母的 49 格配分函数让同时匹配颜色和类型的目标以指数比值胜出，融合层信噪比瞬间拉高到 7.669 倍！S8N6 领先 FiLM 14.9 个点，但纯乘性掩码会把挡路箱子压暗到 0.60 倍。
第三行青绿框是我们的 v2 几何改进：Map-Inject 保持原始特征 H 1.0 倍无损并拼接 4 张热图；MH-Relational 叠加无位置过拟合的 49×49 自我中心关系自注意力，让目标与障碍直接协同推理！
""")

# ================================================================== SLIDE 6: controlled protocol (1:00, B)
s = new_slide("Controlled experimental protocol & causal probes")
yb, hb = 1.24, 0.5
segs = [(0.35, 3.3, "train [0:4000]", BLUE), (3.65, 1.25, "val [4000:4500]", GOLD), (4.9, 1.3, "test [4500:4800]", GREEN),
        (6.2, 1.3, "train [4800:5300]", BLUE)]
for xx, ww, lab, col in segs:
    box(s, xx, yb, ww, hb, lab, fill=col, line=WHITE, color=WHITE, size=12, shape=MSO_SHAPE.RECTANGLE)
box(s, 7.7, yb, 1.95, hb, "OOD S7N4 / S8N6\n[10000:10300]", fill=TEAL, line=WHITE, color=WHITE, size=11.5, shape=MSO_SHAPE.RECTANGLE)
tb(s, 0.35, yb + hb + 0.02, 9.3, 0.3, [[("4,500 train + 500 val = 5,000 expert eps", {"size": 13, "color": GREY}), ("   ·   300 unseen test seeds per env (S6N3, S7N4, S8N6) + 10/20/50/100% scaling", {"size": 12.5, "color": TEAL, "bold": True})]])
rows = [
    ("Budget & selection", ["5,000 expert eps · 30 epochs · wd=0 · max val action acc (test seeds never seen)"]),
    ("Architecture grid", ["14 controlled models (223k–231k p): 12 Attention/fusion designs vs BC30 & Channel FiLM controls"]),
    ("Causal probes", ["(1) 183-state counterfactual swap  (2) 150-scene hidden SNR & background drift  (3) 300-ep failure taxonomy"]),
    ("Uncertainty", ["n = 3 init seeds · mean ± SD (ddof=0) · same-seed paired Δ vs BC30 & Channel FiLM"]),
]
yy = 2.22
for k, v in rows:
    tb(s, 0.35, yy, 2.25, 0.55, [[(k, {"bold": True, "color": BLUE})]], size=17)
    tb(s, 2.6, yy, 7.05, 0.55, v, size=17, spacing=0)
    yy += 0.65
footer(s, "Identical frozen data/expert_5000.pkl, optimizer, and src/eval_closed_loop.py across all 14 models (full configs in B1–B2).", size=10, w=8.8)
notes(s, """
【讲者 B｜3:45–4:45｜60 秒】
实验协议严格受控。全部 14 个模型共用同一份冻结数据（4,500 训练 + 500 验证），在 300 个未见种子上评估同分布 S6N3、零样本 7×7 四物体 S7N4、高拥挤 8×8 六物体 S8N6，以及步数对齐的 10%、20%、50%、100% 数据效率曲线。
选模只看验证动作准确率。除了闭环成功率，我们还设计了三套因果探针直接解剖网络内部表征：183 态反事实换词探针、150 场景隐藏层信噪比与背景漂移探针、以及 300 局全量失败分类。不确定性按 3 个独立初始化种子报告均值 ± 总体标准差 ddof=0 及同种子配对差。
""")

# ================================================================== SLIDE 7: Quantitative & Mechanistic Evidence (1:30, B)
s = new_slide("Quantitative & mechanistic proof: why Attention wins", title_size=22)
pic(s, A / "p_main_attn.png", 0.25, 1.14, w=4.56)
pic(s, A / "p_mechanics_slide.png", 5.18, 1.14, w=4.32)
ss_s6 = get_ev2("spatial_softmax_attn", "S6N3")["closed_loop_success"]
ss_s7 = get_ev2("spatial_softmax_attn", "S7N4")["closed_loop_success"]
ss_s8 = get_ev2("spatial_softmax_attn", "S8N6")["closed_loop_success"]
rel_s6 = get_ev2("mh_spatial_relational", "S6N3")["closed_loop_success"]
rel_p = ev2["variants"]["mh_spatial_relational"]["paired_vs_bc30"]["S6N3"]
ss_f20 = f20_2["variants"]["v1_spatial_softmax_attn"]["S6N3"]["closed_loop_success"]
ss_f50 = sc_attn["spatial_softmax_step"]["f050"]["S6N3"]["closed_loop_success"]
tb(s, 0.35, 4.18, 9.3, 0.86, [
    [("Equal-param (225,183)  ", {"bold": True, "color": BLUE}),
     (f"Spatial-SS beats FiLM ctrl on S6N3 ({ss_s6['mean']:.3f} vs 0.959), S7N4 ({ss_s7['mean']:.3f} vs 0.900), S8N6 ({ss_s8['mean']:.3f} vs 0.738) & 20% data ({ss_f20['mean']:.3f} vs 0.814; 50% hits {ss_f50['mean']:.3f})", {})],
    [("Design laws verified  ", {"bold": True, "color": TEAL}),
     (f"MH-Rel hits top S6N3 {rel_s6['mean']:.3f} ± {rel_s6['sd']:.3f} (paired +{rel_p['delta_mean']:.3f} vs BC30, 3/3) & SNR 2.43 · NoPE lifts Cross-MHA S8N6 0.747 → 0.809 (+6.2%)", {})],
], size=12.3, spacing=2)
footer(s, "Full 14-model table, SNR probe & 10–100% scaling curves in B2, B4–B6, B9. Raw: b2_attn_v2_{eval,scaling,feature_mechanics}.json", y=5.12, size=9.5, w=9.0)
notes(s, """
【讲者 B｜4:45–6:15｜90 秒】
看定量与机制双重实证。左图虚线左侧是两个基线对照（橙色 BC30 与灰蓝色等参数 Channel FiLM），右侧是我们的蓝色 Spatial-Softmax 与青绿色 MH-Relational。
在精确相等的 225,183 参数下：从 S6N3 到 S8N6，FiLM 从 0.959 暴跌到 0.738；而我们的 Spatial-Softmax 稳在 0.971、0.964 和 0.887，在 S7N4 领先 FiLM 6.4 个点、在 S8N6 领先整整 14.9 个点！在低数据效率上优势同样惊人：20% 数据达 0.898（领先 FiLM 8.4 个点、反超全量 BC30），50% 数据即达 0.958 追平全量 FiLM！
右图散点直接揭示物理原因：横轴是换色词的空间选择性，纵轴是 S8N6 成功率。通道调制 FiLM 和 Chaplot 被困在左下角（选择性仅 1.90 倍）；而我们的空间注意力将选择性翻倍至 3.59 倍，跃升至右上角！紫色虚线箭头验证定律一：仅删去 Cross-MHA 的绝对位置编码（NoPE），背景漂移从 0.823 降到 0.749，S8N6 直接回升 6.2 个点！而叠加关系自注意力的 MH-Rel 创下全场最高 S6N3 0.972（相对 BC30 配对 +0.158，3/3 全胜）和最高部分匹配信噪比 2.43。下面请 C 演示真实闭环视频与失败定律。
""")

# ================================================================== SLIDE 8: 40 s Attention Rollout Video & Failure Laws (1:30, C)
s = new_slide("Attention failure laws & hard limits (40 s rollout clip)", title_size=22)
pic(s, A / "demo_attention_laws.gif", 0.32, 1.14, w=4.88)
pic(s, A / "p_failshift.png", 5.42, 1.14, h=2.80)
tb(s, 0.35, 4.05, 9.3, 0.92, [
    [("Verified in video  ", {"bold": True, "color": TEAL}),
     ("Clip 1 (4502): FiLM 0/3 → MH-Rel 3/3 (7 st) · Clip 2 (4501): v1 SS 1/3 → Map-Inject & MH-Rel 3/3 (8 st; attr. confusion 1.0/300)", {})],
    [("Hard POMDP limit  ", {"bold": True, "color": RED}),
     ("Clip 4 (seed 4568, target behind agent): all single-frame models loop L/R (0/3) → requires memory or DAgger, not single-frame fusion", {})],
], size=12.4, spacing=3)
footer(s, "Left: 3-column deterministic rollouts (demo_attention_laws.gif); Right: 300-ep failure decomposition (b2_attn_v2_failures.json, details in B6 & B10).", y=5.12, size=9.5, w=9.0)
notes(s, """
【讲者 C｜6:15–7:45｜90 秒，含 40.8 秒三栏对比视频（进入本页自动播放）】
谢谢。左边 40 秒视频是我们录制的未见测试种子三栏同步 rollout，每一段直接印证一条注意力机理：
【第 1 段 seed 4502 拥挤紫钥匙】BC30 和中间灰框的 Channel FiLM 双双被黄钥匙干扰超时（0/3），而右侧我们的 MH-Rel 7 步精准锁定紫钥匙（3/3），直观证明空间竞争击败通道广播！
【第 2 段 seed 4501 绕障定律】左侧 v1 Spatial-Softmax 因把挡路黄箱压暗到 0.60 倍卡死超时（1/3）；而中间零失真拼接热图的 Map-Inject 与右侧关系自注意力 MH-Rel 双双 8 步丝滑绕障（3/3）！
【第 3 段 seed 4516 同色紫箱 vs 紫钥】BC30 属性混淆超时（0/3），我们的 Spatial-SS 8 步直达目标（3/3）。
【第 4 段 seed 4568 盲区边界】目标在身后视野外，BC30、FiLM 和 MH-Rel 全部陷入左右转循环（0/3）。
看右图 300 局失败拆解：总失败从 BC30 的 55.7、FiLM 对照的 12.3，一路压到 Spatial-SS 的 8.7 和 MH-Rel 的全场最低 8.3，其中表征失败压至仅 2.3 次（属性混淆仅 1.0 例）！剩下的 6.0 次覆盖循环属于无记忆单帧策略的物理边界，下一步应结合时序记忆或解卡 DAgger。
""")

# ================================================================== SLIDE 9: Conclusion & Design Laws (0:15, C)
s = new_slide("Three Attention design laws, tradeoffs & questions", title_size=22)
tb(s, 0.35, 1.15, 9.3, 2.62, [
    [("1. Why Attention > FiLM  ", {"bold": True, "color": BLUE}),
     ("49-cell softmax lifts pre-Conv2 SNR 1.005 → 7.669× & selectivity 1.90 → 3.59×, curbing O(N-1) drift (S8N6 0.738 → 0.887; 20% data 0.814 → 0.898)", {})],
    [("2. Three design laws  ", {"bold": True, "color": TEAL}),
     ("(a) Translation-equivariant NoPE (+6.2% S8N6); (b) Uncorrupted map injection (drift 0.569, 4501 3/3); (c) Ego relational self-attn (S6 0.972, repr 2.3, 4501 & 4502 3/3)", {})],
    [("3. Honest tradeoff & limits  ", {"bold": True, "color": RED}),
     ("Single-head Softmax wins OOD (0.888) & low data (0.898 @20%); MH-Rel wins in-dist (0.972) & detours (3/3) · n=3 seeds; blind-spot 4568 needs memory/DAgger", {})],
], size=15.0, spacing=9)
box(s, 0.35, 3.92, 9.0, 0.42, "A: task & baseline   ·   B: Attention mechanics & cross-scale proof   ·   C: Attention failure laws & limits", fill=LIGHT, line=GREY, size=13.8, bold=False)
tb(s, 0.35, 4.46, 9.0, 0.62, [[("Thank you — questions?", {"bold": True, "color": NAVY, "size": 26})]], align=PP_ALIGN.CENTER)
notes(s, """
【讲者 C｜7:45–8:00｜15 秒】
总结：假设成立——同等参数下空间竞争注意力借 49 格归一化全面超越通道 FiLM；我们进一步验证了无位置过拟合 NoPE、无损热图注入和关系自注意力三条设计定律，并诚实界定了单头 OOD 优势与多头绕障优势的权衡及盲区边界。谢谢，欢迎提问！
""")

# ================================================================== BACKUP divider
s = prs.slides.add_slide(SECTION_LAYOUT)
s.shapes.title.text = "Backup slides (B1–B13)"
r = s.shapes.title.text_frame.paragraphs[0].runs[0]; r.font.color.rgb = NAVY; r.font.bold = True
for sh in TPL_CONTENT.shapes:
    if sh.is_placeholder and sh.placeholder_format.idx == 12:
        s.shapes._spTree.append(copy.deepcopy(sh._element))
tb(s, 0.5, 3.25, 8.6, 1.4, [
    "B1 config & seed split · B2 14-model master table · B3 param & wd controls · B4 full v1+v2 Attention benchmark",
    "B5 feature-space SNR & drift · B6 4501 obstacle & Token-CoAttn heads · B7 counterfactual probe · B8 mission breakdown",
    "B9 Attention scaling curves (10–100%) · B10 6-case comparison · B11 training/val curves · B12 aux branches · B13 refs & AI use",
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
        ["Param budget", "Spatial-Softmax 225,183 = Channel FiLM 225,183 · Map-Inject 226,335 · MH-Rel 229,572"],
        ["Low-data", "10% / 20% / 50% / 100% step-matched (16,534 transitions / ep via RandomSampler)"],
        ["Software", "Python 3.12.14, PyTorch 2.15.0+cpu, MiniGrid 3.2.0, Gymnasium 1.4.0, NumPy 2.5.4"],
        ["Determinism", "CPU, OMP_NUM_THREADS=1, torch.set_num_threads(1), dual AMD EPYC 7453"]]
table(s, rows, 0.35, 1.12, 9.3, [1.5, 7.8], size=12.5, row_h=0.37)
notes(s, "【备份 B1｜A 负责回答】全部 14 个受控模型的共享超参、步数对齐低数据协议与种子无泄漏验证。")

# ================================================================== B2 master 14-model table
s = new_slide("B2 · Complete 14-model benchmark (3 seeds × 300 eps, ddof=0)", title_size=19)
m_order = [
    ("bc30", "BC30 (Concat ctrl)"),
    ("film_nwd", "Channel FiLM (ctrl)"),
    ("gated_attn_chaplot", "Chaplot GA (ctrl)"),
    ("spatial_attn_gate", "v1: Spatial-Gate"),
    ("spatial_softmax_attn", "Ours: Spatial-Softmax"),
    ("cross_attn_mha", "v1: Cross-MHA (pos)"),
    ("cross_mha_nopos", "Ours: Cross-MHA (NoPE)"),
    ("film_spatial_attn", "v1: FiLM+Spatial"),
    ("film_cross_mha", "v1: FiLM+MHA"),
    ("res_spatial_softmax", "Ours: Res-Spatial-SS"),
    ("mh_res_spatial_softmax", "v2: MH-Res-SS"),
    ("attn_map_inject", "Ours: Attn-Map-Inject"),
    ("token_coattn_spatial", "Ours: Token-CoAttn"),
    ("mh_spatial_relational", "Ours: MH-Spatial-Rel"),
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
table(s, trows, 0.25, 1.08, 9.5, [1.86, 0.70, 1.14, 1.14, 1.14, 1.15, 1.29, 0.54, 0.54], size=10.1, row_h=0.255, hl_rows=(5, 10, 12, 14))
footer(s, "‘—’ = diagnostic full-data ablation (not re-trained at 20%). Highlighted: Spatial-SS, Res-SS, Map-Inject, MH-Spatial-Rel.", y=5.06, size=9.8, w=9.2)
notes(s, "【备份 B2｜B 负责回答】全部 14 个受控模型在同分布、双跨尺度零样本、20% 低数据、失败拆分与 4501/4502 个案上的完整汇总表。")

# ================================================================== B3 Capacity & wd controls
s = new_slide("B3 · Parameter-count & wd controls (Concat vs FiLM vs Attention)", title_size=20)
pic(s, A / "p_ablation.png", 0.35, 1.10, w=9.3)
tb(s, 0.35, 4.58, 9.0, 0.55, [
    [("Equal params (225,183)  ", {"bold": True, "color": BLUE}), ("Concat-Wide (224.7k) = 0.718 → Channel FiLM (225.2k) = 0.959 → Spatial-SS (225.2k) = 0.971 (S8N6: 0.738 → 0.887)", {})],
    [("Zero-wd control  ", {"bold": True, "color": TEAL}), ("wd=1e-4 hurts narrow concat (0.707 vs 0.814), so wd=0 is enforced uniformly across all Attention & baseline runs", {})],
], size=12.6, spacing=1)
notes(s, "【备份 B3｜B 负责回答】同屏对比证明：提升绝非来自参数增加（Concat-Wide 仅 0.718），而是先来自早期乘性绑定、再来自跨网格竞争与关系归纳偏置（Spatial-SS 0.971、MH-Rel 0.972）。")

# ================================================================== B4 Full v1+v2 Attention Benchmark
s = new_slide("B4 · Full v1 + v2 Attention benchmark (3 scales, 20%, failures)", title_size=19)
pic(s, A / "b2_attn_v2_benchmark.png", 1.55, 1.08, h=3.92)
footer(s, "Source: reports/b2_attn_v2_{eval,f020_eval,probe,failures}.json · FiLM shown in slate grey as channel control.", y=5.10, size=10, w=8.8)
notes(s, "【备份 B4｜B 负责回答】全部 12 个代表架构在三尺度闭环、20% 低数据及失败分类上的总览图。")

# ================================================================== B5 Feature-Space Mechanistic Proof
s = new_slide("B5 · Feature-space proof: hidden SNR, spatial selectivity & drift", title_size=20)
pic(s, A / "b2_attn_v2_feature_mechanics.png", 0.25, 1.12, w=9.5)
tb(s, 0.35, 4.04, 9.3, 0.90, [
    [("Key proof  ", {"bold": True, "color": BLUE}), ("Channel FiLM h_mod T/maxD = 1.005 (S6) / 0.939 (S8) vs Spatial-Softmax 7.669 / 5.193 at identical 225,183 params", {})],
    [("Drift control  ", {"bold": True, "color": TEAL}), ("Removing pos_emb (NoPE) cuts Cross-MHA S8N6 drift 0.823 → 0.749 (+6.2% S8N6); Map-Inject achieves min drift 0.569", {})],
], size=12.8, spacing=2)
footer(s, "150 visible-target scenes × 3 seeds per env · Code: branches/b2_attn/probe_feature_mechanics.py · b2_attn_v2_feature_mechanics.json", y=5.12, size=9.8, w=9.0)
notes(s, "【备份 B5｜B 负责回答】隐藏层特征几何铁证：直接量化空间竞争如何把融合层信噪比拉高 7 倍以上并抑制背景漂移。")

# ================================================================== B6 Diagnostic Mechanisms
s = new_slide("B6 · Attention mechanisms: 4501 obstacle, NoPE & CoAttn heads", title_size=19)
pic(s, A / "b2_attn_v2_mechanisms.png", 1.35, 1.08, h=3.92)
footer(s, "Panel (a) includes Map-Inject (1.0× raw h1) & MH-Rel; Panel (c) shows Token-CoAttn emergent <color>+<type> vs corridor heads.", y=5.10, size=10, w=9.1)
notes(s, "【备份 B6｜B 负责回答】(a) 对比 v1 压暗障碍至 0.6× 与 Map-Inject (1.0×) / MH-Rel (2.3×) 保留障碍；(b) NoPE 跨尺度曲线；(c) Token-CoAttn 无监督涌现句法-空间分工。")

# ================================================================== B7 Counterfactual probe
s = new_slide("B7 · Counterfactual language probe (183 test states)", title_size=22)
pic(s, A / "p_probe.png", 0.35, 1.10, w=5.65)
tb(s, 6.12, 1.22, 3.55, 3.5, [
    [("Protocol  ", {"bold": True, "color": BLUE}), ("fix (image, heading); swap mission to another object present in the room; measure greedy-action flip rate", {})],
    [("Attention wins  ", {"bold": True, "color": TEAL}), ("Spatial-SS reaches 0.295 color & 0.367 type flip rate (MH-Rel 0.275), beating FiLM ctrl (0.268 / 0.299) & BC30 (0.051)", {})],
], size=15.0, spacing=12)
footer(s, "Sources: b2_attn_probe.json, b2_attn_v2_probe.json (bars = mean ± SD over 3 seeds, ddof=0).", size=10.5)
notes(s, "【备份 B7｜B 负责回答】183 态反事实换词探针：Spatial-Softmax 在颜色交换（0.295）、类型交换（0.367）和综合敏感度上全面领先 Channel FiLM 与 Chaplot GA。")

# ================================================================== B8 Mission breakdown
s = new_slide("B8 · Mission-type breakdown (pooled 900 test episodes)", title_size=22)
pic(s, A / "p_mission.png", 0.35, 1.10, w=9.3)
tb(s, 0.35, 4.42, 9.3, 0.42, [[("Unique-target 'the' (n=804): BC30 0.797 → FiLM ctrl 0.954 → Spatial-SS 0.968 / MH-Rel 0.970 · weakest color 'green': 0.793 → 0.933 → 0.956 (MH-Rel)", {"bold": True, "color": DARK})]], size=13.2)
footer(s, "Missions with <20 train demos not interpreted. Source: per_mission in b2_attn_eval.json, b2_attn_v2_eval.json", y=5.06, size=10.2)
notes(s, "【备份 B8｜A 负责回答】900 局分词元汇总：在占比最大的唯一目标 'the'（804 局）及最难颜色词上，Spatial-SS（0.968）与 MH-Rel（0.970）均稳定超越 Channel FiLM（0.954）。")

# ================================================================== B9 Data efficiency
s = new_slide("B9 · Step-matched scaling curves: 10 / 20 / 50 / 100% demos", title_size=21)
pic(s, A / "p_dataeff.png", 0.35, 1.10, w=6.05)
tb(s, 6.55, 1.18, 3.12, 3.65, [
    [("10% Spatial-SS  ", {"bold": True, "color": BLUE}), ("0.802 ± 0.007 (450 eps) — nearly matches 100% BC30 (0.814) with 10× less data!", {})],
    [("20% & 50%  ", {"bold": True, "color": BLUE}), ("20% hits 0.898 (+8.4% vs FiLM 0.814); 50% hits 0.958 = 100% FiLM (0.959)!", {})],
    [("Bias–variance  ", {"bold": True, "color": TEAL}), ("1-head Softmax leads at 10–50%; higher-capacity MH-Rel crosses to #1 (0.972) at 100%", {})],
], size=14.0, spacing=10)
footer(s, "Step-matched = 16,534 transitions per epoch via RandomSampler. Raw: b2_attn_v2_scaling.json, b5_data_eff_eval.json", y=5.08, size=10.2)
notes(s, "【备份 B9｜C 负责回答】完整的 10/20/50/100% 缩放曲线展示了极经典的偏差-方差规律：单头 Spatial-Softmax 在 10%（0.802）、20%（0.898）、50%（0.958，追平全量 FiLM）全程领跑；多头关系注意力 MH-Rel 在 10%（0.717）略低，20% 起反超 FiLM（0.841），100% 登顶全场第一（0.972）。")

# ================================================================== B10 6-case comparison
s = new_slide("B10 · Diagnostic grounding & detour cases (3 seeds per model)", title_size=21)
cases_a = json.load(open(A / "cases_attn.json"))
case_meta = [
    ("4502", "crowded yellow keys vs purple key"),
    ("4501", "yellow box blocks direct path"),
    ("4516", "same-color purple key vs box"),
    ("4565", "yellow balls vs yellow box"),
    ("4560", "nearer grey ball distractor"),
    ("4568", "target behind agent (blind spot)"),
]
pos = [(0.35, 1.12), (3.45, 1.12), (6.55, 1.12), (0.35, 2.98), (3.45, 2.98), (6.55, 2.98)]
for (x, y), (sd, desc) in zip(pos, case_meta):
    c = cases_a[sd]
    n_bc = sum(x_[0] for x_ in c["bc30"])
    n_fl = sum(x_[0] for x_ in c["film"])
    n_ss = sum(x_[0] for x_ in c["ss"])
    n_rl = sum(x_[0] for x_ in c["mhrel"])
    pic(s, A / f"case_{sd}_t0.png", x, y, w=1.26)
    attn_col = TEAL if (n_ss == 3 or n_rl == 3) else RED
    tb(s, x + 1.30, y - 0.04, 1.82, 1.75, [
        [(f"seed {sd}", {"bold": True, "color": NAVY, "size": 12.5})],
        [(c["mission"].replace("go to ", ""), {"size": 11.5, "italic": True})],
        [(desc, {"size": 10.5, "color": GREY})],
        [(f"BC30 {n_bc}/3 · FiLM {n_fl}/3", {"size": 11.0, "color": SLATE, "bold": True})],
        [(f"SS {n_ss}/3 · MH-Rel {n_rl}/3", {"size": 11.5, "color": attn_col, "bold": True})],
    ], spacing=1)
tb(s, 0.35, 4.66, 9.2, 0.45, [[("Key contrasts: on 4502 FiLM fails 0/3 while SS & MH-Rel solve 3/3; on 4501 v1 SS drops to 1/3 while Map-Inject (3/3) & MH-Rel (3/3) fix detouring; on 4565 FiLM 2/3 → SS & MH-Rel 3/3.", {"size": 10.8, "color": TEAL, "bold": True})]])
notes(s, "【备份 B10｜C 负责回答】六个代表性诊断种子，每一格均同时列出 BC30、Channel FiLM、Spatial-SS 与 MH-Rel 的 3 种子通过率，直观展示 Attention 对 4502、4501、4565 的决定性优势及 4568 的共同盲区。")

# ================================================================== B11 Training / val curves
s = new_slide("B11 · Training CE & validation accuracy curves (3 seeds)", title_size=20)
pic(s, A / "p_loss_backup.png", 0.35, 1.10, w=9.3)
tb(s, 0.35, 4.48, 9.1, 0.55, [[("Spatial-Softmax & MH-Spatial-Rel surpass 0.95 val action acc by epoch 6–8 (vs epoch 13 for FiLM ctrl and 0.873 ceiling for BC30). All checkpoints selected strictly by max val action acc.", {"size": 13.2})]])
notes(s, "【备份 B11｜A 负责回答】四模型训练交叉熵与验证准确率曲线对比：我们的 Spatial-Softmax 与 MH-Rel 不仅收敛上限更高（0.981–0.982），收敛速度也显著快于 Channel FiLM 与 BC30。")

# ================================================================== B12 Auxiliary branches
s = new_slide("B12 · Auxiliary branches (isolated non-fusion controls)", title_size=22)
rows = [["Branch", "What changed", "S6N3 Success (n=3)", "Take-away"],
        ["B1 language encoder", "GRU / Transformer / pos-MLP, concat fusion", "0.819–0.868", "text encoder is not the bottleneck"],
        ["B4 curriculum", "easy→hard pacing, multi-scale envs (concat)", "0.790–0.857", "pacing does not fix binding"],
        ["B3 DAgger", "loop-unsticking queries (+8,526 labels)", "0.890 ± 0.015", "fixes state coverage, not grounding"],
        ["FiLM + DAgger", "Channel FiLM + B3 queries (+6,005 labels)", "0.970 ± 0.007", "exceeds 5,000-ep label budget"],
        ["Ours: Attention v1/v2", "Spatial-SS / MH-Spatial-Rel (zero extra labels!)", "0.971–0.972", "beats FiLM+DAgger within frozen budget"]]
table(s, rows, 0.35, 1.15, 9.3, [1.85, 3.3, 1.6, 2.55], size=12.5, row_h=0.5, hl_rows=(5,))
footer(s, "Sources: b1_langenc_eval.json, b4_curriculum_eval.json, b3_v2_eval_s6n3.json, b2_attn_v2_eval.json.", y=4.4, size=10.5)
notes(s, "【备份 B12｜C 负责回答】横向对照证明：我们的空间与关系注意力在不增加任何专家查询的冻结预算下达到 0.971–0.972，已经超越了额外消耗 6,005 个在线标签的 FiLM+DAgger（0.970）！")

# ================================================================== B13 References & AI use
s = new_slide("B13 · References and AI-use disclosure", title_size=22)
tb(s, 0.35, 1.08, 9.3, 2.95, [
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
notes(s, "【备份 B13｜C 负责回答】全部文献引用（编号 [1]–[8] 与正片完全对齐）与生成式 AI 使用披露。")

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
