"""Build AIAA4220 Project 3 presentation ON the official template
(Expanded 15-Core-Slide Spacious Edition: 100% Attention-Protagonist Narrative;
rich method, mechanistic proof, and diagnostic sections unpacked across 2–3 breathable slides each).

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
TEAL = RGBColor(0x0F, 0x76, 0x6E); PURPLE = RGBColor(0x6D, 0x28, 0xD9); PALEP = RGBColor(0xF3, 0xE8, 0xFF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF); PALEO = RGBColor(0xFB, 0xEB, 0xDA); PALER = RGBColor(0xFD, 0xEC, 0xEA)
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


def new_slide(title, title_size=22.5):
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


def tb(slide, x, y, w, h, paras, size=16.5, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=5, font=FONT):
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


def box(slide, x, y, w, h, text, fill=LIGHT, line=GREY, size=14, color=DARK, bold=True, shape=MSO_SHAPE.ROUNDED_RECTANGLE, lw=1.3, align=PP_ALIGN.CENTER, bold_first=False, first_color=None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line; sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08); tf.margin_top = tf.margin_bottom = Inches(0.04)
    lines = text.split("\n") if isinstance(text, str) else text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if bold_first and i == 0:
            p.space_after = Pt(2)
        r = p.add_run(); r.text = ln
        r.font.size = Pt(size + (0.6 if bold_first and i == 0 else 0))
        r.font.bold = True if (bold or (bold_first and i == 0)) else False
        r.font.color.rgb = (first_color or line) if (bold_first and i == 0) else color
        r.font.name = FONT
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


def footer(slide, text, y=5.08, w=9.0, size=10.0):
    tb(slide, 0.35, y, w, 0.3, [[(text, {"color": GREY, "size": size})]])


# ==============================================================================
# SPEAKER A · SLIDES 1–5 (0:00–2:30 = 150 s): Task, Baseline & Experimental Rigor
# ==============================================================================

# ------------------------------------------------------------------ SLIDE 1: Title (0:15, A)
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
            ("Track 2 · BabyAI GoToLocal · Cross-Cell Competition, Geometry Preservation & Relational Self-Attention", 16, GREY, False),
            ("[Member 1 · ID]     [Member 2 · ID]     [Member 3 · ID]", 20, DARK, True),
            ("AIAA 4220 Project 3 · Embodied AI", 17.5, GREY, False),
        ]
        for i, (t_, sz, col, b_) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER; p.space_after = Pt(8)
            r = p.add_run(); r.text = t_; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b_
notes(s1, """
【讲者 A｜0:00–0:15｜15 秒】
大家好，我们是赛道 2 BabyAI 小组。我们的核心主线是早期空间与关系注意力机制：我们将从算子机理与隐藏层实测出发，论证为什么跨网格竞争与自我中心关系自注意力能在同等参数预算下全面超越晚期拼接和通道调制 FiLM。
""")

# ------------------------------------------------------------------ SLIDE 2: Task & Motivating Challenge (0:35, A)
s = new_slide("Task & motivating grounding challenge: BabyAI-GoToLocal")
pic(s, A / "p_task.png", 0.35, 1.15, h=3.82)
tb(s, 4.65, 1.18, 5.00, 3.80, [
    [("Environment  ", {"bold": True, "color": BLUE}), ("6×6 room, N=3 colored objects (key / ball / box); 7×7×3 partial ego-view + 5-token mission → 7 actions within 36 steps", {})],
    [("Crowded trap  ", {"bold": True, "color": ORANGE}), ("seed 4502 (left): target purple key flanked by 2 yellow keys — both late-concat BC30 & channel FiLM fail 0/3!", {})],
    [("Hypothesis  ", {"bold": True, "color": TEAL}), ("early cross-cell spatial & relational attention beats BC30 by > 0.045 (3/3 seeds) AND beats equal-param channel FiLM across S7N4 / S8N6 & low-data regimes", {"bold": True})],
], size=16.5, spacing=11)
footer(s, "Lit cells = agent's 7×7 egocentric view. Env: BabyAI [1], MiniGrid 3.2.0 [8].", y=5.08)
notes(s, """
【讲者 A｜0:15–0:50｜35 秒】
任务环境是 BabyAI-GoToLocalS6N3：在 6×6 房间内随机生成 3 个带颜色和类型的物体，智能体仅能看到面前 7×7×3 的局部符号视野、5 词指令和朝向，需在 36 步内走到目标正前方并面朝它。
请看左图经典拥挤难例 seed 4502：指令要求去找角落的紫色钥匙，但面前紧挨着两把同类异色的黄色钥匙——无论晚期拼接基线还是全局通道调制 FiLM，在这一局都是 0/3 全军覆没！
因此我们的可证伪假说是：引入跨 49 格竞争的空间与关系注意力，不仅在 S6N3 上 3/3 超过受控基线 0.045，更能在零样本大地图 S7N4、S8N6 和低数据下显著击败等参数的通道调制 FiLM。
""")

# ------------------------------------------------------------------ SLIDE 3: Exact Baseline & Late-Concat Bottleneck (0:35, A)
s = new_slide("Prescribed baseline (BCNet) & why late concatenation fails")
r1, r2, r3, bh = 1.16, 1.92, 2.62, 0.56
row1 = [(0.35, 0.9, "obs\n7×7×3"), (1.45, 1.5, "Embed(11,8)×3\n→ 24×7×7"), (3.15, 1.42, "Conv 3×3 (32)\n+ ReLU"),
        (4.77, 1.42, "Conv 3×3 (32)\n+ ReLU"), (6.39, 0.8, "flatten\n1,568")]
for i, (x, w, t_) in enumerate(row1):
    box(s, x, r1, w, bh, t_, size=13.5)
    if i:
        px, pw, _ = row1[i - 1]; arrow(s, px + pw, r1 + bh / 2, x, r1 + bh / 2)
box(s, 0.35, r2, 0.9, bh, "mission\n5 tokens", size=13)
box(s, 1.45, r2, 3.82, bh, "Embed(14,32) · masked mean → ℓ (32)", size=13.5)
arrow(s, 1.25, r2 + bh / 2, 1.45, r2 + bh / 2)
box(s, 0.35, r3, 0.9, bh, "heading", size=13.5)
box(s, 1.45, r3, 1.57, bh, "one-hot → 4", size=13.5)
arrow(s, 1.25, r3 + bh / 2, 1.45, r3 + bh / 2)
cx = 7.38
box(s, cx, r1, 0.95, r3 + bh - r1, "late\nconcat\n1,604", fill=PALEO, line=ORANGE, size=14, color=ORANGE, lw=2)
arrow(s, 7.19, r1 + bh / 2, cx, r1 + bh / 2); arrow(s, 5.27, r2 + bh / 2, cx, r2 + bh / 2); arrow(s, 3.02, r3 + bh / 2, cx, r3 + bh / 2)
box(s, 8.52, r1, 1.13, 0.85, "Linear 128\n+ ReLU", size=13.5)
box(s, 8.52, r1 + 1.16, 1.13, 0.85, "Linear 7\n→ argmax", size=13.5)
arrow(s, cx + 0.95, r1 + 0.42, 8.52, r1 + 0.42); arrow(s, 9.085, r1 + 0.85, 9.085, r1 + 1.16)
tb(s, 2.75, r3 + 0.04, 4.55, 0.45, [[("vision is 100% mission-blind (Δh2 = 0) until here →", {"color": ORANGE, "bold": True, "size": 14.2})]], align=PP_ALIGN.RIGHT)
tb(s, 0.35, 3.38, 9.3, 1.58, [
    [("Root bottleneck  ", {"bold": True, "color": ORANGE}), ("additive Linear(1604→128) cannot directly compute multiplicative (color ∧ type) binding → color-swap flip rate 0.051, 38.3 repr. failures / 300", {})],
    [("Frozen setup  ", {"bold": True, "color": BLUE}), ("BabyAIBot 5,000 demos · CE loss · Adam 1e-3 · 223,071 params · greedy argmax · PyTorch 2.15 1-thread CPU", {})],
], size=16.5, spacing=8)
footer(s, "BabyAI [1], MiniGrid [8]. Disjoint seed range guarantees 5,000 unique leak-free episodes (details in B1).")
notes(s, """
【讲者 A｜0:50–1:25｜35 秒】
冻结基线是行为克隆网络 BCNet，共 223,071 个参数：三通道符号图像经过两层 3×3 卷积直接展平为 1,568 维，直到橙色框才与 32 维指令均值向量做晚期拼接。
它的根本数学缺陷在于：两层视觉卷积对语言完全盲视，更换颜色词时特征变化 Δh2 恒等于 0；展平后的单层加性投影无法直接表达“颜色 ∧ 类型”的乘性合取绑定，导致反事实换色翻转率仅 5.1%，每 300 局出现多达 38.3 次表征接地失败。
""")

# ------------------------------------------------------------------ SLIDE 4: Baseline Reproduction Evidence (0:35, A)
s = new_slide("Baseline reproduction, early-stopping audit & sanity checks")
pic(s, A / "p_baseline.png", 0.35, 1.14, w=9.3)
tb(s, 0.35, 3.92, 9.3, 1.10, [
    [("Early-stop audit  ", {"bold": True, "color": BLUE}), (f"handbook patience=5: {msd([r['closed_loop_success'] for r in p5])} (seeds 1–2 stop at ep 13/14) → full 30-epoch BC30: {msd(AGG['bc30']['closed_loop_success']['values'])} (fairer, stronger baseline)", {})],
    [("Scale collapse  ", {"bold": True, "color": ORANGE}), ("always-forward gets 56.4% val acc yet 0.060 Success; BC30 drops to 0.697 on S7N4 and collapses to 0.490 on S8N6", {})],
], size=16.0, spacing=5)
footer(s, "3 init seeds × 300 unseen test seeds; bars = mean ± 1 SD (ddof=0), ○ □ △ = seeds 0/1/2. Raw: baseline_eval.json, b2_attn_v2_eval.json", size=9.8)
notes(s, """
【讲者 A｜1:25–2:00｜35 秒】
这是 3 个初始化种子、每种子 300 个未见测试种子的基线复现。
左图红虚线揭示了一个关键细节：手册默认的 patience=5 会让种子 1 和 2 在第 13、14 轮过早停止，成功率仅 0.747 ± 0.093。为建立更严格的对照，我们保持“最大验证动作准确率选模”不变，跑满 30 轮得到受控基线 BC30：S6N3 提升至 0.814 ± 0.045，但在零样本 S7N4 跌至 0.697，到 8×8 六物体 S8N6 直接崩盘至 0.490。
""")

# ------------------------------------------------------------------ SLIDE 5: Controlled Protocol & 3-Layer Causal Probes (0:30, A)
s = new_slide("Controlled experimental protocol & three-layer causal probes")
yb, hb = 1.20, 0.48
segs = [(0.35, 3.3, "train [0:4000]", BLUE), (3.65, 1.25, "val [4000:4500]", GOLD), (4.9, 1.3, "test [4500:4800]", GREEN),
        (6.2, 1.3, "train [4800:5300]", BLUE)]
for xx, ww, lab, col in segs:
    box(s, xx, yb, ww, hb, lab, fill=col, line=WHITE, color=WHITE, size=12, shape=MSO_SHAPE.RECTANGLE)
box(s, 7.7, yb, 1.95, hb, "Zero-shot S7N4 / S8N6\n[10000:10300]", fill=TEAL, line=WHITE, color=WHITE, size=11.2, shape=MSO_SHAPE.RECTANGLE)
tb(s, 0.35, yb + hb + 0.04, 9.3, 0.30, [[("4,500 train + 500 val = 5,000 unique expert eps", {"size": 13, "color": GREY}), ("   ·   300 unseen test seeds per env + 10/20/50/100% step-matched scaling", {"size": 12.8, "color": TEAL, "bold": True})]])
rows_p = [
    ("Strict fairness", ["14 models (223k–231k params) share identical data, Conv/MLP, Adam 1e-3, wd=0, 30 epochs & val-acc selection"]),
    ("Probe 1 · Behavioral", ["183-state counterfactual mission swap (greedy action-change rate under color / type replacement)"]),
    ("Probe 2 · Feature-space", ["150 visible-target scenes × 3 seeds: pre-Conv2 Target/Distractor SNR, spatial selectivity & background drift"]),
    ("Probe 3 · Closed-loop", ["300-episode trajectory failure taxonomy (representation binding vs POMDP state-coverage loops); n=3, ddof=0"]),
]
yy = 2.18
for k, v in rows_p:
    tb(s, 0.35, yy, 2.55, 0.56, [[(k, {"bold": True, "color": BLUE})]], size=16.0)
    tb(s, 2.92, yy, 6.73, 0.56, v, size=16.0, spacing=0)
    yy += 0.68
footer(s, "Zero hyperparameter retuning across architectures; every claim is backed by raw JSON logs in reports/ (B1–B2).")
notes(s, """
【讲者 A｜2:00–2:30｜30 秒】
实验协议做到 100% 受控：全部 14 个模型共用同一份无泄漏冻结数据、优化器与验证选模准则，在同分布 S6N3、零样本 S7N4 / S8N6 及 10%–100% 步数对齐缩放曲线上评估。
为了不仅看成功率涨了多少、更彻底搞懂“为什么能赢”，我们设计了行为换词、隐藏层特征信噪比与背景漂移、以及 300 局闭环失败分类三层因果探针。下面由 B 分三页深入讲解我们的注意力设计与数学原理。
""")


# ==============================================================================
# SPEAKER B · SLIDES 6–11 (2:30–5:45 = 195 s):
#   - Slides 6–8 (3 slides): Method & Architectural Laws
#   - Slides 9–11 (3 slides): Cross-Scale, Sample-Efficiency & Feature-Space Proof
# ==============================================================================

# ------------------------------------------------------------------ SLIDE 6: Method I — Spatial Competition vs Channel Broadcast (0:35, B)
s = new_slide("Method I · Cross-cell spatial competition vs channel broadcast", title_size=21.5)
py, pbh = 1.14, 0.40
p_nodes = [
    (0.35, 1.15, "obs 7×7×3", LIGHT, GREY, DARK),
    (1.66, 1.68, "Embed + Conv1\n→ H (32×7×7)", LIGHT, GREY, DARK),
    (3.50, 2.58, "Early Fusion Site\n(ℓ ∈ ℝ³² × 7×7 grid)", PALEB, BLUE, BLUE),
    (6.24, 1.48, "Conv2 (32)\n+ ReLU", LIGHT, GREY, DARK),
    (7.88, 1.77, "flatten + MLP\n→ 7 actions", LIGHT, GREY, DARK),
]
for i, (xx, ww, tt, fl, ln, tc) in enumerate(p_nodes):
    box(s, xx, py, ww, pbh, tt, fill=fl, line=ln, color=tc, size=10.5, lw=2.0 if i == 2 else 1.2)
    if i:
        px, pw = p_nodes[i - 1][0], p_nodes[i - 1][1]
        arrow(s, px + pw, py + pbh / 2, xx, py + pbh / 2, color=BLUE if i in (2, 3) else GREY, w=1.8)

# Left two spacious comparison cards (equal 225,183 params!)
box(s, 0.35, 1.64, 6.45, 1.20,
    ["Control · Channel FiLM  (225,183 params — uniform broadcast)",
     "H'_{c,i,j} = (1 + Δγ_c(ℓ)) · H_{c,i,j} + Δβ_c(ℓ)     ∀ (i,j) ∈ 7×7",
     "• Zero cross-cell competition: same channel weights on all 49 cells → h_mod T/maxD = 1.005",
     "• Additive Δβ leaks 0.787 drift; distractor noise scales as O(N−1) (seed 4502: 0/3)"],
    fill=PALESLATE, line=SLATE, color=DARK, size=10.8, bold=False, lw=1.4, align=PP_ALIGN.LEFT, bold_first=True)

box(s, 0.35, 2.94, 6.45, 1.24,
    ["Ours · Spatial-Softmax Attention  (225,183 params — exact parameter tie!)",
     "α = softmax_{49}( ⟨W_q ℓ, W_k H_{i,j}⟩ / √C ),     H'_{:,i,j} = (49 · α_{i,j}) · H_{:,i,j}",
     "• Global 49-cell competition: conjunctive (color ∧ type) suppresses distractors by exp(Δs)",
     "• Lifts pre-Conv2 h_mod T/maxD to 7.669×!  Zero additive bias on empty cells (seed 4502: 3/3)"],
    fill=PALEB, line=BLUE, color=DARK, size=10.8, bold=False, lw=2.2, align=PP_ALIGN.LEFT, bold_first=True)

# Right: Real 7x7 heatmap on seed 4502
pic(s, A / "p_map_4502.png", 7.02, 1.64, w=2.52)

tb(s, 0.35, 4.30, 9.3, 0.68, [
    [("Why equal-param Attention wins  ", {"bold": True, "color": BLUE}),
     ("in seed 4502 (right), 49·α locks 6.26 onto the purple key vs 1.79 / 1.70 on flanking yellow keys — turning O(N−1) clutter into sharp winner-take-all localization", {})],
], size=13.8, spacing=2)
footer(s, "Code: branches/b2_attn/attn_net.py · Literature: Perez et al. [2], Xu et al. [4], Mott et al. [5].", y=5.08)
notes(s, """
【讲者 B｜2:30–3:05｜35 秒】
谢谢。我们先看第一页方法：同样在 Conv1 与 Conv2 之间插入早期融合，且参数量精确相等同为 225,183，为什么我们的 Spatial-Softmax 能全面击败 Channel FiLM？
看上方灰框：Channel FiLM 把同一组通道缩放 Δγ 和偏置 Δβ 无差别广播到全部 49 个网格，格子与格子之间零竞争，实测融合层目标与最强干扰物范数比只有 1.005，且 Δβ 在空白背景注入 0.787 的全局漂移，物体一多干扰就按 O(N-1) 累加。
再看下方蓝框我们的 Spatial-Softmax：49 格全局配分函数引入跨位置赢家通吃竞争，同时满足颜色和类型的目标得分高出 Δs，经指数放大后直接把半匹配干扰物压下去！如右图真实热力图所示，在 seed 4502 中紫钥匙获得 6.26 倍增益，两把黄钥匙被压制在 1.7 左右，融合层信噪比瞬间拉升至 7.669 倍！
""")

# ------------------------------------------------------------------ SLIDE 7: Method II — Fixing Naive Attention Pitfalls (Laws 1 & 2) (0:35, B)
s = new_slide("Method II · Fixing naive Attention pitfalls (Design Laws 1 & 2)", title_size=21.5)
tb(s, 0.35, 1.13, 9.3, 0.34, [
    [("Key insight  ", {"bold": True, "color": NAVY}), ("naive cross-attention & pure multiplicative masks suffer two geometric flaws in navigation:", {})],
], size=14.2)

# Left Card: Law 1 (NoPE vs pos_emb)
box(s, 0.35, 1.54, 4.50, 1.28,
    ["Pitfall 1 · Learned 2D Absolute pos_emb",
     "Standard Cross-MHA adds P ∈ ℝ^{1×49×32} trained on 6×6 rooms:",
     "• Memorizes 6×6 wall/corner priors inside ego-view",
     "• Worst OOD background drift (0.823) → S8N6 drops to 0.747"],
    fill=PALER, line=RED, color=DARK, size=10.8, bold=False, lw=1.5, align=PP_ALIGN.LEFT, bold_first=True)
arrow(s, 2.60, 2.84, 2.60, 3.04, color=PURPLE, w=2.4)
box(s, 0.35, 3.06, 4.50, 1.34,
    ["Design Law 1 · Translation-Equivariant NoPE",
     "Remove pos_emb completely (cross_mha_nopos, 227,360 p):",
     "• Conv1 3×3 already encodes local structure equivariantly",
     "• Cuts drift 0.823 → 0.749 & lifts S8N6 +6.2% (0.747 → 0.809)!"],
    fill=PALEP, line=PURPLE, color=DARK, size=10.8, bold=False, lw=2.2, align=PP_ALIGN.LEFT, bold_first=True)

# Right Card: Law 2 (Attn-Map-Inject vs Obstacle Erasure)
box(s, 5.15, 1.54, 4.50, 1.28,
    ["Pitfall 2 · Multiplicative Obstacle Erasure",
     "Pure (49·α) ⊙ H multiplies non-target cells toward zero:",
     "• On seed 4501, blocking yellow_box shrinks to 0.60× (seed 2: 0.16×)",
     "• Conv2 loses obstacle geometry → detour solves only 1/3 seeds!"],
    fill=PALEO, line=ORANGE, color=DARK, size=10.8, bold=False, lw=1.5, align=PP_ALIGN.LEFT, bold_first=True)
arrow(s, 7.40, 2.84, 7.40, 3.04, color=TEAL, w=2.4)
box(s, 5.15, 3.06, 4.50, 1.34,
    ["Design Law 2 · Uncorrupted Map Injection",
     "Attn-Map-Inject (226,335 p):  Conv2_{36→32}([ H ; 49·α^{(1..4)} ])",
     "• Raw visual tensor H stays 100% intact (1.0× obstacle/wall floor)",
     "• Lowest drift (0.569) & solves seed 4501 3/3 in optimal 8 steps!"],
    fill=PALET, line=TEAL, color=DARK, size=10.8, bold=False, lw=2.2, align=PP_ALIGN.LEFT, bold_first=True)

tb(s, 0.35, 4.48, 9.3, 0.52, [
    [("Takeaway  ", {"bold": True, "color": TEAL}),
     ("decoupling 'where the target is' (explicit spatial maps 49·α) from 'where obstacles sit' (unmodified H) eliminates destructive feature erasure", {})],
], size=13.8)
footer(s, "Code: branches/b2_attn/attn_v2_net.py · Motivated by Wang et al. Residual Attention [4] & Hashemi Dijujin et al. [7].", y=5.08)
notes(s, """
【讲者 B｜3:05–3:40｜35 秒】
然而我们并没有停留在基础注意力，而是进一步诊断出直接套用标准注意力时的两大几何陷阱，并提出两条改进定律：
左列是陷阱一：标准 Cross-MHA 会加上可学习的 49 格绝对位置编码 pos_emb，它在 6×6 小房间里过拟合了墙壁分布，导致 8×8 大地图背景漂移高达 0.823，成功率跌至 0.747。定律一（NoPE）：删去绝对位置编码、保留卷积天然的平移等变性，单这一项改动就让 S8N6 直接暴涨 6.2 个点至 0.809！
右列是陷阱二：纯乘性 Spatial-Softmax 在放大目标的同时，会把不匹配指令的挡路障碍物乘向 0——在 seed 4501 中挡路黄箱子被压暗到 0.60 倍，导致绕障成功率跌到 1/3。定律二（Attn-Map-Inject）：完全不破坏原始特征 H（保持 1.0 倍无损），而是把 4 头注意力热图作为显式空间指针通道拼接进 Conv2(36→32)，不仅取得全场最低背景漂移 0.569，更把 4501 绕障拉回 3/3 满分！
""")

# ------------------------------------------------------------------ SLIDE 8: Method III — Egocentric Relational & Co-Attention (Law 3) (0:30, B)
s = new_slide("Method III · Egocentric relational & token co-attention (Law 3)", title_size=21.5)
# Clean visual pipeline for MH-Spatial-Relational
ry, rbh = 1.20, 0.62
r_boxes = [
    (0.35, 1.55, "Conv1 features\nH (32×7×7)", LIGHT, GREY, DARK),
    (2.12, 2.25, "4-Head Residual\nSpatial-Softmax\nH·(1 + w ⊙ (49α − 1))", PALEB, BLUE, BLUE),
    (4.59, 2.35, "+ Smooth Ego Coords\nLinear(2→32) on [-1,1]²\n49×49 MH Self-Attn", PALET, TEAL, TEAL),
    (7.16, 2.49, "LN(H_res + γ_rel·Attn)\n→ Conv2 + MLP\n(MH-Rel · 229,572 p)", PALET, TEAL, DARK),
]
for i, (xx, ww, tt, fl, ln, tc) in enumerate(r_boxes):
    box(s, xx, ry, ww, rbh, tt, fill=fl, line=ln, color=tc, size=10.6, lw=2.0 if i in (1, 2) else 1.4)
    if i:
        px, pw = r_boxes[i - 1][0], r_boxes[i - 1][1]
        arrow(s, px + pw, ry + rbh / 2, xx, ry + rbh / 2, color=TEAL if i >= 2 else BLUE, w=2.0)

# Two clean explanation blocks below
box(s, 0.35, 1.98, 4.52, 2.02,
    ["Design Law 3 · Egocentric Relational Self-Attention",
     "• Local Conv2 (3×3) plans myopically when target & obstacle sit >2 cells apart",
     "• 49×49 self-attention broadcasts target bearing directly to blocking obstacle & corridor cells",
     "• Smooth coords (u,v) ∈ [-1,1]² preserve ego geometry without 6×6 grid overfitting (Zambaldi et al. [5])"],
    fill=PALET, line=TEAL, color=DARK, size=10.8, bold=False, lw=1.8, align=PP_ALIGN.LEFT, bold_first=True)

box(s, 5.13, 1.98, 4.52, 2.02,
    ["Complementary · Token-CoAttn-Spatial (227,435 p)",
     "• Replaces pooled vector ℓ with word-level bilinear co-attention over 5 tokens × 49 grid cells (Lu/Kim [6])",
     "• Per-head token softmax + spatial softmax_49",
     "• Unsupervised role split: Heads 1–2 bind <color>+<type> (0.70–0.73), Head 3 tracks free corridors!"],
    fill=PALEB, line=BLUE, color=DARK, size=10.8, bold=False, lw=1.6, align=PP_ALIGN.LEFT, bold_first=True)

tb(s, 0.35, 4.12, 9.3, 0.84, [
    [("Controlled ablation proof  ", {"bold": True, "color": TEAL}),
     ("vs its non-relational parent MH-Res-Softmax, adding 49×49 relational self-attention lifts S6N3 0.959 → 0.972 (#1 of 14 models), cuts repr errors 5.7 → 2.3/300, and solves seed 4501 (0/3 → 3/3)!", {})],
], size=13.8)
footer(s, "Code: branches/b2_attn/attn_v2_net.py · Literature: Zambaldi et al. ICLR 2019 [5], Liu et al. CoordConv [6], Lu et al. [6].", y=5.08)
notes(s, """
【讲者 B｜3:40–4:10｜30 秒】
第三页方法介绍我们的集大成架构——定律三：自我中心关系自注意力（MH-Spatial-Relational，229,572 参数）。
仅有目标定位还不够，当目标和挡路障碍相隔超过两格时，单靠 3×3 局部卷积容易短视卡死。因此我们在多头残差空间 Softmax 之上，注入平滑归一化自我中心坐标 [-1,1]²，并叠加一层 49×49 多头关系自注意力，让被点亮的目标格与挡路障碍格在进入决策前直接交互相对方位！严格消融证明：相比无关系层的父模型 MH-Res，S6N3 从 0.959 跃升至全场第一的 0.972，4501 绕障从 0/3 跃升至 3/3！右侧我们还设计了词级双线性协同注意力 Token-CoAttn，自动涌现出属性头与走廊导航头的分工。下面用三页展示定量与隐藏层铁证。
""")

# ------------------------------------------------------------------ SLIDE 9: Results I — Cross-Scale Generalization (0:35, B)
s = new_slide("Results I · Cross-scale generalization (S6N3 → S7N4 → S8N6)", title_size=21.5)
pic(s, A / "p_scale_core.png", 0.35, 1.14, w=9.3)
ss_s6 = get_ev2("spatial_softmax_attn", "S6N3")["closed_loop_success"]
ss_s7 = get_ev2("spatial_softmax_attn", "S7N4")["closed_loop_success"]
ss_s8 = get_ev2("spatial_softmax_attn", "S8N6")["closed_loop_success"]
rel_s6 = get_ev2("mh_spatial_relational", "S6N3")["closed_loop_success"]
rel_p = ev2["variants"]["mh_spatial_relational"]["paired_vs_bc30"]["S6N3"]
tb(s, 0.35, 4.10, 9.3, 0.92, [
    [("Equal-param victory (225,183)  ", {"bold": True, "color": BLUE}),
     (f"Spatial-SS beats Channel FiLM ctrl on S6N3 ({ss_s6['mean']:.3f} vs 0.959), S7N4 ({ss_s7['mean']:.3f} vs 0.900, +6.4%) & crowded S8N6 ({ss_s8['mean']:.3f} vs 0.738, +14.9%; Res-SS hits 0.888, +15.0%!)", {})],
    [("Peak in-dist precision  ", {"bold": True, "color": TEAL}),
     (f"MH-Spatial-Rel reaches #1 S6N3 = {rel_s6['mean']:.3f} ± {rel_s6['sd']:.3f} (same-seed paired +{rel_p['delta_mean']:.3f} ± {rel_p['delta_sd']:.3f} vs BC30, 3/3 seeds > +0.11)", {})],
], size=14.4, spacing=4)
footer(s, "Bars = mean ± SD across n=3 init seeds (ddof=0), 300 unseen episodes per env. Full 14-model table in B2 & B4.", y=5.08)
notes(s, """
【讲者 B｜4:10–4:45｜35 秒】
先看第一组核心实证：从训练分布 6×6 三物体 S6N3，零样本外推到 7×7 四物体 S7N4 与高拥挤 8×8 六物体 S8N6。
虚线左侧是两个冻结对照，右侧是我们的注意力家族。随着干扰物从 3 个增加到 6 个，橙色 BC30 崩盘至 0.490，灰蓝色等参数对照 Channel FiLM 也从 0.959 暴跌 22 个点至 0.738！
反观我们的蓝色 Spatial-Softmax（同样精确 225,183 参数）：三尺度稳居 0.971、0.964 和 0.887，在 S7N4 领先 FiLM 6.4 个点，在 S8N6 领先整整 14.9 个点（残差版 Res-SS 更达 0.888，领先 15.0 个点）！而青绿色 MH-Relational 创下全部 14 个模型的最高同分布成功率 0.972 ± 0.004，相对 BC30 同种子配对提升 +0.158，3 个种子全部大比分胜出。
""")

# ------------------------------------------------------------------ SLIDE 10: Results II — Step-Matched Sample Efficiency (0:30, B)
s = new_slide("Results II · Sample efficiency across 10%–100% expert demos", title_size=21.5)
pic(s, A / "p_scaling_core.png", 0.35, 1.14, w=9.3)
tb(s, 0.35, 4.10, 9.3, 0.92, [
    [("Half the data = full FiLM  ", {"bold": True, "color": BLUE}),
     ("at 50% demos (2,250 eps), Spatial-SS hits 0.958 ± 0.006 on S6N3 (matching 100% FiLM 0.959) and 0.911 on S7N4 (beating 100% FiLM 0.900; MH-Rel hits 0.914!)", {})],
    [("Extreme low-data regime  ", {"bold": True, "color": TEAL}),
     ("at 20% data (900 eps), Spatial-SS beats FiLM by +8.4% on S6N3 (0.898 vs 0.814) & +10.4% on S7N4 (0.854 vs 0.750); even 10% S7N4 (0.732) beats 100% BC30 (0.697)!", {})],
], size=14.2, spacing=4)
footer(s, "All fractions step-matched to 16,534 gradient updates/epoch via RandomSampler. Raw: b2_attn_v2_scaling.json, b5_data_eff_{eval,s7n4}.json.", y=5.08)
notes(s, """
【讲者 B｜4:45–5:15｜30 秒】
第二组实证把全套步数对齐的数据效率曲线搬上正片，涵盖 10%、20%、50% 到 100% 专家轨迹：
看左图同分布 S6N3：我们的单头 Spatial-Softmax 在全区间领跑——仅用 10% 数据（450 局）就达到 0.802，逼近全量 BC30；20% 数据达 0.898，领先同预算 FiLM 8.4 个点；50% 数据（2,250 局）即达 0.958，仅用一半数据就追平了 100% 全量数据的 FiLM（0.959）！
看右图零样本 S7N4 外推优势更加悬殊：20% 数据下领先 FiLM 10.4 个点（0.854 vs 0.750）；50% 数据下 Spatial-SS（0.911）与 MH-Rel（0.914）双双超越了 100% 全量训练的 FiLM（0.900）！
""")

# ------------------------------------------------------------------ SLIDE 11: Results III — Direct Hidden-Layer Mechanistic Proof (0:30, B)
s = new_slide("Results III · Hidden-layer SNR, selectivity & drift proof", title_size=21.5)
pic(s, A / "p_probe_core.png", 0.35, 1.14, w=9.3)
tb(s, 0.35, 4.10, 9.3, 0.92, [
    [("Why FiLM degrades  ", {"bold": True, "color": SLATE}),
     ("in Panel (a), Channel FiLM & Chaplot GA sit pinned at 1.005× / 0.939× pre-Conv2 T/maxD (zero spatial separation) with 1.90× selectivity, trapping them in bottom-left of Panel (b)", {})],
    [("Why Attention scales  ", {"bold": True, "color": BLUE}),
     ("Spatial-SS boosts pre-Conv2 SNR to 7.67× (S6) / 5.19× (S8) & selectivity to 3.59×; MH-Rel achieves top post-Conv2 partial-match SNR (2.43×); purple arrow proves NoPE (+6.2%)", {})],
], size=14.0, spacing=4)
footer(s, "Probed over 150 unseen visible-target scenes × 3 seeds per env · Code: probe_feature_mechanics.py · Full 3-panel plot in B5.", y=5.08)
notes(s, """
【讲者 B｜5:15–5:45｜30 秒】
为什么会有如此显著的跨尺度与样本效率差距？第三组实证直接打开网络黑箱，在 150 个未见场景上测量隐藏层特征几何：
看左图融合层输出的目标/最强干扰物范数比：BC30、Channel FiLM 和 Chaplot 全部被钉死在红色 1.0 倍无分离基线上（FiLM 在 S8N6 甚至跌破 1 至 0.939 倍）；而我们的 Spatial-Softmax 直接将分离比拉高到 7.67 倍（S8N6 仍保持 5.19 倍）！
看右图：横轴是换色词空间选择性，纵轴是 S8N6 成功率。通道调制被困在左下角（选择性仅 1.90 倍），而空间注意力将选择性翻倍至 3.59 倍跃居右上角；紫色虚线箭头直观印证定律一 NoPE 削减背景漂移并推高 6.2 个点，MH-Rel 则取得全场最高后层部分匹配信噪比 2.43。下面交由 C 剖析个案机理、闭环演示与失败边界。
""")


# ==============================================================================
# SPEAKER C · SLIDES 12–15 (5:45–8:00 = 135 s):
#   Diagnostics, Modular Demo, Failure Taxonomy & Design Laws
# ==============================================================================

# ------------------------------------------------------------------ SLIDE 12: Case Diagnostics — 4501 Obstacle & Token-CoAttn Heads (0:35, C)
s = new_slide("Diagnostics · Obstacle preservation & emergent CoAttn heads", title_size=21.5)
pic(s, A / "p_diag_core.png", 0.35, 1.14, w=9.3)
tb(s, 0.35, 4.06, 9.3, 0.90, [
    [("Obstacle erasure solved  ", {"bold": True, "color": TEAL}),
     ("v1 Spatial-SS dims blocking yellow_box on seed 4501 to 0.6× (1/3 solved); Map-Inject keeps 1.0× raw h1 & MH-Rel preserves 2.3× obstacle features → both solve 3/3 in 8 steps!", {})],
    [("Emergent role split  ", {"bold": True, "color": BLUE}),
     ("without supervision, Token-CoAttn Heads 1–2 lock 0.70 & 0.73 weight onto <color>+<type>, while Head 3 attends 'the'/'go' (0.32/0.17) to highlight free corridors!", {})],
], size=13.2, spacing=3)
footer(s, "Source: reports/b2_attn_v2_diagnostics.json · Full 4-head spatial heatmaps on seed 4501 in Backup B6.", y=5.08)
notes(s, """
【讲者 C｜5:45–6:20｜35 秒】
谢谢。我们无需依赖动画就能在静态特征层精准看到改进机理：
看左图 seed 4501 绕障诊断：目标蓝钥匙正前方挡着一个黄箱子。第一代 v1 Spatial-Softmax 虽然把目标放大 6.6 倍，却把挡路黄箱压暗到 0.6 倍，导致网络“看不见障碍”只有 1/3 种子成功；而我们的 Map-Inject 保持原始特征 1.0 倍完全无损，MH-Rel 在放大目标 4.9 倍的同时保留黄箱 2.3 倍与墙壁 2.4 倍并传递方位关系，两者均实现 3/3 种子 8 步最优绕障！
看右图词级协同注意力 Token-CoAttn 在 300 局未见测试上的词槽权重：完全无监督下，Head 1 和 Head 2 自动把 70%–73% 的注意力集中到颜色和类型词上负责锁目标，而 Head 3 关注冠词与动词去照亮绕行走廊！
""")

# ------------------------------------------------------------------ SLIDE 13: Modular Closed-Loop Demo Overview (0:30, C)
s = new_slide("Closed-loop rollout demonstration across 4 diagnostic seeds", title_size=21.5)
pic(s, A / "demo_attention_laws.gif", 0.35, 1.18, w=4.85)
# Clean 4-card static digest on the right so the slide works standalone or with any future video version
demo_cards = [
    ("Clip 1 · seed 4502 ('purple key')", "Crowded yellow-key distractors: BC30 0/3 & Channel FiLM 0/3 timeout → Ours (Spatial-SS & MH-Rel) solve 3/3 in 7 steps", PALEB, BLUE),
    ("Clip 2 · seed 4501 ('blue key')", "Blocked direct path: v1 Spatial-SS 1/3 → Ours (Map-Inject & MH-Rel) detour cleanly 3/3 in 8 steps", PALET, TEAL),
    ("Clip 3 · seed 4516 ('purple box')", "Same-color purple key vs purple box: BC30 0/3 attribute confusion → Ours solve 3/3 in 8 steps", PALEB, BLUE),
    ("Clip 4 · seed 4568 ('blue box')", "Target behind agent (outside 7×7 view): BC30, FiLM & Attention all loop L/R (0/3) — POMDP memory boundary", PALER, RED),
]
cy_d = 1.18
for ttl_c, desc_c, fl_c, ln_c in demo_cards:
    box(s, 5.38, cy_d, 4.27, 0.86, [ttl_c, desc_c], fill=fl_c, line=ln_c, color=DARK, size=10.3, bold=False, lw=1.5, align=PP_ALIGN.LEFT, bold_first=True)
    cy_d += 0.95
footer(s, "Left: 3-column deterministic closed-loop rollouts on unseen test seeds [4500:4800]; Right: per-case 3-seed solve summary (B10).", y=5.08)
notes(s, """
【讲者 C｜6:20–6:50｜30 秒】
这一页汇总了四类代表性未见测试种子的真实闭环表现（左侧同步播放确定性三栏轨迹对比，右侧列出 3 种子统计结果）：
第一类拥挤同类干扰（4502）：BC30 和 Channel FiLM 均 0/3 超时，我们的 Spatial-SS 与 MH-Rel 3/3 七步直达；
第二类正前方障碍阻挡（4501）：Map-Inject 与 MH-Rel 修复了 v1 的障碍擦除，3/3 八步绕障；
第三类同色异类混淆（4516）：3/3 精准区分紫钥匙与紫箱子；
第四类目标在身后的视野盲区（4568）：所有单帧模型均 0/3 陷入左右转循环，引出下一页的失败边界分析。
""")

# ------------------------------------------------------------------ SLIDE 14: 300-Episode Failure Decomposition & Hard POMDP Limit (0:35, C)
s = new_slide("300-episode failure taxonomy & the single-frame POMDP limit", title_size=21.5)
pic(s, A / "p_failshift.png", 0.35, 1.14, h=2.92)
pic(s, A / "case_4568_t0.png", 4.78, 1.22, w=1.42)
box(s, 6.32, 1.22, 3.33, 1.42,
    ["Blind-Spot Aliasing (seed 4568)",
     "• Target blue_box sits behind agent (outside 7×7 ego-cone)",
     "• Deterministic argmax maps aliased views to L↔R loops (0/3 across all 14 single-frame models)"],
    fill=PALER, line=RED, color=DARK, size=10.4, bold=False, lw=1.5, align=PP_ALIGN.LEFT, bold_first=True)
box(s, 4.78, 2.76, 4.87, 1.28,
    ["Quantitative Failure Breakdown (/300 unseen S6N3 episodes)",
     "• Representation errors:  BC30 38.3 → FiLM 6.3 → Spatial-SS 3.7 → MH-Rel 2.3 (attribute_confusion = 1.0 / 300!)",
     "• State-coverage loops:  ~5.0–6.0 / 300 physical floor — requires recurrent memory (ConvGRU) or DAgger (B12)"],
    fill=LIGHT, line=NAVY, color=DARK, size=10.5, bold=False, lw=1.5, align=PP_ALIGN.LEFT, bold_first=True)
tb(s, 0.35, 4.18, 9.3, 0.78, [
    [("Calibrated diagnosis  ", {"bold": True, "color": TEAL}),
     ("early spatial & relational attention virtually eliminates visual-language grounding errors (94% reduction vs BC30, 63% reduction vs FiLM), isolating partial observability as the sole remaining bottleneck", {})],
], size=14.2)
footer(s, "Source: reports/b2_attn_v2_failures.json, b3_v2_eval_s6n3.json (mean of 3 seeds × 300 unseen episodes).", y=5.08)
notes(s, """
【讲者 C｜6:50–7:25｜35 秒】
我们进一步对全部 300 局未见测试的失败轨迹做严格归因拆解（左图）：
红色代表表征接地失败，蓝色代表视野盲区导致的状态覆盖循环。可以看到，总失败数从 BC30 的 55.7 次、Channel FiLM 对照的 12.3 次，进一步压低到 Spatial-Softmax 的 8.7 次与 MH-Relational 的全场最低 8.3 次！其中表征绑定失败从 38.3 次锐减 94% 至仅 2.3 次（属性混淆每 300 局仅剩 1.0 例，比 FiLM 再砍掉近三分之二）！
而右上角 seed 4568 解释了为什么蓝色覆盖失败稳定在 5 到 6 次：当目标出生在智能体正后方、完全处于 7×7 视野盲区时，任何无记忆单帧确定性策略都会遭遇观测混叠而左右摆头。这科学界定了单帧跨模态融合的物理上限——最后 2% 必须靠时序记忆（如 ConvGRU）或在线解卡 DAgger 解决。
""")

# ------------------------------------------------------------------ SLIDE 15: Conclusions, Design Laws & Honest Tradeoffs (0:35, C)
s = new_slide("Conclusions: three Attention design laws & honest tradeoffs", title_size=21.5)
box(s, 0.35, 1.15, 9.30, 0.86,
    ["1 · Spatial Competition Beats Channel Broadcast at Identical Parameters (225,183 p)",
     "49-cell softmax lifts pre-Conv2 T/maxD SNR 1.005× → 7.669× & selectivity 1.90× → 3.59×, beating FiLM by +14.9% on S8N6 (0.887 vs 0.738) and matching 100% FiLM with 50% data (0.958)"],
    fill=PALEB, line=BLUE, color=DARK, size=11.0, bold=False, lw=1.8, align=PP_ALIGN.LEFT, bold_first=True)
box(s, 0.35, 2.10, 9.30, 0.90,
    ["2 · Three Verified v2 Geometric Laws Fix Naive Attention Failure Modes",
     "(a) NoPE removes 6×6 pos_emb overfitting (+6.2% S8N6)   ·   (b) Map-Inject [H; 49α] preserves 1.0× obstacle geometry (min drift 0.569, 4501 3/3)   ·   (c) Ego MH-Relational reaches top S6N3 (0.972) & 2.3/300 repr errors"],
    fill=PALET, line=TEAL, color=DARK, size=10.9, bold=False, lw=1.8, align=PP_ALIGN.LEFT, bold_first=True)
box(s, 0.35, 3.09, 9.30, 0.84,
    ["3 · Bias–Variance Tradeoff, Scope Limits & Next Step",
     "1-head Spatial-SS wins OOD (0.888) & low data (0.802 @10%, 0.898 @20%); higher-capacity MH-Rel wins in-dist (0.972) & detours (3/3) · Next: pair MH-Rel with recurrent ConvGRU memory for blind-spot coverage"],
    fill=LIGHT, line=SLATE, color=DARK, size=10.9, bold=False, lw=1.4, align=PP_ALIGN.LEFT, bold_first=True)

box(s, 0.35, 4.04, 9.30, 0.38, "A: task, baseline & protocol (1–5)   ·   B: Attention laws & mechanistic proof (6–11)   ·   C: diagnostics & limits (12–15)", fill=WHITE, line=GREY, size=11.2, bold=False)
tb(s, 0.35, 4.52, 9.30, 0.55, [[("Thank you — happy to take questions!", {"bold": True, "color": NAVY, "size": 23})]], align=PP_ALIGN.CENTER)
notes(s, """
【讲者 C｜7:25–8:00｜35 秒】
最后总结我们的三点核心结论：
第一，假说全面成立：在逐位相等的 225,183 参数下，跨 49 格竞争的空间注意力将融合层信噪比从通道 FiLM 的 1.005 倍拉升至 7.67 倍，在 S8N6 大地图领先 14.9 个点，且仅用 50% 数据即追平全量 FiLM；
第二，我们诊断并验证了三条具身注意力设计定律：无位置过拟合 NoPE、零失真热图拼接 Map-Inject、以及自我中心关系自注意力 MH-Relational（创下 0.972 最高成功率与 2.3/300 最低表征错误）；
第三，我们诚实揭示了偏差-方差权衡与单帧盲区物理边界：单头空间 Softmax 最适合零样本跨尺度与小样本，多头关系注意力最擅长复杂遮挡绕障，下一步将结合时序记忆攻克最后 6/300 的盲区循环。谢谢大家，欢迎提问！
""")


# ==============================================================================
# BACKUP SLIDES (Slide 16 Divider + B1–B13 Supporting Evidence)
# ==============================================================================
s = prs.slides.add_slide(SECTION_LAYOUT)
s.shapes.title.text = "Backup slides (B1–B13)"
r = s.shapes.title.text_frame.paragraphs[0].runs[0]; r.font.color.rgb = NAVY; r.font.bold = True
for sh in TPL_CONTENT.shapes:
    if sh.is_placeholder and sh.placeholder_format.idx == 12:
        s.shapes._spTree.append(copy.deepcopy(sh._element))
tb(s, 0.5, 3.25, 8.8, 1.4, [
    "B1 config & seed split · B2 14-model master table · B3 param & wd controls · B4 full v1+v2 Attention benchmark",
    "B5 full 3-panel feature SNR & drift · B6 4-head CoAttn spatial maps · B7 counterfactual probe · B8 mission breakdown",
    "B9 bias–variance scaling details · B10 6-case seed matrix · B11 training/val curves · B12 aux branches · B13 refs & AI use",
], size=13.8, color=GREY)
notes(s, "【备份页，不计入 8 分钟】仅用于 Q&A 深度追问。")

# ------------------------------------------------------------------ B1 config
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
table(s, rows, 0.35, 1.14, 9.3, [1.5, 7.8], size=12.5, row_h=0.37)
notes(s, "【备份 B1｜A 负责回答】全部 14 个受控模型的共享超参、步数对齐低数据协议与种子无泄漏验证。")

# ------------------------------------------------------------------ B2 master 14-model table
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
table(s, trows, 0.25, 1.12, 9.5, [1.86, 0.70, 1.14, 1.14, 1.14, 1.15, 1.29, 0.54, 0.54], size=10.1, row_h=0.252, hl_rows=(5, 10, 12, 14))
footer(s, "‘—’ = diagnostic full-data ablation (not re-trained at 20%). Highlighted: Spatial-SS, Res-SS, Map-Inject, MH-Spatial-Rel.", y=5.06, size=9.8, w=9.2)
notes(s, "【备份 B2｜B 负责回答】全部 14 个受控模型在同分布、双跨尺度零样本、20% 低数据、失败拆分与 4501/4502 个案上的完整汇总表。")

# ------------------------------------------------------------------ B3 Capacity & wd controls
s = new_slide("B3 · Parameter-count & wd controls (Concat vs FiLM vs Attention)", title_size=20)
pic(s, A / "p_ablation.png", 0.35, 1.14, w=9.3)
tb(s, 0.35, 4.58, 9.0, 0.52, [
    [("Equal params (225,183)  ", {"bold": True, "color": BLUE}), ("Concat-Wide (224.7k) = 0.718 → Channel FiLM (225.2k) = 0.959 → Spatial-SS (225.2k) = 0.971 (S8N6: 0.738 → 0.887)", {})],
    [("Zero-wd control  ", {"bold": True, "color": TEAL}), ("wd=1e-4 hurts narrow concat (0.707 vs 0.814), so wd=0 is enforced uniformly across all Attention & baseline runs", {})],
], size=12.6, spacing=1)
notes(s, "【备份 B3｜B 负责回答】同屏对比证明：提升绝非来自参数增加（Concat-Wide 仅 0.718），而是来自早期跨网格竞争与关系归纳偏置。")

# ------------------------------------------------------------------ B4 Full v1+v2 Attention Benchmark
s = new_slide("B4 · Full v1 + v2 Attention benchmark (3 scales, 20%, failures)", title_size=19)
pic(s, A / "b2_attn_v2_benchmark.png", 1.55, 1.12, h=3.88)
footer(s, "Source: reports/b2_attn_v2_{eval,f020_eval,probe,failures}.json · FiLM shown in slate grey as channel control.", y=5.10, size=10, w=8.8)
notes(s, "【备份 B4｜B 负责回答】全部 12 个代表架构在三尺度闭环、20% 低数据及失败分类上的总览图。")

# ------------------------------------------------------------------ B5 Feature-Space Mechanistic Proof (Full 3-panel incl. h2 partial-match SNR)
s = new_slide("B5 · Full 3-panel feature probe (incl. post-Conv2 partial-match SNR)", title_size=19.5)
pic(s, A / "b2_attn_v2_feature_mechanics.png", 0.25, 1.14, w=9.5)
tb(s, 0.35, 4.06, 9.3, 0.90, [
    [("Panel (b) partial-match SNR  ", {"bold": True, "color": BLUE}), ("on S8N6 distractors sharing color OR type with target, MH-Rel reaches top h2 SNR 2.43× (Token-CoAttn 2.71×, Spatial-SS 2.27×) vs FiLM 1.51× & BC30 1.03×", {})],
    [("Drift ranking  ", {"bold": True, "color": TEAL}), ("Map-Inject achieves lowest background drift (0.569), followed by MH-Rel (0.728) & NoPE (0.749) vs Cross-MHA pos_emb (0.823)", {})],
], size=12.6, spacing=2)
footer(s, "150 visible-target scenes × 3 seeds per env · Code: branches/b2_attn/probe_feature_mechanics.py · b2_attn_v2_feature_mechanics.json", y=5.10, size=9.8, w=9.0)
notes(s, "【备份 B5｜B 负责回答】完整三联特征探针图（补充正片第 11 页未展开的中间面板 h2 同色/同类半匹配干扰物信噪比）。")

# ------------------------------------------------------------------ B6 Diagnostic Mechanisms (4-head spatial heatmaps)
s = new_slide("B6 · Token-CoAttn 4-head spatial maps & NoPE OOD curve", title_size=19.5)
pic(s, A / "b2_attn_v2_mechanisms.png", 1.35, 1.12, h=3.88)
footer(s, "Bottom row: Token-CoAttn per-head 7×7 spatial maps on seed 4501 (Heads 1–2 lock blue_key; Head 3 lights corridor 15.46).", y=5.10, size=10, w=9.1)
notes(s, "【备份 B6｜C 负责回答】补充正片第 12 页底部未展开的四头 7×7 空间热力图：Head 2 精准聚焦目标蓝钥匙（5.19 vs 黄箱 0.02），Head 3 照亮黄箱右侧通行走廊（15.46）。")

# ------------------------------------------------------------------ B7 Counterfactual probe
s = new_slide("B7 · Counterfactual language probe (183 test states)", title_size=22)
pic(s, A / "p_probe.png", 0.35, 1.14, w=5.65)
tb(s, 6.12, 1.24, 3.55, 3.5, [
    [("Protocol  ", {"bold": True, "color": BLUE}), ("fix (image, heading); swap mission to another object present in the room; measure greedy-action flip rate", {})],
    [("Attention wins  ", {"bold": True, "color": TEAL}), ("Spatial-SS reaches 0.295 color & 0.367 type flip rate (MH-Rel 0.275), beating FiLM ctrl (0.268 / 0.299) & BC30 (0.051)", {})],
], size=15.0, spacing=12)
footer(s, "Sources: b2_attn_probe.json, b2_attn_v2_probe.json (bars = mean ± SD over 3 seeds, ddof=0).", size=10.5)
notes(s, "【备份 B7｜B 负责回答】183 态反事实换词探针：Spatial-Softmax 在颜色交换（0.295）、类型交换（0.367）和综合敏感度上全面领先 Channel FiLM 与 Chaplot GA。")

# ------------------------------------------------------------------ B8 Mission breakdown
s = new_slide("B8 · Mission-type breakdown (pooled 900 test episodes)", title_size=22)
pic(s, A / "p_mission.png", 0.35, 1.14, w=9.3)
tb(s, 0.35, 4.44, 9.3, 0.42, [[("Unique-target 'the' (n=804): BC30 0.797 → FiLM ctrl 0.954 → Spatial-SS 0.968 / MH-Rel 0.970 · weakest color 'green': 0.793 → 0.933 → 0.956 (MH-Rel)", {"bold": True, "color": DARK})]], size=13.2)
footer(s, "Missions with <20 train demos not interpreted. Source: per_mission in b2_attn_eval.json, b2_attn_v2_eval.json", y=5.06, size=10.2)
notes(s, "【备份 B8｜A 负责回答】900 局分词元汇总：在占比最大的唯一目标 'the'（804 局）及最难颜色词上，Spatial-SS（0.968）与 MH-Rel（0.970）均稳定超越 Channel FiLM（0.954）。")

# ------------------------------------------------------------------ B9 Bias-variance scaling details
s = new_slide("B9 · Bias–variance crossover on S6N3 scaling (10–100% demos)", title_size=21)
pic(s, A / "p_dataeff.png", 0.35, 1.14, w=6.05)
tb(s, 6.55, 1.20, 3.12, 3.65, [
    [("10%–50% regime  ", {"bold": True, "color": BLUE}), ("parameter-free 1-head Spatial-SS dominates low data: 0.802 (10%), 0.898 (20%), 0.958 (50%)", {})],
    [("100% crossover  ", {"bold": True, "color": TEAL}), ("49×49 MH-Rel needs ~900+ eps to fit relational projections (0.717 @10% → 0.841 @20% → 0.943 @50% → #1 0.972 @100%)", {})],
], size=14.0, spacing=10)
footer(s, "Step-matched = 16,534 transitions per epoch via RandomSampler. Raw: b2_attn_v2_scaling.json, b5_data_eff_eval.json", y=5.08, size=10.2)
notes(s, "【备份 B9｜B 负责回答】放大展示单头 Spatial-Softmax 与多头关系注意力 MH-Rel 在 10%–100% 样本量下的经典偏差-方差交叉规律。")

# ------------------------------------------------------------------ B10 6-case comparison
s = new_slide("B10 · Six diagnostic grounding & detour cases (3 seeds / model)", title_size=21)
cases_a = json.load(open(A / "cases_attn.json"))
case_meta = [
    ("4502", "crowded yellow keys vs purple key"),
    ("4501", "yellow box blocks direct path"),
    ("4516", "same-color purple key vs box"),
    ("4565", "yellow balls vs yellow box"),
    ("4560", "nearer grey ball distractor"),
    ("4568", "target behind agent (blind spot)"),
]
pos = [(0.35, 1.14), (3.45, 1.14), (6.55, 1.14), (0.35, 2.98), (3.45, 2.98), (6.55, 2.98)]
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
notes(s, "【备份 B10｜C 负责回答】六个代表性诊断种子，每一格均同时列出 BC30、Channel FiLM、Spatial-SS 与 MH-Rel 的 3 种子通过率。")

# ------------------------------------------------------------------ B11 Training / val curves
s = new_slide("B11 · Training CE & validation accuracy curves (3 seeds)", title_size=20)
pic(s, A / "p_loss_backup.png", 0.35, 1.14, w=9.3)
tb(s, 0.35, 4.48, 9.1, 0.55, [[("Spatial-Softmax & MH-Spatial-Rel surpass 0.95 val action acc by epoch 6–8 (vs epoch 13 for FiLM ctrl and 0.873 ceiling for BC30). All checkpoints selected strictly by max val action acc.", {"size": 13.2})]])
notes(s, "【备份 B11｜A 负责回答】四模型训练交叉熵与验证准确率曲线对比：Spatial-Softmax 与 MH-Rel 收敛速度与验证上限均显著优于 Channel FiLM 与 BC30。")

# ------------------------------------------------------------------ B12 Auxiliary branches
s = new_slide("B12 · Auxiliary branches (isolated non-fusion controls)", title_size=22)
rows = [["Branch", "What changed", "S6N3 Success (n=3)", "Take-away"],
        ["B1 language encoder", "GRU / Transformer / pos-MLP, concat fusion", "0.819–0.868", "text encoder is not the bottleneck"],
        ["B4 curriculum", "easy→hard pacing, multi-scale envs (concat)", "0.790–0.857", "pacing does not fix binding"],
        ["B3 DAgger", "loop-unsticking queries (+8,526 labels)", "0.890 ± 0.015", "fixes state coverage, not grounding"],
        ["FiLM + DAgger", "Channel FiLM + B3 queries (+6,005 labels)", "0.970 ± 0.007", "exceeds 5,000-ep label budget"],
        ["Ours: Attention v1/v2", "Spatial-SS / MH-Spatial-Rel (zero extra labels!)", "0.971–0.972", "beats FiLM+DAgger within frozen budget"]]
table(s, rows, 0.35, 1.16, 9.3, [1.85, 3.3, 1.6, 2.55], size=12.5, row_h=0.5, hl_rows=(5,))
footer(s, "Sources: b1_langenc_eval.json, b4_curriculum_eval.json, b3_v2_eval_s6n3.json, b2_attn_v2_eval.json.", y=4.4, size=10.5)
notes(s, "【备份 B12｜C 负责回答】横向对照证明：我们的空间与关系注意力在不增加任何专家查询的冻结预算下达到 0.971–0.972，已经超越了额外消耗 6,005 个在线标签的 FiLM+DAgger（0.970）！")

# ------------------------------------------------------------------ B13 References & AI use
s = new_slide("B13 · References and AI-use disclosure", title_size=22)
tb(s, 0.35, 1.12, 9.3, 2.92, [
    "[1] Chevalier-Boisvert et al., BabyAI (FiLM + AttGRU), ICLR 2019 (arXiv:1810.08272).",
    "[2] Perez et al., FiLM: Visual reasoning with a general conditioning layer, AAAI 2018 (arXiv:1709.07871).",
    "[3] Chaplot et al., Gated-Attention architectures for task-oriented language grounding, AAAI 2018.",
    "[4] Xu et al., Show, Attend and Tell (soft spatial attention), ICML 2015 · Wang et al., Residual Attention Network, CVPR 2017.",
    "[5] Mott et al., Attention Augmented Agents, NeurIPS 2019 · Zambaldi et al., Relational Deep RL, ICLR 2019.",
    "[6] Liu et al., CoordConv, NeurIPS 2018 · Lu et al., Co-Attention, NIPS 2016 · Kim et al., BAN, NeurIPS 2018.",
    "[7] Hashemi Dijujin et al., Inductive biases for zero-shot systematic generalization in language-informed RL, arXiv:2501.15270.",
    "[8] Farama Foundation, MiniGrid 3.2.0 / Gymnasium 1.4.0 · Ross et al., DAgger, AISTATS 2011.",
], size=12, spacing=2.5)
tb(s, 0.35, 4.04, 8.8, 1.0, [
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
