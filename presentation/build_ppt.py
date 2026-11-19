"""Build AIAA4220 Project 3 presentation ON the official template.

Run:  python3 scripts/make_plots.py && python3 scripts/make_case_figs.py && python3 scripts/make_demo_gif.py
      python3 build_ppt.py
Inputs : src/AIAA4220_Project3_Presentation_Template.pptx (official template, copied from 4090)
         assets/*.png|gif (generated from raw repository JSONs / rollouts), src/remote/json/*.json
Output : AIAA4220_Project3_Presentation.pptx
"""
import copy, json
from pathlib import Path
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
A = ROOT / "assets"; J = ROOT / "src/remote/json"
TEMPLATE = ROOT / "src/AIAA4220_Project3_Presentation_Template.pptx"
OUT = ROOT / "AIAA4220_Project3_Presentation.pptx"

NAVY = RGBColor(0x11, 0x2B, 0x5C); BLUE = RGBColor(0x11, 0x4C, 0xB7); GOLD = RGBColor(0x82, 0x5D, 0x1C)
DARK = RGBColor(0x22, 0x22, 0x22); GREY = RGBColor(0x55, 0x5F, 0x6B); LIGHT = RGBColor(0xF2, 0xF4, 0xF8)
GREEN = RGBColor(0x0E, 0x96, 0x64); ORANGE = RGBColor(0xD9, 0x73, 0x0D); RED = RGBColor(0xC0, 0x39, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF); PALEG = RGBColor(0xE3, 0xF4, 0xEC); PALEO = RGBColor(0xFB, 0xEB, 0xDA)
FONT = "Arial"

# ------------------------------------------------------------------ raw numbers
def load(f):
    return json.load(open(J / f))
b2 = load("b2_film_eval.json"); b2s7 = load("b2_film_s7n4.json"); base = load("baseline_eval.json"); s7 = load("s7n4_transfer.json")
AGG, AGG7 = b2["aggregate"], b2s7["aggregate"]
def msd(v, nd=3):
    return f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=0):.{nd}f}"
p5 = [r for r in base["runs"] if "baseline_seed" in r["run"]]
p5_7 = [r["closed_loop_success"] for r in s7["runs"] if "baseline_seed" in r["run"]]

prs = Presentation(TEMPLATE)
SW, SH = prs.slide_width, prs.slide_height
tpl_slides = list(prs.slides)
TPL_CONTENT = tpl_slides[2]            # "Task and testable hypothesis" slide: title + body + sldNum + separator
LAYOUT = TPL_CONTENT.slide_layout
SECTION_LAYOUT = [l for l in prs.slide_layouts if l.name == "SECTION_HEADER"][0]


# ------------------------------------------------------------------ helpers
def _strip_placeholders(slide):
    for sh in list(slide.placeholders):
        sh._element.getparent().remove(sh._element)


def new_slide(title, title_size=24):
    s = prs.slides.add_slide(LAYOUT)
    _strip_placeholders(s)
    for sh in TPL_CONTENT.shapes:
        if sh.is_placeholder and sh.placeholder_format.idx == 1:
            continue  # drop template body prompt
        s.shapes._spTree.append(copy.deepcopy(sh._element))
    t = s.shapes.title
    bp = t.text_frame._txBody.find(qn("a:bodyPr"))
    for c in list(bp):
        bp.remove(c)
    t.text_frame.text = title
    r = t.text_frame.paragraphs[0].runs[0]
    r.font.size = Pt(title_size); r.font.color.rgb = NAVY; r.font.bold = True; r.font.name = FONT
    return s


def tb(slide, x, y, w, h, paras, size=18, color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=4, font=FONT):
    """paras: list of paragraphs; each paragraph = str or list of (text, dict(bold,color,size,italic))."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(spacing)
        runs = [(p, {})] if isinstance(p, str) else p
        for text, st in runs:
            r = para.add_run(); r.text = text
            f = r.font; f.name = font; f.size = Pt(st.get("size", size)); f.bold = st.get("bold", False)
            f.italic = st.get("italic", False); f.color.rgb = st.get("color", color)
    return box


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


def table(slide, rows, x, y, w, col_w, size=13, header_fill=NAVY, row_h=0.3, bold_first_col=True, hl_rows=()):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gt.table
    tblPr = gt._element.graphic.graphicData.tbl.tblPr
    style = tblPr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # "No Style, Table Grid"
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            cell = t.cell(i, j); cell.text = ""
            cell.margin_left = cell.margin_right = Inches(0.05); cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(rows[i][j]); r.font.name = FONT; r.font.size = Pt(size)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill; r.font.bold = True; r.font.color.rgb = WHITE
            else:
                cell.fill.fore_color.rgb = PALEG if i in hl_rows else (WHITE if i % 2 else LIGHT)
                r.font.color.rgb = DARK; r.font.bold = (j == 0 and bold_first_col) or i in hl_rows
    return gt


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


def footer(slide, text, y=5.08, w=7.95, size=11):
    tb(slide, 0.35, y, w, 0.3, [[(text, {"color": GREY, "size": size})]])


# ================================================================== SLIDE 1: title (template slide 1, edited in place)
s1 = tpl_slides[0]
for sh in s1.placeholders:
    idx = sh.placeholder_format.idx
    if idx == 0:
        tf = sh.text_frame; tf.text = "Early FiLM Fusion Overcomes Color-Shortcut Collapse in BabyAI Imitation"
        r = tf.paragraphs[0].runs[0]; r.font.size = Pt(32); r.font.bold = True; r.font.color.rgb = NAVY
    elif idx == 1:
        tf = sh.text_frame
        bp = tf._txBody.find(qn("a:bodyPr"))
        for c in list(bp):
            bp.remove(c)
        for p in tf.paragraphs[1:]:
            p._p.getparent().remove(p._p)
        for r_ in list(tf.paragraphs[0].runs):
            r_._r.getparent().remove(r_._r)
        lines = [("Track 2 · BabyAI GoToLocal · Primary modification: FiLM cross-modal fusion", 18, GREY, False),
                 ("[Member 1 · ID]     [Member 2 · ID]     [Member 3 · ID]", 20, DARK, True),
                 ("AIAA 4220 Project 3", 18, GREY, False)]
        for i, (t_, sz, col, b_) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER; p.space_after = Pt(8)
            r = p.add_run(); r.text = t_; r.font.size = Pt(sz); r.font.color.rgb = col; r.font.bold = b_
notes(s1, """
【讲者 A｜0:00–0:15｜15 秒】
大家好，我们是赛道 2 BabyAI 小组。我们只做了一个主修改：早期 FiLM 跨模态融合。
问题是：在网络前端加一层 FiLM，能不能解决基线"看不懂颜色词"的问题？下面我先介绍任务和基线。
""")

# ================================================================== SLIDE 2: task & hypothesis (0:30, A)
s = new_slide("Task and testable hypothesis")
pic(s, A / "p_task.png", 0.35, 1.12, h=3.95)
tb(s, 4.75, 1.2, 4.9, 3.6, [
    [("Task  ", {"bold": True, "color": BLUE}), ("BabyAI-GoToLocalS6N3-v0: face the named object within 36 steps", {})],
    [("Input  ", {"bold": True, "color": BLUE}), ("7×7×3 symbolic view + 5-token mission + heading → 7 actions", {})],
    [("Question  ", {"bold": True, "color": BLUE}), ("does early FiLM fusion fix color grounding?", {})],
    [("Hypothesis  ", {"bold": True, "color": GREEN}), ("early channel-wise FiLM gating lifts S6N3 closed-loop Success by > 0.045 (1 BC30 SD) on all 3 seeds, else refuted", {"bold": True})],
], size=19, spacing=14)
footer(s, "Lit cells = agent's 7×7 egocentric view. Env: MiniGrid 3.2.0 / BabyAI [1, 5].", y=5.08, w=7.9)
notes(s, """
【讲者 A｜0:15–0:45｜30 秒】
环境是 BabyAI-GoToLocalS6N3：智能体只看到 7×7×3 的符号化局部视野、5 个词的指令和自身朝向，从 7 个动作里选一个；36 步内站到目标旁边并面朝它就算成功。
比如这一局要找紫色箱子，旁边却有同色的紫色钥匙。
我们的假设是：用早期、按通道的 FiLM 门控让指令直接调制视觉特征，会让 S6N3 闭环成功率比受控基线高出一个基线标准差、也就是 0.045 以上，而且 3 个种子全部成立；否则假设被否定。
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
tb(s, 3.2, r3 + 0.04, 4.1, 0.5, [[("language meets vision only here →", {"color": ORANGE, "bold": True, "size": 15})]], align=PP_ALIGN.RIGHT)
tb(s, 0.35, 3.35, 9.3, 1.6, [
    [("Algorithm  ", {"bold": True, "color": BLUE}), ("BC on BabyAIBot demos · cross-entropy · Adam 1e-3 · 223,071 params", {})],
    [("Task logic  ", {"bold": True, "color": BLUE}), ("return 1 − 0.9·t/36 on success, 0 on timeout · greedy argmax", {})],
    [("Setup  ", {"bold": True, "color": BLUE}), ("5,000 eps · ≤30 epochs · 3 seeds · PyTorch 2.15 1-thread EPYC CPU · MiniGrid 3.2.0 · commits 018a695 / 6d91f9c", {})],
], size=18, spacing=6)
footer(s, "BabyAI [1, 5]. Only deviation: disjoint seed range added to keep 5,000 unique, leak-free episodes (backup B1).")
notes(s, """
【讲者 A｜0:45–1:45｜60 秒】
基线是手册规定的行为克隆，专家数据来自 BabyAIBot。网络 BCNet 共 223,071 个参数：
图像的三个类别通道各自查 11×8 的嵌入表，得到 24×7×7，经过两层不池化的 3×3 卷积，展平成 1,568 维；
指令用 14 词词表的 32 维词嵌入做掩码平均；朝向是 4 维 one-hot。
三者在这里拼接成 1,604 维，经过 128 维隐藏层输出 7 个动作，用交叉熵训练，Adam 学习率 1e-3、批大小 128，测试时贪心取 argmax。
请注意橙色这一块：语言和视觉只在最后这一次拼接里相遇——这正是我们后面要改的地方。
任务逻辑：成功时回报是 1 减 0.9 乘以 t 除以 36，超时为 0。
环境用 PyTorch 2.15 CPU 单线程保证逐字节可复现，MiniGrid 3.2.0，提交号如图。唯一的偏差是种子划分，细节在备份页 B1。
""")

# ================================================================== SLIDE 4: baseline reproduction (1:00, A)
s = new_slide("Baseline reproduction evidence")
pic(s, A / "p_baseline.png", 0.35, 1.08, w=9.3)
tb(s, 0.35, 3.82, 9.3, 1.2, [
    [("Discrepancy  ", {"bold": True, "color": BLUE}), (f"prescribed patience=5: {msd([r['closed_loop_success'] for r in p5])}; seeds 1–2 stop at epochs 13/14 → BC30 (same val-acc rule, 30 epochs): {msd(AGG['bc30']['closed_loop_success']['values'])}", {})],
    [("Sanity  ", {"bold": True, "color": BLUE}), ("always-forward: 56.4% val acc but 0.060 Success; random walk 0.543", {})],
], size=17, spacing=3)
footer(s, "3 init seeds × 300 unseen test seeds; bars = mean, error/shading = ±1 SD (ddof=0), ○ □ △ = seeds 0/1/2. Raw: baseline_eval.json, b2_film_eval.json, runs/*/history.json", size=10, w=8.05)
notes(s, """
【讲者 A｜1:45–2:45｜60 秒】
这是 3 个初始化种子、每个种子 300 个未见测试种子的基线复现。
左图是 30 个 epoch 的训练和验证交叉熵，红色虚线是 patience=5 早停：种子 1 和 2 在验证平台期的第 13、14 个 epoch 就停了。
按手册原样复现的 patience=5 基线成功率是 0.747 ± 0.093。我们保持"按验证动作准确率选模"的规则不变，只是跑满 30 个 epoch，得到受控基线 BC30：0.814 ± 0.045。它更强、方差更小，后面的主比较都用它。
中间是合理性检查：永远前进有 56.4% 的动作准确率，但成功率只有 0.060；随机游走反而有 0.543。说明动作准确率不是任务指标，也解释了为什么我们的成功率高于约 0.31 的非正式参考值——这个房间很小。
右图是每个种子在 S6N3 和零样本 S7N4 上的结果。所有数字都对应 reports 目录下的原始 JSON。下面由 B 介绍修改。
""")

# ================================================================== SLIDE 5: one controlled modification (1:00, B)
s = new_slide("One controlled modification: early FiLM fusion")
y1, y2, bh = 1.15, 1.95, 0.52
tb(s, 0.28, y1 + 0.1, 1.1, 0.4, [[("BCNet", {"bold": True, "color": ORANGE})]], size=14)
tb(s, 0.28, y2 + 0.1, 1.1, 0.4, [[("FiLM-nwd", {"bold": True, "color": GREEN})]], size=14)
seq_b = [(1.35, 1.05, "Embed"), (2.65, 1.05, "Conv1"), (5.25, 1.05, "Conv2"), (6.55, 1.05, "flatten"), (7.85, 1.8, "concat + MLP")]
for i, (x, w, t_) in enumerate(seq_b):
    box(s, x, y1, w, bh, t_, size=14)
    if i:
        px, pw, _ = seq_b[i - 1]; arrow(s, px + pw, y1 + bh / 2, x, y1 + bh / 2)
seq_f = [(1.35, 1.05, "Embed"), (2.65, 1.05, "Conv1"), (3.95, 1.05, "FiLM"), (5.25, 1.05, "Conv2"), (6.55, 1.05, "flatten"), (7.85, 1.8, "concat + MLP")]
for i, (x, w, t_) in enumerate(seq_f):
    if t_ == "FiLM":
        box(s, x, y2, w, bh, t_, fill=PALEG, line=GREEN, color=GREEN, size=15, lw=2.25)
    else:
        box(s, x, y2, w, bh, t_, size=14)
    if i:
        px, pw, _ = seq_f[i - 1]; arrow(s, px + pw, y2 + bh / 2, x, y2 + bh / 2)
box(s, 2.3, 2.68, 2.9, 0.46, "ℓ → Linear(32→64), zero-init", fill=PALEG, line=GREEN, color=GREEN, size=14)
arrow(s, 4.475, 2.68, 4.475, y2 + bh, color=GREEN)
tb(s, 5.3, 2.6, 4.4, 0.6, [[("H′ = (1 + Δγ) ⊙ H + Δβ", {"bold": True, "color": GREEN, "size": 20})]])
tb(s, 0.35, 3.25, 9.3, 1.8, [
    [("Change  ", {"bold": True, "color": GREEN}), ("one zero-init FiLM layer after Conv1 (+2,112 params, +0.95%)", {})],
    [("Why  ", {"bold": True, "color": BLUE}), ("color gates channels before Conv2 mixes cells; concat flips only 5.1% of actions on color swaps", {})],
    [("Fixed  ", {"bold": True, "color": BLUE}), ("data, encoders, head, optimizer, 30 epochs, selection, evaluator", {})],
    [("Scope  ", {"bold": True, "color": BLUE}), ("handbook Track 2 option: cross-modal fusion (FiLM) [3]", {})],
], size=17, spacing=3)
footer(s, "Code: branches/b2_film/film_net.py (FiLMBCNet), train_film.py; variants, Attention & fusion ablations in backup B3–B4.", size=10.5, w=8.05)
notes(s, """
【讲者 B｜2:45–3:45｜60 秒】
谢谢。我们的唯一修改来自手册赛道 2 菜单中的"跨模态融合：用 FiLM 让指令调制视觉特征"。
做法是在第一层卷积之后插入一层 FiLM：指令向量 ℓ 经过一个 32 到 64 维的线性层，输出每个通道的 Δγ 和 Δβ，再对特征图做 (1+Δγ)⊙H+Δβ。
这个线性层零初始化，所以训练开始时 FiLM 网络和基线完全等价，只多了 2,112 个参数，约 0.95%。
为什么会有用？颜色信息需要在第二层卷积把各个格子混合之前，就被指令"门控"——比如指令说紫色箱子，就放大和紫色、箱子相关的通道。而晚期拼接的基线几乎不看颜色词：把指令里的颜色换成场景里另一个物体的颜色，动作只有 5.1% 会改变。
选 FiLM 而非 Attention 作为主修改是因为它仅增 0.95% 参数且保留障碍物空间信息，与 Attention 及融合的对照见备份页 B4。其他一切保持不变：数据、嵌入、卷积、MLP 头和拼接、优化器、30 个 epoch、选模规则和评估代码。
""")

# ================================================================== SLIDE 6: controlled protocol (1:00, B)
s = new_slide("Controlled experimental protocol")
# seed-split bar (segment widths not to scale so every label is readable)
yb, hb = 1.3, 0.5
segs = [(0.35, 3.3, "train [0:4000]", BLUE), (3.65, 1.25, "val [4000:4500]", GOLD), (4.9, 1.3, "test [4500:4800]", GREEN),
        (6.2, 1.3, "train [4800:5300]", BLUE)]
for xx, ww, lab, col in segs:
    box(s, xx, yb, ww, hb, lab, fill=col, line=WHITE, color=WHITE, size=12, shape=MSO_SHAPE.RECTANGLE)
box(s, 7.7, yb, 1.95, hb, "S7N4 test\n[10000:10300]", fill=GREEN, line=WHITE, color=WHITE, size=12, shape=MSO_SHAPE.RECTANGLE)
tb(s, 0.35, yb + hb + 0.02, 9.3, 0.3, [[("4,500 train + 500 val = 5,000 expert eps", {"size": 13, "color": GREY}), ("   ·   300 unseen test seeds per model, S6N3 and S7N4", {"size": 13, "color": GREEN, "bold": True})]])
rows = [("Same budget", ["5,000 expert eps · ≤30 epochs · same evaluator"]),
        ("Selection", ["best val action acc; test seeds never used"]),
        ("Metrics", ["primary: closed-loop Success", "secondary: return · med. len · val acc · S7N4 · data eff."]),
        ("Uncertainty", ["n = 3 init seeds · mean ± SD (ddof=0) · paired Δ"]),
        ("Primary pair", ["FiLM-nwd vs BC30 (both wd=0); patience=5 shown"])]
yy = 2.22
for k, v in rows:
    tb(s, 0.35, yy, 2.0, 0.5, [[(k, {"bold": True, "color": BLUE})]], size=18)
    tb(s, 2.35, yy, 7.3, 0.4 + 0.32 * (len(v) - 1), v, size=18, spacing=0)
    yy += 0.5 + 0.32 * (len(v) - 1)
footer(s, "Seed lists, configs and run IDs: backup B1/B2. Same frozen data/expert_5000.pkl and src/eval_closed_loop.py for every method.", size=10, w=8.05)
notes(s, """
【讲者 B｜3:45–4:45｜60 秒】
实验协议。两种方法共用同一个冻结的数据文件：5,000 条 BabyAIBot 成功专家轨迹，4,500 条训练、500 条验证。
上面这条是种子划分：训练用 0 到 4000 和 4800 到 5300，验证用 4000 到 4500，测试是 300 个从未见过的种子 4500 到 4799；零样本迁移用 S7N4 的种子 10000 到 10299。
选模只看验证动作准确率，测试种子在训练和选模中从未使用。主指标是闭环成功率；次指标按手册包括平均回报、成功局中位步数、验证动作准确率、S7N4 迁移成功率和数据效率，另外报告超时数。
不确定性的独立单位是 3 个网络初始化种子，报告均值 ± 总体标准差，ddof=0，和仓库 JSON 一致；同时报告同种子的配对差。
主比较是 FiLM-nwd 对 BC30：两者都不用权重衰减、都跑满 30 个 epoch，唯一差别就是那一层 FiLM；patience=5 基线也一并展示。
""")

# ================================================================== SLIDE 7: quantitative results (1:30, B)
s = new_slide("Quantitative results: FiLM-nwd vs controlled BC30")
pic(s, A / "p_main.png", 0.3, 1.08, h=3.3)
pic(s, A / "p_curves.png", 5.0, 1.06, w=4.4)
pr = b2["paired"]["film_nwd"]; pr7 = b2s7["paired"]["film_nwd"]
dl = " / ".join(f"+{r['delta']:.3f}" for r in pr["rows"])
tb(s, 5.05, 3.48, 4.6, 0.95, [
    [("Paired Δ S6N3  ", {"bold": True, "color": GREEN}), (f"+{pr['delta_mean']:.3f} ± {pr['delta_sd']:.3f} (3/3)", {"bold": True})],
    [(f"per seed: {dl}", {"size": 15, "color": GREY})],
    [("Paired Δ S7N4  ", {"bold": True, "color": GREEN}), (f"+{pr7['delta_mean']:.3f} ± {pr7['delta_sd']:.3f} (3/3)", {"bold": True})],
], size=17, spacing=1)
tb(s, 0.35, 4.4, 8.0, 0.9, [
    [("Hypothesis supported: ", {"bold": True, "color": NAVY}), ("all 3 paired seed gains > 0.045 (1 BC30 SD)", {"bold": True})],
    [("Secondary: ", {"bold": True, "color": BLUE}),
     (f"return {AGG['bc30']['mean_return']['mean']:.3f} → {AGG['film_nwd']['mean_return']['mean']:.3f} · median length {AGG['bc30']['median_len_success']['mean']:.1f} → {AGG['film_nwd']['median_len_success']['mean']:.1f} steps · timeouts {AGG['bc30']['n_timeout']['mean']:.1f} → {AGG['film_nwd']['n_timeout']['mean']:.1f}/300 · data eff.: 20%-data FiLM 0.814 = 100% BC30 (B7)", {})],
], size=15, spacing=2)
notes(s, """
【讲者 B｜4:45–6:15｜90 秒】
结果。左图三组方法：实心是 S6N3，斜线是零样本 S7N4；误差棒是 ±1 个标准差，三种标记是三个种子，每个点都是 300 个未见种子的闭环评估。
FiLM-nwd 在 S6N3 上从 BC30 的 0.814 ± 0.045 提升到 0.959 ± 0.006。同种子配对提升 +0.144 ± 0.044，三个种子分别是 +0.087、+0.153、+0.193，全部为正，最小的一个也超过 0.045 的判据——假设成立，种子间标准差也缩小了约七倍。
迁移到更大的 S7N4 房间提升更大：0.697 到 0.900，配对 +0.203，同样 3/3。
右上是验证准确率曲线：同样 30 个 epoch，FiLM 在第 10 个 epoch 左右就超过 0.93，BC30 到最后也只有 0.87。
次指标方向一致：平均回报从 0.723 到 0.871，成功局的中位步数从 4 步降到 3 步，每 300 局超时从 55.7 降到 12.3；数据效率上，只用 20% 演示的 FiLM 已达到 0.814，追平全量 BC30，细节在备份 B7。
相对 patience=5 基线的提升是 +0.212，但我们以更保守的 BC30 作为主比较。需要说明 n=3 很小，我们不做参数检验，只报告效应量和全部三个种子。
等参数量的宽拼接对照、FiLM vs Attention 及融合消融放在备份页 B3–B4。接下来由 C 讲成功和失败案例。
""")

# ================================================================== SLIDE 8: successes / failures / limitations (1:30 incl. 40 s video, C)
s = new_slide("Successes, failures and limitations (40 s clip)")
pic(s, A / "demo_bc30_vs_film.gif", 0.35, 1.1, w=5.0)
pic(s, A / "p_failshift.png", 5.5, 1.08, h=3.0)
tb(s, 0.35, 4.2, 9.3, 0.4, [
    [("Reduced  ", {"bold": True, "color": GREEN}), ("attribute confusion 19.3 → 4.7, distractor priority 19.0 → 1.7 per 300 eps", {})]], size=17)
tb(s, 0.35, 4.58, 7.95, 0.7, [
    [("Preserved  ", {"bold": True, "color": RED}), ("out-of-view L/R loops (seed 4568, FiLM 0/3) → next test: loop-unsticking DAgger", {})],
], size=17)
notes(s, """
【讲者 C｜6:15–7:45｜90 秒，含 40 秒视频（进入本页即自动播放）】
谢谢。视频约 40 秒，左边是 BC30，右边是 FiLM，同一个未见测试种子、同一条指令。
【视频第 1 段】成功案例：两者都用 8 步绕过挡路的黄色箱子。注意 FiLM 在这一局只有 2/3 个种子成功——我们不挑最好的种子。
【第 2 段】"去紫色箱子"：BC30 被同色的紫色钥匙吸住，左右摆动直到超时；FiLM 8 步到达，3 个种子都成功。这是属性混淆。
【第 3 段】"去绿色球"：BC30 被更近的灰色球吸引；FiLM 两步完成。这是干扰物优先。
【第 4 段】FiLM 也失败的一局：目标在初始视野之外，策略在左转和右转之间无限循环。
【视频结束】右图把每 300 局的失败分成两类：表征类，也就是语言接地错误，从 38.3 降到 6.3，减少 83.5%；状态覆盖类从 17.3 降到 6.0，剩下的失败约一半就是这种原地振荡。
这是观察；我们的解释是：离线行为克隆从未见过这些朝向状态，所以左右两个动作打平。因此下一个诊断实验是预算匹配的一轮"解卡" DAgger。另外三个以上语言接地失败案例在备份页 B8。
""")

# ================================================================== SLIDE 9: conclusion (0:15, C)
s = new_slide("Conclusion and questions")
tb(s, 0.35, 1.2, 9.3, 2.6, [
    [("Supported  ", {"bold": True, "color": GREEN}), ("S6N3 Success 0.814 ± 0.045 → 0.959 ± 0.006 (3/3 seeds); S7N4 0.697 → 0.900", {})],
    [("Why  ", {"bold": True, "color": BLUE}), ("color sensitivity 0.051 → 0.268; param-matched concat: 0.718", {})],
    [("Limits  ", {"bold": True, "color": RED}), ("n = 3 seeds; out-of-view loops remain → test DAgger next", {})],
], size=20, spacing=16)
box(s, 0.35, 3.75, 7.9, 0.5, "A: baseline  ·  B: FiLM & ablations  ·  C: failures & video", fill=LIGHT, line=GREY, size=18, bold=False)
tb(s, 0.35, 4.35, 7.9, 0.7, [[("Thank you — questions?", {"bold": True, "color": NAVY, "size": 28})]], align=PP_ALIGN.CENTER)
notes(s, """
【讲者 C｜7:45–8:00｜15 秒】
总结：假设成立——S6N3 成功率从 0.814 提升到 0.959，三个种子全部提升；机制是颜色敏感度提升约五倍，而等参数的宽拼接没有用。局限是只有 3 个种子，视野外振荡仍在。谢谢，欢迎提问。
""")

# ================================================================== BACKUP divider
s = prs.slides.add_slide(SECTION_LAYOUT)
s.shapes.title.text = "Backup slides (B1–B11)"
r = s.shapes.title.text_frame.paragraphs[0].runs[0]; r.font.color.rgb = NAVY; r.font.bold = True
for sh in TPL_CONTENT.shapes:
    if sh.is_placeholder and sh.placeholder_format.idx == 12:
        s.shapes._spTree.append(copy.deepcopy(sh._element))
tb(s, 0.6, 3.3, 8.4, 1.2, [
    "B1 config & seed split · B2 per-seed table · B3 capacity / wd ablation · B4 FiLM vs Attention & fusion · B5 language probe · B6 mission breakdown",
    "B7 data efficiency · B8 grounding failure cases · B9 loss curves · B10 auxiliary branches · B11 references & AI use"], size=15, color=GREY)
notes(s, "【备份页，不计入 8 分钟】仅用于 Q&A。")

# ================================================================== B1 config
s = new_slide("B1 · Configuration, seed split and documented deviation", title_size=22)
rows = [["Item", "Setting (identical for baseline and FiLM)"],
        ["Environment", "BabyAI-GoToLocalS6N3-v0 (H=36) · zero-shot GoToLocalS7N4-v0 (H=49)"],
        ["Expert data", "BabyAIBot, 5,000 successful eps = 4,500 train (16,534 steps) + 500 val (1,844 steps)"],
        ["Seed split", "train [0:4000] ∪ [4800:5300] · val [4000:4500] · test [4500:4800] · S7N4 [10000:10300]"],
        ["Deviation", "cycling [0:4000] would duplicate 1,000 eps or exceed 5,000 → disjoint [4800:5300]; assert_no_leak() passes"],
        ["Training", "Adam lr 1e-3, batch 128, ≤30 epochs; patience=5 (prescribed) and BC30 (30 ep); select max val action acc"],
        ["Models", "BCNet 223,071 · FiLM 225,183 (+2,112) · FiLM-only 221,087 · Concat-Wide 224,683 params"],
        ["Software", "Python 3.12.14, PyTorch 2.15.0+cpu, MiniGrid 3.2.0, Gymnasium 1.4.0, NumPy 2.5.4"],
        ["Determinism", "CPU, torch.set_num_threads(1), dual AMD EPYC 7453 (~55 s per seed)"],
        ["Provenance", "commits 018a695 (baseline), 6d91f9c (FiLM + BC30), 84dfe11 (aux branches); raw JSON in reports/"]]
table(s, rows, 0.35, 1.15, 9.3, [1.55, 7.75], size=13, row_h=0.37)
notes(s, "【备份 B1｜A 负责回答】种子划分偏差：每个环境种子对应唯一确定的一局，验证集需要 500 个独立种子；若在 [0:4000] 内循环会产生 1,000 条重复或超出 5,000 条上限，所以补充不相交的 [4800:5300]，并用 assert_no_leak() 验证训练/验证/测试无交集。")

# ================================================================== B2 per-seed results
s = new_slide("B2 · Per-seed results (300 unseen seeds per run, mean ± SD ddof=0)", title_size=20)
def row(name, key=None, p5mode=False):
    if p5mode:
        v = [r["closed_loop_success"] for r in p5]; ret = [r["mean_return"] for r in p5]; va = [r["val_action_acc"] for r in p5]
        to = [r["n_timeout"] for r in p5]; s7v = p5_7; params = "223,071"
    else:
        g = AGG[key]; v = g["closed_loop_success"]["values"]; ret = g["mean_return"]["values"]; va = g["val_action_acc"]["values"]
        to = g["n_timeout"]["values"]; s7v = AGG7[key]["closed_loop_success"]["values"] if key in AGG7 else None
        params = {"bc30": "223,071", "concat_nwd": "223,071", "concat_wd": "223,071", "concat_wide_wd": "224,683",
                  "film_only_wd": "221,087"}.get(key, "225,183")
    return [name, f"{v[0]:.3f}", f"{v[1]:.3f}", f"{v[2]:.3f}", msd(v), msd(s7v) if s7v else "—", f"{np.mean(ret):.3f}",
            f"{np.mean(va):.3f}", f"{np.mean(to):.1f}", params]
rows = [["Method", "s0", "s1", "s2", "S6N3 mean±SD", "S7N4 mean±SD", "Return", "Val acc", "Timeouts", "Params"],
        row("BC patience=5", p5mode=True), row("BC30 (control)", "bc30"), row("FiLM-nwd (primary)", "film_nwd"),
        row("FiLM-wd", "film_wd"), row("FiLM-only (wd)", "film_only_wd"), row("FiLM-res (wd)", "film_res_wd"),
        row("Concat-nwd", "concat_nwd"), row("Concat-wd", "concat_wd"), row("Concat-Wide (wd)", "concat_wide_wd")]
table(s, rows, 0.35, 1.12, 9.3, [1.72, 0.56, 0.56, 0.56, 1.38, 1.38, 0.72, 0.72, 0.92, 0.78], size=12, row_h=0.33, hl_rows=(3,))
footer(s, f"Paired Δ vs BC30 (same seed): FiLM-nwd S6N3 {dl} (3/3 > 0); S7N4 " + " / ".join(f"+{r['delta']:.3f}" for r in pr7["rows"]) +
       ". Sources: baseline_eval.json, s7n4_transfer.json, b2_film_eval.json, b2_film_s7n4.json.", y=4.55, size=11, w=8.0)
notes(s, "【备份 B2｜B 负责回答】所有方法、所有种子的完整结果都在这里，绝不只展示最佳运行。标准差为总体标准差 ddof=0（与仓库 JSON 一致）；换算成样本标准差 ddof=1 需乘以 √(3/2)≈1.225，例如 BC30 的 0.045 变为 0.056，FiLM 的 0.006 变为 0.008，结论不变。")

# ================================================================== B3 ablations
s = new_slide("B3 · Capacity control and FiLM / weight-decay ablations", title_size=22)
pic(s, A / "p_ablation.png", 0.35, 1.08, w=9.3)
tb(s, 0.35, 4.6, 7.95, 0.6, [
    [("Capacity  ", {"bold": True, "color": BLUE}), ("Concat-Wide 0.718 vs FiLM-wd 0.946 at same wd; FiLM-only (fewer params than BCNet) 0.947", {})],
    [("Weight decay  ", {"bold": True, "color": BLUE}), ("1e-4 hurts concat (0.836 → 0.707) but barely FiLM (0.959 vs 0.946)", {})],
], size=14, spacing=1)
notes(s, "【备份 B3｜B 负责回答】为什么不是参数更多？等参数的 Concat-Wide（224,683 参数，与 FiLM 的 225,183 相当）只有 0.718 ± 0.023，与同样 wd=1e-4 的 FiLM-wd 0.946 相差 0.228；去掉拼接、参数比基线还少 1,984 个的 FiLM-only 仍有 0.947 ± 0.007。权重衰减：wd=1e-4 让窄主干的拼接模型从 0.836 降到 0.707（惩罚 Linear(1604,128)），而 FiLM 几乎不受影响。")


# ================================================================== B4 FiLM vs Attention & FiLM+Attention
s = new_slide("B4 · FiLM vs Attention and FiLM+Attention fusion", title_size=21)
pic(s, A / "p_attn_ablation.png", 0.30, 1.05, w=5.48)
pic(s, A / "p_attn_heatmap.png", 5.88, 1.02, h=3.22)
tb(s, 0.35, 4.22, 9.3, 0.75, [
    [("Spatial-Softmax  ", {"bold": True, "color": BLUE}),
     ("+2,112 p (tie): 0.971 S6 / 0.964 S7 / 0.887 S8 · solves 4502 3/3 ✓, but 4501 detour drops 2/3 → 1/3", {})],
    [("FiLM + Attention  ", {"bold": True, "color": GREEN}),
     ("FiLM+Spatial (0.959 / S7 0.939) & FiLM+MHA (0.963 / S7 0.926) beat FiLM on OOD, not Spatial-Softmax", {})],
], size=12.5, spacing=3)
footer(s, "3 seeds × 300 eps (ddof=0). 20%-data: Spatial-Gate 0.850, Cross-MHA 0.867, FiLM+Spatial 0.848 vs FiLM 0.814 · b2_attn_*.json", y=5.04, size=10, w=8.8)
notes(s, "【备份 B4｜B 负责回答】为什么主修改选 FiLM 而不是 Attention，两者能否融合？我们在完全相同预算下对比了三种 Attention 和两种串联融合：等参数的 Spatial-Softmax（+2,112 参数）在 S6N3 达 0.971 ± 0.003、S7N4 达 0.964、S8N6 达 0.887，右图热力图可见它把 6.26 高权重精准压在 seed 4502 拥挤紫钥匙上，3/3 修复了 FiLM 失败的 4502；但代价是把 4501 挡路黄箱子压低到 0.46，绕障通过率从 FiLM 的 2/3 降到 1/3。两者可以串联融合：FiLM+Spatial（0.959/S7 0.939）与 FiLM+MHA（0.963/S7 0.926）都提升了 FiLM 的跨房间迁移，但未超越独立 Spatial-Softmax，因为早期通道抑制削弱了后续空间门控对比度。")

# ================================================================== B5 probe
s = new_slide("B5 · Counterfactual language probe (183 test states)", title_size=22)
pic(s, A / "p_probe.png", 0.35, 1.1, w=5.4)
tb(s, 5.95, 1.25, 3.7, 3.4, [
    [("Probe  ", {"bold": True, "color": BLUE}), ("fix (image, heading); swap the mission to another object that exists in the scene; count greedy-action changes", {})],
    [("Result  ", {"bold": True, "color": GREEN}), ("concat ignores 93.5% of color swaps; FiLM is 4.1× more color-sensitive (5.3× vs BC30)", {})],
], size=16, spacing=12)
footer(s, "Source: reports/b2_language_sensitivity.json (probe_language.py). Bars = mean ± SD over runs (ddof=0).", size=11)
notes(s, "【备份 B5｜B 负责回答】这是机制证据：固定图像和朝向，只把指令换成场景里真实存在的另一个物体（同类异色 138 对、同色异类 59 对），统计贪心动作改变的比例。拼接模型颜色交换只改变 6.5% 的动作（BC30 为 5.1%），FiLM 为 26.8%。类型敏感度两者差距较小（0.218→0.299），说明问题主要出在颜色绑定。")

# ================================================================== B5 mission breakdown
s = new_slide("B6 · Mission-type breakdown (pooled 900 test episodes)", title_size=22)
pic(s, A / "p_mission.png", 0.6, 1.08, w=8.8)
tb(s, 0.35, 4.35, 9.3, 0.4, [[("BC30 → FiLM: 'the' (unique target) 0.797 → 0.954 · 'a' 0.958 → 1.000 · blue 0.762 → 0.944", {})]], size=15)
footer(s, "Missions with <20 train demos ('go to a red box' 15, 'go to a purple key' 19) are not interpreted. Source: per_mission in b2_film_eval.json, baseline_eval.json", y=4.75, size=11)
notes(s, "【备份 B6｜A 负责回答】手册要求的 mission 类型分解。'the' 表示场景中唯一匹配目标，占 804 局，BC30 0.797、FiLM 0.954；'a' 表示有多个匹配，任一即可，占 96 局。FiLM 让六种颜色都超过 0.933。训练样本少于 20 条的两种 mission 不做结论。")

# ================================================================== B6 data efficiency
s = new_slide("B7 · Data efficiency: 10 / 20 / 50 / 100% of demonstrations", title_size=22)
pic(s, A / "p_dataeff.png", 0.35, 1.08, w=6.0)
tb(s, 6.5, 1.25, 3.15, 3.4, [
    [("20% data  ", {"bold": True, "color": GREEN}), ("step-matched FiLM 0.814 ± 0.023 = 100%-data BC30 (5× fewer demos)", {})],
    [("Prescribed BC  ", {"bold": True, "color": BLUE}), ("patience=5: 0.732 ± 0.007 (20%), 0.747 ± 0.012 (50%)", {})],
    [("S7N4 at 20%  ", {"bold": True, "color": BLUE}), ("FiLM 0.750 ± 0.052 vs BC 0.627 ± 0.021", {})],
], size=15, spacing=12)
footer(s, "Step-matched = 3,900 gradient updates per fraction. Raw: b5_data_eff_eval.json, b5_data_eff_s7n4.json, data_efficiency.csv", size=11)
notes(s, "【备份 B7｜C 负责回答】数据效率是手册要求的次指标。小数据下朴素 30 epoch 会混入'训练步数不足'的混杂，因此对齐 3,900 个梯度步：BC 在 20%/50% 只有 0.742/0.751，FiLM 为 0.814/0.929，20% 数据即追平全量 BC30。")

# ================================================================== B7 failure cases
s = new_slide("B8 · Language-grounding failure cases (≥3 required)", title_size=22)
cases = json.load(open(A / "cases.json"))
info = {"4516": "attribute confusion: same-color purple key", "4565": "attribute confusion: yellow balls vs yellow box",
        "4570": "attribute confusion: locks onto red ball", "4582": "attribute confusion: purple vs blue box",
        "4560": "distractor priority: nearer grey ball", "4568": "state coverage: target out of view, L/R loop"}
pos = [(0.35, 1.12), (3.45, 1.12), (6.55, 1.12), (0.35, 2.98), (3.45, 2.98), (6.55, 2.98)]
for (x, y), sd in zip(pos, ["4516", "4565", "4570", "4582", "4560", "4568"]):
    c = cases[sd]; nfix = sum(f[0] for f in c["film"]); okst = [f[1] for f in c["film"] if f[0]]
    pic(s, A / f"case_{sd}_t0.png", x, y, w=1.3)
    film_txt = (f"FiLM {nfix}/3 ✓" if nfix else "FiLM 0/3 ✗") + (f" ({min(okst)}–{max(okst)} st.)" if okst and min(okst) != max(okst) else (f" ({okst[0]} steps)" if okst else ""))
    tb(s, x + 1.33, y - 0.04, 1.8, 1.75, [
        [(f"seed {sd}", {"bold": True, "color": NAVY, "size": 13})],
        [(c["mission"].replace("go to ", ""), {"size": 12, "italic": True})],
        [(info[sd], {"size": 11, "color": GREY})],
        [("BC30: timeout 36", {"size": 12, "color": RED, "bold": True})],
        [(film_txt, {"size": 12, "color": GREEN if nfix else RED, "bold": True})],
    ], spacing=1)
tb(s, 0.35, 4.62, 8.0, 0.6, [[("BC30 = runs/b3_bc30_seed0; FiLM counts over runs/b2_film_nwd_seed{0,1,2}, re-rolled read-only. On 4502 (crowded purple key): FiLM 0/3, whereas Spatial-Softmax/Gate solve 3/3 & Cross-MHA 2/3 (see B4); 4587/4684 target hidden at t=0.", {"size": 11, "color": GREY})]])
notes(s, "【备份 B8｜C 负责回答】手册要求至少 3 个语言接地失败案例，这里给出 5 个（4516、4565、4570、4582 为属性混淆，4560 为干扰物优先）外加 1 个状态覆盖失败 4568。所有 FiLM 结果都按 3 个种子统计：4565 只有 2/3 修复，4568 为 0/3，我们如实报告。FiLM 自身仍失败的接地案例是 4502（两把黄色钥匙夹住紫色钥匙，空间全局的 γ、β 无法区分拥挤的同类物体），已实测验证（见 B4）：Spatial-Softmax 与 Gate 均 3/3 修复 4502，但会轻微削弱 4501 绕障。")

# ================================================================== B8 loss curves
s = new_slide("B9 · Training / validation loss curves (mean ± SD over 3 seeds)", title_size=22)
pic(s, A / "p_loss_backup.png", 0.35, 1.1, w=9.3)
be = {r["run"]: r["best_epoch"] for r in b2["runs"] + b2["baseline_runs"]}
tb(s, 0.35, 4.5, 7.95, 0.6, [[("Checkpoint = max val action acc. Best epochs: BC30 " +
    "/".join(str(be[f"runs/b3_bc30_seed{i}"]) for i in range(3)) + ", FiLM-nwd " +
    "/".join(str(be[f"runs/b2_film_nwd_seed{i}"]) for i in range(3)) + ", patience=5 26/8/9.", {})]], size=15)
notes(s, "【备份 B9｜A 负责回答】手册要求的训练/验证损失曲线。patience=5 在种子 1、2 的验证平台期过早停止（第 13、14 个 epoch，最佳 epoch 为 8、9），这是 BC30 作为受控基线的原因。FiLM 的验证损失在中后期略有回升，但我们只按验证动作准确率选模，测试集从不参与。")

# ================================================================== B9 aux branches
s = new_slide("B10 · Auxiliary branches (not the primary comparison)", title_size=22)
rows = [["Branch", "What changed", "S6N3 Success (n=3)", "Take-away"],
        ["B1 language encoder", "GRU / Transformer / pos-MLP, concat fusion", "0.819–0.868", "encoder is not the bottleneck"],
        ["B4 curriculum", "easy→hard pacing, multi-scale envs (concat)", "0.790–0.857", "pacing does not fix binding"],
        ["B3 DAgger", "loop-unsticking queries (+8,526 labels)", "0.890 ± 0.015", "fixes state coverage, not grounding"],
        ["FiLM + DAgger", "B2 FiLM + B3 queries (+6,005 labels)", "0.970 ± 0.007", "next step; exceeds 5,000-ep label budget"],
        ["B5 data efficiency", "20 / 50 / 100% demos, step-matched", "see B7", "FiLM 20% = BC30 100%"]]
table(s, rows, 0.35, 1.15, 9.3, [1.8, 3.2, 1.7, 2.6], size=13, row_h=0.5, hl_rows=(4,))
footer(s, "Sources: b1_langenc_eval.json, b4_curriculum_eval.json, b3_v2_eval_s6n3.json (extra_oracle_queries_mean), b5_data_eff_eval.json.", y=4.4, size=11)
notes(s, "【备份 B10｜C 负责回答】为什么 DAgger 不是主修改？手册的预算是最多 5,000 条成功专家轨迹；DAgger 需要额外向 BabyAIBot 查询约 6,000～8,500 个状态标签，等于改变了数据预算，而且它是交互式数据收集，不是对基线的单一受控修改。FiLM+DAgger 达到 0.970 ± 0.007、每 300 局失败 9.0 局，我们把它作为下一步，而不是主结论。")

# ================================================================== B10 references
s = new_slide("B11 · References and AI-use disclosure", title_size=22)
tb(s, 0.35, 1.12, 9.3, 2.5, [
    "[1] Chevalier-Boisvert et al., BabyAI: A platform to study the sample efficiency of grounded language learning, ICLR 2019.",
    "[2] Hui et al., BabyAI 1.1, arXiv:2007.12770, 2020.",
    "[3] Perez, Strub, de Vries, Dumoulin, Courville, FiLM: Visual reasoning with a general conditioning layer, AAAI 2018.",
    "[4] Perez, FiLM: A retrospective, ML Retrospectives @ NeurIPS 2019.",
    "[5] Farama Foundation, MiniGrid & BabyAI v3.2.0 (GoToLocal, BabyAIBot).",
    "[6] Ross, Gordon, Bagnell, A reduction of imitation learning ... (DAgger), AISTATS 2011.",
    "[7] Towers et al., Gymnasium v1.4.0, arXiv:2407.17032, 2024.",
], size=13, spacing=3)
tb(s, 0.35, 3.62, 8.0, 1.4, [
    [("AI use (Handbook §7): ", {"bold": True, "color": BLUE}),
     ("Claude/Gemini IDE agents helped scaffold plotting/evaluation scripts, layout this deck and polish wording. "
      "No numbers, trajectories or citations were generated by AI: every value comes from runs/* and reports/*.json executed on our server and checked by all members.", {})],
], size=13)
notes(s, "【备份 B11｜C 负责回答】参考文献与生成式 AI 使用披露，与书面报告附录 A 一致。")

# ================================================================== drop template slides 2..10
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst)[1:10]:
    prs.part.drop_rel(sldId.rId); sldIdLst.remove(sldId)
# refresh cached text of <a:fld type="slidenum"> so static inspection matches rendered numbers
for k, sl in enumerate(prs.slides, 1):
    for fld in sl._element.iter(qn("a:fld")):
        if fld.get("type") == "slidenum":
            fld.find(qn("a:t")).text = str(k)
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
