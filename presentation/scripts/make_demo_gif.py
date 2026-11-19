"""Build the 35-45 s 3-column Attention-protagonist demo GIF (Slide 8).

Input : src/rollouts_attn.npz streamed deterministically from 4090.
Output: assets/demo_attention_laws.gif (first frame = static poster shown in PDF) + assets/demo_attention_poster.png
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "src/rollouts_attn.npz")
meta = json.loads(str(d["meta"]))

F = "/System/Library/Fonts/Supplemental/Arial.ttf"
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
f_title = ImageFont.truetype(FB, 28)
f_sub = ImageFont.truetype(F, 18)
f_lab = ImageFont.truetype(FB, 21)
f_stat = ImageFont.truetype(FB, 19)
f_plb = ImageFont.truetype(FB, 18)
f_pl = ImageFont.truetype(F, 16)

W, H = 900, 500
PANEL = 252
STEP_MS = 185

NAVY = (17, 43, 92)
GREY = (85, 95, 107)
RED = (192, 57, 43)
GREEN = (14, 150, 100)
COL_BC = (217, 115, 13)
COL_FILM = (100, 116, 139)     # neutral slate control for Channel FiLM
COL_SS = (37, 99, 235)         # royal blue for Spatial-Softmax
COL_MAP = (234, 88, 12)        # warm orange for Map-Inject
COL_REL = (15, 118, 110)       # deep teal protagonist for MH-Spatial-Relational

CLIPS = [
    (4502, "Law 0 · Spatial competition beats Channel FiLM on crowded keys", [
        ("bc30", "BC30 (Concat) [0/3]", COL_BC),
        ("film", "Channel FiLM [0/3]", COL_FILM),
        ("mhrel", "Ours: MH-Rel [3/3]", COL_REL),
    ]),
    (4501, "Laws 2–3 · Map-Inject & MH-Rel fix v1 Spatial-SS obstacle erasure", [
        ("ss", "v1 Spatial-SS [1/3]", COL_SS),
        ("mapinj", "Ours: Map-Inject [3/3]", COL_MAP),
        ("mhrel", "Ours: MH-Rel [3/3]", COL_REL),
    ]),
    (4516, "Conjunctive grounding · Same-color purple key vs purple box", [
        ("bc30", "BC30 (Concat) [0/3]", COL_BC),
        ("ss", "Ours: Spatial-SS [3/3]", COL_SS),
        ("mhrel", "Ours: MH-Rel [2/3]", COL_REL),
    ]),
    (4568, "Hard POMDP boundary · Target behind agent (out of 7×7 view)", [
        ("bc30", "BC30 (s0 loop) [1/3]", COL_BC),
        ("film", "Channel FiLM [0/3]", COL_FILM),
        ("mhrel", "Ours: MH-Rel [0/3]", COL_REL),
    ]),
]


def centered(draw, y, text, font, fill=NAVY):
    w = draw.textlength(text, font=font)
    assert w <= W - 16, f"Too wide ({w}): {text}"
    draw.text(((W - w) / 2, y), text, font=font, fill=fill)


def poster():
    """Static first frame (rendered in PDF & slide thumbnail): seed 4502 step 7 where FiLM fails 0/3 & Attention wins 3/3."""
    s, t = 4502, 7
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    centered(dr, 8, f"seed {s} · '{meta[f'bc30_s0_{s}']['mission']}' · step {t}", f_title)
    centered(dr, 44, "Crowded yellow-key distractors: Channel FiLM fails (0/3), Spatial & Relational Attention solve 3/3", f_sub, COL_REL)

    P = 236
    xs = [34, 332, 630]
    cols = [
        ("bc30", "BC30 (Concat) [0/3]", COL_BC),
        ("film", "Channel FiLM [0/3]", COL_FILM),
        ("mhrel", "Ours: MH-Rel [3/3]", COL_REL),
    ]
    for x, (tag, lab, col) in zip(xs, cols):
        m = meta[f"{tag}_s0_{s}"]
        frames = d[f"{tag}_{s}"]
        dr.text((x, 74), lab, font=f_lab, fill=col)
        k = min(t, len(frames) - 1)
        img = Image.fromarray(frames[k]).resize((384, 384), Image.NEAREST).resize((P, P), Image.LANCZOS)
        im.paste(img, (x, 104))
        dr.rectangle([x - 3, 101, x + P + 2, 104 + P + 2], outline=col, width=3)
        if k == len(frames) - 1 and m["success"]:
            st, c = f"SUCCESS: {m['steps']} steps", GREEN
        else:
            st, c = "confused → timeout 36", RED
        dr.text((x, 104 + P + 6), st, font=f_stat, fill=c)

    dr.text((28, 380), "40 s Attention rollout playlist (unseen test seeds, [seeds solved / 3]):", font=f_plb, fill=NAVY)
    lines = [
        "1. 4502 keys: FiLM timeout [0/3] → MH-Rel 7 st [3/3]",
        "2. 4501 detour: v1 SS [1/3] → Map-Inj & MH-Rel 8 st [3/3]",
        "3. 4516 box: BC30 [0/3] → Spatial-SS 8 st [3/3]",
        "4. 4568 blind spot: all 1-frame models L/R loop [0/3]",
    ]
    for j, it in enumerate(lines):
        assert dr.textlength(it, font=f_pl) < 428, it
        dr.text((28 + (j % 2) * 442, 412 + (j // 2) * 30), it, font=f_pl, fill=GREY)
    return im


def frame(ci, s, subtitle, spec, t):
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    centered(dr, 8, f"{ci}/4 · seed {s} · '{meta[f'bc30_s0_{s}']['mission']}'", f_title)
    centered(dr, 44, subtitle, f_sub, GREY)

    xs = [30, 324, 618]
    for x, (tag, lab, col) in zip(xs, spec):
        m = meta[f"{tag}_s0_{s}"]
        frs = d[f"{tag}_{s}"]
        dr.text((x, 82), lab, font=f_lab, fill=col)
        k = min(t, len(frs) - 1)
        img = Image.fromarray(frs[k]).resize((384, 384), Image.NEAREST).resize((PANEL, PANEL), Image.LANCZOS)
        im.paste(img, (x, 114))
        dr.rectangle([x - 3, 111, x + PANEL + 2, 114 + PANEL + 2], outline=col, width=3)
        done = k == len(frs) - 1
        if done and m["success"]:
            st, c = f"SUCCESS: {m['steps']} steps", GREEN
        elif done:
            st, c = "TIMEOUT (36 steps)", RED
        else:
            st, c = f"step {k}/36", GREY
        dr.text((x, 114 + PANEL + 10), st, font=f_stat, fill=c)
    return im


frames, durs = [], []
p = poster()
p.save(ROOT / "assets/demo_attention_poster.png")
frames.append(p); durs.append(2800)
for ci, (s, sub, spec) in enumerate(CLIPS, 1):
    n = max(len(d[f"{tag}_{s}"]) for tag, _, _ in spec)
    for t in range(n):
        frames.append(frame(ci, s, sub, spec, t))
        durs.append(1000 if t == 0 else STEP_MS)
    durs[-1] = 2200
pal = [fr.convert("P", palette=Image.ADAPTIVE, colors=128) for fr in frames]
out = ROOT / "assets/demo_attention_laws.gif"
pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
tot = sum(durs) / 1000
assert 30 <= tot <= 55, tot
print(out, len(frames), "frames", round(tot, 1), "s", out.stat().st_size // 1024, "KB")
