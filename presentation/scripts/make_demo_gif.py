"""Build the 30-60 s side-by-side demo GIF (BC30 seed0 vs FiLM-nwd seed0, same unseen test seeds).

Input: src/rollouts.npz streamed read-only from 4090 by scripts/remote_rollout.py.
Output: assets/demo_bc30_vs_film.gif (first frame = static poster, shown in PDF) + assets/demo_poster.png
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "src/rollouts.npz")
meta = json.loads(str(d["meta"]))

F = "/System/Library/Fonts/Supplemental/Arial.ttf"
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
f_title = ImageFont.truetype(FB, 31)
f_sub = ImageFont.truetype(F, 25)
f_lab = ImageFont.truetype(FB, 27)
f_stat = ImageFont.truetype(FB, 26)
f_case = ImageFont.truetype(FB, 27)
f_case2 = ImageFont.truetype(F, 24)

W, H = 800, 490
PANEL = 320
STEP_MS = 200        # 0.2 s per environment step
GREEN, RED, GREY, NAVY = (16, 140, 80), (200, 40, 40), (90, 90, 90), (20, 40, 80)
COL_BC, COL_FILM = (217, 115, 13), (14, 150, 100)

CASES = [
    (4501, "Success kept: obstacle detour"),
    (4516, "Attribute confusion: purple key vs box"),
    (4560, "Distractor priority: nearer grey ball"),
    (4568, "Residual failure: target out of view"),
]


def nfilm(s):
    return sum(meta[f"{t}_{s}"]["success"] for t in ("film", "film_s1", "film_s2"))


def centered(draw, y, text, font, fill=NAVY):
    w = draw.textlength(text, font=font)
    draw.text(((W - w) / 2, y), text, font=font, fill=fill)


def outcome(m):
    return f"{m['steps']} steps" if m["success"] else "timeout"


def poster():
    """Static first frame (what PDF/static view shows): real side-by-side rollout frame + 4-clip playlist."""
    s, t = 4516, 8
    b, f = meta[f"bc30_{s}"], meta[f"film_{s}"]
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    centered(dr, 6, f"seed {s} · '{b['mission']}' · step {t}", f_title)
    P = 262
    xs = [(W // 2 - P - 30), (W // 2 + 30)]
    for x, frames, m, lab, col in [(xs[0], d[f"bc30_{s}"], b, "BC30 baseline", COL_BC), (xs[1], d[f"film_{s}"], f, "FiLM-nwd", COL_FILM)]:
        dr.text((x, 44), lab, font=f_lab, fill=col)
        k = min(t, len(frames) - 1)
        img = Image.fromarray(frames[k]).resize((384, 384), Image.NEAREST).resize((P, P), Image.LANCZOS)
        im.paste(img, (x, 78))
        dr.rectangle([x - 3, 75, x + P + 2, 78 + P + 2], outline=col, width=3)
        if k == len(frames) - 1 and m["success"]:
            st, c = f"SUCCESS: {m['steps']} steps", GREEN
        else:
            st, c = "oscillating → timeout at 36", RED
        dr.text((x, 78 + P + 6), st, font=ImageFont.truetype(FB, 22), fill=c)
    f_pl = ImageFont.truetype(F, 20); f_plb = ImageFont.truetype(FB, 20)
    dr.text((16, 380), "40 s playlist (BC30 → FiLM, FiLM seeds solved):", font=f_plb, fill=NAVY)
    def so(m):
        return f"{m['steps']} st" if m["success"] else "timeout"
    labs = ["detour", "attribute", "distractor", "blind loop"]
    items = [f"{j + 1}. {c[0]} {labs[j]}: {so(meta[f'bc30_{c[0]}'])} → {so(meta[f'film_{c[0]}'])} [{nfilm(c[0])}/3]" for j, c in enumerate(CASES)]
    for j, it in enumerate(items):
        dr.text((16 + (j % 2) * 400, 412 + (j // 2) * 32), it, font=f_pl, fill=GREY)
        assert dr.textlength(it, font=f_pl) < 384, it
    return im


def frame(case_idx, s, t):
    b, f = meta[f"bc30_{s}"], meta[f"film_{s}"]
    fb, ff = d[f"bc30_{s}"], d[f"film_{s}"]
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    centered(dr, 6, f"{case_idx}/4 · seed {s} · '{b['mission']}'", f_title)
    centered(dr, 44, f"{CASES[case_idx - 1][1]} (FiLM {nfilm(s)}/3 seeds)", f_sub, GREY)
    xs = [(W // 2 - PANEL - 24), (W // 2 + 24)]
    for x, frames, m, lab, col in [(xs[0], fb, b, "BC30 baseline", COL_BC), (xs[1], ff, f, "FiLM-nwd", COL_FILM)]:
        dr.text((x, 80), lab, font=f_lab, fill=col)
        k = min(t, len(frames) - 1)
        img = Image.fromarray(frames[k]).resize((384, 384), Image.NEAREST).resize((PANEL, PANEL), Image.LANCZOS)
        im.paste(img, (x, 116))
        dr.rectangle([x - 3, 113, x + PANEL + 2, 116 + PANEL + 2], outline=col, width=3)
        done = k == len(frames) - 1
        if done and m["success"]:
            st, c = f"SUCCESS: {m['steps']} steps", GREEN
        elif done:
            st, c = "TIMEOUT (36 steps)", RED
        else:
            st, c = f"step {k}/36", GREY
        dr.text((x, 116 + PANEL + 10), st, font=f_stat, fill=c)
    return im


frames, durs = [], []
p = poster(); p.save(ROOT / "assets/demo_poster.png")
frames.append(p); durs.append(3000)
for ci, (s, _) in enumerate(CASES, 1):
    n = max(len(d[f"bc30_{s}"]), len(d[f"film_{s}"]))
    for t in range(n):
        frames.append(frame(ci, s, t))
        durs.append(1200 if t == 0 else STEP_MS)
    durs[-1] = 2500  # hold final outcome
pal = [fr.convert("P", palette=Image.ADAPTIVE, colors=128) for fr in frames]
out = ROOT / "assets/demo_bc30_vs_film.gif"
pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
tot = sum(durs) / 1000
assert 30 <= tot <= 60, tot
print(out, len(frames), "frames", tot, "s", out.stat().st_size // 1024, "KB")
