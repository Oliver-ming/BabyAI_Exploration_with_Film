"""Task-example figure (slide 2) and per-case thumbnails (backup) from the streamed rollouts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"
d = np.load(ROOT / "src/rollouts.npz")
meta = json.loads(str(d["meta"]))
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 14, "savefig.dpi": 250})

# --- annotated task example: seed 4516 'go to the purple box', t=0 ---
img = d["film_4516"][0]
T = 32  # px per tile
fig, ax = plt.subplots(figsize=(4.0, 3.7))
ax.imshow(img, extent=(0, 6, 6, 0), interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
kw = dict(fontsize=13.5, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=2, color="#111"), ha="center")
ax.annotate("target:\npurple box", xy=(1.5, 3.5), xytext=(1.5, 5.45), color="#7B2FBE",
            bbox=dict(fc="white", ec="#7B2FBE", lw=1.5, pad=0.25), **kw)
ax.annotate("distractor:\npurple key", xy=(3.5, 3.5), xytext=(4.1, 5.45), color="#7B2FBE",
            bbox=dict(fc="white", ec="#7B2FBE", lw=1.5, pad=0.25), **kw)
ax.annotate("agent (facing east)", xy=(4.5, 2.5), xytext=(3.0, 0.45), color="#C0392B",
            bbox=dict(fc="white", ec="#C0392B", lw=1.5, pad=0.25), **kw)
ax.set_title("seed 4516: 'go to the purple box'", fontsize=14, fontweight="bold", pad=6)
fig.tight_layout(pad=0.2); fig.savefig(A / "p_task.png"); plt.close(fig)

# --- t=0 thumbnails for the failure-case backup slide ---
for s in [4516, 4565, 4570, 4582, 4560, 4568, 4501]:
    Image.fromarray(d[f"film_{s}"][0]).resize((384, 384), Image.NEAREST).save(A / f"case_{s}_t0.png")
cases = {}
for s in [4501, 4516, 4565, 4570, 4582, 4560, 4630, 4716, 4568]:
    cases[s] = dict(mission=meta[f"bc30_{s}"]["mission"],
                    bc30=(meta[f"bc30_{s}"]["success"], meta[f"bc30_{s}"]["steps"]),
                    film=[(meta[f"{t}_{s}"]["success"], meta[f"{t}_{s}"]["steps"]) for t in ("film", "film_s1", "film_s2")])
json.dump(cases, open(A / "cases.json", "w"), indent=1)
print("ok")
