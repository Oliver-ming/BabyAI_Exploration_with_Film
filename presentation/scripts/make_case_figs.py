"""Task-example figure (slide 2: seed 4502 crowded keys) and per-case thumbnails + Attention solve metadata (backup B10)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"
d = np.load(ROOT / "src/rollouts_attn.npz")
meta = json.loads(str(d["meta"]))
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 14, "savefig.dpi": 250})

# --- annotated task example: seed 4502 'go to the purple key' (FiLM 0/3 vs Attention 3/3), t=2 (facing room) ---
img = d["mhrel_4502"][2]  # after turning to illuminate the room
fig, ax = plt.subplots(figsize=(4.05, 3.75))
ax.imshow(img, extent=(0, 6, 6, 0), interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
kw = dict(fontsize=12.8, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=2, color="#111"), ha="center")
ax.annotate("target:\npurple key", xy=(1.5, 1.5), xytext=(1.35, 5.35), color="#7B2FBE",
            bbox=dict(fc="white", ec="#7B2FBE", lw=1.6, pad=0.25), **kw)
ax.annotate("2 yellow-key\ndistractors", xy=(2.9, 2.0), xytext=(4.35, 5.35), color="#B45309",
            bbox=dict(fc="white", ec="#B45309", lw=1.5, pad=0.25), **kw)
ax.annotate("agent (7×7 view)", xy=(4.5, 3.5), xytext=(3.8, 0.52), color="#C0392B",
            bbox=dict(fc="white", ec="#C0392B", lw=1.5, pad=0.25), **kw)
ax.set_title("seed 4502: 'go to the purple key'", fontsize=13.8, fontweight="bold", pad=6, color="#112B5C")
fig.tight_layout(pad=0.2); fig.savefig(A / "p_task.png"); plt.close(fig)

# --- t=0 thumbnails & multi-model 3-seed solve table for backup B10 ---
ALL_SEEDS = [4502, 4501, 4516, 4565, 4570, 4582, 4560, 4568]
cases_attn = {}
for s in ALL_SEEDS:
    Image.fromarray(d[f"mhrel_{s}"][0]).resize((384, 384), Image.NEAREST).save(A / f"case_{s}_t0.png")
    entry = {"mission": meta[f"bc30_s0_{s}"]["mission"]}
    for tag in ("bc30", "film", "ss", "mapinj", "mhrel"):
        entry[tag] = [(meta[f"{tag}_s{k}_{s}"]["success"], meta[f"{tag}_s{k}_{s}"]["steps"]) for k in range(3)]
    cases_attn[str(s)] = entry

json.dump(cases_attn, open(A / "cases_attn.json", "w"), indent=1)
print("saved p_task.png and cases_attn.json cleanly")
