# Executed READ-ONLY on 4090 via `ssh 4090 'cd p3-babyai && PYTHONDONTWRITEBYTECODE=1 env_embodied/bin/python -' < this`.
# Writes nothing on the server: rendered frames are streamed to stdout as a compressed npz.
import sys, io, json
sys.dont_write_bytecode = True
sys.path.insert(0, "branches/b2_film"); sys.path.insert(0, "branches/_shared"); sys.path.insert(0, "src")
import numpy as np
import gymnasium as gym, minigrid  # noqa
import torch
torch.set_num_threads(1)
from film_policy import load_any_run, make_policy_fn

ACTS = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
RUNS = {"bc30": "runs/b3_bc30_seed0", "film": "runs/b2_film_nwd_seed0",
        "film_s1": "runs/b2_film_nwd_seed1", "film_s2": "runs/b2_film_nwd_seed2"}
SEEDS = [4501, 4516, 4565, 4570, 4582, 4560, 4630, 4716, 4568]  # the 9 CASE_ANALYSIS cases
env = gym.make("BabyAI-GoToLocalS6N3-v0", render_mode="rgb_array")
out, meta = {}, {}
for tag, run in RUNS.items():
    m, _, _ = load_any_run(run); pol = make_policy_fn(m)
    for s in SEEDS:
        obs, _ = env.reset(seed=s); frames = [env.render()]; acts = []; term = trunc = False; R = 0.0
        while not (term or trunc):
            a = int(pol(obs)); acts.append(ACTS[a])
            obs, r, term, trunc, _ = env.step(a); R += r
            frames.append(env.render())
        if tag in ("bc30", "film"):
            out[f"{tag}_{s}"] = np.stack(frames).astype(np.uint8)
        meta[f"{tag}_{s}"] = dict(run=run, seed=s, mission=obs["mission"], success=bool(term),
                                  steps=len(acts), ret=round(float(R), 4), actions=acts)
env.close()
buf = io.BytesIO()
np.savez_compressed(buf, meta=json.dumps(meta), **out)
sys.stdout.buffer.write(buf.getvalue())
