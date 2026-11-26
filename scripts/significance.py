"""Paired bootstrap (5000 resamples) of per-scenario ADE@6s differences -> outputs/significance.md"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage import config as C  # noqa: E402
from sage.data import load_scenes  # noqa: E402

PAIRS = [("hist", "raw"), ("fut", "raw"), ("cot", "fut"), ("fut_noaa", "fut"), ("futg", "fut"), ("node", "fut"),
         ("node", "futg")]


def main():
    gt = {(r := json.loads(l))["sid"]: np.array(r["gt"]) for l in open("data/cache/val_records.jsonl")}
    sc = {s["sid"]: s for s in load_scenes("data/cache/val_scenes.pkl")}
    lead = {}
    for sid, s in sc.items():
        p = s["agent_pos"][1:, C.T0][s["agent_valid"][1:, C.T0].astype(bool)]
        lead[sid] = bool(((p[:, 0] > 0) & (p[:, 0] < 25) & (abs(p[:, 1]) < 2.5)).any())
    E = {}
    for v in {x for p in PAIRS for x in p}:
        P = {(d := json.loads(l))["sid"]: np.array(d["text_traj"]) for l in open(f"outputs/llm_{v}/eval_val_pred.jsonl")}
        E[v] = {s: np.linalg.norm(P[s] - gt[s], axis=1).mean() for s in gt}
    rng = np.random.default_rng(0)
    lines = ["| comparison (A - B, ADE@6s) | subset | n | diff (m) | 95% CI | rel. |", "|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        for name, f in (("all", lambda s: True), ("lead vehicle <25 m", lambda s: lead[s])):
            ss = [s for s in gt if f(s)]
            d = np.array([E[a][s] - E[b][s] for s in ss])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rel = 100 * d.mean() / np.mean([E[b][s] for s in ss])
            sig = " *" if lo > 0 or hi < 0 else ""
            lines.append(f"| {a} - {b} | {name} | {len(ss)} | {d.mean():+.3f}{sig} | [{lo:+.3f}, {hi:+.3f}] | {rel:+.1f}% |")
    lines.append("\n`*` = 95% CI excludes 0. Negative = A is better.")
    print("\n".join(lines))
    open("outputs/significance.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
