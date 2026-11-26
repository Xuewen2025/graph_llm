"""ADE@6s by scenario type (where relational reasoning should matter) -> outputs/results_by_scene.md"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage import config as C  # noqa: E402
from sage.data import load_scenes  # noqa: E402
from sage.serialize import ego_kinematics  # noqa: E402

RUNS = ["raw", "hist", "fut", "cot", "futg", "fut_noaa", "node", "full", "full_v2"]


def main():
    sc = {s["sid"]: s for s in load_scenes("data/cache/val_scenes.pkl")}
    gt = {(r := json.loads(l))["sid"]: np.array(r["gt"]) for l in open("data/cache/val_records.jsonl")}
    cv, lead = {}, {}
    for sid, s in sc.items():
        v0 = ego_kinematics(s)[0]
        c = np.stack([v0 * C.DT * np.arange(1, C.N_PLAN + 1), np.zeros(C.N_PLAN)], 1)
        cv[sid] = np.linalg.norm(c - gt[sid], axis=1).mean()
        p = s["agent_pos"][1:, C.T0][s["agent_valid"][1:, C.T0].astype(bool)]
        lead[sid] = bool(((p[:, 0] > 0) & (p[:, 0] < 25) & (abs(p[:, 1]) < 2.5)).any())
    thr = np.quantile(list(cv.values()), 0.8)
    groups = {"all": lambda s: True, "lead vehicle <25 m ahead": lambda s: lead[s],
              "no lead vehicle": lambda s: not lead[s], "hard (const-vel err top 20%)": lambda s: cv[s] >= thr}
    runs = [r for r in RUNS if os.path.exists(f"outputs/llm_{r}/eval_val_pred.jsonl")]
    lines = ["| ADE@6s | n | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
    P = {r: {(d := json.loads(l))["sid"]: (np.array(d["head_traj"]) if d.get("head_traj") else np.array(d["text_traj"]))
             for l in open(f"outputs/llm_{r}/eval_val_pred.jsonl")} for r in runs}
    runs = [r for r in runs if len(P[r]) >= len(gt)]  # skip evaluations still in progress
    lines = ["| ADE@6s | n | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
    for g, f in groups.items():
        ss = [s for s in gt if f(s)]
        lines.append(f"| {g} | {len(ss)} | " + " | ".join(
            f"{np.mean([np.linalg.norm(P[r][s] - gt[s], axis=1).mean() for s in ss]):.3f}" for r in runs) + " |")
    lines.append("\n(full / full_v2 use the head trajectory)")
    print("\n".join(lines))
    open("outputs/results_by_scene.md", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
