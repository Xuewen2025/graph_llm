"""P1 comparison: GNN planner runs vs LLM runs. ADE@6s by subset + paired bootstrap -> outputs/p1_results.md

    python scripts/compare_p1.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage import config as C  # noqa: E402
from sage.data import load_scenes  # noqa: E402
from sage.serialize import ego_kinematics  # noqa: E402

RUNS = {"llm_raw": "outputs/llm_raw", "llm_fut": "outputs/llm_fut", "llm_node": "outputs/llm_node",
        "gnn_plan": "outputs/gnn_plan", "gnn_plan_nonav": "outputs/gnn_plan_nonav",
        "gnn_plan_scratch": "outputs/gnn_plan_scratch", "gnn_plan_noaa": "outputs/gnn_plan_noaa"}
PAIRS = [("gnn_plan", "llm_raw"), ("gnn_plan", "llm_fut"), ("gnn_plan_nonav", "gnn_plan"),
         ("gnn_plan_scratch", "gnn_plan"), ("gnn_plan_noaa", "gnn_plan")]


def main(RUNS=RUNS, PAIRS=PAIRS, out="outputs/p1_results.md", extra_groups=None):
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
    groups = {"all": lambda s: True, "lead vehicle <25 m": lambda s: lead[s], "hard (CV err top 20%)": lambda s: cv[s] >= thr}
    if extra_groups:
        groups.update(extra_groups(sc))

    runs = {k: v for k, v in RUNS.items() if os.path.exists(f"{v}/eval_val.json")}
    E, M = {}, {}
    for k, v in runs.items():
        P = {(d := json.loads(l))["sid"]: np.array(d["text_traj"]) for l in open(f"{v}/eval_val_pred.jsonl")}
        E[k] = {s: np.linalg.norm(P[s] - gt[s], axis=1).mean() for s in gt}
        M[k] = json.load(open(f"{v}/eval_val.json"))["text"]
    L = ["| run | ADE@3s | ADE@6s | FDE@6s | " + " | ".join(f"{g} (n={sum(map(f, gt))})" for g, f in groups.items() if g != "all")
         + " | FVR | Collision |", "|---" * (7 + len(groups) - 1) + "|"]
    for k in runs:
        sub = " | ".join(f"{np.mean([E[k][s] for s in gt if f(s)]):.3f}" for g, f in groups.items() if g != "all")
        m = M[k]
        L.append(f"| {k} | {m['ADE@3s']:.3f} | {m['ADE@6s']:.3f} | {m['FDE@6s']:.3f} | {sub} | {m['FVR']*100:.1f}% | "
                 f"{m['Collision']*100:.2f}% |")
    rng = np.random.default_rng(0)
    L += ["", "| A - B (ADE@6s) | subset | n | diff (m) | 95% CI | rel. |", "|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        if a not in E or b not in E:
            continue
        for g, f in groups.items():
            ss = [s for s in gt if f(s)]
            d = np.array([E[a][s] - E[b][s] for s in ss])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sig = " *" if lo > 0 or hi < 0 else ""
            L.append(f"| {a} - {b} | {g} | {len(ss)} | {d.mean():+.3f}{sig} | [{lo:+.3f}, {hi:+.3f}] | "
                     f"{100 * d.mean() / np.mean([E[b][s] for s in ss]):+.1f}% |")
    L.append("\n`*` = 95% CI excludes 0. Negative = A is better.")
    print("\n".join(L))
    open(out, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
