"""Collect eval json files into the ablation table (Table IV style)."""
import json
import os

ROWS = [("raw (zero-shot)", "outputs/llm_raw/zeroshot_val.json", "text"),
        ("Baseline (raw LLM, SFT)", "outputs/llm_raw/eval_val.json", "text"),
        ("+ Historical/Current Graph", "outputs/llm_hist/eval_val.json", "text"),
        ("+ Future Graph Prediction", "outputs/llm_fut/eval_val.json", "text"),
        ("+ CoT Prompt", "outputs/llm_cot/eval_val.json", "text"),
        ("+ Full Model (text traj)", "outputs/llm_full/eval_val.json", "text"),
        ("+ Full Model (head traj)", "outputs/llm_full/eval_val.json", "head"),
        ("Full v2: [PLAN] no CE, alpha2=1 (text traj)", "outputs/llm_full_v2/eval_val.json", "text"),
        ("Full v2: [PLAN] no CE, alpha2=1 (head traj)", "outputs/llm_full_v2/eval_val.json", "head"),
        ("[A] Future graph text + graph tokens", "outputs/llm_futg/eval_val.json", "text"),
        ("[B] Future graph, predictor w/o agent-agent edges", "outputs/llm_fut_noaa/eval_val.json", "text"),
        ("[N] Node-level graph tokens aligned to names + encoder FT", "outputs/llm_node/eval_val.json", "text"),
        ("[C] 2k train: Baseline", "outputs/llm_raw_2k/eval_val.json", "text"),
        ("[C] 2k train: + Hist graph", "outputs/llm_hist_2k/eval_val.json", "text"),
        ("[C] 2k train: + Future graph", "outputs/llm_fut_2k/eval_val.json", "text")]


def main():
    keys = ["ADE@3s", "FDE@3s", "ADE@6s", "FDE@6s", "FVR", "Collision"]
    lines = ["| Variant | " + " | ".join(keys) + " | parse fail |", "|" + "---|" * (len(keys) + 2)]
    ref = None
    for name, p, k in ROWS:
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        ref = m["const_vel_ref"]
        r = m[k]
        lines.append(f"| {name} | " + " | ".join(
            f"{100 * r[x]:.2f}%" if x in ("FVR", "Collision") else f"{r[x]:.3f}" for x in keys) +
            f" | {100 * m['parse_fail_rate']:.1f}% |")
    if ref:
        lines.append("| (ref) constant velocity | " + " | ".join(
            f"{100 * ref[x]:.2f}%" if x in ("FVR", "Collision") else f"{ref[x]:.3f}" for x in keys) + " | - |")
        lines.append(f"\nGT FVR (thresholds = 95th pct of train GT): {100 * ref['FVR_GT']:.2f}%")
    # Table I: model scale (baseline vs SAGE-Drive)
    scale = [("Qwen3-0.6B", "outputs/llm_raw/eval_val.json", "outputs/llm_full/eval_val.json"),
             ("Qwen3-1.7B", "outputs/llm_raw_1.7b/eval_val.json", "outputs/llm_full_1.7b/eval_val.json"),
             ("Qwen3-4B", "outputs/llm_raw_4b/eval_val.json", "outputs/llm_full_4b/eval_val.json")]
    rows = []
    for name, pb, pf in scale:
        if not (os.path.exists(pb) and os.path.exists(pf)):
            continue
        b, f = json.load(open(pb))["text"], json.load(open(pf))
        best = min(("text", "head"), key=lambda k: f[k]["ADE@6s"] if k in f else 1e9)
        f = f[best]
        rows.append(f"| {name} | {b['ADE@6s']:.3f} | {f['ADE@6s']:.3f} ({best}) | "
                    f"{100 * (1 - f['ADE@6s'] / b['ADE@6s']):.1f}% | {100 * b['FVR']:.2f}% | {100 * f['FVR']:.2f}% |")
    if rows:
        lines += ["", "Model scale (0.6B: 3 epochs; 1.7B/4B: 1 epoch, same budget for baseline and full)",
                  "| Model | Baseline ADE@6s | +SAGE-Drive ADE@6s | Improvement | Baseline FVR | +SAGE-Drive FVR |",
                  "|---|---|---|---|---|---|"] + rows
    print("\n".join(lines))
    open("outputs/results.md", "w").write("\n".join(lines) + "\n")
    import subprocess, sys  # noqa: E401  (per-scenario-type breakdown -> outputs/results_by_scene.md)
    subprocess.run([sys.executable, "scripts/stratify.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
