"""Generate CoT reasoning labels with an OpenRouter teacher (falls back to the rule-based teacher).

    python scripts/gen_labels.py --split train --model ~deepseek/deepseek-v4-flash-latest --workers 48
    python scripts/gen_labels.py --split train --teacher rule        # free, deterministic
Resumable: already-labelled sids in the output file are skipped.
Output: data/cache/{split}_cot.jsonl  {sid, reasoning, source}
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import load_scenes  # noqa: E402
from sage.serialize import ego_kinematics  # noqa: E402
from sage.teacher import OpenRouterTeacher, rule_reasoning  # noqa: E402


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--teacher", default="llm", choices=["llm", "rule"])
    ap.add_argument("--model", default="~deepseek/deepseek-v4-flash-latest")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--budget_usd", type=float, default=15.0, help="stop when this much was spent")
    a = ap.parse_args()
    load_env()

    recs = [json.loads(l) for l in open(f"{a.cache}/{a.split}_records.jsonl")]
    if a.max:
        recs = recs[: a.max]
    kin = {s["sid"]: ego_kinematics(s) for s in load_scenes(f"{a.cache}/{a.split}_scenes.pkl")}
    out_path = f"{a.cache}/{a.split}_cot.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["sid"] for l in open(out_path)}
    todo = [r for r in recs if r["sid"] not in done]
    print(f"{len(done)} done, {len(todo)} to label with {a.teacher}:{a.model if a.teacher == 'llm' else ''}")

    teacher = OpenRouterTeacher(a.model) if a.teacher == "llm" else None
    spent, n_fallback = 0.0, 0

    def job(r):
        v0, acc, _ = kin[r["sid"]]
        if teacher is not None:
            txt, usage = teacher(r["fut"], np.array(r["gt"]), v0)
            if txt:
                return r["sid"], txt, "llm", usage.get("cost", 0.0)
        return r["sid"], rule_reasoning(r, v0, acc), "rule", 0.0

    with open(out_path, "a") as f, ThreadPoolExecutor(a.workers if teacher else 1) as ex:
        futs = [ex.submit(job, r) for r in todo]
        pbar = tqdm(as_completed(futs), total=len(futs))
        for fu in pbar:
            sid, txt, src, cost = fu.result()
            spent += cost or 0.0
            n_fallback += src == "rule" and teacher is not None
            f.write(json.dumps(dict(sid=sid, reasoning=txt, source=src)) + "\n")
            f.flush()
            pbar.set_postfix(usd=f"{spent:.3f}", fallback=n_fallback)
            if spent > a.budget_usd:
                print("budget reached, stopping")
                for x in futs:
                    x.cancel()
                break
    print(f"spent ${spent:.3f}, rule fallbacks {n_fallback}")


if __name__ == "__main__":
    main()
