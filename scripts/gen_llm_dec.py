"""P2: generate LLM decisions on val -> {out}/dec_val.jsonl {sid, dec (None if unparsable), output}

    python scripts/gen_llm_dec.py --ckpt outputs/llm_dec/final.pt --out outputs/llm_dec
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.decision import parse  # noqa: E402
from sage.llm import SageLLM  # noqa: E402
from train_llm import build_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/llm_dec/final.pt")
    ap.add_argument("--out", default="outputs/llm_dec")
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--split", default="val")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--variant", default="dec", choices=["dec", "instr"])
    ap.add_argument("--instr_set", default="seen", choices=["seen", "unseen"])
    ap.add_argument("--tag", default=None, help="output name, default dec_{split}")
    a = ap.parse_args()
    model = SageLLM(a.llm, a.variant).cuda().eval()
    r = model.load_state_dict(torch.load(a.ckpt, map_location="cuda"), strict=False)
    assert not r.unexpected_keys
    dl = build_loader(a, a.split, model, a.bs, False, a.max)
    lab = {(x := json.loads(l))["sid"]: x["dec"] for l in open(f"{a.cache}/{a.split}_dec.jsonl")}
    n, fail, hit = 0, 0, {k: 0 for k in ("lane", "lead", "v3", "v6")}
    with open(f"{a.out}/{a.tag or 'dec_' + a.split}.jsonl", "w") as f:
        for b in tqdm(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                texts, _ = model.plan(b)
            for sid, t in zip(b["sid"], texts):
                d = parse(t)
                n += 1
                if d is None:
                    fail += 1
                else:
                    for k in hit:
                        hit[k] += d[k] == lab[sid][k]
                f.write(json.dumps(dict(sid=sid, dec=d, output=t)) + "\n")
    res = dict(n=n, parse_fail=fail, acc={k: v / n for k, v in hit.items()})
    print(json.dumps(res))
    json.dump(res, open(f"{a.out}/{a.tag or 'dec_' + a.split}_acc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
