"""D: frozen-LM sentence embeddings of the instructions (mean-pooled last hidden state of the base Qwen3-0.6B),
the text encoder of the GNN baseline.  -> data/cache/{split}_instr_emb.pkl {sid: {set: float16 [K, d]}}

    python scripts/embed_instructions.py --split train
"""
import argparse
import json
import pickle

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.llm)
    tok.padding_side = "right"
    m = AutoModel.from_pretrained(a.llm, dtype=torch.bfloat16).cuda().eval()
    rows = [json.loads(l) for l in open(f"{a.cache}/{a.split}_instr.jsonl")]
    flat = [(r["sid"], k, i, t) for r in rows for k in ("seen", "unseen") if k in r for i, t in enumerate(r[k])]
    out = {}
    with torch.no_grad():
        for s in range(0, len(flat), 256):
            ch = flat[s:s + 256]
            b = tok([t for *_, t in ch], return_tensors="pt", padding=True).to("cuda")
            h = m(**b).last_hidden_state.float()
            msk = b["attention_mask"][..., None].float()
            e = ((h * msk).sum(1) / msk.sum(1)).cpu().numpy().astype(np.float16)
            for (sid, k, i, _), v in zip(ch, e):
                out.setdefault(sid, {}).setdefault(k, []).append(v)
    out = {sid: {k: np.stack(v) for k, v in d.items()} for sid, d in out.items()}
    with open(f"{a.cache}/{a.split}_instr_emb.pkl", "wb") as f:
        pickle.dump(out, f)
    print(a.split, len(out), "dim", next(iter(out.values()))["seen"].shape)


if __name__ == "__main__":
    main()
