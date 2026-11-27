"""Train a SAGE-Drive LLM variant (single GPU; run variants in parallel on different GPUs).

Full model, two-stage alignment (Sec. III-C-2c):
    python scripts/train_llm.py --variant full --epochs1 1 --epochs2 2 --alpha1 1.0 --alpha2 0.1
Ablation variants (text-only, CE loss):
    python scripts/train_llm.py --variant raw|hist|fut|cot --epochs1 0 --epochs2 3
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sage.data import SageDataset, load_scenes  # noqa: E402
from sage.llm import VARIANTS, Collator, SageLLM  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def build_loader(a, split, model, bs, shuffle, max_n=0, ddp=False):
    import pickle
    v = VARIANTS[a.variant]
    recs = load_jsonl(f"{a.cache}/{split}_records.jsonl")
    cot = {}
    if v["cot"] and (split == "train" or os.path.exists(f"{a.cache}/{split}_cot.jsonl")):
        cot = {r["sid"]: r["reasoning"] for r in load_jsonl(f"{a.cache}/{split}_cot.jsonl")}
        if split == "train":
            recs = [r for r in recs if r["sid"] in cot]
    if v.get("dec"):
        dec = {r["sid"]: r["text"] for r in load_jsonl(f"{a.cache}/{split}_dec.jsonl")}
        for r in recs:
            r["dec_text"] = dec[r["sid"]]
    if v.get("instr"):
        ins = {r["sid"]: r for r in load_jsonl(f"{a.cache}/{split}_instr.jsonl")}
        key = "seen" if split == "train" else getattr(a, "instr_set", "seen")
        for r in recs:
            r["instr_list"] = ins[r["sid"]][key]
    if max_n:
        recs = recs[:max_n]
    scenes, preds = [], {}
    if v["graph_tokens"]:
        scenes = load_scenes(f"{a.cache}/{split}_scenes.pkl")
        with open(f"{a.cache}/{split}_pred.pkl", "rb") as f:
            preds = pickle.load(f)
    nmap = {}
    if v.get("node"):
        nmap = {r["sid"]: r["nodes"] for r in load_jsonl(f"{a.cache}/{split}_nodemap.jsonl")}
    ds = SageDataset(scenes, preds, recs, with_graphs=v["graph_tokens"])
    sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=shuffle) if ddp else None
    return DataLoader(ds, bs, shuffle=shuffle and sampler is None, sampler=sampler, num_workers=8,
                      collate_fn=Collator(model.tok, a.variant, cot, node_map=nmap), drop_last=shuffle, persistent_workers=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--llm", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs1", type=float, default=1, help="stage I epochs (large alpha)")
    ap.add_argument("--epochs2", type=float, default=2, help="stage II epochs (small alpha)")
    ap.add_argument("--alpha1", type=float, default=1.0)
    ap.add_argument("--alpha2", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_new", type=float, default=1e-3, help="lr for aligner / head (from scratch)")
    ap.add_argument("--lr_graph", type=float, default=1e-4, help="lr for the graph encoder when fine-tuned")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--q_tokens", type=int, default=8)
    ap.add_argument("--train_graph", action="store_true", help="also fine-tune the graph encoder")
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--init", default=None, help="initialise trainable weights (e.g. LoRA) from a checkpoint")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (less memory)")
    a = ap.parse_args()
    a.out = a.out or f"outputs/llm_{a.variant}"
    # multi-GPU: launch with torchrun --nproc_per_node K (global batch = K * bs * accum)
    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    rank = int(os.environ.get("RANK", 0))
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    main_proc = rank == 0
    if main_proc:
        os.makedirs(a.out, exist_ok=True)
        json.dump(dict(vars(a), world=int(os.environ.get("WORLD_SIZE", 1))), open(f"{a.out}/args.json", "w"), indent=1)
    torch.manual_seed(0)

    model = SageLLM(a.llm, a.variant, lora_r=a.lora_r, q_tokens=a.q_tokens, train_graph=a.train_graph).cuda()
    if a.init:
        r = model.load_state_dict(torch.load(a.init, map_location="cuda"), strict=False)
        assert not r.unexpected_keys, r.unexpected_keys
        if main_proc:
            print("init from", a.init, flush=True)
    if a.grad_ckpt:
        model.llm.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dl = build_loader(a, "train", model, a.bs, True, a.max_train, ddp)
    new_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "llm." not in n and not n.startswith("predictor.")]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "llm." in n]
    graph_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("predictor.")]
    if main_proc:
        print(f"trainable: lora {sum(p.numel() for p in lora_params)/1e6:.2f}M, "
              f"new {sum(p.numel() for p in new_params)/1e6:.2f}M, train samples {len(dl.dataset)}", flush=True)
    groups = [{"params": lora_params, "lr": a.lr}, {"params": new_params, "lr": a.lr_new}]
    if graph_params:  # pre-trained graph encoder: fine-tune with a small lr
        groups.append({"params": graph_params, "lr": a.lr_graph})
        if main_proc:
            print(f"fine-tuning graph encoder: {sum(p.numel() for p in graph_params)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    two_stage = VARIANTS[a.variant]["head"]
    if not two_stage:  # text-only variants: a single SFT stage with the same total budget
        a.epochs2, a.epochs1 = a.epochs1 + a.epochs2, 0
    steps_per_epoch = len(dl) // a.accum
    s1 = int(a.epochs1 * steps_per_epoch)
    total = s1 + int(a.epochs2 * steps_per_epoch)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    step, t0, epoch = 0, time.time(), 0
    model.train()
    net = DDP(model, device_ids=[torch.cuda.current_device()]) if ddp else model
    log = open(f"{a.out}/train_log.jsonl", "a") if main_proc else None
    acc_hist = {}
    while step < total:
        if ddp:
            dl.sampler.set_epoch(epoch)
        epoch += 1
        for it, b in enumerate(dl):
            alpha = a.alpha1 if step < s1 else a.alpha2
            sync = (it + 1) % a.accum == 0
            ctx = net.no_sync() if (ddp and not sync) else contextlib.nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = net(b, alpha=alpha)
                (out["loss"] / a.accum).backward()
            for k, v in out.items():
                acc_hist.setdefault(k, []).append(v.item())
            if (it + 1) % a.accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step == s1 and two_stage and main_proc:
                torch.save(model.trainable_state_dict(), f"{a.out}/stage1.pt")
                print("== stage I done -> stage II (alpha", a.alpha2, ")", flush=True)
            if step % 25 == 0 and main_proc:
                m = {k: sum(v) / len(v) for k, v in acc_hist.items()}
                acc_hist = {}
                m.update(step=step, total=total, stage=1 if step <= s1 else 2, alpha=alpha,
                         lr=sched.get_last_lr()[0], min=(time.time() - t0) / 60)
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
                log.write(json.dumps(m) + "\n"); log.flush()
            if step >= total:
                break
    if main_proc:
        torch.save(model.trainable_state_dict(), f"{a.out}/final.pt")
        print("saved", f"{a.out}/final.pt")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
