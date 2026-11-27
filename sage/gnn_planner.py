"""P1: pure-GNN ego planner (no LLM).

STGraphPredictor (GATv2 + GRU + AR future prediction for all agents) + an ego planning head:
  query  = h~_ego + nav embedding + ego kinematic state
  keys   = h~_n + Enc(predicted future states of node n)      (all nodes; acts like the "future graph")
  2 cross-attention blocks -> MLP -> 12 x (dx, dy) displacements -> cumsum -> 6 s plan (ego frame).
Trained with L = w_ego * L_ego(12 pts) + w_aux * L_predictor (all agents, 3 s).
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .data import PredictorDataset, collate_predictor, predictor_loss
from .models import STGraphPredictor
from .scene import ego_future

NAV = ["go straight", "turn left", "turn right", "stop / stay"]


class PlannerDataset(PredictorDataset):
    def __init__(self, scenes, records_path):
        super().__init__(scenes)
        nav = {(r := json.loads(l))["sid"]: r["nav"] for l in open(records_path)}
        self.nav = [NAV.index(nav[s["sid"]]) for s in scenes]

    def __getitem__(self, i):
        it = super().__getitem__(i)
        it["ego_gt"] = ego_future(self.scenes[i]).astype(np.float32)
        it["nav"] = self.nav[i]
        it["sid"] = self.scenes[i]["sid"]
        return it


def collate_planner(items):
    b = collate_predictor(items)
    b["ego_gt"] = torch.from_numpy(np.stack([it["ego_gt"] for it in items]))
    b["nav"] = torch.tensor([it["nav"] for it in items])
    b["sid"] = [it["sid"] for it in items]
    return b


class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, q, kv, mask):
        q = self.n1(q + self.attn(q, kv, kv, key_padding_mask=~mask)[0])
        return self.n2(q + self.ffn(q))


class EgoGNNPlanner(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, drop_etypes=(), use_nav=True, use_fut=True, n_plan=C.N_PLAN):
        super().__init__()
        self.use_nav, self.use_fut, self.n_plan = use_nav, use_fut, n_plan
        self.pred = STGraphPredictor(d, layers, heads, drop_etypes=drop_etypes)
        self.nav_emb = nn.Embedding(len(NAV), d)
        self.ego_in = nn.Linear(C.NODE_DIM, d)
        self.fut_enc = nn.Sequential(nn.Linear(C.TF * 6, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                  nn.Linear(256, n_plan * 2))

    def forward(self, b):
        out, H, ht = self.pred(b)                                    # out [B,N,TF,6], ht [B,N,d]
        B, N, _ = ht.shape
        kv = ht
        if self.use_fut:
            dyn = (b["ntype"] == C.NT_AGENT)[..., None].to(ht.dtype)
            kv = kv + dyn * self.fut_enc(out.reshape(B, N, -1))
        mask = b["nvalid"][:, -1] & (b["ntype"] >= 0)
        q = ht[:, :1] + self.ego_in(b["x"][:, -1, :1])
        if self.use_nav:
            q = q + self.nav_emb(b["nav"])[:, None]
        for blk in self.blocks:
            q = blk(q, kv, mask)
        z = torch.cat([q[:, 0], ht[:, 0]], -1)
        disp = self.head(z).view(B, self.n_plan, 2) * (C.DT * C.VEL_SCALE)   # metres per step
        # residual over constant velocity (current ego velocity, ego frame)
        v0 = b["x"][:, -1, 0, 2:4] * C.VEL_SCALE
        disp = disp + (v0 * C.DT)[:, None]
        return disp.cumsum(1), out


def planner_loss(plan, out, b, w_ego=1.0, w_aux=1.0):
    le = F.smooth_l1_loss(plan, b["ego_gt"], reduction="none").sum(-1).mean()
    la, m = predictor_loss(out, b) if w_aux > 0 else (torch.zeros((), device=plan.device), {})
    with torch.no_grad():
        m["ego_ade6"] = (plan - b["ego_gt"]).norm(dim=-1).mean().item()
    return w_ego * le + w_aux * la, m


# ============================================================================ P2: decision-conditioned decoder
class DecisionDataset(PlannerDataset):
    """+ decision labels (sage/decision.py) and the candidate nodes named in the prompt."""

    def __init__(self, scenes, records_path, dec_path, nodemap_path):
        super().__init__(scenes, records_path)
        dec = {(r := json.loads(l))["sid"]: r["dec"] for l in open(dec_path)}
        nm = {(r := json.loads(l))["sid"]: r["nodes"] for l in open(nodemap_path)}
        self.dec = [dec[s["sid"]] for s in scenes]
        self.names = [nm[s["sid"]] for s in scenes]
        self.override = None  # sid -> decision dict (e.g. LLM decisions) used instead of the labels
        self.instr = None     # sid -> {set: [K, De]} instruction embeddings (D); random variant if train
        self.instr_set, self.train = "seen", False

    def __getitem__(self, i):
        from .decision import to_index
        it = super().__getitem__(i)
        names = self.names[i]
        it["dec"] = to_index(self.dec[i], names)
        if self.override is not None:
            d = self.override.get(it["sid"])
            it["dec_ovr"] = to_index(d, names) if d is not None else (-2, -2, -2, -2)  # -2 = missing
        if self.instr is not None:
            e = self.instr[it["sid"]][self.instr_set]
            it["instr"] = e[np.random.randint(len(e))] if self.train else e[0]
        it["lane_cand"] = [v for k, v in names.items() if k[0] == "L"]
        it["agent_cand"] = [v for k, v in names.items() if k[0] == "A"]
        return it


def collate_decision(items):
    b = collate_planner(items)
    B, N = b["ntype"].shape
    b["dec"] = torch.tensor([it["dec"] for it in items])                          # [B,4]
    if "dec_ovr" in items[0]:
        b["dec_ovr"] = torch.tensor([it["dec_ovr"] for it in items])
    lc, ac = torch.zeros(B, N, dtype=torch.bool), torch.zeros(B, N, dtype=torch.bool)
    for k, it in enumerate(items):
        lc[k, it["lane_cand"]] = True
        ac[k, it["agent_cand"]] = True
    b["lane_cand"], b["agent_cand"] = lc, ac
    if "instr" in items[0]:
        b["instr"] = torch.from_numpy(np.stack([it["instr"] for it in items]).astype(np.float32))
    return b


class DecisionGNNPlanner(EgoGNNPlanner):
    """EgoGNNPlanner + (a) decision heads (pointer over named lanes / agents + 'none', speed bins) and
    (b) a decoder conditioned on a decision (GT labels in training; GT / own / LLM decisions at test)."""

    def __init__(self, nv=11, instr_dim=0, **kw):
        super().__init__(**kw)
        d = self.nav_emb.embedding_dim
        self.instr_proj = (nn.Sequential(nn.LayerNorm(instr_dim), nn.Linear(instr_dim, d), nn.GELU(), nn.Linear(d, d))
                           if instr_dim else None)
        self.ptr_q = nn.ModuleDict({k: nn.Linear(d, d) for k in ("lane", "lead")})
        self.ptr_k = nn.ModuleDict({k: nn.Linear(d, d) for k in ("lane", "lead")})
        self.none_logit = nn.ModuleDict({k: nn.Linear(d, 1) for k in ("lane", "lead")})
        self.v_cls = nn.ModuleDict({k: nn.Linear(d, nv) for k in ("v3", "v6")})
        self.c_node = nn.ModuleDict({k: nn.Linear(d, d) for k in ("lane", "lead")})
        self.c_none = nn.ParameterDict({k: nn.Parameter(torch.zeros(d)) for k in ("lane", "lead")})
        self.c_v = nn.ModuleDict({k: nn.Embedding(nv, d) for k in ("v3", "v6")})
        self.dec_blocks = nn.ModuleList([Block(d) for _ in range(2)])

    def encode(self, b):
        out, H, ht = self.pred(b)
        B, N, _ = ht.shape
        kv = ht
        if self.use_fut:
            dyn = (b["ntype"] == C.NT_AGENT)[..., None].to(ht.dtype)
            kv = kv + dyn * self.fut_enc(out.reshape(B, N, -1))
        mask = b["nvalid"][:, -1] & (b["ntype"] >= 0)
        q = ht[:, :1] + self.ego_in(b["x"][:, -1, :1])
        if self.use_nav:
            q = q + self.nav_emb(b["nav"])[:, None]
        if self.instr_proj is not None:
            q = q + self.instr_proj(b["instr"])[:, None]
        for blk in self.blocks:
            q = blk(q, kv, mask)
        return out, ht, kv, mask, q

    def decision_logits(self, q, kv, b):
        """-> dict: lane/lead logits [B, N+1] (last = none), v3/v6 logits [B, nv]"""
        lg = {}
        for k, cand in (("lane", b["lane_cand"]), ("lead", b["agent_cand"])):
            s = (self.ptr_k[k](kv) * self.ptr_q[k](q)).sum(-1) / kv.shape[-1] ** 0.5       # [B,N]
            s = s.masked_fill(~cand, -1e4)
            lg[k] = torch.cat([s, self.none_logit[k](q[:, 0])], -1)
        for k in ("v3", "v6"):
            lg[k] = self.v_cls[k](q[:, 0])
        return lg

    @staticmethod
    def argmax_decision(lg):
        out = []
        for k in ("lane", "lead"):
            a = lg[k].argmax(-1)
            out.append(torch.where(a == lg[k].shape[-1] - 1, torch.full_like(a, -1), a))
        out += [lg["v3"].argmax(-1), lg["v6"].argmax(-1)]
        return torch.stack(out, -1)                                                    # [B,4]

    def decode(self, q, kv, mask, ht, b, dec):
        B = q.shape[0]
        ar = torch.arange(B, device=q.device)
        c = 0
        for j, k in enumerate(("lane", "lead")):
            idx = dec[:, j]
            node = self.c_node[k](kv[ar, idx.clamp(min=0)])
            c = c + torch.where((idx >= 0)[:, None], node, self.c_none[k][None])
        c = c + self.c_v["v3"](dec[:, 2]) + self.c_v["v6"](dec[:, 3])
        q = q + c[:, None]
        for blk in self.dec_blocks:
            q = blk(q, kv, mask)
        z = torch.cat([q[:, 0], ht[:, 0]], -1)
        disp = self.head(z).view(B, self.n_plan, 2) * (C.DT * C.VEL_SCALE)
        v0 = b["x"][:, -1, 0, 2:4] * C.VEL_SCALE
        return (disp + (v0 * C.DT)[:, None]).cumsum(1)

    def forward(self, b, source="gt"):
        """source: 'gt' (labels), 'own' (argmax of own decision heads), 'ovr' (b['dec_ovr'], own as fallback)"""
        out, ht, kv, mask, q = self.encode(b)
        lg = self.decision_logits(q, kv, b)
        own = self.argmax_decision(lg)
        if source == "gt":
            dec = b["dec"]
        elif source == "own":
            dec = own
        else:
            dec = torch.where(b["dec_ovr"] == -2, own, b["dec_ovr"])
        return self.decode(q, kv, mask, ht, b, dec), out, lg, own


def decision_loss(lg, dec):
    ls = 0
    acc = {}
    for j, k in enumerate(("lane", "lead", "v3", "v6")):
        t = dec[:, j]
        if k in ("lane", "lead"):
            t = torch.where(t < 0, torch.full_like(t, lg[k].shape[-1] - 1), t)
        ls = ls + F.cross_entropy(lg[k].float(), t)
        acc[f"acc_{k}"] = (lg[k].argmax(-1) == t).float().mean().item()
    return ls, acc
