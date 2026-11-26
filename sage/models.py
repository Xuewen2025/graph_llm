"""Spatio-temporal graph predictor (Sec. III-A-2) and graph-language aligner (Sec. III-C-2)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


# ============================================================================ batching
def collate_graph_seqs(seqs):
    """seqs: list (batch) of list (time) of graph dicts sharing the node layout within a sample.

    Returns tensors: x [B,T,N,D], ntype [B,N], nvalid [B,T,N], ei [2,E] (flattened node ids
    (b*T+t)*N+n), ea [E,De], and n_agents [B].
    """
    B, T = len(seqs), len(seqs[0])
    N = max(len(s[0]["x"]) for s in seqs)
    x = np.zeros((B, T, N, C.NODE_DIM), np.float32)
    ntype = np.full((B, N), -1, np.int64)
    nvalid = np.zeros((B, T, N), bool)
    eis, eas = [], []
    for b, s in enumerate(seqs):
        n = len(s[0]["x"])
        ntype[b, :n] = s[0]["ntype"]
        for t, g in enumerate(s):
            x[b, t, :n] = g["x"]
            nvalid[b, t, :n] = g["nvalid"]
            eis.append(g["ei"] + (b * T + t) * N)
            eas.append(g["ea"])
    return dict(x=torch.from_numpy(x), ntype=torch.from_numpy(ntype), nvalid=torch.from_numpy(nvalid),
                ei=torch.from_numpy(np.concatenate(eis, 1)), ea=torch.from_numpy(np.concatenate(eas, 0)),
                n_agents=torch.tensor([len(s[0]["pos"]) for s in seqs]))


def to_dev(d, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in d.items()}


# ============================================================================ GATv2
class GATv2Layer(nn.Module):
    """GATv2 with relational edge features (Eq. 6) and attention aggregation (Eq. 7).

    alpha_ij = softmax_j( a^T LeakyReLU(W1 h_i + W2 h_j + Wr r_ij) ),  h_i' = sigma(sum_j alpha_ij W h_j)
    Multi-head, plus residual + LayerNorm + FFN for stable stacking.
    """

    def __init__(self, d, heads=4, edge_dim=C.EDGE_DIM, dropout=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.W1 = nn.Linear(d, d)
        self.W2 = nn.Linear(d, d, bias=False)
        self.Wr = nn.Linear(edge_dim, d, bias=False)
        self.W = nn.Linear(d, d)
        self.a = nn.Parameter(torch.randn(heads, self.dh) / math.sqrt(self.dh))
        self.out = nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, ei, ea):
        Nn = h.shape[0]
        j, i = ei[0], ei[1]
        e = F.leaky_relu(self.W1(h)[i] + self.W2(h)[j] + self.Wr(ea), 0.2).view(-1, self.h, self.dh)
        s = (e * self.a).sum(-1)                                     # [E,H]
        smax = torch.full((Nn, self.h), -1e9, device=h.device, dtype=s.dtype)
        smax = smax.scatter_reduce(0, i[:, None].expand_as(s), s, "amax", include_self=True)
        w = torch.exp(s - smax[i])
        den = torch.zeros((Nn, self.h), device=h.device, dtype=s.dtype).index_add_(0, i, w)
        alpha = w / (den[i] + 1e-9)
        msg = alpha[..., None] * self.W(h)[j].view(-1, self.h, self.dh)
        agg = torch.zeros((Nn, self.h, self.dh), device=h.device, dtype=h.dtype).index_add_(0, i, msg.to(h.dtype))
        h = self.n1(h + self.drop(self.out(F.elu(agg.view(Nn, -1)))))
        return self.n2(h + self.drop(self.ffn(h)))


class SpatialGraphEncoder(nn.Module):
    """f_graph: type-specific linear projection f_enc (Eq. 5) + stacked GATv2 (Eq. 6-7)."""

    def __init__(self, d=128, layers=3, heads=4, drop_etypes=()):
        super().__init__()
        self.d = d
        # ablation: remove edge types (e.g. "agent-agent") from message passing, same parameters otherwise
        self.drop_idx = [8 + len(C.RELATIONS) + C.EDGE_TYPES.index(t) for t in drop_etypes]
        self.enc = nn.ModuleList([nn.Linear(C.NODE_DIM, d) for _ in C.NODE_TYPES])
        self.type_emb = nn.Embedding(len(C.NODE_TYPES), d)
        self.layers = nn.ModuleList([GATv2Layer(d, heads) for _ in range(layers)])

    def forward(self, b):
        """b: collated batch -> H [B,T,N,d]"""
        x, ntype = b["x"], b["ntype"]
        B, T, N, _ = x.shape
        nt = ntype[:, None, :].expand(B, T, N).reshape(-1)
        xf = x.reshape(-1, x.shape[-1])
        h = torch.zeros(xf.shape[0], self.d, device=x.device, dtype=x.dtype)
        for k, enc in enumerate(self.enc):
            m = nt == k
            if m.any():
                h[m] = enc(xf[m]) + self.type_emb.weight[k]
        ei, ea = b["ei"], b["ea"]
        if self.drop_idx:
            keep = (ea[:, self.drop_idx] < 0.5).all(1)
            ei, ea = ei[:, keep], ea[keep]
        for layer in self.layers:
            h = layer(h, ei, ea)
        h = h * b["nvalid"].reshape(-1, 1)
        return h.view(B, T, N, self.d)


# ============================================================================ predictor
class STGraphPredictor(nn.Module):
    """Spatial GATv2 encoding -> node-wise GRU (Eq. 8) -> autoregressive f_pred (Eq. 9-10)."""

    def __init__(self, d=128, layers=3, heads=4, tf=C.TF, drop_etypes=()):
        super().__init__()
        self.tf = tf
        self.encoder = SpatialGraphEncoder(d, layers, heads, drop_etypes)
        self.gru = nn.GRUCell(d, d)
        self.pred = nn.Sequential(nn.Linear(6 + d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                  nn.Linear(256, 6))

    def temporal(self, H, b):
        """H [B,T,N,d] -> h_tilde [B,N,d]; GRU for dynamic agents, last embedding for static nodes."""
        B, T, N, d = H.shape
        hs = torch.zeros(B * N, d, device=H.device, dtype=H.dtype)
        valid = b["nvalid"].reshape(B, T, N)
        for t in range(T):
            nh = self.gru(H[:, t].reshape(B * N, d), hs)
            v = valid[:, t].reshape(-1, 1).to(H.dtype)
            hs = v * nh + (1 - v) * hs
        hs = hs.view(B, N, d)
        dyn = (b["ntype"] == C.NT_AGENT)[..., None]
        return torch.where(dyn, hs, H[:, -1])

    @staticmethod
    def kin_state(x):
        """normalised kinematic slice of agent node features: px,py,vx,vy,cos,sin"""
        return x[..., :6]

    def forward(self, b):
        H = self.encoder(b)
        ht = self.temporal(H, b)                      # [B,N,d]
        s = self.kin_state(b["x"][:, -1])             # x^_{t0} = x_{t0}   (Eq. 9)
        outs = []
        for _ in range(self.tf):                      # Eq. 10
            o = self.pred(torch.cat([s, ht], -1))
            pos = s[..., :2] + o[..., :2] * (C.DT * C.VEL_SCALE / C.POS_SCALE)
            vel = o[..., 2:4]
            cs = F.normalize(o[..., 4:6] + s[..., 4:6], dim=-1)
            s = torch.cat([pos, vel, cs], -1)
            outs.append(s)
        out = torch.stack(outs, 2)                    # [B,N,TF,6] normalised
        return out, H, ht

    @staticmethod
    def denorm(out):
        """-> [..., 5] (px, py, vx, vy, heading) in meters / m/s / rad"""
        return torch.cat([out[..., :2] * C.POS_SCALE, out[..., 2:4] * C.VEL_SCALE,
                          torch.atan2(out[..., 5], out[..., 4])[..., None]], -1)


# ============================================================================ aligner
class AttnPool(nn.Module):
    """Pool(.) : attention-based pooling with Q learnable queries over a node-token set."""

    def __init__(self, d, q=8, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(q, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens, mask):
        """tokens [B,L,d], mask [B,L] True=valid -> [B,Q,d]"""
        q = self.q[None].expand(tokens.shape[0], -1, -1)
        o, _ = self.attn(q, tokens, tokens, key_padding_mask=~mask)
        return self.norm(o + q)


class GraphLanguageAligner(nn.Module):
    """z_k = Proj(Pool(H_k)), k in {hist, fut}   (Eq. 14-15)."""

    def __init__(self, dg, dl, q=8, max_t=16):
        super().__init__()
        self.time_emb = nn.Embedding(max_t, dg)
        self.pool = nn.ModuleDict({k: AttnPool(dg, q) for k in ("hist", "fut")})
        self.proj = nn.Sequential(nn.Linear(dg, dl), nn.GELU(), nn.Linear(dl, dl))

    def _tokens(self, H, valid, t_off):
        B, T, N, d = H.shape
        te = self.time_emb(torch.arange(T, device=H.device) + t_off)[None, :, None]
        return (H + te).reshape(B, T * N, d), valid.reshape(B, T * N)

    def forward(self, H_hist, v_hist, ht, H_fut=None, v_fut=None):
        th, mh = self._tokens(H_hist, v_hist, 0)
        th = torch.cat([th, ht], 1)                                        # + temporal context h~
        mh = torch.cat([mh, v_hist[:, -1]], 1)
        z = [self.proj(self.pool["hist"](th, mh))]
        if H_fut is not None:
            tf, mf = self._tokens(H_fut, v_fut, H_hist.shape[1])
            z.append(self.proj(self.pool["fut"](tf, mf)))
        return torch.cat(z, 1)                                              # [B, Q or 2Q, dl]
