"""Datasets shared by the predictor and the LLM stages."""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C
from .graph import future_graphs, history_graphs


def load_scenes(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def future_targets(scene):
    """GT dynamic-node states for t0+1..t0+TF: [Na,TF,5] and validity [Na,TF]."""
    sl = slice(C.T0 + 1, C.T0 + 1 + C.TF)
    st = np.concatenate([scene["agent_pos"][:, sl], scene["agent_vel"][:, sl],
                         scene["agent_head"][:, sl, None]], -1).astype(np.float32)
    v = scene["agent_valid"][:, sl] & scene["agent_valid"][:, C.T0:C.T0 + 1]
    return st, v


class PredictorDataset(Dataset):
    def __init__(self, scenes):
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        s = self.scenes[i]
        st, v = future_targets(s)
        return dict(hist=history_graphs(s), tgt=st, tgt_valid=v)


def collate_predictor(items):
    from .models import collate_graph_seqs
    b = collate_graph_seqs([it["hist"] for it in items])
    B, N = b["ntype"].shape
    tgt = np.zeros((B, N, C.TF, 5), np.float32)
    tv = np.zeros((B, N, C.TF), bool)
    for k, it in enumerate(items):
        n = len(it["tgt"])
        tgt[k, :n], tv[k, :n] = it["tgt"], it["tgt_valid"]
    b["tgt"], b["tgt_valid"] = torch.from_numpy(tgt), torch.from_numpy(tv)
    return b


def predictor_loss(out, b):
    from .models import STGraphPredictor
    pr = STGraphPredictor.denorm(out)                     # [B,N,TF,5]
    tg, m = b["tgt"], b["tgt_valid"].float()
    lp = torch.nn.functional.smooth_l1_loss(pr[..., :2], tg[..., :2], reduction="none").sum(-1)
    lv = torch.nn.functional.smooth_l1_loss(pr[..., 2:4], tg[..., 2:4], reduction="none").sum(-1)
    dh = torch.atan2(torch.sin(pr[..., 4] - tg[..., 4]), torch.cos(pr[..., 4] - tg[..., 4]))
    # heading of slow / static agents is noisy -> weight heading by speed
    w_h = (tg[..., 2:4].norm(dim=-1) > 1.0).float()
    loss = ((lp + 0.5 * lv + w_h * dh.abs()) * m).sum() / m.sum().clamp(min=1)
    with torch.no_grad():
        err = (pr[..., :2] - tg[..., :2]).norm(dim=-1)
        ade = (err * m).sum() / m.sum().clamp(min=1)
        fm = m[..., -1]
        fde = (err[..., -1] * fm).sum() / fm.sum().clamp(min=1)
    return loss, dict(ade=ade.item(), fde=fde.item())


# ----------------------------------------------------------------------------- LLM stage
class SageDataset(Dataset):
    """Returns prompt / target text + history & predicted future graphs for one scenario."""

    def __init__(self, scenes, preds, records, with_graphs=True, with_fut=True):
        self.scenes = {s["sid"]: s for s in scenes}
        self.preds = preds
        self.records = records
        self.with_graphs, self.with_fut = with_graphs, with_fut

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        out = dict(rec=r, gt=np.array(r["gt"], np.float32))
        if self.with_graphs:
            s = self.scenes[r["sid"]]
            out["hist"] = history_graphs(s)
            if self.with_fut:
                out["fut"] = future_graphs(s, self.preds[r["sid"]])
        return out
