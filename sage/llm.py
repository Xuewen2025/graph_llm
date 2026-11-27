"""LLM planner with projection-based graph-language alignment (Sec. III-B, III-C).

Input sequence (full model):
    [z_hist (Q tokens)] [z_fut (Q tokens)] [prompt: instruction + serialised graphs]
    [Reasoning: ...] [PLAN] [Trajectory: [(x,y),...]] <eos>

* z_k = Proj(Pool(H_k)) from the (pre-trained) spatio-temporal graph encoder (Eq. 14-16).
* LoRA-adapted LLM M_{theta, dtheta} (Eq. 17).
* L = L_traj + alpha * L_text (Eq. 18-19). L_traj is a SmoothL1 regression on a trajectory head that
  reads the hidden state of a learned [PLAN] embedding placed *after* the reasoning and *before* the
  textual trajectory, so the head is conditioned on the CoT but cannot copy the teacher-forced
  trajectory text during training (no train/test mismatch).
"""
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config as C
from .models import GraphLanguageAligner, STGraphPredictor, collate_graph_seqs, to_dev
from .serialize import build_prompt, fmt_traj

VARIANTS = {
    # ablation ladder (Table IV)
    "raw":  dict(text="raw",  cot=False, graph_tokens=False, head=False),  # Baseline (raw LLM)
    "hist": dict(text="hist", cot=False, graph_tokens=False, head=False),  # + historical/current graph
    "fut":  dict(text="fut",  cot=False, graph_tokens=False, head=False),  # + future graph prediction
    "cot":  dict(text="fut",  cot=True,  graph_tokens=False, head=False),  # + CoT prompting
    "full": dict(text="fut",  cot=True,  graph_tokens=True,  head=True),   # + graph-language alignment
    # extra ablation: graph tokens only (isolates the graph-embedding input; no CoT, no regression head)
    "futg": dict(text="fut",  cot=False, graph_tokens=True,  head=False),
    # node-level graph tokens: the GNN embedding of each named node (EGO, A_k, L_k, X_k) is inserted right
    # after its name in the prompt; graph encoder fine-tuned end-to-end
    "node": dict(text="fut",  cot=False, graph_tokens=True,  head=False, node=True, train_graph=True),
    # P2: LLM outputs only a discrete decision pointing at prompt nodes (sage/decision.py); a graph decoder
    # turns it into a trajectory
    "dec":  dict(text="fut",  cot=False, graph_tokens=False, head=False, dec=True),
    # D: decision conditioned on a natural-language instruction (sage/instruct.py)
    "instr": dict(text="fut", cot=False, graph_tokens=False, head=False, dec=True, instr=True),
}
DEC_INSTR = (" Instead of waypoints, output only the driving decision: the lane (L_k) the ego will be on in 6 s, "
             "the agent (A_k) it must follow or yield to on its path, and its speed at 3 s and 6 s.")
NODE_TOKEN = "<|vision_pad|>"  # placeholder id in the prompt, replaced by the node embedding
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TRAJ_SCALE = 5.0  # per-step displacement scale (m / 0.5 s)


def make_texts(rec, reasoning, v):
    prompt = build_prompt(rec[v["text"]], v["cot"], fut=v["text"] == "fut")
    reason = f"Reasoning:\n{reasoning}\n" if v["cot"] else ""
    traj = f"Trajectory: {fmt_traj(np.array(rec['gt']))}"
    if v.get("dec"):
        prompt = prompt.replace("\n\n", DEC_INSTR + "\n\n", 1)
        traj = rec["dec_text"]
    if v.get("instr"):
        ins = random.choice(rec["instr_list"])
        prompt = prompt.replace("\n\n### Answer:\n", f"\nInstruction: {ins}\n\n### Answer:\n")
    return prompt, reason, traj


class Collator:
    """Tokenises in DataLoader workers and collates graph sequences."""

    def __init__(self, tok, variant, cot_map=None, max_prompt=2400, node_map=None):
        self.tok, self.v, self.cot = tok, VARIANTS[variant], cot_map or {}
        self.max_prompt = max_prompt
        self.node_map = node_map or {}
        self.node_id = tok.convert_tokens_to_ids(NODE_TOKEN)

    def tokenize_with_nodes(self, p, names):
        """insert a placeholder token after the first (definition) occurrence of each node name."""
        hits = []
        for name, idx in names.items():
            pat = rf"\b{name} crosswalk" if name[0] == "X" else rf"(?m)^{name}(?=[: ])"
            m = re.search(pat, p)
            if m:
                hits.append((m.start() + len(name), idx))
        hits.sort()
        ids, nodes, last = [], [], 0
        for pos, idx in hits:
            ids += self.tok(p[last:pos], add_special_tokens=False).input_ids + [self.node_id]
            nodes.append(idx); last = pos
        ids += self.tok(p[last:], add_special_tokens=False).input_ids
        return ids, nodes

    def __call__(self, items):
        b = dict(sid=[], prompt_ids=[], reason_ids=[], traj_ids=[], gt=[], node_idx=[])
        for it in items:
            r = it["rec"]
            p, rs, tj = make_texts(r, self.cot.get(r["sid"], ""), self.v)
            b["sid"].append(r["sid"])
            if self.v.get("node"):
                ids, nodes = self.tokenize_with_nodes(p, self.node_map[r["sid"]])
                ids = ids[-self.max_prompt:]
                b["node_idx"].append(nodes[len(nodes) - ids.count(self.node_id):])
            else:
                ids = self.tok(p, add_special_tokens=False).input_ids[-self.max_prompt:]
            b["prompt_ids"].append(ids)
            b["reason_ids"].append(self.tok(rs, add_special_tokens=False).input_ids if rs else [])
            b["traj_ids"].append(self.tok(tj, add_special_tokens=False).input_ids + [self.tok.eos_token_id])
            b["gt"].append(it["gt"])
        b["gt"] = torch.from_numpy(np.stack(b["gt"]))
        if self.v["graph_tokens"]:
            b["hist"] = collate_graph_seqs([it["hist"] for it in items])
            b["fut"] = collate_graph_seqs([it["fut"] for it in items])
        return b


class SageLLM(nn.Module):
    def __init__(self, llm_name="Qwen/Qwen3-0.6B", variant="full", predictor_ckpt="outputs/predictor.pt",
                 lora_r=16, q_tokens=8, train_graph=False):
        super().__init__()
        self.v = VARIANTS[variant]
        self.variant = variant
        self.tok = AutoTokenizer.from_pretrained(llm_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        self.llm = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                   target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        dl = base.config.hidden_size
        self.train_graph = train_graph or self.v.get("train_graph", False)
        train_graph = self.train_graph
        if self.v["graph_tokens"]:
            ck = torch.load(predictor_ckpt, map_location="cpu")
            self.predictor = STGraphPredictor(**ck["cfg"])
            self.predictor.load_state_dict(ck["model"])
            self.predictor.requires_grad_(train_graph)
            self.aligner = GraphLanguageAligner(ck["cfg"]["d"], dl, q_tokens)
        if self.v.get("node"):
            dg = ck["cfg"]["d"]  # node token = Proj([H_t0, GRU state, mean predicted-future embedding])
            self.node_proj = nn.Sequential(nn.LayerNorm(3 * dg), nn.Linear(3 * dg, dl), nn.GELU(), nn.Linear(dl, dl))
            self.node_id = self.tok.convert_tokens_to_ids(NODE_TOKEN)
        if self.v["head"]:
            self.plan_emb = nn.Parameter(torch.randn(dl) * 0.02)
            self.head = nn.Sequential(nn.Linear(dl, 512), nn.GELU(), nn.Linear(512, C.N_PLAN * 2))

    def decode_head(self, h):
        """head predicts per-step displacements; waypoints = cumulative sum (smoother than absolute xy)"""
        return torch.cumsum(self.head(h.float()).view(-1, C.N_PLAN, 2) * TRAJ_SCALE, dim=1)

    # ------------------------------------------------------------------ helpers
    @property
    def dev(self):
        return next(self.llm.parameters()).device

    def _backbone(self):
        return self.llm.base_model.model.model

    def _lm_head(self):
        return self.llm.base_model.model.lm_head

    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names}

    def graph_tokens(self, b):
        hist, fut = to_dev(b["hist"], self.dev), to_dev(b["fut"], self.dev)
        with torch.set_grad_enabled(self.train_graph and self.training):
            if not self.train_graph:
                self.predictor.eval()
            Hh = self.predictor.encoder(hist)                    # f_graph(G_t), t in T_hist
            ht = self.predictor.temporal(Hh, hist)
            Hf = self.predictor.encoder(fut)                     # f_graph(G^_t0+tau)
        z = self.aligner(Hh, hist["nvalid"], ht, Hf, fut["nvalid"]).to(torch.bfloat16)   # [B, 2Q, dl]
        nodes = None
        if self.v.get("node"):
            N = min(Hh.shape[2], Hf.shape[2])
            vf = fut["nvalid"][:, :, :N, None].float()
            hf = (Hf[:, :, :N] * vf).sum(1) / vf.sum(1).clamp(min=1)
            nodes = self.node_proj(torch.cat([Hh[:, -1, :N], ht[:, :N], hf], -1).float()).to(torch.bfloat16)
        return z, nodes                                           # nodes: [B, N, dl] or None

    def assemble(self, b, parts, z=None, left_pad=False, nodes=None):
        """parts: which text segments to include: subset of ('reason','plan','traj').

        Returns inputs_embeds [B,L,d], attention_mask, labels, plan position."""
        emb_layer = self.llm.get_input_embeddings()
        B = len(b["prompt_ids"])
        seqs, labs, plan_pos = [], [], []
        for i in range(B):
            chunks, lab = [], []
            if z is not None:
                chunks.append(z[i]); lab += [-100] * z.shape[1]
            ids = torch.tensor(b["prompt_ids"][i], device=self.dev)
            e = emb_layer(ids)
            if nodes is not None and b["node_idx"][i]:
                pos = (ids == self.node_id).nonzero()[:, 0]
                idx = torch.tensor(b["node_idx"][i], device=self.dev)
                e = e.index_put((pos,), nodes[i, idx].to(e.dtype))
            chunks.append(e); lab += [-100] * len(ids)
            if "reason" in parts and len(b["reason_ids"][i]):
                ids = torch.tensor(b["reason_ids"][i], device=self.dev)
                chunks.append(emb_layer(ids)); lab += ids.tolist()
            if "plan" in parts:
                plan_pos.append(len(lab))
                chunks.append(self.plan_emb.to(torch.bfloat16)[None]); lab += [-100]
            if "traj" in parts:
                ids = torch.tensor(b["traj_ids"][i], device=self.dev)
                chunks.append(emb_layer(ids)); lab += ids.tolist()
                if "plan" in parts:  # [PLAN] state serves the regression head only: no next-token loss on it
                    lab[plan_pos[-1] + 1] = -100  # (the fixed "Trajectory:" prefix is forced at inference)
            seqs.append(torch.cat(chunks, 0)); labs.append(lab)
        L = max(len(s) for s in seqs)
        d = seqs[0].shape[1]
        emb = torch.zeros(B, L, d, device=self.dev, dtype=torch.bfloat16)
        att = torch.zeros(B, L, device=self.dev, dtype=torch.long)
        lab = torch.full((B, L), -100, device=self.dev, dtype=torch.long)
        for i, s in enumerate(seqs):
            n = len(s)
            sl = slice(L - n, L) if left_pad else slice(0, n)
            emb[i, sl], att[i, sl] = s, 1
            lab[i, sl] = torch.tensor(labs[i], device=self.dev)
            if left_pad and plan_pos:
                plan_pos[i] += L - n
        return emb, att, lab, (torch.tensor(plan_pos, device=self.dev) if plan_pos else None)

    # ------------------------------------------------------------------ training
    def forward(self, b, alpha=1.0):
        z, nodes = self.graph_tokens(b) if self.v["graph_tokens"] else (None, None)
        parts = ("reason", "plan", "traj") if self.v["head"] else ("reason", "traj")
        emb, att, lab, pp = self.assemble(b, parts, z, nodes=nodes)
        hid = self._backbone()(inputs_embeds=emb, attention_mask=att).last_hidden_state
        sel = lab[:, 1:] != -100
        logits = self._lm_head()(hid[:, :-1][sel]).float()      # only on target positions (memory)
        l_text = F.cross_entropy(logits, lab[:, 1:][sel])
        out = dict(l_text=l_text)
        if self.v["head"]:
            traj = self.decode_head(hid[torch.arange(len(pp)), pp])
            l_traj = F.smooth_l1_loss(traj, b["gt"].to(self.dev), reduction="none").sum(-1).mean()
            out["l_traj"] = l_traj
            out["loss"] = l_traj + alpha * l_text
        else:
            out["loss"] = l_text
        return out

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def plan(self, b, max_new_tokens=None):
        """Returns generated texts and (if head) regressed trajectories [B,N,2]."""
        if max_new_tokens is None:
            max_new_tokens = 420 if self.v["cot"] else (40 if self.v.get("dec") else 160)
        z, nodes = self.graph_tokens(b) if self.v["graph_tokens"] else (None, None)
        emb, att, _, _ = self.assemble(b, (), z, left_pad=True, nodes=nodes)
        gen = self.llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=self.tok.pad_token_id,
                                eos_token_id=self.tok.eos_token_id)
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        head_traj = None
        if self.v["head"]:
            # Training sequence is  reasoning -> [PLAN] -> "Trajectory: ..."; the [PLAN] embedding is not a
            # token, so the LLM cannot emit it. Cut the generated text after the reasoning, insert [PLAN],
            # then (a) read the head and (b) generate the textual trajectory from exactly the training context.
            reasons = [self._cut_reasoning(t) if self.v["cot"] else "" for t in texts]
            # The hidden state at [PLAN] is shared with the regression head and predicts the first text token
            # poorly, so the fixed format prefix "Trajectory:" is forced and only the numbers are generated.
            prefix = "Trajectory:"
            pre_ids = self.tok(prefix, add_special_tokens=False).input_ids
            b2 = dict(b, reason_ids=[self.tok(r, add_special_tokens=False).input_ids if r else [] for r in reasons],
                      traj_ids=[pre_ids] * len(reasons))
            emb, att, _, pp = self.assemble(b2, ("reason", "plan", "traj"), z, left_pad=True, nodes=nodes)
            hid = self._backbone()(inputs_embeds=emb, attention_mask=att).last_hidden_state
            head_traj = self.decode_head(hid[torch.arange(len(pp)), pp]).float().cpu().numpy()
            gen = self.llm.generate(inputs_embeds=emb, attention_mask=att, max_new_tokens=160, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
            trajs = self.tok.batch_decode(gen, skip_special_tokens=True)
            texts = [r + prefix + t for r, t in zip(reasons, trajs)]
        return texts, head_traj

    @staticmethod
    def _cut_reasoning(t):
        """generated text -> 'Reasoning:\\n...4. Intent decision: ...\\n' (the training reason segment)"""
        k = t.find("Intent decision:")
        if k >= 0:
            e = t.find("\n", k)
            body = t[:e] if e >= 0 else t
        else:
            body = t
            for m in ("Trajectory:", "Answer:"):
                body = body.split(m)[0]
        body = body.rstrip()
        if not body.startswith("Reasoning:"):
            body = "Reasoning:\n" + body.lstrip()
        return body + "\n"
