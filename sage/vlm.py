"""Experiment 1: does a VLM plan better from a scene GRAPH than from a scene IMAGE carrying the same information?

Backbone: Qwen2.5-VL-3B-Instruct, LoRA on the language model (vision tower frozen, merger trainable).
Variants (all share the instruction, the ego state line and the navigation command):
    raw     : unstructured object / lane lists as text (no relations)              -- text reference
    hist    : serialised history/current scene graph (nodes, relations, history)   -- graph as text
    img     : BEV raster (sage/bev.py) of the same scene                           -- image
    imghist : BEV raster + graph text
"""
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .bev import IMG_DESC
from .serialize import INSTR, fmt_traj

VL_VARIANTS = {
    "raw": dict(text="raw", img=False),
    "hist": dict(text="hist", img=False),
    "img": dict(text=None, img=True),
    "imghist": dict(text="hist", img=True),
    "ego": dict(text=None, img=False),   # control: ego state + navigation only, no scene
}
LORA_RE = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
N_IMG_TOK = 256  # 448x448 -> 32x32 patches -> 2x2 merge


def ego_nav(rec):
    lines = rec["raw"].split("\n")
    ego = next(l for l in lines if l.startswith("EGO:"))
    nav = next(l for l in lines if l.startswith("Navigation:"))
    return f"Scene (ego frame: x forward, y left; meters, m/s).\n{ego}\n{nav}"


def make_prompt(rec, v):
    parts = [INSTR]
    if v["img"]:
        parts.append(IMG_DESC + "\n<|vision_start|>" + "<|image_pad|>" * N_IMG_TOK + "<|vision_end|>")
    parts.append(rec[v["text"]] if v["text"] else ego_nav(rec))
    user = "\n\n".join(parts)
    return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


class VLCollator:
    def __init__(self, proc, variant, img_dir):
        self.proc, self.tok, self.v, self.img_dir = proc, proc.tokenizer, VL_VARIANTS[variant], img_dir
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, recs):
        b = dict(sid=[], prompt_ids=[], target_ids=[], gt=[])
        imgs = []
        for r in recs:
            b["sid"].append(r["sid"])
            b["prompt_ids"].append(self.tok(make_prompt(r, self.v), add_special_tokens=False).input_ids)
            b["target_ids"].append(self.tok(f"Trajectory: {fmt_traj(np.array(r['gt']))}",
                                            add_special_tokens=False).input_ids + [self.eos])
            b["gt"].append(np.array(r["gt"], np.float32))
            if self.v["img"]:
                imgs.append(Image.open(f"{self.img_dir}/{r['sid']}.png").convert("RGB"))
        b["gt"] = torch.from_numpy(np.stack(b["gt"]))
        if imgs:
            o = self.proc.image_processor(images=imgs, return_tensors="pt")
            b["pixel_values"], b["image_grid_thw"] = o["pixel_values"], o["image_grid_thw"]
            assert int(o["image_grid_thw"][0].prod()) // 4 == N_IMG_TOK
        return b


class VLMPlanner(nn.Module):
    def __init__(self, name="Qwen/Qwen2.5-VL-3B-Instruct", variant="img", lora_r=16, grad_ckpt=False):
        super().__init__()
        self.v, self.variant = VL_VARIANTS[variant], variant
        self.proc = AutoProcessor.from_pretrained(name)
        self.tok = self.proc.tokenizer
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16,
                                                                   attn_implementation="sdpa")
        if grad_ckpt:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        self.llm = get_peft_model(base, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                   target_modules=LORA_RE, task_type="CAUSAL_LM"))
        if self.v["img"]:  # the vision-language projector is adapted, like the graph projector of the LLM variants
            self.merger.float().requires_grad_(True)

    @property
    def merger(self):
        return self.core.visual.merger

    @property
    def core(self):
        return self.llm.base_model.model.model

    @property
    def dev(self):
        return next(self.llm.parameters()).device

    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names}

    def _pad(self, seqs, left, val):
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), val, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            sl = slice(L - len(s), L) if left else slice(0, len(s))
            out[i, sl] = torch.tensor(s)
            att[i, sl] = 1
        return out.to(self.dev), att.to(self.dev)

    def _vis(self, b):
        if not self.v["img"]:
            return {}
        return dict(pixel_values=b["pixel_values"].to(self.dev, torch.bfloat16),
                    image_grid_thw=b["image_grid_thw"].to(self.dev))

    def forward(self, b):
        seqs = [p + t for p, t in zip(b["prompt_ids"], b["target_ids"])]
        ids, att = self._pad(seqs, False, self.tok.pad_token_id)
        lab = torch.full_like(ids, -100)
        for i, (p, t) in enumerate(zip(b["prompt_ids"], b["target_ids"])):
            lab[i, len(p):len(p) + len(t)] = torch.tensor(t)
        hid = self.core(input_ids=ids, attention_mask=att, **self._vis(b)).last_hidden_state
        sel = lab[:, 1:] != -100
        logits = self.llm.base_model.model.lm_head(hid[:, :-1][sel]).float()
        return dict(loss=F.cross_entropy(logits, lab[:, 1:][sel]))

    @torch.no_grad()
    def plan(self, b, max_new_tokens=160):
        ids, att = self._pad(b["prompt_ids"], True, self.tok.pad_token_id)
        gen = self.llm.generate(input_ids=ids, attention_mask=att, **self._vis(b), max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=self.tok.pad_token_id,
                                eos_token_id=self.tok.convert_tokens_to_ids("<|im_end|>"))
        return self.tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)
