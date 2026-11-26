"""End-to-end SAGE-Drive planner: scene dict -> graphs -> future graphs -> LLM plan.

The scene dict follows `sage.scene.load_scene` (ego-centric, 0.5 s sub-steps, t0 = index 9),
so any data source (AV2, nuPlan adapter, simulator) only needs to produce that dict.
"""
import numpy as np
import torch

from . import config as C
from .graph import future_graphs, history_graphs
from .llm import Collator, SageLLM
from .models import STGraphPredictor, collate_graph_seqs, to_dev
from .serialize import graph_prompt, parse_traj, raw_prompt


class SageDrivePlanner:
    def __init__(self, ckpt, variant="full", predictor_ckpt="outputs/predictor.pt", llm="Qwen/Qwen3-0.6B",
                 use_head=True, device="cuda"):
        self.model = SageLLM(llm, variant, predictor_ckpt=predictor_ckpt).to(device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
        ck = torch.load(predictor_ckpt, map_location="cpu")
        self.predictor = STGraphPredictor(**ck["cfg"]).to(device).eval()
        self.predictor.load_state_dict(ck["model"])
        self.collate = Collator(self.model.tok, variant)
        self.variant, self.use_head, self.device = variant, use_head, device

    @torch.no_grad()
    def predict_future(self, scene):
        hist = history_graphs(scene)
        out, _, _ = self.predictor(to_dev(collate_graph_seqs([hist]), self.device))
        return hist, STGraphPredictor.denorm(out)[0, : len(scene["agent_pos"])].cpu().numpy()

    @torch.no_grad()
    def plan(self, scene, nav="go straight"):
        """returns (waypoints [N,2] in the ego frame at t0, generated text)"""
        hist, pred = self.predict_future(scene)
        fut = future_graphs(scene, pred)
        rec = dict(sid=scene.get("sid", "x"), gt=np.zeros((C.N_PLAN, 2)).tolist(), nav=nav,
                   raw=raw_prompt(scene, nav), hist=graph_prompt(scene, hist, None, nav),
                   fut=graph_prompt(scene, hist, fut, nav))
        item = dict(rec=rec, gt=np.zeros((C.N_PLAN, 2), np.float32), hist=hist, fut=fut)
        b = self.collate([item])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            texts, head = self.model.plan(b)
        if self.use_head and head is not None:
            return head[0], texts[0]
        p = parse_traj(texts[0])
        if p is None:
            v0 = np.linalg.norm(scene["agent_vel"][0, C.T0])
            p = np.stack([v0 * C.DT * np.arange(1, C.N_PLAN + 1), np.zeros(C.N_PLAN)], 1)
        return p, texts[0]
