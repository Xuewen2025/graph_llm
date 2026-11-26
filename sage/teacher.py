"""Reasoning-supervised label generation (Sec. III-C-1, Eq. 13).

(y_reason, y_text) = T(G_hist, G^_fut, s_ego)

Teacher = any OpenAI-compatible chat model on OpenRouter (paper: GPT-4o). The teacher also sees
the executed expert trajectory, so it *explains* the demonstrated behaviour instead of inventing
a possibly inconsistent plan (rationalisation, as in GPT-Driver / DriveLM label pipelines).
A deterministic rule-based teacher is the fallback (no API / invalid response).
"""
import os
import re
import time

import numpy as np
import requests

from . import config as C
from .serialize import fmt_traj

SYSTEM = ("You are an expert autonomous-driving planner who writes concise chain-of-thought rationales "
          "used as training labels for a smaller planning model.")

USER_TMPL = """Below is a serialized spatio-temporal traffic scene graph (history, current, and predicted future graphs).

{scene}

The ego vehicle actually executed this {T:.0f}-second plan ({N} waypoints, {dt} s apart, ego frame x forward / y left):
{traj}
Summary of the executed plan: {summary}

Write the reasoning that leads to this plan, as exactly 4 numbered lines:
1. Trend analysis: ego motion trend and the key agents' trends from the history graphs.
2. Interaction assessment: the most relevant agents/lanes/crosswalks and their relation to EGO (use node ids like A1, L2, X1).
3. Conflict anticipation: potential conflicts in the predicted future graphs (or state that there are none).
4. Intent decision: the resulting high-level decision (lane keeping/lane change/turn, accelerate/keep speed/decelerate/stop, target speed).
Rules: at most 110 words in total; reason from the scene only; never mention that the plan or future trajectory was given to you; do not output the waypoints."""


def plan_summary(gt, v0):
    v = np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), gt]), axis=0), axis=1) / C.DT
    lat = gt[-1, 1]
    h = np.degrees(np.arctan2(gt[-1, 1] - gt[-3, 1], gt[-1, 0] - gt[-3, 0])) if np.linalg.norm(gt[-1] - gt[-3]) > 0.5 else 0.0
    if v.max() < 0.5:
        lon = "remains stopped"
    elif v[-1] < 0.5:
        lon = f"decelerates from {v0:.1f} m/s to a stop"
    elif v[-1] > v0 + 1.0:
        lon = f"accelerates from {v0:.1f} to {v[-1]:.1f} m/s"
    elif v[-1] < v0 - 1.0:
        lon = f"decelerates from {v0:.1f} to {v[-1]:.1f} m/s"
    else:
        lon = f"keeps about {v0:.1f} m/s"
    if h > 30:
        latm = "turns left"
    elif h < -30:
        latm = "turns right"
    elif abs(lat) > 2.0:
        latm = f"shifts {'left' if lat > 0 else 'right'} by {abs(lat):.1f} m (lane change / nudge)"
    else:
        latm = "keeps its lane"
    return f"ego {lon} and {latm}; travels {gt[-1, 0]:.1f} m forward, {gt[-1, 1]:.1f} m lateral."


def valid_reasoning(txt):
    lines = [l for l in txt.strip().splitlines() if l.strip()]
    return len(lines) >= 4 and all(re.match(rf"\s*\**{k}\.", lines[k - 1]) for k in range(1, 5)) and len(txt) < 1500


def clean(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).replace("**", "").strip()
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return "\n".join(lines[:4])


class OpenRouterTeacher:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model="qwen/qwen3.7-flash", key=None, max_tokens=400, temperature=0.3):
        self.model = model
        self.key = key or os.environ.get("OPENROUTER_API_KEY")
        assert self.key, "set OPENROUTER_API_KEY (e.g. in .env)"
        self.max_tokens, self.temperature = max_tokens, temperature
        self.sess = requests.Session()

    def __call__(self, scene_text, gt, v0, retries=4):
        msg = USER_TMPL.format(scene=scene_text, T=C.T_PLAN, N=C.N_PLAN, dt=C.DT, traj=fmt_traj(gt),
                               summary=plan_summary(gt, v0))
        body = dict(model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
                    reasoning={"enabled": False, "exclude": True})
        for k in range(retries):
            try:
                r = self.sess.post(self.URL, json=body, timeout=120,
                                   headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** k * 2)
                    continue
                r.raise_for_status()
                j = r.json()
                txt = clean(j["choices"][0]["message"].get("content") or "")
                if valid_reasoning(txt):
                    return txt, j.get("usage", {})
            except Exception:
                time.sleep(2 ** k)
        return None, {}


# ----------------------------------------------------------------------------- rule-based fallback
def rule_reasoning(rec, v0, acc):
    """Deterministic template CoT from the serialised graph text + executed plan summary."""
    scene = rec["fut"]
    gt = np.array(rec["gt"])
    trend = "accelerating" if acc > 0.5 else "decelerating" if acc < -0.5 else "at steady speed"
    hist = re.search(r"History \(-2s -> -1s -> 0s\): (.*)", scene)
    l1 = f"1. Trend analysis: EGO moves {trend} at {v0:.1f} m/s" + \
         (f"; nearby agents: {hist.group(1).split(';')[0].strip()}." if hist else ".")
    edges = re.search(r"Edges @t0: (.*)", scene)
    trip = re.findall(r"\((A\d+), (in_front_of|left_of|behind|right_of), EGO, d=([\d.]+)", edges.group(1)) if edges else []
    same = re.search(r"\(([A\d/]+), same_lane_as, EGO\)", scene)
    lead = [t for t in trip if t[1] == "in_front_of" and same and t[0] in same.group(1).split("/")]
    if lead:
        l2 = f"2. Interaction assessment: {lead[0][0]} leads EGO in the same lane at {lead[0][2]} m."
    elif trip:
        l2 = f"2. Interaction assessment: closest relation is {trip[0][0]} {trip[0][1].replace('_', ' ')} EGO at {trip[0][2]} m."
    else:
        l2 = "2. Interaction assessment: no agent interacts closely with EGO."
    conf = re.findall(r"(A\d+): [^|]*\| rel to EGO: ([^\n]*!)", scene)
    l3 = ("3. Conflict anticipation: predicted close interaction with " + ", ".join(c[0] for c in conf[:3]) +
          " in the future graphs; keep a safe gap.") if conf else \
        "3. Conflict anticipation: the predicted future graphs show no conflict with EGO."
    l4 = f"4. Intent decision: {rec['nav']}; {plan_summary(gt, v0).replace('ego ', 'EGO ', 1)}"
    return "\n".join([l1, l2, l3, l4])
