"""Serialise multi-temporal scene graphs into compact relational triplets + state tables
(Sec. III-B), and the raw-text baseline serialisation (graph-free).
"""
import re

import numpy as np

from . import config as C
from .graph import ET

MAX_PROMPT_AGENTS = 10
CLS_NAME = ["vehicle", "pedestrian", "cyclist", "bus", "object"]
REL_TXT = ["in_front_of", "left_of", "behind", "right_of"]


# ----------------------------------------------------------------------------- utils
def f1(x):
    x = round(float(x), 1)
    return f"{x + 0.0:.1f}" if x != 0 else "0.0"


def fmt_traj(p):
    return "[" + ",".join(f"({f1(x)},{f1(y)})" for x, y in p) + "]"


_NUM = r"(-?\d+(?:\.\d+)?)"


def parse_traj(text, n=C.N_PLAN):
    """parse the last 'Trajectory: [(x,y),...]' block; pads by repeating the last point."""
    m = list(re.finditer(r"Trajectory:\s*\[(.*?)(\]|$)", text, re.S))
    seg = m[-1].group(1) if m else text
    pts = re.findall(r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)", seg)
    if not pts:
        return None
    p = np.array([[float(a), float(b)] for a, b in pts[:n]], np.float32)
    if len(p) < n:
        p = np.concatenate([p, np.repeat(p[-1:], n - len(p), 0)])
    return p


def deg(h):
    return int(round(np.degrees(h)))


def nav_command(fut):
    """High-level route command (the 'optional task instruction'), derived from the route the
    ego follows. Comparable to the mission goal given to GPT-Driver / nuPlan planners."""
    d = fut[-1]
    if np.linalg.norm(d) < 2.0:
        return "stop / stay"
    h = np.arctan2(fut[-1, 1] - fut[-3, 1], fut[-1, 0] - fut[-3, 0]) if np.linalg.norm(fut[-1] - fut[-3]) > 0.5 \
        else np.arctan2(d[1], d[0])
    if h > np.radians(30):
        return "turn left"
    if h < -np.radians(30):
        return "turn right"
    return "go straight"


def ego_kinematics(scene):
    v = scene["agent_vel"][0]
    sp = np.linalg.norm(v, axis=1)
    acc = (sp[C.T0] - sp[C.T0 - 1]) / C.DT
    yr = (scene["agent_head"][0, C.T0] - scene["agent_head"][0, C.T0 - 1])
    yr = (yr + np.pi) % (2 * np.pi) - np.pi
    return sp[C.T0], acc, yr / C.DT


def _ego_block(scene):
    sp, acc, yr = ego_kinematics(scene)
    past = scene["agent_pos"][0, C.HIST_IDX]
    return (f"EGO: speed {sp:.1f} m/s, acc {acc:+.1f} m/s2, yaw-rate {np.degrees(yr):+.0f} deg/s; "
            f"past positions (-2.0s..0s) {fmt_traj(past)}")


def select_agents(scene, g_now):
    """prompt agents: valid at t0, ego-connected first, then by distance."""
    valid = scene["agent_valid"][:, C.T0]
    d = np.linalg.norm(scene["agent_pos"][:, C.T0], axis=1)
    linked = set(g_now["ei"][0][(g_now["ei"][1] == 0) & (g_now["etype"] == ET["agent-agent"])].tolist())
    idx = [i for i in range(1, len(d)) if valid[i]]
    idx.sort(key=lambda i: (i not in linked, d[i]))
    return idx[:MAX_PROMPT_AGENTS]


# ----------------------------------------------------------------------------- raw baseline
def raw_prompt(scene, nav):
    """Graph-free baseline: plain text-serialised object/map lists, no relations, no graphs."""
    lines = ["Scene (ego frame: x forward, y left; meters, m/s).", _ego_block(scene), "Objects:"]
    valid = scene["agent_valid"][:, C.T0]
    d = np.linalg.norm(scene["agent_pos"][:, C.T0], axis=1)
    idx = sorted([i for i in range(1, len(d)) if valid[i]], key=lambda i: d[i])[:MAX_PROMPT_AGENTS]
    for k, i in enumerate(idx):
        p, v = scene["agent_pos"][i, C.T0], scene["agent_vel"][i, C.T0]
        pp = scene["agent_pos"][i, C.T0 - 2] if scene["agent_valid"][i, C.T0 - 2] else p
        lines.append(f"- {CLS_NAME[scene['agent_cls'][i]]} at ({f1(p[0])},{f1(p[1])}), "
                     f"1s ago ({f1(pp[0])},{f1(pp[1])}), velocity ({f1(v[0])},{f1(v[1])})")
    lines.append("Lane centerlines (start->end):")
    lp = scene["lane_pts"]
    order = np.argsort(np.linalg.norm(lp, axis=-1).min(1))[:8]
    for l in order:
        lines.append(f"- ({lp[l,0,0]:.0f},{lp[l,0,1]:.0f})->({lp[l,-1,0]:.0f},{lp[l,-1,1]:.0f})")
    lines.append(f"Navigation: {nav}.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- graph serialisation
def _lane_desc(scene, l, names):
    lp = scene["lane_pts"][l]
    d = lp[-1] - lp[0]
    ln = np.linalg.norm(np.diff(lp, axis=0), axis=1).sum()
    nb = []
    for key, tag in (("lane_left", "left"), ("lane_right", "right")):
        j = scene[key][l]
        if j >= 0 and j in names:
            nb.append(f"{tag}={names[j]}")
    succ = [names[b] for a, b in scene["lane_succ"] if a == l and b in names]
    if succ:
        nb.append("next=" + "/".join(succ[:3]))
    inter = ", intersection" if scene["lane_inter"][l] else ""
    return f"{names[l]}: heading {deg(np.arctan2(d[1], d[0]))} deg, length {ln:.0f} m{inter}" + \
        (f"; {', '.join(nb)}" if nb else "")


def _pair(g, j, i):
    """agent-agent edge j->i at graph g (relation of j w.r.t. i)"""
    m = (g["ei"][0] == j) & (g["ei"][1] == i) & (g["etype"] == ET["agent-agent"])
    k = np.where(m)[0]
    return None if len(k) == 0 else k[0]


def _ttc(t):
    return "inf" if not np.isfinite(t) else f"{t:.1f}s"


def prompt_nodes(scene, g0):
    """agents (graph node ids) and lanes (lane ids) named in the graph prompt: A1.., L1.."""
    ag = select_agents(scene, g0)
    # lanes referenced: ego lane + its neighbours/successors + lanes of selected agents
    lanes = []
    def addl(l):
        if l >= 0 and l not in lanes:
            lanes.append(int(l))
    el = g0["agent_lane"][0]
    addl(el)
    if el >= 0:
        addl(scene["lane_left"][el]); addl(scene["lane_right"][el])
        for a, b in scene["lane_succ"]:
            if a == el:
                addl(b)
    for i in ag:
        addl(g0["agent_lane"][i])
    return ag, lanes[:10]


def node_map(scene, g0):
    """prompt name -> graph node index (agents, then lanes, then crosswalks; see graph.make_graph)."""
    ag, lanes = prompt_nodes(scene, g0)
    Na, M = len(scene["agent_pos"]), len(scene["lane_pts"])
    m = {"EGO": 0, **{f"A{k+1}": int(i) for k, i in enumerate(ag)}, **{f"L{k+1}": Na + l for k, l in enumerate(lanes)}}
    m.update({f"X{x+1}": Na + M + x for x in range(len(scene["xw_seg"]))})
    return m


def graph_prompt(scene, hist, fut=None, nav="go straight"):
    """hist: list of history graphs (last = current); fut: list of predicted future graphs."""
    g0 = hist[-1]
    Na = len(scene["agent_pos"])
    ag, lanes = prompt_nodes(scene, g0)
    aname = {0: "EGO", **{i: f"A{k+1}" for k, i in enumerate(ag)}}
    el = g0["agent_lane"][0]
    lname = {l: f"L{k+1}" for k, l in enumerate(lanes)}

    out = ["Spatio-temporal scene graph (ego frame: x forward, y left; meters, m/s).", _ego_block(scene)]
    if el >= 0:
        out[-1] += f"; on {lname[el]}"
    out.append("Agent nodes @t0:")
    for i in ag:
        p, v = g0["pos"][i], g0["vel"][i]
        lane = g0["agent_lane"][i]
        out.append(f"{aname[i]} {CLS_NAME[scene['agent_cls'][i]]}: pos ({f1(p[0])},{f1(p[1])}), "
                   f"speed {np.linalg.norm(v):.1f}, heading {deg(g0['head'][i])} deg"
                   + (f", on {lname[lane]}" if lane in lname else ""))
    if lanes:
        out.append("Lane nodes:")
        out += [_lane_desc(scene, l, lname) for l in lanes]
    if len(scene["xw_seg"]):
        xs = []
        m = g0["etype"] == ET["xw-agent"]
        ref = {int(s_) for s_, d_ in zip(g0["ei"][0][m], g0["ei"][1][m]) if d_ in aname}
        for x in range(len(scene["xw_seg"])):
            seg = scene["xw_seg"][x]
            d = np.linalg.norm(seg.mean(0))
            if d < 30 or (Na + len(scene["lane_pts"]) + x) in ref:
                xs.append(f"X{x+1} crosswalk center ({seg.mean(0)[0]:.0f},{seg.mean(0)[1]:.0f})")
        if xs:
            out.append("Regulatory nodes: " + "; ".join(xs))

    # relational triplets at t0 (ego-centred + crosswalk influence)
    trip = []
    for i in ag:
        k = _pair(g0, i, 0)
        if k is not None:
            trip.append(f"({aname[i]}, {REL_TXT[g0['rel'][k]]}, EGO, d={g0['dist'][k]:.1f}, ttc={_ttc(g0['ttc'][k])})")
    if el >= 0:
        same = [aname[i] for i in ag if g0["agent_lane"][i] == el]
        if same:
            trip.append(f"({'/'.join(same)}, same_lane_as, EGO)")
    m = g0["etype"] == ET["xw-agent"]
    for s, d in zip(g0["ei"][0][m], g0["ei"][1][m]):
        if d in aname:
            trip.append(f"(X{s - Na - len(scene['lane_pts']) + 1}, regulates, {aname[d]})")
    out.append("Edges @t0: " + (" ".join(trip) if trip else "none"))

    # temporal evolution of ego relations over the history graphs
    ev = []
    for i in ag:
        seq = []
        for g in hist[::2]:  # -2s, -1s, 0s
            k = _pair(g, i, 0)
            seq.append("-" if k is None else f"{REL_TXT[g['rel'][k]][:-3]} {g['dist'][k]:.0f}")
        if any(s != "-" for s in seq):
            ev.append(f"{aname[i]}: " + " -> ".join(seq))
    if ev:
        out.append("History (-2s -> -1s -> 0s): " + "; ".join(ev))

    if fut is not None:
        out.append("Predicted future graphs (+1s, +2s, +3s):")
        sel = [1, 3, 5]
        ep = [fut[k]["pos"][0] for k in sel]
        out.append("EGO forecast: " + " ".join(f"({f1(p[0])},{f1(p[1])})" for p in ep))
        for i in ag:
            ps = [fut[k]["pos"][i] for k in sel]
            line = f"{aname[i]}: " + " ".join(f"({f1(p[0])},{f1(p[1])})" for p in ps)
            rels = []
            for k in sel:
                e = _pair(fut[k], i, 0)
                if e is not None:
                    flag = "!" if (fut[k]["ttc"][e] < 3 or fut[k]["dist"][e] < 6) else ""
                    rels.append(f"{REL_TXT[fut[k]['rel'][e]][:-3]} {fut[k]['dist'][e]:.0f}{flag}")
                else:
                    rels.append("-")
            line += " | rel to EGO: " + " -> ".join(rels)
            out.append(line)
    out.append(f"Navigation: {nav}.")
    return "\n".join(out)


INSTR = ("You are the planner of an autonomous vehicle. Plan the ego trajectory for the next "
         f"{C.T_PLAN:.0f} s as {C.N_PLAN} waypoints at {C.DT} s intervals in the ego frame. The plan must be safe, "
         "smooth, obey traffic rules and be kinematically feasible (bounded acceleration and curvature).")
FUT_INSTR = " Avoid conflicts implied by the predicted future graphs."
COT_INSTR = (" First reason step by step: 1. Trend analysis, 2. Interaction assessment, "
             "3. Conflict anticipation, 4. Intent decision. Then output the trajectory.")


def build_prompt(scene_text, cot, fut=False):
    return f"{INSTR}{FUT_INSTR if fut else ''}{COT_INSTR if cot else ''}\n\n{scene_text}\n\n### Answer:\n"
