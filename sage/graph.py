"""Traffic scene graph construction G_t = (V_t, E_t)   (Sec. III-A-1, Eq. 2-4).

The same rules are used for historical/current graphs (from observed states) and for
reconstructed future graphs (from predicted agent states, Eq. 11), so the structure is
consistent across time.
"""
import numpy as np

from . import config as C

ET = {n: i for i, n in enumerate(C.EDGE_TYPES)}
LANE_CLASSES = (0, 2, 3)  # vehicle, cyclist, bus may occupy lanes


def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)  # R(-ang) v


def relation_of(dx, dy):
    """discrete language relation of j w.r.t. i from displacement in i's frame"""
    ang = np.degrees(np.arctan2(dy, dx))
    rel = np.full(np.shape(ang), 2, np.int64)  # behind
    rel[np.abs(ang) <= 45] = 0
    rel[(ang > 45) & (ang <= 135)] = 1
    rel[(ang < -45) & (ang >= -135)] = 3
    return rel


# ----------------------------------------------------------------------------- nodes
def static_node_feats(scene):
    pts = scene["lane_pts"]
    M = len(pts)
    lf = np.zeros((M, C.NODE_DIM), np.float32)
    if M:
        st, md, en = pts[:, 0], pts[:, C.LANE_PTS // 2], pts[:, -1]
        d = en - st
        ln = np.linalg.norm(np.diff(pts, axis=1), axis=-1).sum(1)
        h = np.arctan2(d[:, 1], d[:, 0])
        lf[:, 0:2], lf[:, 2:4], lf[:, 4:6] = st / C.POS_SCALE, md / C.POS_SCALE, en / C.POS_SCALE
        lf[:, 6], lf[:, 7] = np.cos(h), np.sin(h)
        lf[:, 8] = ln / C.POS_SCALE
        lf[:, 9] = scene["lane_width"] / 5.0
        lf[:, 10] = scene["lane_inter"]
        lf[np.arange(M), 11 + scene["lane_type"]] = 1
        lf[:, 14] = scene["lane_has_left"]
        lf[:, 15] = scene["lane_has_right"]
    seg = scene["xw_seg"]
    X = len(seg)
    xf = np.zeros((X, C.NODE_DIM), np.float32)
    if X:
        d = seg[:, 1] - seg[:, 0]
        h = np.arctan2(d[:, 1], d[:, 0])
        xf[:, 0:2], xf[:, 2:4] = seg[:, 0] / C.POS_SCALE, seg[:, 1] / C.POS_SCALE
        xf[:, 4:6] = seg.mean(1) / C.POS_SCALE
        xf[:, 6], xf[:, 7] = np.cos(h), np.sin(h)
        xf[:, 8] = np.linalg.norm(d, axis=1) / C.POS_SCALE
        xf[:, 9] = scene["xw_width"] / 10.0
    return lf, xf


def agent_node_feats(pos, vel, head, acc, cls, valid):
    Na = len(pos)
    f = np.zeros((Na, C.NODE_DIM), np.float32)
    f[:, 0:2] = pos / C.POS_SCALE
    f[:, 2:4] = vel / C.VEL_SCALE
    f[:, 4], f[:, 5] = np.cos(head), np.sin(head)
    f[:, 6] = np.linalg.norm(vel, axis=1) / C.VEL_SCALE
    f[:, 7] = np.clip(acc / C.ACC_SCALE, -3, 3)
    f[np.arange(Na), 8 + cls] = 1
    f[0, 13] = 1  # ego flag
    f[:, 14] = 1.0
    f[~valid] = 0
    return f


# ----------------------------------------------------------------------------- edges
def _closest_on_polylines(p, pts):
    """p [Na,2], pts [M,P,2] -> dist [Na,M], proj point [Na,M,2], seg heading [Na,M]"""
    a, b = pts[:, :-1], pts[:, 1:]                      # [M,S,2]
    ab = b - a
    ap = p[:, None, None, :] - a[None]                  # [Na,M,S,2]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(p[:, None, None] - proj, axis=-1)  # [Na,M,S]
    k = d.argmin(-1)
    ii, jj = np.meshgrid(np.arange(len(p)), np.arange(len(pts)), indexing="ij")
    hd = np.arctan2(ab[..., 1], ab[..., 0])             # [M,S]
    return d[ii, jj, k], proj[ii, jj, k], hd[jj, k]


def build_edges(scene, pos, vel, head, valid):
    """Relational conditions R(i,j,t) and edge features r_ij = [r_geo, r_lang].

    Returns dict(ei [2,E] (src j, dst i), ea [E,EDGE_DIM], etype [E], rel [E], ttc [E], dist [E],
                 agent_lane [Na]).
    Node index layout: agents [0,Na), lanes [Na,Na+M), crosswalks [Na+M, Na+M+X).
    """
    Na, M, X = len(pos), len(scene["lane_pts"]), len(scene["xw_seg"])
    cls = scene["agent_cls"]
    src, dst, typ = [], [], []
    ps, vs, hs, pd_, vd, hd = [], [], [], [], [], []

    def add(s, d, t, p_s, v_s, h_s, p_d, v_d, h_d):
        n = len(s)
        if n == 0:
            return
        src.append(s); dst.append(d); typ.append(np.full(n, t))
        ps.append(p_s); vs.append(v_s); hs.append(h_s); pd_.append(p_d); vd.append(v_d); hd.append(h_d)

    # agent <-> agent: spatial proximity
    vi = np.where(valid)[0]
    if len(vi) > 1:
        D = np.linalg.norm(pos[vi, None] - pos[None, vi], axis=-1)
        jj, ii = np.where((D < C.AA_RADIUS) & ~np.eye(len(vi), dtype=bool))
        j, i = vi[jj], vi[ii]
        add(j, i, ET["agent-agent"], pos[j], vel[j], head[j], pos[i], vel[i], head[i])

    # lane topology
    agent_lane = -np.ones(Na, np.int64)
    if M:
        lp = scene["lane_pts"]
        lmid = lp[:, C.LANE_PTS // 2]
        dd = lp[:, -1] - lp[:, 0]
        lh = np.arctan2(dd[:, 1], dd[:, 0])
        z = np.zeros((0, 2))

        def lane_pair(a, b, t):  # edge src b -> dst a
            if len(a):
                add(Na + b, Na + a, t, lmid[b], np.zeros((len(b), 2)), lh[b], lmid[a], np.zeros((len(a), 2)), lh[a])

        sc = scene["lane_succ"]
        if len(sc):
            lane_pair(sc[:, 0], sc[:, 1], ET["lane-next"])
            lane_pair(sc[:, 1], sc[:, 0], ET["lane-prev"])
        for arr, t in ((scene["lane_left"], ET["lane-left"]), (scene["lane_right"], ET["lane-right"])):
            a = np.where(arr >= 0)[0]
            lane_pair(a, arr[a], t)

        # agent <-> lane occupancy
        cand = np.where(valid & np.isin(cls, LANE_CLASSES))[0]
        if len(cand):
            d, proj, sh = _closest_on_polylines(pos[cand], lp)
            cosd = np.cos(head[cand][:, None] - sh)
            score = d + 3 * (1 - cosd)
            score[(d > C.LANE_ASSIGN_DIST) | (cosd < 0.5)] = np.inf
            best = score.argmin(1)
            ok = np.isfinite(score[np.arange(len(cand)), best])
            a, l = cand[ok], best[ok]
            agent_lane[a] = l
            pp, hh = proj[ok, l], sh[ok, l]
            zv = np.zeros((len(a), 2))
            add(a, Na + l, ET["agent-on-lane"], pos[a], vel[a], head[a], pp, zv, hh)
            add(Na + l, a, ET["lane-has-agent"], pp, zv, hh, pos[a], vel[a], head[a])

    # crosswalk <-> agent (regulatory influence)
    if X and len(vi):
        seg = scene["xw_seg"]
        d, proj, sh = _closest_on_polylines(pos[vi], seg)
        aa, xx = np.where(d < C.XW_NEAR_DIST)
        if len(aa):
            a = vi[aa]
            pp, hh = proj[aa, xx], sh[aa, xx]
            zv = np.zeros((len(a), 2))
            add(Na + M + xx, a, ET["xw-agent"], pp, zv, hh, pos[a], vel[a], head[a])
            add(a, Na + M + xx, ET["agent-xw"], pos[a], vel[a], head[a], pp, zv, hh)

    if not src:
        return dict(ei=np.zeros((2, 0), np.int64), ea=np.zeros((0, C.EDGE_DIM), np.float32),
                    etype=np.zeros(0, np.int64), rel=np.zeros(0, np.int64), ttc=np.zeros(0),
                    dist=np.zeros(0), agent_lane=agent_lane)
    src, dst, typ = map(np.concatenate, (src, dst, typ))
    ps, vs, hs, pd_, vd, hd = map(lambda x: np.concatenate(x).astype(np.float64), (ps, vs, hs, pd_, vd, hd))

    # r_geo in the frame of the target node i
    dp = _rot(ps - pd_, hd)
    dv = _rot(vs - vd, hd)
    dist = np.linalg.norm(dp, axis=1)
    closing = -(dp * dv).sum(1) / np.maximum(dist, 1e-3)
    ttc = np.where((typ == ET["agent-agent"]) & (closing > 0.1), dist / np.maximum(closing, 1e-3), np.inf)
    dh = hs - hd
    rel = relation_of(dp[:, 0], dp[:, 1])
    E = len(src)
    ea = np.zeros((E, C.EDGE_DIM), np.float32)
    ea[:, 0:2] = dp / C.POS_SCALE
    ea[:, 2] = dist / C.POS_SCALE
    ea[:, 3:5] = dv / C.VEL_SCALE
    ea[:, 5] = np.where(np.isfinite(ttc), 1.0 / (1.0 + ttc), 0.0)
    ea[:, 6], ea[:, 7] = np.cos(dh), np.sin(dh)
    # r_lang: discretised spatial relation + edge (relation) type, deterministically derived
    ea[np.arange(E), 8 + rel] = 1
    ea[np.arange(E), 8 + len(C.RELATIONS) + typ] = 1
    return dict(ei=np.stack([src, dst]).astype(np.int64), ea=ea, etype=typ.astype(np.int64),
                rel=rel, ttc=ttc, dist=dist, agent_lane=agent_lane)


# ----------------------------------------------------------------------------- graphs
def make_graph(scene, pos, vel, head, acc, valid, static=None):
    lf, xf = static if static is not None else static_node_feats(scene)
    af = agent_node_feats(pos, vel, head, acc, scene["agent_cls"], valid)
    x = np.concatenate([af, lf, xf], 0)
    ntype = np.concatenate([np.full(len(af), C.NT_AGENT), np.full(len(lf), C.NT_LANE),
                            np.full(len(xf), C.NT_XW)]).astype(np.int64)
    nvalid = np.concatenate([valid, np.ones(len(lf) + len(xf), bool)])
    g = build_edges(scene, pos, vel, head, valid)
    g.update(x=x, ntype=ntype, nvalid=nvalid, pos=pos, vel=vel, head=head, avalid=valid)
    return g


def _acc(scene, s):
    v = np.linalg.norm(scene["agent_vel"][:, s], axis=1)
    if s == 0:
        return np.zeros_like(v)
    vp = np.linalg.norm(scene["agent_vel"][:, s - 1], axis=1)
    ok = scene["agent_valid"][:, s - 1]
    return np.where(ok, (v - vp) / C.DT, 0.0)


def history_graphs(scene):
    """{G_t}, t in T_hist (observed)."""
    static = static_node_feats(scene)
    out = []
    for s in C.HIST_IDX:
        out.append(make_graph(scene, scene["agent_pos"][:, s], scene["agent_vel"][:, s],
                              scene["agent_head"][:, s], _acc(scene, s), scene["agent_valid"][:, s], static))
    return out


def future_graphs(scene, pred):
    """Reconstruct predicted future graphs {G^_{t0+tau}} from predicted dynamic states (Eq. 11).

    pred: [Na, TF, 5] = (px, py, vx, vy, heading) in ego frame. Static nodes are copied.
    """
    static = static_node_feats(scene)
    valid = scene["agent_valid"][:, C.T0]
    prev_v = np.linalg.norm(scene["agent_vel"][:, C.T0], axis=1)
    out = []
    for k in range(pred.shape[1]):
        v = np.linalg.norm(pred[:, k, 2:4], axis=1)
        acc = (v - prev_v) / C.DT
        prev_v = v
        out.append(make_graph(scene, pred[:, k, :2], pred[:, k, 2:4], pred[:, k, 4], acc, valid, static))
    return out
