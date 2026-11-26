"""Open-loop metrics: ADE / FDE, Feasibility Violation Rate (FVR), collision rate."""
import numpy as np

from . import config as C


def kinematics(traj, v0):
    """traj [N,2] ego-frame waypoints at dt (starting point = origin, initial velocity v0 along +x).

    Returns per-timestep |a_lon|, |a_lat|, |kappa| (each [N]).
    """
    p = np.concatenate([np.zeros((1, 2)), traj], 0)
    vel = np.diff(p, axis=0) / C.DT                                 # [N,2]
    vel = np.concatenate([np.array([[v0, 0.0]]), vel], 0)           # prepend current velocity
    sp = np.linalg.norm(vel, axis=1)
    h = np.arctan2(vel[:, 1], vel[:, 0])
    h[0] = 0.0
    # heading is undefined when (almost) standing still -> hold previous heading
    for k in range(1, len(h)):
        if sp[k] < 0.5:
            h[k] = h[k - 1]
    dh = (np.diff(h) + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = dh / C.DT
    a_lon = np.diff(sp) / C.DT
    vm = 0.5 * (sp[1:] + sp[:-1])
    a_lat = vm * yaw_rate
    kappa = np.where(vm > 1.0, yaw_rate / np.maximum(vm, 1e-3), 0.0)
    return np.abs(a_lon), np.abs(a_lat), np.abs(kappa)


def fvr_thresholds(gts, v0s, q=95):
    """95th percentile of GT kinematics on the training set (paper: 95th percentile of nuPlan)."""
    al, at, k = zip(*[kinematics(g, v) for g, v in zip(gts, v0s)])
    return dict(a_max=float(np.percentile(np.concatenate(al), q)),
                alat_max=float(np.percentile(np.concatenate(at), q)),
                kappa_max=float(np.percentile(np.concatenate(k), q)))


def fvr(traj, v0, th):
    al, at, k = kinematics(traj, v0)
    viol = (al > th["a_max"]) | (at > th["alat_max"]) | (k > th["kappa_max"])
    return viol.mean()


def collision(traj, scene, radius=2.0):
    """1 if the plan comes within `radius` m of any other agent's GT future position (open-loop proxy)."""
    pos = scene["agent_pos"][1:, C.T0 + 1:C.T0 + 1 + C.N_PLAN]
    val = scene["agent_valid"][1:, C.T0 + 1:C.T0 + 1 + C.N_PLAN]
    if len(pos) == 0:
        return 0.0
    d = np.linalg.norm(pos - traj[None], axis=-1)
    return float(((d < radius) & val).any())


def summarize(preds, gts, v0s, th, scenes=None):
    preds, gts = np.asarray(preds), np.asarray(gts)
    err = np.linalg.norm(preds - gts, axis=-1)                       # [S,N]
    h3 = int(3.0 / C.DT)
    out = {
        "ADE@6s": err.mean(), "FDE@6s": err[:, -1].mean(),
        "ADE@3s": err[:, :h3].mean(), "FDE@3s": err[:, h3 - 1].mean(),
        "FVR": float(np.mean([fvr(p, v, th) for p, v in zip(preds, v0s)])),
        "FVR_GT": float(np.mean([fvr(g, v, th) for g, v in zip(gts, v0s)])),
    }
    if scenes is not None:
        out["Collision"] = float(np.mean([collision(p, s) for p, s in zip(preds, scenes)]))
    return {k: float(v) for k, v in out.items()}
