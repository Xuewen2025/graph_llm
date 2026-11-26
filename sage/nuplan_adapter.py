"""nuPlan closed-loop adapter (STUB -- not tested; requires nuplan-devkit + nuPlan data).

Converts nuPlan `PlannerInput` into the SAGE scene dict and wraps `SageDrivePlanner` as an
`AbstractPlanner`, so it can be run with the official simulator on Val14:

    python nuplan/planning/script/run_simulation.py +simulation=closed_loop_nonreactive_agents \
        planner=sage_drive scenario_filter=val14_split ...

Notes
* nuPlan history is 20 Hz; it is resampled to the 0.5 s sub-steps used here (2 s of history needed).
* Models are trained on AV2 here; for the Val14 numbers retrain on nuPlan scenes converted with
  `scene_from_planner_input` (same dict schema), since lane/agent statistics differ between datasets.
* nuPlan provides traffic-light status; it can be added as signal nodes (NT_XW slot or a new type).
"""
import numpy as np

from . import config as C
from .planner import SageDrivePlanner

try:
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInput
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    _HAS_NUPLAN = True
except ImportError:  # keep importable without the devkit
    AbstractPlanner, _HAS_NUPLAN = object, False

_TYPE = {"VEHICLE": 0, "PEDESTRIAN": 1, "BICYCLE": 2}


def scene_from_planner_input(inp, map_api, max_agents=C.MAX_AGENTS):
    """PlannerInput -> scene dict (ego frame at t0, 0.5 s sub-steps, only history filled)."""
    egos = list(inp.history.ego_states)
    obs = list(inp.history.observations)
    S = len(C.RAW_STEPS)
    # pick history samples every 0.5 s ending at the current one
    step = max(1, int(round(0.5 / (egos[-1].time_seconds - egos[-2].time_seconds))))
    picks = list(range(len(egos) - 1, -1, -step))[: C.T0 + 1][::-1]
    e0 = egos[-1].rear_axle
    yaw, o = e0.heading, np.array([e0.x, e0.y])
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    tf = lambda p: (np.asarray(p) - o) @ R.T  # noqa: E731

    tokens = {}
    for o_ in obs[-1].tracked_objects.tracked_objects:
        d = np.hypot(o_.center.x - e0.x, o_.center.y - e0.y)
        if d < C.AGENT_RADIUS:
            tokens[o_.track_token] = d
    ids = ["AV"] + sorted(tokens, key=tokens.get)[:max_agents]
    Na = len(ids)
    pos = np.zeros((Na, S, 2), np.float32); vel = np.zeros((Na, S, 2), np.float32)
    head = np.zeros((Na, S), np.float32); valid = np.zeros((Na, S), bool); cls = np.zeros(Na, np.int64)
    off = C.T0 + 1 - len(picks)
    for k, h in enumerate(picks):
        si = off + k
        e = egos[h]
        pos[0, si] = tf([e.rear_axle.x, e.rear_axle.y])
        v = e.dynamic_car_state.center_velocity_2d
        vel[0, si] = np.array([v.x, v.y]) @ np.array([[np.cos(e.rear_axle.heading - yaw), np.sin(e.rear_axle.heading - yaw)],
                                                        [-np.sin(e.rear_axle.heading - yaw), np.cos(e.rear_axle.heading - yaw)]])
        head[0, si] = e.rear_axle.heading - yaw
        valid[0, si] = True
        by_tok = {t.track_token: t for t in obs[h].tracked_objects.tracked_objects}
        for a, tid in enumerate(ids[1:], 1):
            t = by_tok.get(tid)
            if t is None:
                continue
            pos[a, si] = tf([t.center.x, t.center.y])
            vel[a, si] = np.array([t.velocity.x, t.velocity.y]) @ R.T
            head[a, si] = t.center.heading - yaw
            valid[a, si] = True
            cls[a] = _TYPE.get(t.tracked_object_type.name, 4)

    # map: lanes + lane connectors + crosswalks around the ego
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.CROSSWALK]
    objs = map_api.get_proximal_map_objects(e0.point, C.LANE_RADIUS, layers)
    lanes = objs[SemanticMapLayer.LANE] + objs[SemanticMapLayer.LANE_CONNECTOR]
    lane_pts, lids = [], []
    for ln in lanes:
        pts = np.array([[p.x, p.y] for p in ln.baseline_path.discrete_path])
        idx = np.linspace(0, len(pts) - 1, C.LANE_PTS).round().astype(int)
        lane_pts.append(tf(pts[idx])); lids.append(ln.id)
    order = np.argsort([np.linalg.norm(p, axis=1).min() for p in lane_pts])[: C.MAX_LANES]
    lane_pts = [lane_pts[i] for i in order]; lanes = [lanes[i] for i in order]; lids = [lids[i] for i in order]
    l2i = {l: i for i, l in enumerate(lids)}
    M = len(lanes)
    succ = [(i, l2i[e.id]) for i, ln in enumerate(lanes) for e in ln.outgoing_edges if e.id in l2i]
    left, right = -np.ones(M, np.int64), -np.ones(M, np.int64)
    for i, ln in enumerate(lanes):
        adj = getattr(ln, "adjacent_edges", (None, None))
        if adj[0] is not None and adj[0].id in l2i:
            left[i] = l2i[adj[0].id]
        if adj[1] is not None and adj[1].id in l2i:
            right[i] = l2i[adj[1].id]
    xw = []
    for cw in objs[SemanticMapLayer.CROSSWALK][: C.MAX_XW]:
        ext = np.array(cw.polygon.exterior.coords)[:4]
        a, b = tf((ext[0] + ext[1]) / 2), tf((ext[2] + ext[3]) / 2)
        xw.append(np.stack([a, b]))
    return dict(
        sid="nuplan", agent_ids=np.array(ids), agent_pos=pos, agent_vel=vel, agent_head=head,
        agent_valid=valid, agent_cls=cls, lane_ids=np.arange(M), lane_pts=np.array(lane_pts, np.float32).reshape(M, C.LANE_PTS, 2),
        lane_width=np.full(M, 3.5, np.float32), lane_inter=np.array([ln.__class__.__name__.endswith("Connector") for ln in lanes]),
        lane_type=np.zeros(M, np.int64), lane_left=left, lane_right=right, lane_has_left=left >= 0,
        lane_has_right=right >= 0, lane_succ=np.array(succ, np.int64).reshape(-1, 2),
        xw_seg=np.array(xw, np.float32).reshape(-1, 2, 2), xw_width=np.full(len(xw), 4.0, np.float32))


class SageDriveNuPlanPlanner(AbstractPlanner):
    requires_scenario = False

    def __init__(self, ckpt, predictor_ckpt="outputs/predictor.pt", nav="go straight"):
        assert _HAS_NUPLAN, "install nuplan-devkit"
        self.planner = SageDrivePlanner(ckpt, predictor_ckpt=predictor_ckpt)
        self.nav = nav

    def name(self):
        return "sage_drive"

    def observation_type(self):
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        return DetectionsTracks

    def initialize(self, initialization):
        self.map_api = initialization.map_api
        self.route = initialization.route_roadblock_ids  # TODO: derive nav command from the route

    def compute_planner_trajectory(self, current_input: "PlannerInput"):
        scene = scene_from_planner_input(current_input, self.map_api)
        wps, _ = self.planner.plan(scene, self.nav)
        ego = current_input.history.ego_states[-1]
        x0, y0, h0 = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
        c, s = np.cos(h0), np.sin(h0)
        pts = np.concatenate([[[0.0, 0.0]], wps])
        states = [ego]
        for k in range(1, len(pts)):
            d = pts[k] - pts[k - 1]
            hd = h0 + np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 0.1 else states[-1].rear_axle.heading
            gx, gy = x0 + c * pts[k, 0] - s * pts[k, 1], y0 + s * pts[k, 0] + c * pts[k, 1]
            v = np.linalg.norm(d) / C.DT
            states.append(EgoState.build_from_rear_axle(
                StateSE2(gx, gy, hd), StateVector2D(v, 0.0), StateVector2D(0.0, 0.0), 0.0,
                TimePoint(ego.time_point.time_us + int(k * C.DT * 1e6)), ego.car_footprint.vehicle_parameters))
        return InterpolatedTrajectory(states)
