# task2b_planar_module.py
from __future__ import annotations

import math
import random
import inspect
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from shapely.geometry import Point as ShPoint, Polygon, LineString
from shapely.geometry import LineString as ShLineString

# ============================================================
# Externe Klassen aus deinem Framework/Projekt
# ============================================================
# Passe diese Imports ggf. an deine Projektstruktur an.
try:
    from IPBenchmark import Benchmark
except Exception as e:
    raise ImportError(
        "Konnte Benchmark nicht importieren. Stelle sicher, dass "
        "`from IPBenchmark import Benchmark` in deinem Projekt funktioniert."
    ) from e

# EnhancedLazyPRM kommt aus deinem Projekt (z.B. IPPLazyPRM_Task2.py / IPLazyPRM Erweiterung)
# -> wird im Notebook importiert/übergeben. Hier NICHT hard-abhängig machen.


# ============================================================
# 0) Benchmark-Factory (robust gegen unterschiedliche Signaturen)
# ============================================================
def _make_benchmark(name, collisionChecker, start, goal, description=None, level=1):
    if description is None:
        description = name
    startList = [start]
    goalList  = [goal]

    sig = inspect.signature(Benchmark.__init__)
    params = list(sig.parameters.keys())[1:]  # ohne self

    mapping = {
        "name": name,
        "collisionChecker": collisionChecker,
        "startList": startList,
        "goalList": goalList,
        "description": description,
        "level": level,
    }

    args = []
    for p in params:
        if p in mapping:
            args.append(mapping[p])
        else:
            if sig.parameters[p].default is inspect._empty:
                raise TypeError(f"Benchmark.__init__ hat unbekannten Pflichtparameter: {p}")
            args.append(sig.parameters[p].default)

    b = Benchmark(*args)

    # Komfortfelder
    if hasattr(b, "name"):
        b.name = name
    else:
        setattr(b, "name", name)
    setattr(b, "start", start)
    setattr(b, "goal", goal)

    return b


# ============================================================
# 1) Gemeinsames CollisionChecker-Interface (Framework-kompatibel)
# ============================================================
class CCBase:
    """
    Adapter-Basis:
      - getDim()
      - getEnvironmentLimits()
      - pointInCollision(q)
      - lineInCollision(q1, q2)  (Alias: edgeInCollision)
    """
    def __init__(self, bounds, obstacles):
        # bounds: (xmin, ymin, xmax, ymax) fürs Workspace-Sampling (Point-Robot)
        self.bounds = bounds
        self.obstacles = list(obstacles) if obstacles is not None else []
        self.counter = 0

    def getDim(self):
        raise NotImplementedError

    def getEnvironmentLimits(self):
        xmin, ymin, xmax, ymax = self.bounds
        return [[float(xmin), float(xmax)], [float(ymin), float(ymax)]]

    def lineInCollision(self, q1, q2):
        return self.edgeInCollision(q1, q2)

    def edgeInCollision(self, q1, q2):
        # Default: diskretisiere Segment -> point checks
        q1 = np.asarray(q1, float)
        q2 = np.asarray(q2, float)
        steps = max(2, int(np.linalg.norm(q2 - q1) / 0.05))
        for t in np.linspace(0.0, 1.0, steps):
            q = (1 - t) * q1 + t * q2
            if self.pointInCollision(q.tolist()):
                return True
        return False


# ============================================================
# 2) 2DoF Point-Robot (q=[x,y]) in 2D Workspace
# ============================================================
class PointRobot2DoFCollisionChecker(CCBase):
    def __init__(self, bounds, obstacles, min_clearance=0.0):
        super().__init__(bounds=bounds, obstacles=obstacles)
        self.min_clearance = float(min_clearance)

        # Für Abstand: Hindernisse "aufblasen" (Minkowski Summe)
        if self.min_clearance > 0.0:
            self._inflated_obstacles = [obs.buffer(self.min_clearance) for obs in self.obstacles]
        else:
            self._inflated_obstacles = list(self.obstacles)

    def getDim(self):
        return 2

    def pointInCollision(self, q):
        self.counter += 1
        p = ShPoint(float(q[0]), float(q[1]))

        xmin, ymin, xmax, ymax = self.bounds
        if p.x < xmin or p.x > xmax or p.y < ymin or p.y > ymax:
            return True

        for obs in self._inflated_obstacles:
            if obs.contains(p) or obs.touches(p):
                return True
        return False

    def edgeInCollision(self, q1, q2):
        self.counter += 1
        seg = LineString([(float(q1[0]), float(q1[1])), (float(q2[0]), float(q2[1]))])

        xmin, ymin, xmax, ymax = self.bounds
        if (seg.bounds[0] < xmin) or (seg.bounds[2] > xmax) or (seg.bounds[1] < ymin) or (seg.bounds[3] > ymax):
            return True

        for obs in self._inflated_obstacles:
            if seg.intersects(obs):
                return True
        return False


# ============================================================
# 3) 2DoF Planar Manipulator (q=[theta1,theta2]) – Kollision im Workspace
# ============================================================
class PlanarRobot2DoFCollisionChecker(CCBase):
    """
    Konfiguration q=[theta1,theta2] (Radiant)
    Abstand/Roboterdicke wird robust als "Capsule" der Links modelliert:
      link_capsule = LineString(...).buffer(min_clearance)
    Damit gilt: Link muss mindestens min_clearance Abstand zum Hindernis halten.
    """
    def __init__(self, theta_limits, obstacles, base=(0.0, 0.0), L1=1.6, L2=1.2, min_clearance=0.0):
        super().__init__(bounds=(-math.pi, -math.pi, math.pi, math.pi), obstacles=obstacles)
        self.theta_limits = theta_limits
        self.base = base
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.min_clearance = float(min_clearance)

    def getDim(self):
        return 2

    def getEnvironmentLimits(self):
        (a1, b1), (a2, b2) = self.theta_limits
        return [[float(a1), float(b1)], [float(a2), float(b2)]]

    def _fk_points(self, q):
        t1, t2 = float(q[0]), float(q[1])
        x0, y0 = self.base
        x1 = x0 + self.L1 * math.cos(t1)
        y1 = y0 + self.L1 * math.sin(t1)
        x2 = x1 + self.L2 * math.cos(t1 + t2)
        y2 = y1 + self.L2 * math.sin(t1 + t2)
        return (x0, y0), (x1, y1), (x2, y2)

    def pointInCollision(self, q):
        self.counter += 1
        p0, p1, p2 = self._fk_points(q)

        link1 = LineString([p0, p1])
        link2 = LineString([p1, p2])

        # Clearance: Links als Kapseln (robust)
        if self.min_clearance > 0.0:
            link1 = link1.buffer(self.min_clearance, cap_style=1, join_style=1)
            link2 = link2.buffer(self.min_clearance, cap_style=1, join_style=1)

        for obs in self.obstacles:
            if link1.intersects(obs) or link2.intersects(obs):
                return True
        return False

    def edgeInCollision(self, q1, q2):
        q1 = np.asarray(q1, float)
        q2 = np.asarray(q2, float)
        steps = max(8, int(np.linalg.norm(q2 - q1) / 0.05))
        for t in np.linspace(0.0, 1.0, steps):
            q = (1 - t) * q1 + t * q2
            if self.pointInCollision(q.tolist()):
                return True
        return False


# ============================================================
# 4) Benchmarks (Task1-Style) + Builder
# ============================================================
def _rounded_box(xmin, ymin, xmax, ymax, r=0.15):
    rect = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    return rect.buffer(r).buffer(-r)


def build_2b_benchmarks_from_task1style(
    robot_radius=0.0,
    safety_margin=0.0
):
    """
    robot_radius + safety_margin => CLEARANCE (in Workspace-Einheiten)
    Clearance gilt:
      - Point-Robot: Hindernisse werden inflated (Minkowski) geprüft
      - Planar-2R: Links werden als Capsules buffer(min_clearance) geprüft
    """
    CLEARANCE = float(robot_radius) + float(safety_margin)

    # A) Workspace obstacles
    bounds = (-3.0, -3.0, 3.0, 3.0)

    obs_easy = [
        LineString([(-2.0, 0.0), (-0.8, 0.0)]).buffer(0.5),
        LineString([( 2.0, 0.0), ( 2.0, 1.0)]).buffer(0.2),
        LineString([(-1.0, 2.0), ( 1.0, 2.0)]).buffer(0.1),
    ]

    obs_passage = [
        LineString([(1.8, -3.0), (1.8, -0.8)]).buffer(0.2),
        LineString([(1.8,  0.8), (1.8,  3.0)]).buffer(0.2),
    ]

    # B) Point 2DoF (q=[x,y]) – Clearance aktiv
    ccP_easy    = PointRobot2DoFCollisionChecker(bounds=bounds, obstacles=obs_easy,    min_clearance=CLEARANCE)
    ccP_passage = PointRobot2DoFCollisionChecker(bounds=bounds, obstacles=obs_passage, min_clearance=CLEARANCE)

    bP_easy = _make_benchmark(
        "Point 2-DoF (Easy)", ccP_easy,
        start=[-2.6,  2.2], goal=[ 2.6, -2.2],
        description=f"Point 2DoF easy (clearance={CLEARANCE:.2f})", level=1
    )
    bP_passage = _make_benchmark(
        "Point 2-DoF (Passage)", ccP_passage,
        start=[-2.6,  0.0], goal=[ 2.6,  0.0],
        description=f"Point 2DoF passage (clearance={CLEARANCE:.2f})", level=2
    )

    # C) Planar 2DoF (2R) – Clearance aktiv (WICHTIG: für BEIDE Benchmarks setzen)
    theta_limits = [(-math.pi, math.pi), (-math.pi, math.pi)]

    ccA_easy = PlanarRobot2DoFCollisionChecker(
        theta_limits=theta_limits,
        obstacles=obs_easy,
        base=(0.0, 0.0),
        L1=1.6,
        L2=1.2,
        min_clearance=CLEARANCE
    )
    ccA_passage = PlanarRobot2DoFCollisionChecker(
        theta_limits=theta_limits,
        obstacles=obs_passage,
        base=(0.0, 0.0),
        L1=1.6,
        L2=1.2,
        min_clearance=CLEARANCE
    )

    bA_easy = _make_benchmark(
        "Manipulator 2-DoF (Easy)", ccA_easy,
        start=[2.0, 0.5], goal=[-2.0, -0.5],
        description=f"Planar 2DoF easy (clearance={CLEARANCE:.2f})", level=1
    )
    bA_passage = _make_benchmark(
        "Manipulator 2-DoF (Passage)", ccA_passage,
        start=[math.pi, 0.0], goal=[0.0, 0.0],
        description=f"Planar 2DoF passage (clearance={CLEARANCE:.2f})", level=2
    )

    return [bP_easy, bP_passage, bA_easy, bA_passage]


def sample_free_angles(cc, max_tries=20000):
    lims = cc.getEnvironmentLimits()
    for _ in range(max_tries):
        q = [random.uniform(*lims[0]), random.uniform(*lims[1])]
        if not cc.pointInCollision(q):
            return q
    raise RuntimeError("No collision-free angles found (max_tries reached).")


def ensure_valid_start_goal(b, cc, max_tries=20000):
    """
    Stellt sicher, dass Start und Goal kollisionsfrei sind.
    Wenn nicht, wird jeweils ein neuer Start/Goal im Winkelraum gesampelt.
    Aktualisiert b.startList/b.goalList sowie b.start/b.goal.
    """
    start_list = getattr(b, "startList", [b.start])
    goal_list  = getattr(b, "goalList",  [b.goal])

    # START fixen
    if cc.pointInCollision(start_list[0]):
        print("Start ist in Kollision -> sample neuen kollisionsfreien Start ...")
        qS = sample_free_angles(cc, max_tries=max_tries)
        b.startList = [qS]
        b.start = qS
        start_list = [qS]
        print("Neuer Start:", qS, "collision?", cc.pointInCollision(qS))

    # GOAL fixen
    if cc.pointInCollision(goal_list[0]):
        print("Goal ist in Kollision -> sample neues kollisionsfreies Goal ...")
        qG = sample_free_angles(cc, max_tries=max_tries)
        b.goalList = [qG]
        b.goal = qG
        goal_list = [qG]
        print("Neues Goal:", qG, "collision?", cc.pointInCollision(qG))

    return start_list, goal_list


# ============================================================
# 5) Runner-Utilities
# ============================================================
def run_one_2b(planner_class, benchmark, config, seed=None):
    import time
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    if hasattr(benchmark.collisionChecker, "counter"):
        benchmark.collisionChecker.counter = 0

    planner = planner_class(benchmark.collisionChecker)

    t0 = time.time()
    err = None
    path = []
    try:
        start_list = getattr(benchmark, "startList", [benchmark.start])
        goal_list  = getattr(benchmark, "goalList",  [benchmark.goal])
        path = planner.planPath(start_list, goal_list, config)
        ok = bool(path) and len(path) >= 2
    except Exception as e:
        ok = False
        err = str(e)

    dt = time.time() - t0
    return {
        "success": ok,
        "time_s": float(dt),
        "collision_checks": int(getattr(benchmark.collisionChecker, "counter", -1)),
        "path": path,
        "err": err,
        "planner": planner,
    }


def path_positions_from_graph(graph, path_nodes):
    if not path_nodes:
        return []
    out = []
    for n in path_nodes:
        if n not in graph.nodes:
            continue
        pos = graph.nodes[n].get("pos", None)
        if pos is None:
            continue
        out.append(list(pos))
    return out


# ============================================================
# 6) Animation / Visualisierung (Workspace + C-Space)
# ============================================================
def densify_path(path_xy, step=0.05):
    if path_xy is None or len(path_xy) < 2:
        return np.asarray(path_xy, float) if path_xy is not None else np.zeros((0, 2))
    pts = np.asarray(path_xy, dtype=float)
    dense = [pts[0]]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        dist = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(dist / step)))
        for k in range(1, n + 1):
            t = k / n
            dense.append((1 - t) * a + t * b)
    return np.asarray(dense)


def _graph_pos(G):
    return nx.get_node_attributes(G, "pos")


def _draw_polygon(ax, poly, alpha=0.25):
    if poly is None:
        return
    if getattr(poly, "is_empty", False):
        return
    if poly.geom_type == "Polygon":
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=alpha)
    elif poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            x, y = p.exterior.xy
            ax.fill(x, y, alpha=alpha)


def _draw_obstacles(ax, obstacles, alpha=0.25):
    if obstacles is None:
        return
    for obs in obstacles:
        _draw_polygon(ax, obs, alpha=alpha)


def _draw_obstacles_inflated(ax, obstacles, clearance, alpha=0.10):
    """
    Visualisiert die Sicherheitszone (Obstacle buffer(clearance)).
    Nur Visualisierung; die echte Kollision macht der CollisionChecker.
    """
    if obstacles is None:
        return
    clearance = float(clearance)
    if clearance <= 0:
        return
    for obs in obstacles:
        infl = obs.buffer(clearance)
        _draw_polygon(ax, infl, alpha=alpha)


def _plot_roadmap(ax, G, edge_alpha=0.12, node_size=18):
    pos = _graph_pos(G)
    for u, v in G.edges():
        if u in pos and v in pos:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax.plot(x, y, linewidth=1, alpha=edge_alpha)
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    ax.scatter(xs, ys, s=node_size)


def _plot_path(ax, path_xy, lw=3, alpha=1.0):
    if path_xy is None or len(path_xy) < 2:
        return
    p = np.asarray(path_xy, float)
    ax.plot(p[:, 0], p[:, 1], linewidth=lw, alpha=alpha)


def _fk_2r(q, L1, L2, base=(0.0, 0.0)):
    t1, t2 = float(q[0]), float(q[1])
    x0, y0 = base
    x1 = x0 + L1 * math.cos(t1)
    y1 = y0 + L1 * math.sin(t1)
    x2 = x1 + L2 * math.cos(t1 + t2)
    y2 = y1 + L2 * math.sin(t1 + t2)
    return (x0, y0), (x1, y1), (x2, y2)


def animate_workspace_and_cspace(
    benchmark,
    planner,
    path_q,
    q_limits,
    ws_limits,
    robot_kind="arm",        # "arm" oder "point"
    interval_ms=40,
    densify_step=0.03,
    arm_L1=1.6,
    arm_L2=1.2,
    show_clearance=False,
    start_q=None,
    goal_q=None
):
    cc = benchmark.collisionChecker
    obstacles = getattr(cc, "obstacles", None)
    clearance = float(getattr(cc, "min_clearance", 0.0) or 0.0)

    # Pfad verdichten
    path_q = densify_path(path_q, step=densify_step)

    fig, (ax_ws, ax_q) = plt.subplots(1, 2, figsize=(12, 5))
    ax_ws.set_aspect("equal", adjustable="box")

    xmin, xmax, ymin, ymax = ws_limits
    ax_ws.set_xlim(xmin, xmax)
    ax_ws.set_ylim(ymin, ymax)

    (q1min, q1max), (q2min, q2max) = q_limits
    ax_q.set_xlim(q1min, q1max)
    ax_q.set_ylim(q2min, q2max)
    ax_q.set_aspect("equal", adjustable="box")

    ax_ws.set_title(f"Workspace: {getattr(benchmark, 'name', 'Benchmark')}")
    ax_q.set_title("Configuration Space")

    # Obstacles im Workspace
    _draw_obstacles(ax_ws, obstacles, alpha=0.25)
    if show_clearance and clearance > 0.0:
        _draw_obstacles_inflated(ax_ws, obstacles, clearance, alpha=0.10)

    # Roadmap & Pfad im C-space
    _plot_roadmap(ax_q, planner.graph, edge_alpha=0.10, node_size=14)
    _plot_path(ax_q, path_q, lw=3, alpha=1.0)

    # Marker in C-space (bewegter Punkt)
    dot_q, = ax_q.plot([], [], marker="o", markersize=7, linestyle="None")

    # Start/Goal im C-space anzeigen
    if start_q is not None:
        ax_q.scatter([start_q[0]], [start_q[1]], s=90, marker="o", label="Start (C)")
    if goal_q is not None:
        ax_q.scatter([goal_q[0]], [goal_q[1]], s=90, marker="X", label="Goal (C)")

    if robot_kind == "point":
        # Point robot: Workspace == C-space
        if start_q is not None:
            ax_ws.scatter([start_q[0]], [start_q[1]], s=90, marker="o", label="Start (W)")
        if goal_q is not None:
            ax_ws.scatter([goal_q[0]], [goal_q[1]], s=90, marker="X", label="Goal (W)")

        dot_ws, = ax_ws.plot([], [], marker="o", markersize=8, linestyle="None")
        _plot_path(ax_ws, path_q, lw=3, alpha=0.6)

        ax_ws.legend(loc="upper right")
        ax_q.legend(loc="upper right")

        def init():
            dot_q.set_data([], [])
            dot_ws.set_data([], [])
            return (dot_q, dot_ws)

        def update(i):
            q = path_q[i]
            dot_q.set_data([q[0]], [q[1]])
            dot_ws.set_data([q[0]], [q[1]])
            return (dot_q, dot_ws)

        ani = FuncAnimation(fig, update, frames=len(path_q), init_func=init,
                            interval=interval_ms, blit=True)
        plt.close(fig)
        return ani

    # Arm im Workspace
    link1_line, = ax_ws.plot([], [], linewidth=4)
    link2_line, = ax_ws.plot([], [], linewidth=4)
    ee_dot,     = ax_ws.plot([], [], marker="o", markersize=6, linestyle="None")

    base_xy = getattr(cc, "base", (0.0, 0.0))

    # Start/Goal im WORKSPACE anzeigen (Endeffektor)
    if start_q is not None:
        _, _, p2s = _fk_2r(start_q, arm_L1, arm_L2, base=base_xy)
        ax_ws.scatter([p2s[0]], [p2s[1]], s=90, marker="o", label="Start")

    if goal_q is not None:
        _, _, p2g = _fk_2r(goal_q, arm_L1, arm_L2, base=base_xy)
        ax_ws.scatter([p2g[0]], [p2g[1]], s=90, marker="X", label="Goal")

    ax_ws.legend(loc="upper right")
    ax_q.legend(loc="upper right")

    def init():
        dot_q.set_data([], [])
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        ee_dot.set_data([], [])
        return (dot_q, link1_line, link2_line, ee_dot)

    def update(i):
        q = path_q[i]
        dot_q.set_data([q[0]], [q[1]])

        p0, p1, p2 = _fk_2r(q, arm_L1, arm_L2, base=base_xy)
        link1_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        link2_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        ee_dot.set_data([p2[0]], [p2[1]])
        return (dot_q, link1_line, link2_line, ee_dot)

    ani = FuncAnimation(fig, update, frames=len(path_q), init_func=init,
                        interval=interval_ms, blit=True)
    plt.close(fig)
    return ani


def set_benchmark_start_goal(b, start, goal, check=True):
    b.startList = [list(start)]
    b.goalList  = [list(goal)]
    b.start = list(start)
    b.goal  = list(goal)

    if check:
        cc = b.collisionChecker
        if cc.pointInCollision(b.startList[0]):
            raise ValueError(f"{getattr(b, 'name','Benchmark')}: Start kollidiert: {b.startList[0]}")
        if cc.pointInCollision(b.goalList[0]):
            raise ValueError(f"{getattr(b, 'name','Benchmark')}: Goal kollidiert: {b.goalList[0]}")
