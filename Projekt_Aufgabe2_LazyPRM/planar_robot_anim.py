# planar_robot_anim.py
from __future__ import annotations

import math
import random
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import networkx as nx

from IPython.display import HTML, display


# ============================================================
# Densify / Graph utils
# ============================================================
def densify_path(path_xy, step: float = 0.05) -> np.ndarray:
    if path_xy is None or len(path_xy) < 2:
        return np.asarray(path_xy) if path_xy is not None else np.zeros((0, 2))
    pts = np.asarray(path_xy, dtype=float)
    dense = [pts[0]]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        dist = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(dist / float(step))))
        for k in range(1, n + 1):
            t = k / n
            dense.append((1 - t) * a + t * b)
    return np.asarray(dense, dtype=float)


def path_positions_from_graph(graph: nx.Graph, path_nodes: List[Any]) -> List[List[float]]:
    if not path_nodes:
        return []
    out = []
    for n in path_nodes:
        if n not in graph.nodes:
            continue
        pos = graph.nodes[n].get("pos", None)
        if pos is None:
            continue
        out.append([float(pos[0]), float(pos[1])])
    return out


def _graph_pos(G: nx.Graph) -> Dict[Any, List[float]]:
    return nx.get_node_attributes(G, "pos")


# ============================================================
# Drawing helpers
# ============================================================
def _draw_polygon(ax, poly, alpha: float = 0.25):
    if poly is None:
        return
    if getattr(poly, "is_empty", False):
        return
    gtype = getattr(poly, "geom_type", "")
    if gtype == "Polygon":
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=alpha)
    elif gtype == "MultiPolygon":
        for p in poly.geoms:
            x, y = p.exterior.xy
            ax.fill(x, y, alpha=alpha)


def draw_obstacles(ax, obstacles, alpha: float = 0.25):
    if obstacles is None:
        return
    for obs in obstacles:
        _draw_polygon(ax, obs, alpha=alpha)


def draw_obstacles_inflated(ax, obstacles, clearance: float, alpha: float = 0.10):
    """
    Visualisiert die Sicherheitszone (obs.buffer(clearance)).
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


def plot_roadmap(ax, G: nx.Graph, edge_alpha: float = 0.12, node_size: int = 18):
    pos = _graph_pos(G)
    for u, v in G.edges():
        if u in pos and v in pos:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                linewidth=1,
                alpha=edge_alpha,
            )
    if pos:
        xs = [pos[n][0] for n in pos]
        ys = [pos[n][1] for n in pos]
        ax.scatter(xs, ys, s=node_size)


def plot_path(ax, path_xy, lw: int = 3, alpha: float = 1.0):
    if path_xy is None or len(path_xy) < 2:
        return
    p = np.asarray(path_xy, dtype=float)
    ax.plot(p[:, 0], p[:, 1], linewidth=lw, alpha=alpha)


# ============================================================
# FK for planar 2R
# ============================================================
def fk_2r(q, L1: float, L2: float, base=(0.0, 0.0)):
    t1, t2 = float(q[0]), float(q[1])
    x0, y0 = float(base[0]), float(base[1])
    x1 = x0 + float(L1) * math.cos(t1)
    y1 = y0 + float(L1) * math.sin(t1)
    x2 = x1 + float(L2) * math.cos(t1 + t2)
    y2 = y1 + float(L2) * math.sin(t1 + t2)
    return (x0, y0), (x1, y1), (x2, y2)


# ============================================================
# Start/Goal helpers (optional)
# ============================================================
def set_benchmark_start_goal(benchmark, start_q, goal_q, check: bool = True):
    """
    Setzt Start/Goal als C-space Winkel (Radiant).
    """
    benchmark.startList = [list(start_q)]
    benchmark.goalList = [list(goal_q)]
    benchmark.start = list(start_q)
    benchmark.goal = list(goal_q)

    if check:
        cc = benchmark.collisionChecker
        if cc.pointInCollision(benchmark.startList[0]):
            raise ValueError(f"{getattr(benchmark,'name','bench')}: Start kollidiert: {benchmark.startList[0]}")
        if cc.pointInCollision(benchmark.goalList[0]):
            raise ValueError(f"{getattr(benchmark,'name','bench')}: Goal kollidiert: {benchmark.goalList[0]}")


def sample_free_angles(cc, max_tries: int = 20000):
    lims = cc.getEnvironmentLimits()
    for _ in range(int(max_tries)):
        q = [random.uniform(*lims[0]), random.uniform(*lims[1])]
        if not cc.pointInCollision(q):
            return q
    raise RuntimeError("No collision-free angles found (max_tries reached).")


def ensure_valid_start_goal(benchmark, cc, max_tries: int = 20000, verbose: bool = True):
    """
    Wenn Start oder Goal kollidiert: resample kollisionsfrei.
    Achtung: Das ist eine 'Fix'-Funktion. Im Report solltest du erwähnen,
    wenn du damit Start/Goal geändert hast.
    """
    start_list = getattr(benchmark, "startList", [getattr(benchmark, "start", None)])
    goal_list = getattr(benchmark, "goalList", [getattr(benchmark, "goal", None)])

    if start_list[0] is None or goal_list[0] is None:
        raise ValueError("Benchmark has no start/goal set.")

    # START
    if cc.pointInCollision(start_list[0]):
        if verbose:
            print("Start ist in Kollision -> sample neuen kollisionsfreien Start ...")
        qS = sample_free_angles(cc, max_tries=max_tries)
        benchmark.startList = [qS]
        benchmark.start = qS
        start_list = [qS]
        if verbose:
            print("Neuer Start:", qS, "collision?", cc.pointInCollision(qS))

    # GOAL
    if cc.pointInCollision(goal_list[0]):
        if verbose:
            print("Goal ist in Kollision -> sample neues kollisionsfreies Goal ...")
        qG = sample_free_angles(cc, max_tries=max_tries)
        benchmark.goalList = [qG]
        benchmark.goal = qG
        goal_list = [qG]
        if verbose:
            print("Neues Goal:", qG, "collision?", cc.pointInCollision(qG))

    return start_list, goal_list


# ============================================================
# Core animation (Workspace + C-space)
# ============================================================
def animate_workspace_and_cspace(
    benchmark,
    planner,
    path_q,  # list[[q1,q2]] in radians
    q_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    ws_limits: Tuple[float, float, float, float],
    interval_ms: int = 40,
    densify_step: float = 0.03,
    arm_L1: float = 1.6,
    arm_L2: float = 1.2,
    show_clearance: bool = False,
    start_q: Optional[List[float]] = None,
    goal_q: Optional[List[float]] = None,
    title_ws: Optional[str] = None,
    title_q: str = "Configuration Space",
):
    cc = benchmark.collisionChecker
    obstacles = getattr(cc, "obstacles", None)
    clearance = float(getattr(cc, "min_clearance", 0.0) or 0.0)
    base_xy = getattr(cc, "base", (0.0, 0.0))

    path_q = densify_path(path_q, step=densify_step)
    if len(path_q) < 2:
        print(f"[{getattr(benchmark,'name','bench')}] ❌ path_q too short for animation.")
        return None

    fig, (ax_ws, ax_q) = plt.subplots(1, 2, figsize=(12, 5))
    ax_ws.set_aspect("equal", adjustable="box")

    xmin, xmax, ymin, ymax = ws_limits
    ax_ws.set_xlim(xmin, xmax)
    ax_ws.set_ylim(ymin, ymax)

    (q1min, q1max), (q2min, q2max) = q_limits
    ax_q.set_xlim(q1min, q1max)
    ax_q.set_ylim(q2min, q2max)
    ax_q.set_aspect("equal", adjustable="box")

    # Titles
    if title_ws is None:
        title_ws = f"Workspace: {getattr(benchmark,'name','Planar')}"
    ax_ws.set_title(title_ws)
    ax_q.set_title(title_q)

    # Workspace obstacles
    draw_obstacles(ax_ws, obstacles, alpha=0.25)
    if show_clearance and clearance > 0.0:
        draw_obstacles_inflated(ax_ws, obstacles, clearance, alpha=0.10)

    # C-space roadmap + path
    plot_roadmap(ax_q, planner.graph, edge_alpha=0.10, node_size=14)
    plot_path(ax_q, path_q, lw=3, alpha=1.0)

    # Start/Goal in C-space
    if start_q is None:
        start_q = getattr(benchmark, "startList", [getattr(benchmark, "start", None)])[0]
    if goal_q is None:
        goal_q = getattr(benchmark, "goalList", [getattr(benchmark, "goal", None)])[0]

    if start_q is not None:
        ax_q.scatter([start_q[0]], [start_q[1]], s=90, marker="o", label="Start (C)")
    if goal_q is not None:
        ax_q.scatter([goal_q[0]], [goal_q[1]], s=90, marker="X", label="Goal (C)")
    ax_q.legend(loc="upper right")

    # Workspace start/goal: show EE positions for start/goal
    if start_q is not None:
        _, _, p2s = fk_2r(start_q, arm_L1, arm_L2, base=base_xy)
        ax_ws.scatter([p2s[0]], [p2s[1]], s=90, marker="o", label="Start (EE)")
    if goal_q is not None:
        _, _, p2g = fk_2r(goal_q, arm_L1, arm_L2, base=base_xy)
        ax_ws.scatter([p2g[0]], [p2g[1]], s=90, marker="X", label="Goal (EE)")

    # Animated items
    dot_q, = ax_q.plot([], [], marker="o", markersize=7, linestyle="None")

    link1_line, = ax_ws.plot([], [], linewidth=4)
    link2_line, = ax_ws.plot([], [], linewidth=4)
    ee_dot,     = ax_ws.plot([], [], marker="o", markersize=6, linestyle="None")

    ax_ws.legend(loc="upper right")

    def init():
        dot_q.set_data([], [])
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        ee_dot.set_data([], [])
        return (dot_q, link1_line, link2_line, ee_dot)

    def update(i: int):
        q = path_q[i]
        dot_q.set_data([q[0]], [q[1]])

        p0, p1, p2 = fk_2r(q, arm_L1, arm_L2, base=base_xy)
        link1_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        link2_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        ee_dot.set_data([p2[0]], [p2[1]])
        return (dot_q, link1_line, link2_line, ee_dot)

    ani = FuncAnimation(fig, update, frames=len(path_q), init_func=init,
                        interval=int(interval_ms), blit=True)
    plt.close(fig)
    return ani


# ============================================================
# High-level driver: animate per benchmark & mode from seed map
# ============================================================
def animate_planar_from_seedmap(
    bench_list,  # list[(label, bench_obj)]
    mode_order: List[str],
    seed_map: Dict[Tuple[str, str], int],
    configs: Dict[str, dict],
    planner_class,
    q_limits=(( -math.pi, math.pi), (-math.pi, math.pi)),
    ws_limits=(-3.0, 3.0, -3.0, 3.0),
    interp_densify_step: float = 0.03,
    interval_ms: int = 30,
    show_clearance: bool = False,
    fix_start_goal_if_invalid: bool = False,
    fix_max_tries: int = 20000,
    show_debug: bool = False,
):
    """
    - seeds kommen aus seed_map[(bench.name, mode)]
    - pro (bench, mode) wird ein frischer planner instanziiert und planPath() ausgeführt
    - anschließend wird Workspace+C-Space animiert
    """
    for bench_label, b in bench_list:
        bname = getattr(b, "name", bench_label)
        cc = b.collisionChecker

        print("\n" + "#" * 90)
        print("BENCHMARK:", bench_label, "|", bname)
        print("#" * 90)

        for mode in mode_order:
            seed = seed_map.get((bname, mode), None)
            if seed is None:
                print(f"SKIP: {bname} | {mode} (kein erfolgreicher Run in df)")
                continue

            # Seeding
            random.seed(int(seed))
            np.random.seed(int(seed))

            # Optional: start/goal fix
            if fix_start_goal_if_invalid:
                ensure_valid_start_goal(b, cc, max_tries=fix_max_tries, verbose=True)

            # Build config copy, include overrides (safe for mode4)
            cfg = dict(configs.get(mode, {}) or {})
            start_q = getattr(b, "startList", [getattr(b, "start", None)])[0]
            goal_q  = getattr(b, "goalList",  [getattr(b, "goal", None)])[0]
            if start_q is not None:
                cfg["start_override"] = start_q
            if goal_q is not None:
                cfg["goal_override"] = goal_q

            # Plan
            planner = planner_class(cc)
            path_nodes = planner.planPath(getattr(b, "startList", [b.start]), getattr(b, "goalList", [b.goal]), cfg)

            ok = bool(path_nodes) and len(path_nodes) >= 2
            print("\n" + "=" * 70)
            print(f"ANIMATION: {bench_label} | {mode} | seed={seed} | success={ok} | path_nodes={len(path_nodes) if path_nodes else 0}")

            if show_debug:
                G = planner.graph
                print("Graph nodes:", G.number_of_nodes(), "| edges:", G.number_of_edges())
                print("Start collision?", cc.pointInCollision(start_q) if start_q is not None else None)
                print("Goal  collision?", cc.pointInCollision(goal_q) if goal_q is not None else None)

            if not ok:
                continue

            path_q = path_positions_from_graph(planner.graph, path_nodes)

            ani = animate_workspace_and_cspace(
                benchmark=b,
                planner=planner,
                path_q=path_q,
                q_limits=q_limits,
                ws_limits=ws_limits,
                interval_ms=interval_ms,
                densify_step=interp_densify_step,
                arm_L1=float(getattr(cc, "L1", 1.6)),
                arm_L2=float(getattr(cc, "L2", 1.2)),
                show_clearance=show_clearance,
                start_q=start_q,
                goal_q=goal_q,
                title_ws=f"Workspace: {bname} | {mode} | seed={seed}",
                title_q="Configuration Space",
            )

            if ani is not None:
                display(HTML(ani.to_jshtml()))
