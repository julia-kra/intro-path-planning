# point_robot_anim.py
from __future__ import annotations

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import matplotlib.patches as patches

from IPython.display import HTML, display


# ============================================================
# OOB-Guard (wie Evaluation)
# ============================================================
def ensure_oob_is_collision(collisionChecker):
    if getattr(collisionChecker, "_oob_guard_enabled", False):
        return

    orig_point_in_collision = collisionChecker.pointInCollision

    def wrapped_point_in_collision(pos, _cc=collisionChecker, _orig=orig_point_in_collision):
        (xmin, xmax), (ymin, ymax) = _cc.getEnvironmentLimits()
        if pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax:
            if hasattr(_cc, "counter"):
                _cc.counter += 1
            return True
        return _orig(pos)

    collisionChecker.pointInCollision = wrapped_point_in_collision
    collisionChecker._oob_guard_enabled = True


# ============================================================
# Helpers fürs Zeichnen/Animation (Prof-Stil)
# ============================================================
def interpolate_line(a, b, step=0.5):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dist = float(np.linalg.norm(b - a))
    if dist < 1e-12:
        return [a.tolist()]
    n = max(1, int(np.ceil(dist / float(step))))
    return [(a + (i / n) * (b - a)).tolist() for i in range(n + 1)]


def get_ws_limits(bench):
    (xmin, xmax), (ymin, ymax) = bench.collisionChecker.getEnvironmentLimits()
    return (xmin, xmax, ymin, ymax)


def draw_obstacles_from_scene(ax, cc, alpha=0.25, edgecolor="white", linewidth=2):
    scene = getattr(cc, "scene", {})
    if not isinstance(scene, dict) or not scene:
        return

    for geom in scene.values():
        if geom is None or getattr(geom, "is_empty", False):
            continue

        gtype = getattr(geom, "geom_type", None)
        if gtype == "Polygon":
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
        elif gtype == "MultiPolygon":
            for poly in geom.geoms:
                xs, ys = poly.exterior.xy
                ax.fill(xs, ys, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
        else:
            # best-effort fallback
            try:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
            except Exception:
                pass


def visualize_prm_in_axis(planner, solution_nodes, ax, roadmap_alpha=0.15, node_alpha=0.15):
    G = planner.graph
    pos = nx.get_node_attributes(G, "pos")
    if not pos:
        return

    # Roadmap edges
    for u, v in G.edges():
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], alpha=roadmap_alpha)

    # Roadmap nodes
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=8, alpha=node_alpha)

    # Solution polyline
    if solution_nodes:
        sol_xy = [pos[n] for n in solution_nodes if n in pos]
        if len(sol_xy) >= 2:
            ax.plot([p[0] for p in sol_xy], [p[1] for p in sol_xy], linewidth=3)


# ============================================================
# Robot-Parameter aus Config holen
# ============================================================
def robot_radius_from_mode_config(mode_cfg, cc=None, robot_visual="robot"):
    """
    robot_visual:
      - "robot":     zeichnet robotRadius
      - "clearance": zeichnet robotRadius + safetyMargin (oder cfg["clearance"])
    """
    cfg = mode_cfg or {}

    # 1) explizite Clearance, falls vorhanden
    if robot_visual == "clearance" and ("clearance" in cfg):
        return float(cfg["clearance"])

    # 2) robotRadius + safetyMargin
    r = float(cfg.get("robotRadius", 0.0) or 0.0)
    if robot_visual == "clearance":
        s = float(cfg.get("safetyMargin", 0.0) or 0.0)
        return r + s

    # 3) nur Roboterradius
    if r > 0.0:
        return r

    # 4) Fallback: CollisionChecker (falls min_clearance genutzt wird)
    if cc is not None and hasattr(cc, "min_clearance"):
        return float(getattr(cc, "min_clearance", 0.0) or 0.0)

    return 0.0


# ============================================================
# Animation: PlanPath-Aufruf exakt wie Evaluation
# ============================================================
def animate_point_robot_eval_exact(
    bench,
    planner_class,
    mode_name,
    seed,
    configs,
    interp_step=0.5,
    fig_size=(7, 7),
    obstacle_alpha=0.25,
    robot_visual="robot",     # "robot" oder "clearance"
    draw_robot_fill=False,
    robot_alpha=0.35,
    roadmap_alpha=0.15,
    node_alpha=0.15,
    show_debug=False,
):
    cc = bench.collisionChecker

    # exakt wie Evaluation:
    random.seed(int(seed))
    np.random.seed(int(seed))
    ensure_oob_is_collision(cc)

    planner = planner_class(cc)

    mode_cfg = configs.get(mode_name, {}) or {}
    robot_radius = robot_radius_from_mode_config(mode_cfg, cc=cc, robot_visual=robot_visual)

    sol = planner.planPath(bench.startList, bench.goalList, mode_cfg)

    if show_debug:
        pos = nx.get_node_attributes(planner.graph, "pos")
        sample_log = getattr(planner, "_sample_log", []) or []
        n_nodes = len(pos)
        n_edges = planner.graph.number_of_edges()

        src_corr = sum(1 for r in sample_log if r.get("source") == "start_goal_corridor")
        src_uni = sum(1 for r in sample_log if r.get("source") in ("uniform", "uniform_fallback"))

        print(
            f"[{bench.name} | {mode_name}] nodes={n_nodes} edges={n_edges} | "
            f"stats={getattr(planner,'_stats',None)} | "
            f"log_total={len(sample_log)} corr_src={src_corr} uni_src={src_uni} | "
            f"start={getattr(planner,'_start_for_mode4',None)} goal={getattr(planner,'_goal_for_mode4',None)}"
        )

    if not sol or len(sol) < 2:
        print(f"[{bench.name}] ❌ Kein Pfad gefunden (seed={seed}, mode={mode_name})")
        return None

    sol_pos = [planner.graph.nodes[n]["pos"] for n in sol]

    i_sol = [sol_pos[0]]
    for i in range(1, len(sol_pos)):
        i_sol += interpolate_line(sol_pos[i - 1], sol_pos[i], interp_step)[1:]

    ws_limits = get_ws_limits(bench)
    frames = len(i_sol)

    fig_local = plt.figure(figsize=fig_size)
    ax = fig_local.add_subplot(1, 1, 1)

    start = bench.startList[0]
    goal = bench.goalList[0]

    def _anim(t):
        ax.clear()
        xmin, xmax, ymin, ymax = ws_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        title = f"{bench.name} | {mode_name} | seed={seed}"
        if robot_radius > 0:
            title += f" | {robot_visual}_r={robot_radius:.2f}"
        ax.set_title(title, fontsize=12)

        draw_obstacles_from_scene(ax, cc, alpha=obstacle_alpha)

        ax.scatter([start[0]], [start[1]], s=120, c="green", zorder=5, label="Start")
        ax.scatter([goal[0]], [goal[1]], s=120, c="red", zorder=5, label="Goal")

        visualize_prm_in_axis(
            planner, sol, ax=ax,
            roadmap_alpha=roadmap_alpha,
            node_alpha=node_alpha
        )

        p = i_sol[t]
        ax.scatter([p[0]], [p[1]], s=30, c="blue", zorder=7)

        if robot_radius > 0:
            circ = patches.Circle(
                (float(p[0]), float(p[1])),
                radius=float(robot_radius),
                fill=bool(draw_robot_fill),
                alpha=float(robot_alpha) if draw_robot_fill else 1.0,
                linewidth=2.0,
                edgecolor="blue",
                facecolor="blue" if draw_robot_fill else "none",
                zorder=6,
                label="Robot shape"
            )
            ax.add_patch(circ)

        ax.legend(loc="upper right")

    ani = animation.FuncAnimation(fig_local, _anim, frames=frames)
    display(HTML(ani.to_jshtml()))
    plt.close(fig_local)
    return ani


# ============================================================
# Convenience: Evaluation -> SeedMap -> Animation pro Benchmark
# ============================================================
def animate_benchmarks_from_seedmap(
    bench_list,                       # list[(label, bench_obj)]
    mode_order,
    seed_map,
    configs,
    planner_class,
    interp_step=0.5,
    fig_size=(7, 7),
    obstacle_alpha=0.25,
    robot_visual="robot",
    draw_robot_fill=False,
    robot_alpha=0.25,
    roadmap_alpha=0.15,
    node_alpha=0.15,
    show_debug=False,
):
    """
    Animiert für alle Benchmarks und Modi die "besten Seeds" aus seed_map[(bench_name, mode)].
    """
    for bench_label, bench_obj in bench_list:
        bname = bench_obj.name
        print("\n" + "#" * 90)
        print("BENCHMARK:", bench_label, "|", bname)
        print("#" * 90)

        for mode_name in mode_order:
            seed = seed_map.get((bname, mode_name), None)
            if seed is None:
                print(f"SKIP: {bname} | {mode_name} (kein erfolgreicher Run in df)")
                continue

            print("\n" + "=" * 70)
            print(f"ANIMATION: {bench_label} | {mode_name} | seed={seed}")

            animate_point_robot_eval_exact(
                bench=bench_obj,
                planner_class=planner_class,
                mode_name=mode_name,
                seed=seed,
                configs=configs,
                interp_step=interp_step,
                fig_size=fig_size,
                obstacle_alpha=obstacle_alpha,
                robot_visual=robot_visual,
                draw_robot_fill=draw_robot_fill,
                robot_alpha=robot_alpha,
                roadmap_alpha=roadmap_alpha,
                node_alpha=node_alpha,
                show_debug=show_debug,
            )
