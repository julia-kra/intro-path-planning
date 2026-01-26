# eval_task2.py
from __future__ import annotations

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# 1) Evaluation Helpers
# ============================================================
def path_length_from_graph(graph, path):
    if not path:
        return np.nan
    pts = [graph.nodes[p]["pos"] for p in path]
    return float(sum(
        np.linalg.norm(np.array(pts[i]) - np.array(pts[i + 1]))
        for i in range(len(pts) - 1)
    ))


def ensure_oob_is_collision(collisionChecker):
    """
    Out-of-bounds => Collision hart machen (falls CC das nicht selbst tut).
    Patcht collisionChecker.pointInCollision einmalig.
    """
    if getattr(collisionChecker, "_oob_guard_enabled", False):
        return

    orig_point_in_collision = collisionChecker.pointInCollision

    def wrapped_point_in_collision(pos, _cc=collisionChecker, _orig=orig_point_in_collision):
        lims = _cc.getEnvironmentLimits()
        (xmin, xmax), (ymin, ymax) = lims
        if pos[0] < xmin or pos[0] > xmax or pos[1] < ymin or pos[1] > ymax:
            if hasattr(_cc, "counter"):
                _cc.counter += 1
            return True
        return _orig(pos)

    collisionChecker.pointInCollision = wrapped_point_in_collision
    collisionChecker._oob_guard_enabled = True


# ============================================================
# 2) Run helpers
# ============================================================
def run_one(planner_class, benchmark, config, seed=None):
    """
    Führt genau einen Run aus. Robust gegen Exceptions.
    Zusätzlich: zählt Seeds (Mode 1/2) und "Nodes aus Seeds" (Mode 1/2), damit wir
    später einen Plot-Run wählen können, in dem Seeds sichtbar sind.
    """
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)

    ensure_oob_is_collision(benchmark.collisionChecker)

    planner = planner_class(benchmark.collisionChecker)

    # optionaler Counter reset
    if hasattr(benchmark.collisionChecker, "counter"):
        benchmark.collisionChecker.counter = 0

    t0 = time.time()
    err = None
    path = []
    try:
        path = planner.planPath(benchmark.startList, benchmark.goalList, config)
    except Exception as e:
        err = str(e)
        path = []
    t1 = time.time()

    collisions = getattr(benchmark.collisionChecker, "counter", np.nan)

    # Graph-Metriken
    try:
        size = len(planner.graph.nodes())
    except Exception:
        size = np.nan

    try:
        length = path_length_from_graph(planner.graph, path)
    except Exception:
        length = np.nan

    success = (path is not None) and (path != [])

    # Seeds und Seed-Nodes zählen
    seed_used_count = len(getattr(planner, "_seed_points_used", []) or [])
    sample_log = getattr(planner, "_sample_log", []) or []
    seed_sampled_nodes_count = sum(
        1 for r in sample_log
        if (r.get("seed_key") is not None) and (r.get("mode") in ("mode1_seed_gauss", "mode2_seed_dist"))
    )

    return {
        "time_s": (t1 - t0),
        "collision_checks": collisions,
        "path_length": length,
        "roadmap_size": size,
        "success": success,
        "error": err,
        "seed_used_count": int(seed_used_count),
        "seed_sampled_nodes_count": int(seed_sampled_nodes_count),
    }


def _modes_for_benchmark(bench_name, configs, bench_mode_map=None):
    if bench_mode_map is None:
        return list(configs.keys())
    return bench_mode_map.get(bench_name, list(configs.keys()))


def run_suite(planner_class, benchmarks, configs, runs=30, base_seed=1234, progress_every=10, bench_mode_map=None):
    rows = []

    total = 0
    for b in benchmarks:
        total += len(_modes_for_benchmark(b.name, configs, bench_mode_map)) * runs

    done = 0
    t0 = time.time()

    for b in benchmarks:
        mode_list = _modes_for_benchmark(b.name, configs, bench_mode_map)

        for mode_name in mode_list:
            cfg = configs[mode_name]

            for r in range(runs):
                seed = int(base_seed + 1000 * r)
                metrics = run_one(planner_class, b, cfg, seed=seed)

                rows.append({
                    "benchmark": b.name,
                    "mode": mode_name,
                    "run": r,
                    "seed": seed,
                    **metrics
                })

                done += 1
                if progress_every and (done % progress_every == 0):
                    elapsed = time.time() - t0
                    print(f"{done}/{total} done, elapsed {elapsed:.1f}s")

    return pd.DataFrame(rows)


# ============================================================
# 3) Aggregation (Summary)
# ============================================================
def summarize(df):
    df_metrics = df.copy()

    metric_cols = ["time_s", "collision_checks", "path_length", "roadmap_size"]
    for c in metric_cols:
        df_metrics.loc[df_metrics["success"] == False, c] = np.nan

    summary = (df_metrics
        .groupby(["benchmark", "mode"])
        .agg(
            runs=("run", "count"),
            success_rate=("success", "mean"),
            success_count=("success", "sum"),

            time_mean=("time_s", "mean"),
            time_std=("time_s", "std"),

            coll_mean=("collision_checks", "mean"),
            coll_std=("collision_checks", "std"),

            len_mean=("path_length", "mean"),
            len_std=("path_length", "std"),

            size_mean=("roadmap_size", "mean"),
            size_std=("roadmap_size", "std"),

            n_errors=("error", lambda s: int(s.notna().sum()))
        )
        .reset_index()
    )
    return summary



# ============================================================
# 4) Plot Layout (1×5)
# ============================================================
METRIC_SPECS = [
    ("success_count", None,      "Erfolgreiche Runs", "seagreen",        "Erfolgreich (#/30)"),
    ("time_mean",     "time_std","Planungszeit (s)",  "gold",            "Planungszeit (s)"),
    ("coll_mean",     "coll_std","Kollisionschecks",  "tomato",          "Kollisionschecks"),
    ("len_mean",      "len_std", "Pfadlänge",         "cornflowerblue",  "Pfadlänge"),
    ("size_mean",     "size_std","Roadmap Größe",     "mediumpurple",    "Roadmap Größe (#Nodes)"),
]

def compute_global_ylims(summary_df, metric_specs, pad=0.10):
    ylims = {}
    for mean_col, std_col, *_ in metric_specs:
        mean_vals = summary_df[mean_col].to_numpy(dtype=float)
        std_vals  = summary_df[std_col].to_numpy(dtype=float)
        comb = mean_vals + np.nan_to_num(std_vals, nan=0.0)
        y_max = np.nanmax(comb) if np.any(np.isfinite(comb)) else 1.0
        if not np.isfinite(y_max) or y_max <= 0:
            y_max = 1.0
        ylims[mean_col] = (0.0, y_max * (1.0 + pad))
    return ylims


def plot_benchmark_like_task1(summary_df, benchmark_name, mode_order):
    sub = summary_df[summary_df["benchmark"] == benchmark_name].copy()

    # Reihenfolge der Modi gemäß mode_order (falls vorhanden)
    present = [m for m in mode_order if m in set(sub["mode"].values)]
    if present:
        sub["mode"] = pd.Categorical(sub["mode"], categories=present, ordered=True)
        sub = sub.sort_values("mode")
    else:
        sub = sub.sort_values("mode")

    labels = sub["mode"].astype(str).tolist()
    x = np.arange(len(labels), dtype=int)

    fig, axs = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f"Aufgabe 2: Vergleich Sampling-Modi – {benchmark_name}", fontsize=14)

    # Y-Limits nur für Metriken mit mean/std (success_count hat keine std)
    GLOBAL_YLIMS = compute_global_ylims(summary_df, METRIC_SPECS[1:], pad=0.10)

    runs_max = int(sub["runs"].max()) if "runs" in sub.columns else 30

    for ax, (mean_col, std_col, title, color, ylabel) in zip(axs, METRIC_SPECS):
        y = sub[mean_col].to_numpy(dtype=float)

        # Sonderfall: Erfolgsanzahl (keine Std) + Werte in Balken schreiben
        if mean_col == "success_count":
            bars = ax.bar(x, y, color=color, alpha=0.85)

            for rect, val in zip(bars, y):
                if np.isnan(val):
                    continue
                h = rect.get_height()

                # Textposition: bei 0 minimal über der Achse, sonst mittig im Balken
                y_text = (h * 0.5) if h > 0 else 0.3
                va = "center" if h > 0 else "bottom"

                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    y_text,
                    f"{int(val)}",
                    ha="center",
                    va=va,
                    fontsize=10,
                    color="black"
                )

            ax.set_ylim(0, runs_max)

        else:
            yerr = sub[std_col].to_numpy(dtype=float) if std_col else None
            yerr_plot = np.nan_to_num(yerr, nan=0.0) if yerr is not None else None
            ax.bar(x, y, yerr=yerr_plot, capsize=4, color=color, alpha=0.85)
            ax.set_ylim(*GLOBAL_YLIMS[mean_col])

        ax.set_title(title)
        ax.set_xlabel("Sampling-Modus")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")

    plt.tight_layout()
    plt.show()


# ============================================================
# 5) Best-Plot pro Mode & Benchmark (mit Sampling-Struktur)
# ============================================================
def _robust_minmax_norm(series, q_low=0.05, q_high=0.95):
    x = series.to_numpy(dtype=float)
    if np.all(~np.isfinite(x)):
        return pd.Series(np.zeros(len(series)), index=series.index)

    lo = np.nanquantile(x, q_low)
    hi = np.nanquantile(x, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)

    xn = (series - lo) / (hi - lo)
    xn = xn.clip(0.0, 1.0).fillna(1.0)
    return xn


def _pick_best_run_multicriteria(df_succ, w_time=0.6, w_len=0.25, w_size=0.15,
                                 q_low=0.05, q_high=0.95):
    d = df_succ.copy()

    for col in ["time_s", "path_length", "roadmap_size"]:
        if col not in d.columns:
            d[col] = np.nan

    d["time_n"] = _robust_minmax_norm(d["time_s"], q_low=q_low, q_high=q_high)
    d["len_n"]  = _robust_minmax_norm(d["path_length"], q_low=q_low, q_high=q_high)
    d["size_n"] = _robust_minmax_norm(d["roadmap_size"], q_low=q_low, q_high=q_high)
    d["score"] = w_time * d["time_n"] + w_len * d["len_n"] + w_size * d["size_n"]

    d = d.sort_values(["score", "time_s"], ascending=[True, True])
    return d.iloc[0]


def extract_best_seed_map(
    df,
    prefer_seed_runs_for_mode12=True,
    w_time=0.6, w_len=0.25, w_size=0.15
):
    """
    Gibt dict zurück: seed_map[(benchmark_name, mode_name)] = seed
    verwendet die gleiche Auswahlregel wie visualize_best_per_mode_and_benchmark().
    """
    seed_map = {}

    for (bench_name, mode), df_bm in df.groupby(["benchmark", "mode"]):
        df_succ = df_bm[df_bm["success"] == True].copy()
        if df_succ.empty:
            continue

        df_pick = df_succ
        if prefer_seed_runs_for_mode12 and mode in ("mode1_seed_gauss", "mode2_seed_dist"):
            df_seed = df_succ[df_succ["seed_used_count"] > 0].copy()
            if not df_seed.empty:
                df_pick = df_seed

        row = _pick_best_run_multicriteria(df_pick, w_time=w_time, w_len=w_len, w_size=w_size)
        seed_map[(bench_name, mode)] = int(row["seed"])

    return seed_map


def visualize_best_per_mode_and_benchmark(
    df, benchmarks, configs,
    planner_class,
    lazyPRMVisualize,
    nodeSize=80,
    mode_order=None, bench_order=None, figsize=(7, 7),
    w_time=0.6, w_len=0.25, w_size=0.15,
    prefer_seed_runs_for_mode12=True,
):
    """
    - Mode 1/2: Seed (X) + Nodes-aus-Seed + Seed->Node-Linien, nur wenn Seeds genutzt wurden
    - Mode 3: Kandidatenwolke + Selected (letzter Schritt)
    - Mode 4: Start–Goal Achse + Korridor-Nodes
    - Baseline: keine Extras
    """
    bench_by_name = {b.name: b for b in benchmarks}

    if bench_order is None:
        bench_order = [b.name for b in benchmarks]
    if mode_order is None:
        mode_order = list(configs.keys())

    for bench_name in bench_order:
        bench = bench_by_name[bench_name]

        modes_present = [m for m in mode_order if ((df["benchmark"] == bench_name) & (df["mode"] == m)).any()]
        for mode in modes_present:
            df_bm = df[(df["benchmark"] == bench_name) & (df["mode"] == mode)].copy()
            if df_bm.empty:
                continue

            df_succ = df_bm[df_bm["success"] == True].copy()

            df_pick = df_succ
            if prefer_seed_runs_for_mode12 and mode in ("mode1_seed_gauss", "mode2_seed_dist"):
                df_seed = df_succ[df_succ["seed_used_count"] > 0].copy()
                if not df_seed.empty:
                    df_pick = df_seed

            if not df_pick.empty:
                row = _pick_best_run_multicriteria(df_pick, w_time=w_time, w_len=w_len, w_size=w_size)
                tag = f"BEST (score={row['score']:.3f}; w_t={w_time}, w_l={w_len}, w_s={w_size})"
            else:
                row = df_bm.sort_values("run").iloc[-1]
                tag = "FALLBACK (no success)"

            run_idx = int(row["run"])
            seed = int(row["seed"])

            random.seed(seed)
            np.random.seed(seed)
            ensure_oob_is_collision(bench.collisionChecker)

            planner = planner_class(bench.collisionChecker)
            try:
                sol = planner.planPath(bench.startList, bench.goalList, configs[mode])
            except Exception as e:
                sol = []
                tag = f"{tag} – ERROR: {e}"

            fig, ax = plt.subplots(figsize=figsize)
            lazyPRMVisualize(planner, sol, ax=ax, nodeSize=nodeSize)

            (xmin, xmax), (ymin, ymax) = bench.collisionChecker.getEnvironmentLimits()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")

            pos = nx.get_node_attributes(planner.graph, "pos")

            # Mode 1/2: Seeds + Seed-Nodes + Links
            if mode in ("mode1_seed_gauss", "mode2_seed_dist"):
                seeds = getattr(planner, "_seed_points_used", []) or []
                sample_log = getattr(planner, "_sample_log", []) or []

                if seeds:
                    sx = [p[0] for p in seeds]
                    sy = [p[1] for p in seeds]
                    ax.scatter(
                        sx, sy, marker="x",
                        s=max(70, int(1.3 * nodeSize)),
                        linewidths=2.5,
                        zorder=30,
                        label="Seed used (Mode 1/2)"
                    )

                    seed_nodes = []
                    seed_links = []

                    seed_map = {}
                    for sp in seeds:
                        k = (round(float(sp[0]), 3), round(float(sp[1]), 3))
                        seed_map[k] = [float(sp[0]), float(sp[1])]

                    for r in sample_log:
                        if r.get("mode") != mode:
                            continue
                        k = r.get("seed_key", None)
                        if k is None:
                            continue
                        nid = int(r["node_id"])
                        if nid not in pos:
                            continue
                        seed_nodes.append(nid)
                        sp = seed_map.get(k, r.get("seed", None))
                        if sp is not None:
                            seed_links.append((sp, pos[nid]))

                    if seed_nodes:
                        nxs = [pos[n][0] for n in seed_nodes]
                        nys = [pos[n][1] for n in seed_nodes]
                        ax.scatter(
                            nxs, nys, marker="o",
                            s=max(35, int(0.8 * nodeSize)),
                            edgecolors="black",
                            linewidths=1.2,
                            zorder=29,
                            label="Nodes sampled from seed"
                        )
                        for sp, npnt in seed_links:
                            ax.plot([sp[0], npnt[0]], [sp[1], npnt[1]],
                                    linewidth=1.0, alpha=0.6, zorder=28)

            # Mode 3: Candidates + Selected
            if mode == "mode3_max_min":
                cand_log = getattr(planner, "_mode3_candidates_log", []) or []
                if cand_log:
                    last = cand_log[-1]
                    cands = last.get("candidates", []) or []
                    sel = last.get("selected", None)

                    if cands:
                        cx = [c[0] for c in cands]
                        cy = [c[1] for c in cands]
                        ax.scatter(cx, cy, marker="o", s=18,
                                   facecolors="none", edgecolors="black", linewidths=1.0,
                                   zorder=20, label="Max–Min candidates")
                    if sel is not None:
                        ax.scatter([sel[0]], [sel[1]], marker="*", s=200,
                                   zorder=21, label="Max–Min selected")

            # Mode 4: Corridor axis + corridor nodes
            if mode == "mode4_start_goal_corr":
                s = np.array(bench.startList[0], dtype=float)
                g = np.array(bench.goalList[0], dtype=float)

                ax.plot([s[0], g[0]], [s[1], g[1]],
                        linestyle="--", linewidth=2, alpha=0.8,
                        zorder=18, label="Start–Goal corridor axis")

                sample_log = getattr(planner, "_sample_log", []) or []
                corr_nodes = []
                for r in sample_log:
                    if r.get("mode") != mode:
                        continue
                    if r.get("source") != "start_goal_corridor":
                        continue
                    nid = int(r["node_id"])
                    if nid in pos:
                        corr_nodes.append(nid)

                if corr_nodes:
                    xs = [pos[n][0] for n in corr_nodes]
                    ys = [pos[n][1] for n in corr_nodes]
                    ax.scatter(xs, ys, marker="^",
                               s=max(25, int(0.7 * nodeSize)),
                               zorder=19, label="Corridor-sampled nodes")

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc="upper right")

            t = float(row["time_s"]) if pd.notna(row.get("time_s", np.nan)) else np.nan
            L = float(row["path_length"]) if pd.notna(row.get("path_length", np.nan)) else np.nan
            S = float(row["roadmap_size"]) if pd.notna(row.get("roadmap_size", np.nan)) else np.nan
            ax.set_title(
                f"{bench_name} – {mode}\n"
                f"{tag}: run={run_idx}, time={t:.3f}s, len={L:.2f}, size={S:.0f}, seed={seed}"
            )

            plt.tight_layout()
            plt.show()
