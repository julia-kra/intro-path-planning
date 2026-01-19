# benchmarks_task2.py
from __future__ import annotations

import math
import random
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon, Point, LineString

from IPBenchmark import Benchmark
from IPEnvironment import CollisionChecker


# ============================================================
# Bestehende Benchmarks (Workspace / Task2 bisher)
# ============================================================

def make_circle_field_benchmark(
    *,
    clearance: float,
    name: str = "CircleField",
    bounds=(0.0, 22.0, 0.0, 22.0),        # (xmin, xmax, ymin, ymax)
    start=(2.0, 20.0),
    goal=(20.0, 2.0),
    n_circles: int = 18,
    r_min: float = 1.2,
    r_max: float = 2.2,
    min_center_dist: float = 1.0,         # zusätzlicher Abstand zwischen Kreismittelpunkten
    inner_keepout: float = 2.2,           # Abstand Start/Ziel
    edge_overshoot: float = 2.5,          # Mittelpunkte dürfen über den Rand hinaus
    seed: int = 42,
    max_tries: int = 20000,
) -> Benchmark:
    """
    Viele runde Hindernisse (Kreise) zufällig, so dass Start/Ziel frei bleiben.
    Kreise dürfen in den Rand ragen bzw. über den Rand hinausgehen.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = bounds

    scene = {}
    centers: list[np.ndarray] = []
    radii: list[float] = []

    start = np.array(start, dtype=float)
    goal  = np.array(goal, dtype=float)

    # Sampling-Bereich für Mittelpunkte: erweitert, damit Kreise in den Rand ragen können
    x_lo = xmin - float(edge_overshoot)
    x_hi = xmax + float(edge_overshoot)
    y_lo = ymin - float(edge_overshoot)
    y_hi = ymax + float(edge_overshoot)

    tries = 0
    while len(centers) < n_circles and tries < max_tries:
        tries += 1

        r = float(rng.uniform(r_min, r_max))
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))
        c = np.array([x, y], dtype=float)

        # Start/Ziel frei halten
        if np.linalg.norm(c - start) < (r + inner_keepout):
            continue
        if np.linalg.norm(c - goal) < (r + inner_keepout):
            continue

        # Abstand zwischen Hindernissen
        ok = True
        for (c2, r2) in zip(centers, radii):
            if np.linalg.norm(c - c2) < (r + r2 + min_center_dist):
                ok = False
                break
        if not ok:
            continue

        centers.append(c)
        radii.append(r)
        scene[f"c{len(centers):02d}"] = Point(x, y).buffer(r)

    if len(centers) < n_circles:
        print(f"[WARN] Nur {len(centers)}/{n_circles} Kreise platziert (Parameter evtl. zu streng).")

    cc = CollisionChecker(scene, limits=[[xmin, xmax], [ymin, ymax]], min_clearance=float(clearance))

    bench = Benchmark(
        name=name,
        collisionChecker=cc,
        startList=[[float(start[0]), float(start[1])]],
        goalList=[[float(goal[0]), float(goal[1])]],
        description=f"Viele runde Hindernisse (n={len(centers)}), Rand-Überlappung erlaubt.",
        level=2,
    )

    # Sanity: Start/Ziel kollisionsfrei
    assert not bench.collisionChecker.pointInCollision(bench.startList[0]), f"{name}: Start kollidiert"
    assert not bench.collisionChecker.pointInCollision(bench.goalList[0]),  f"{name}: Ziel kollidiert"

    return bench


def _find_free_point(cc: CollisionChecker, candidates) -> list[float]:
    """Gibt den ersten kollisionsfreien Kandidaten zurück, sonst Exception."""
    for p in candidates:
        if not cc.pointInCollision(list(p)):
            return [float(p[0]), float(p[1])]
    raise Exception("Kein kollisionsfreier Punkt in candidates gefunden. Hindernisse/Bounds prüfen.")


def make_mode1_wall_tiny_door_closed22(*, clearance: float, extra_block: bool = True) -> Benchmark:
    scene = {}

    xmin, xmax, ymin, ymax = 0.0, 22.0, 0.0, 22.0

    # zentrale Wand: einzige Passage = Door
    x1, x2 = 9.6, 13.4
    door_y1, door_y2 = 10.2, 11.8

    scene["wall_bottom"] = Polygon([(x1, ymin), (x2, ymin), (x2, door_y1), (x1, door_y1)])
    scene["wall_top"]    = Polygon([(x1, door_y2), (x2, door_y2), (x2, ymax), (x1, ymax)])

    if extra_block:
        scene["block_ul"] = Polygon([(2.0, 15.0), (6.0, 15.0), (6.0, 18.5), (2.0, 18.5)])

    cc = CollisionChecker(scene, min_clearance=float(clearance))

    start_candidates = [(1.0, 21.0), (1.5, 20.5), (1.0, 19.5), (2.0, 21.0), (3.0, 21.0), (1.0, 18.0)]
    goal_candidates  = [(21.0, 1.0), (20.5, 1.5), (21.0, 2.0), (19.5, 1.0), (20.0, 3.0)]

    start = _find_free_point(cc, start_candidates)
    goal  = _find_free_point(cc, goal_candidates)

    return Benchmark(
        name="WallTinyDoor",
        collisionChecker=cc,
        startList=[start],
        goalList=[goal],
        description="Zentrale Wand blockiert oben/unten komplett, nur eine kleine Tür offen.",
        level=3,
    )


def make_mode2_wall_tiny_wide_door(*, clearance: float, extra_block: bool = True) -> Benchmark:
    scene = {}

    xmin, xmax, ymin, ymax = 0.0, 22.0, 0.0, 22.0

    x1, x2 = 6.6, 18.4
    door_y1, door_y2 = 10.2, 11.8

    scene["wall_bottom"] = Polygon([(x1, ymin), (x2, ymin), (x2, door_y1), (x1, door_y1)])
    scene["wall_top"]    = Polygon([(x1, door_y2), (x2, door_y2), (x2, ymax), (x1, ymax)])

    if extra_block:
        scene["block_ul"] = Polygon([(2.0, 15.0), (6.0, 15.0), (6.0, 18.5), (2.0, 18.5)])

    cc = CollisionChecker(scene, min_clearance=float(clearance))

    start_candidates = [(1.0, 21.0), (1.5, 20.5), (1.0, 19.5), (2.0, 21.0), (3.0, 21.0), (1.0, 18.0)]
    goal_candidates  = [(21.0, 1.0), (20.5, 1.5), (21.0, 2.0), (19.5, 1.0), (20.0, 3.0)]

    start = _find_free_point(cc, start_candidates)
    goal  = _find_free_point(cc, goal_candidates)

    return Benchmark(
        name="WallTinyWideDoor",
        collisionChecker=cc,
        startList=[start],
        goalList=[goal],
        description="Zentrale Wand blockiert oben/unten komplett, nur eine kleine, breite Tür offen.",
        level=3,
    )


def make_mode3_u_shape_benchmark(
    *,
    clearance: float,
    name: str = "U-Shape",
    bounds=(0.0, 22.0, 0.0, 22.0),
) -> Benchmark:
    xmin, xmax, ymin, ymax = bounds

    scene = {}
    scene["blk_bottom_orange"] = Polygon([
        (xmin + 3.5, ymin + 5.0),
        (xmax - 5.5, ymin + 5.0),
        (xmax - 5.5, ymin + 15.0),
        (xmax - 7.5, ymin + 15.0),
        (xmax - 7.5, ymin + 7.0),
        (xmin + 5.5, ymin + 7.0),
        (xmin + 5.5, ymin + 15.0),
        (xmin + 3.5, ymin + 15.0),
    ])

    start = [[xmin + 1.2, ymin + 1.2]]
    goal  = [[xmax - 12.0, ymax - 12.0]]

    cc = CollisionChecker(scene, min_clearance=float(clearance))

    bench = Benchmark(
        name=name,
        collisionChecker=cc,
        startList=start,
        goalList=goal,
        description="U-Shape.",
        level=4,
    )

    assert not bench.collisionChecker.pointInCollision(start[0]), "Start liegt in Kollision!"
    assert not bench.collisionChecker.pointInCollision(goal[0]),  "Ziel liegt in Kollision!"

    return bench


def make_mode4_snail(
    *,
    clearance: float,
    name: str = "Snail",
    bounds=(0.0, 22.0, 0.0, 22.0),
) -> Benchmark:
    xmin, xmax, ymin, ymax = bounds

    scene = {}
    scene["blk_upper_blue"] = Polygon([
        (xmin + 2.5, ymin + 0.0),
        (xmin + 5.0, ymin + 0.0),
        (xmin + 5.0, ymin + 17.5),
        (xmin + 17.5, ymin + 17.5),
        (xmin + 17.5, ymin + 5.0),
        (xmin + 10.0, ymin + 5.0),
        (xmin + 10.0, ymin + 12.5),
        (xmin + 12.5, ymin + 12.5),
        (xmin + 12.5, ymin + 7.5),
        (xmin + 15.0, ymin + 7.5),
        (xmin + 15.0, ymin + 15.0),
        (xmin + 7.5, ymin + 15.0),
        (xmin + 7.5, ymin + 2.5),
        (xmin + 20.0, ymin + 2.5),
        (xmin + 20.0, ymin + 20.0),
        (xmin + 2.5, ymin + 20.0),
    ])

    start = [[xmin + 1.2, ymin + 1.2]]
    goal  = [[xmax - 11.0, ymax - 11.0]]

    cc = CollisionChecker(scene, min_clearance=float(clearance))

    bench = Benchmark(
        name=name,
        collisionChecker=cc,
        startList=start,
        goalList=goal,
        description="Schneckenförmiger enger Korridor durch große Hindernisfläche.",
        level=4,
    )

    assert not bench.collisionChecker.pointInCollision(start[0]), "Start liegt in Kollision!"
    assert not bench.collisionChecker.pointInCollision(goal[0]),  "Ziel liegt in Kollision!"

    return bench


# ============================================================
# TASK 2b: Point 2DoF + Planar 2R 2DoF (C-space) Benchmarks
# ============================================================

class CCBase:
    """
    Minimal-Interface, das dein Planner/Framework typischerweise braucht:
      - getDim()
      - getEnvironmentLimits()
      - pointInCollision(q)
      - edgeInCollision(q1, q2) / lineInCollision(q1, q2)
    """
    def __init__(self, bounds, obstacles):
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
        q1 = np.asarray(q1, float)
        q2 = np.asarray(q2, float)
        steps = max(2, int(np.linalg.norm(q2 - q1) / 0.05))
        for t in np.linspace(0.0, 1.0, steps):
            q = (1 - t) * q1 + t * q2
            if self.pointInCollision(q.tolist()):
                return True
        return False


class PointRobot2DoFCollisionChecker(CCBase):
    """
    Point Robot in 2D Workspace: q=[x,y]
    Clearance: obstacles werden inflated via buffer(min_clearance)
    """
    def __init__(self, bounds, obstacles, min_clearance=0.0):
        super().__init__(bounds=bounds, obstacles=obstacles)
        self.min_clearance = float(min_clearance)

        if self.min_clearance > 0.0:
            self._inflated_obstacles = [obs.buffer(self.min_clearance) for obs in self.obstacles]
        else:
            self._inflated_obstacles = list(self.obstacles)

    def getDim(self):
        return 2

    def pointInCollision(self, q):
        self.counter += 1
        p = Point(float(q[0]), float(q[1]))

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


class PlanarRobot2DoFCollisionChecker(CCBase):
    """
    Planar 2R: q=[theta1,theta2] in Radiant
    Collision im Workspace gegen shapely obstacles, Links als Capsules:
      capsule = LineString(...).buffer(min_clearance)
    """
    def __init__(
        self,
        theta_limits: List[Tuple[float, float]],
        obstacles,
        base=(0.0, 0.0),
        L1=1.6,
        L2=1.2,
        min_clearance=0.0,
    ):
        super().__init__(bounds=(-math.pi, -math.pi, math.pi, math.pi), obstacles=obstacles)
        self.theta_limits = theta_limits
        self.base = tuple(base)
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


def build_2b_benchmarks_from_task1style(
    robot_radius: float = 0.0,
    safety_margin: float = 0.0,
) -> List[Benchmark]:
    """
    Erzeugt 4 Benchmarks:
      [0] Point Easy
      [1] Point Passage
      [2] Planar Easy
      [3] Planar Passage

    Clearance = robot_radius + safety_margin
    - Point: Obstacles werden inflated
    - Planar: Links werden als Capsules buffer(clearance) geprüft
    """
    clearance = float(robot_radius) + float(safety_margin)

    # Workspace-Limits (für Visualisierung/Obstacles)
    bounds_ws = (-3.0, -3.0, 3.0, 3.0)

    # Hindernisse als shapely Polygone (in Workspace-Koordinaten)
    obs_easy = [
        LineString([(-2.0, 0.0), (-0.8, 0.0)]).buffer(0.5),
        LineString([( 2.0, 0.0), ( 2.0, 1.0)]).buffer(0.2),
        LineString([(-1.0, 2.0), ( 1.0, 2.0)]).buffer(0.1),
    ]

    obs_passage = [
        LineString([(1.8, -3.0), (1.8, -0.8)]).buffer(0.2),
        LineString([(1.8,  0.8), (1.8,  3.0)]).buffer(0.2),
    ]

    # -------------------------
    # Point 2DoF Benchmarks
    # -------------------------
    ccP_easy = PointRobot2DoFCollisionChecker(
        bounds=bounds_ws, obstacles=obs_easy, min_clearance=clearance
    )
    ccP_passage = PointRobot2DoFCollisionChecker(
        bounds=bounds_ws, obstacles=obs_passage, min_clearance=clearance
    )

    bP_easy = Benchmark(
        name="Point 2-DoF (Easy)",
        collisionChecker=ccP_easy,
        startList=[[-2.6,  2.2]],
        goalList=[[ 2.6, -2.2]],
        description=f"Point 2DoF easy (clearance={clearance:.2f})",
        level=1,
    )

    bP_passage = Benchmark(
        name="Point 2-DoF (Passage)",
        collisionChecker=ccP_passage,
        startList=[[-2.6,  0.0]],
        goalList=[[ 2.6,  0.0]],
        description=f"Point 2DoF passage (clearance={clearance:.2f})",
        level=2,
    )

    # -------------------------
    # Planar 2R 2DoF Benchmarks
    # -------------------------
    theta_limits = [(-math.pi, math.pi), (-math.pi, math.pi)]

    ccA_easy = PlanarRobot2DoFCollisionChecker(
        theta_limits=theta_limits,
        obstacles=obs_easy,
        base=(0.0, 0.0),
        L1=1.6,
        L2=1.2,
        min_clearance=clearance,
    )

    ccA_passage = PlanarRobot2DoFCollisionChecker(
        theta_limits=theta_limits,
        obstacles=obs_passage,
        base=(0.0, 0.0),
        L1=1.6,
        L2=1.2,
        min_clearance=clearance,
    )

    # Start/Goal sind Winkel (Radiant!)
    bA_easy = Benchmark(
        name="Manipulator 2-DoF (Easy)",
        collisionChecker=ccA_easy,
        startList=[[-2.0, -1.0]],
        goalList=[[ 2.0,  1.0]],
        description=f"Planar 2DoF easy (clearance={clearance:.2f})",
        level=1,
    )

    bA_passage = Benchmark(
        name="Manipulator 2-DoF (Passage)",
        collisionChecker=ccA_passage,
        startList=[[math.pi, 0.0]],
        goalList=[[0.0, 0.0]],
        description=f"Planar 2DoF passage (clearance={clearance:.2f})",
        level=2,
    )

    # Sanity: Start/Goal collisionfrei
    for b in (bP_easy, bP_passage, bA_easy, bA_passage):
        cc = b.collisionChecker
        assert not cc.pointInCollision(b.startList[0]), f"{b.name}: Start kollidiert {b.startList[0]}"
        assert not cc.pointInCollision(b.goalList[0]),  f"{b.name}: Goal kollidiert {b.goalList[0]}"

    return [bP_easy, bP_passage, bA_easy, bA_passage]
