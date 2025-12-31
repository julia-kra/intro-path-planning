# IPPLazyPRM_Task2.py
import random
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from IPLazyPRM import LazyPRM


class EnhancedLazyPRM(LazyPRM):
    """
    EnhancedLazyPRM (STRICT LAZY):
    - Sampling: KEINE Collision-Checks
    - Edge-Add: KEINE Collision-Checks
    - Collision-Info entsteht nur durch Lazy-Validierung (Kandidatenpfad)

    Mindestabstand/Roboterradius:
    - wird im CollisionChecker über min_clearance umgesetzt
    - Kantenabstand wird über cc.lineInCollision(q1,q2) geprüft
    """

    # ---------- zentraler Zugriff auf CollisionChecker ----------
    def _cc(self):
        if hasattr(self, "_collisionChecker") and self._collisionChecker is not None:
            return self._collisionChecker
        if hasattr(self, "collisionChecker") and self.collisionChecker is not None:
            return self.collisionChecker
        raise AttributeError("EnhancedLazyPRM: No collision checker found (expected _collisionChecker or collisionChecker).")

    def _env_limits(self):
        lims = self._cc().getEnvironmentLimits()
        return [list(l) for l in lims]

    def _in_limits(self, q, lims=None):
        if lims is None:
            lims = self._env_limits()
        (xmin, xmax), (ymin, ymax) = lims
        return (xmin <= q[0] <= xmax) and (ymin <= q[1] <= ymax)

    def _clamp_to_limits(self, q, lims=None):
        if lims is None:
            lims = self._env_limits()
        (xmin, xmax), (ymin, ymax) = lims
        return [min(max(q[0], xmin), xmax), min(max(q[1], ymin), ymax)]

    def _rand_uniform_in_limits(self, lims=None):
        if lims is None:
            lims = self._env_limits()
        (xmin, xmax), (ymin, ymax) = lims
        return [random.uniform(xmin, xmax), random.uniform(ymin, ymax)]

    # ------------------------------------------------------------------------------------
    # Override _getRandomPosition() => IMMER in Benchmark-Bounds
    # STRICT LAZY: keine Collision-Checks
    def _getRandomPosition(self):
        return self._rand_uniform_in_limits(self._env_limits())
    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ZENTRALER Edge-Check für Lazy-Validierung:
    # (bricht strict-lazy NICHT, solange LazyPRM ihn nur auf Kandidatenpfaden nutzt)
    def _edge_in_collision(self, q1, q2):
        return self._cc().lineInCollision(q1, q2)

    # Optional: häufige Hook-Namen in LazyPRM-Implementationen.
    # Falls deine LazyPRM-Basis diese verwendet, greift automatisch der Clearance-Check:
    def _checkEdgeCollision(self, q1, q2):
        return self._edge_in_collision(q1, q2)

    def _isEdgeInCollision(self, q1, q2):
        return self._edge_in_collision(q1, q2)

    def _edgeCollision(self, q1, q2):
        return self._edge_in_collision(q1, q2)
    # ------------------------------------------------------------------

    def planPath(self, startList, goalList, config):
        # Stats + Logs für Visualisierung (wichtig fürs Plotting)
        self._stats = {"seed_none": 0, "uniform": 0, "m1": 0, "m2": 0, "m3": 0, "m4": 0}
        self._config = config or {}

        # Visual-Logs (werden von Plot-Funktionen genutzt)
        self._sample_log = []             # pro Node: Quelle/Mode/Seed-Key etc.
        self._seed_points_used = []       # Liste tatsächlicher Seeds, die genutzt wurden
        self._seed_trees = {}             # seed_key -> {"seed": [x,y], "nodes": [node_ids]}

        # Mode3-Log: Kandidatenwolken
        self._mode3_candidates_log = []   # Liste von {"candidates":[...], "selected":[x,y]}

        # Resets
        self.nonCollidingEdges = []
        self.collidingEdges = []

        # Mode4: robust Start/Goal extrahieren (nicht still scheitern)
        def _as_xy(p):
            if p is None:
                return None
            # list/tuple/np.array mit mindestens 2 Einträgen
            if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                return [float(p[0]), float(p[1])]
            # Objekt mit .x/.y (z.B. Shapely Point)
            if hasattr(p, "x") and hasattr(p, "y"):
                return [float(p.x), float(p.y)]
            return None

        self._start_for_mode4 = None
        self._goal_for_mode4  = None

        try:
            if startList and len(startList) > 0:
                self._start_for_mode4 = _as_xy(startList[0])
            if goalList and len(goalList) > 0:
                self._goal_for_mode4 = _as_xy(goalList[0])
        except Exception as e:
            print(f"[WARN] Mode4 start/goal extraction failed: {e}")

        # Warnen, wenn Mode4 aktiv ist, aber Start/Goal nicht nutzbar sind
        if (self._config.get("enhanceMode") == "mode4_start_goal_corr" and
            (self._start_for_mode4 is None or self._goal_for_mode4 is None)):
            print(f"[WARN] Mode4 active but invalid start/goal. "
                  f"startList[0]={startList[0] if startList else None}, "
                  f"goalList[0]={goalList[0] if goalList else None}")

        
        return super().planPath(startList, goalList, config)

    def _mode_color(self):
        mode = (getattr(self, "_config", {}) or {}).get("enhanceMode", "baseline_uniform")
        return {
            "baseline_uniform": "#888888",
            "mode1_seed_gauss": "#1f77b4",
            "mode2_seed_dist":  "#2ca02c",
            "mode3_max_min":    "#ff7f0e",
            "mode4_start_goal_corr": "#d62728",
        }.get(mode, "#888888")

    # ==============================
    # Seed-Auswahl (Mode 1/2)
    # ==============================
    def _pick_seed_from_colliding_edge(self, lims, strategy="segment_random"):
        if not getattr(self, "collidingEdges", None):
            self._stats["seed_none"] += 1
            return None

        u, v = random.choice(self.collidingEdges)
        if (u not in self.graph.nodes) or (v not in self.graph.nodes):
            self._stats["seed_none"] += 1
            return None
        if ("pos" not in self.graph.nodes[u]) or ("pos" not in self.graph.nodes[v]):
            self._stats["seed_none"] += 1
            return None

        qu = np.array(self.graph.nodes[u]["pos"], dtype=float)
        qv = np.array(self.graph.nodes[v]["pos"], dtype=float)

        if strategy == "endpoint":
            seed = (qu if (random.random() < 0.5) else qv)
        elif strategy == "midpoint":
            seed = 0.5 * (qu + qv)
        else:
            a = random.uniform(0.0, 1.0)
            seed = (1.0 - a) * qu + a * qv

        seed = [float(seed[0]), float(seed[1])]

        if not self._in_limits(seed, lims):
            seed = self._clamp_to_limits(seed, lims)

        return seed

    def _seed_key(self, seed, ndigits=3):
        return (round(float(seed[0]), ndigits), round(float(seed[1]), ndigits))

    # ==============================
    # Bounds-Policy beim Sampling
    # ==============================
    def _apply_bounds_policy(self, q, lims, oob_policy="reject"):
        if self._in_limits(q, lims):
            return q

        if oob_policy == "clamp":
            return self._clamp_to_limits(q, lims)

        return None

    # ==============================
    # Mode 1: Seed Gaussian
    # ==============================
    def _sample_seed_gaussian_from_seed(self, seed, lims):
        cfg = getattr(self, "_config", {}) or {}
        sigma = float(cfg.get("seedSigma", 4.0))
        seedTries = int(cfg.get("seedTries", 6))
        oobPolicy = str(cfg.get("oobPolicy", "reject")).lower()

        seed = np.array(seed, dtype=float)

        for _ in range(seedTries):
            q = (seed + np.random.normal(scale=sigma, size=2)).tolist()
            q2 = self._apply_bounds_policy(q, lims, oob_policy=oobPolicy)
            if q2 is not None:
                self._stats["m1"] += 1
                return q2

        q = (seed + np.random.normal(scale=sigma, size=2)).tolist()
        q = self._clamp_to_limits(q, lims)
        self._stats["m1"] += 1
        return q

    # ==============================
    # Mode 2: Seed Distance
    # ==============================
    def _sample_seed_distance_from_seed(self, seed, lims):
        cfg = getattr(self, "_config", {}) or {}
        seedMaxStep = float(cfg.get("seedMaxStep", 4.0))
        seedBeta = float(cfg.get("seedBeta", 0.9))
        seedTries = int(cfg.get("seedTries", 40))
        oobPolicy = str(cfg.get("oobPolicy", "reject")).lower()

        seed = np.array(seed, dtype=float)

        for _ in range(seedTries):
            direction = np.random.normal(size=2)
            n = np.linalg.norm(direction)
            if n < 1e-12:
                continue
            d = direction / n
            a = random.uniform(0.0, 1.0) * seedBeta * seedMaxStep
            q = (seed + a * d).tolist()

            q2 = self._apply_bounds_policy(q, lims, oob_policy=oobPolicy)
            if q2 is not None:
                self._stats["m2"] += 1
                return q2

        direction = np.random.normal(size=2)
        n = np.linalg.norm(direction) + 1e-12
        d = direction / n
        a = random.uniform(0.0, 1.0) * seedBeta * seedMaxStep
        q = (seed + a * d).tolist()
        q = self._clamp_to_limits(q, lims)
        self._stats["m2"] += 1
        return q

    # ==============================
    # Mode 3: Max–Min
    # ==============================
    def _sample_max_min(self, lims):
        cfg = getattr(self, "_config", {}) or {}
        dispersionCandidates = int(cfg.get("dispersionCandidates", 30))

        candidates = [self._getRandomPosition() for _ in range(max(2, dispersionCandidates))]

        pos_items = nx.get_node_attributes(self.graph, "pos")
        if len(pos_items) == 0:
            selected = candidates[0]
            self._mode3_candidates_log.append({"candidates": candidates, "selected": selected})
            self._stats["m3"] += 1
            return selected

        existing = np.array(list(pos_items.values()), dtype=float)

        best = None
        best_min_dist = -1.0
        for c in candidates:
            cnp = np.array(c, dtype=float)
            dists = np.linalg.norm(existing - cnp, axis=1)
            dmin = float(np.min(dists)) if dists.size > 0 else 0.0
            if dmin > best_min_dist:
                best_min_dist = dmin
                best = c

        selected = best if best is not None else candidates[0]
        self._mode3_candidates_log.append({"candidates": candidates, "selected": selected})
        self._stats["m3"] += 1
        return selected

    # ==============================
    # Mode 4: Start–Goal Corridor Bias
    # ==============================
    def _sample_start_goal_corridor(self, lims, start, goal):
        cfg = getattr(self, "_config", {}) or {}

        corridorSigma = float(cfg.get("corridorSigma", 1.8))
        corridorAlongSigma = float(cfg.get("corridorAlongSigma", 0.45))
        corridorTries = int(cfg.get("corridorTries", 50))
        oobPolicy = str(cfg.get("oobPolicy", "reject")).lower()

        s = np.array(start, dtype=float)
        g = np.array(goal, dtype=float)
        v = (g - s)
        vn = np.linalg.norm(v) + 1e-12
        along = v / vn
        ortho = np.array([-along[1], along[0]], dtype=float)

        for _ in range(corridorTries):
            t = random.uniform(0.0, 1.0)
            base = s + t * v
            off_a = np.random.normal(scale=corridorAlongSigma)
            off_o = np.random.normal(scale=corridorSigma)
            q = (base + off_a * along + off_o * ortho).tolist()

            q2 = self._apply_bounds_policy(q, lims, oob_policy=oobPolicy)
            if q2 is not None:
                self._stats["m4"] += 1
                return q2

        t = random.uniform(0.0, 1.0)
        base = s + t * v
        off_a = np.random.normal(scale=corridorAlongSigma)
        off_o = np.random.normal(scale=corridorSigma)
        q = (base + off_a * along + off_o * ortho).tolist()
        q = self._clamp_to_limits(q, lims)
        self._stats["m4"] += 1
        return q

    # ====================================================================================
    # ROADMAP BUILD (STRICT LAZY: keine collision checks hier)
    # ====================================================================================
    def _buildRoadmap(self, numNodes, kNearest):
        addedNodes = []
        cfg = (getattr(self, "_config", {}) or {})
        mode = str(cfg.get("enhanceMode", "baseline_uniform"))
        lims = self._env_limits()

        seedTreeSize = int(cfg.get("seedTreeSize", max(2, min(6, int(numNodes)))))
        seedPointStrategy = str(cfg.get("seedPointStrategy", "segment_random"))

        if mode in ("mode1_seed_gauss", "mode2_seed_dist"):
            n_left = int(numNodes)
            while n_left > 0:
                seed = self._pick_seed_from_colliding_edge(lims, strategy=seedPointStrategy)
                if seed is None:
                    for _ in range(n_left):
                        pos = self._getRandomPosition()
                        self._stats["uniform"] += 1
                        nid = self.lastGeneratedNodeNumber
                        self.graph.add_node(
                            nid,
                            pos=pos,
                            color=self._mode_color(),
                            sample_mode=mode,
                            sample_source="uniform_fallback",
                        )
                        self._sample_log.append({
                            "node_id": nid, "mode": mode, "source": "uniform_fallback",
                            "seed": None, "seed_key": None,
                        })
                        addedNodes.append(nid)
                        self.lastGeneratedNodeNumber += 1
                    n_left = 0
                    break

                skey = self._seed_key(seed)
                if skey not in self._seed_trees:
                    self._seed_trees[skey] = {"seed": [float(seed[0]), float(seed[1])], "nodes": []}
                    self._seed_points_used.append([float(seed[0]), float(seed[1])])

                burst = min(seedTreeSize, n_left)
                for _ in range(burst):
                    if mode == "mode1_seed_gauss":
                        pos = self._sample_seed_gaussian_from_seed(seed, lims)
                        source = "seed_gauss"
                    else:
                        pos = self._sample_seed_distance_from_seed(seed, lims)
                        source = "seed_dist"

                    nid = self.lastGeneratedNodeNumber
                    self.graph.add_node(
                        nid,
                        pos=pos,
                        color=self._mode_color(),
                        sample_mode=mode,
                        sample_source=source,
                        seed_key=skey,
                        seed_point=[float(seed[0]), float(seed[1])],
                    )

                    self._seed_trees[skey]["nodes"].append(nid)
                    self._sample_log.append({
                        "node_id": nid,
                        "mode": mode,
                        "source": source,
                        "seed": [float(seed[0]), float(seed[1])],
                        "seed_key": skey,
                    })

                    addedNodes.append(nid)
                    self.lastGeneratedNodeNumber += 1

                n_left -= burst

        elif mode == "mode3_max_min":
            for _ in range(int(numNodes)):
                pos = self._sample_max_min(lims)
                nid = self.lastGeneratedNodeNumber
                self.graph.add_node(
                    nid,
                    pos=pos,
                    color=self._mode_color(),
                    sample_mode=mode,
                    sample_source="max_min",
                )
                self._sample_log.append({
                    "node_id": nid, "mode": mode, "source": "max_min",
                    "seed": None, "seed_key": None,
                })
                addedNodes.append(nid)
                self.lastGeneratedNodeNumber += 1

        elif mode == "mode4_start_goal_corr":
            # >>> FIX: Start/Goal aus dem planPath()-Kontext verwenden
            start = getattr(self, "_start_for_mode4", None)
            goal  = getattr(self, "_goal_for_mode4", None)
        
            # Optional: weiterhin Overrides zulassen
            start = self._config.get("start_override", start)
            goal  = self._config.get("goal_override", goal)
        
            if start is None or goal is None:
                # zur Not uniform, aber das sollte nun praktisch nie passieren
                for _ in range(int(numNodes)):
                    pos = self._getRandomPosition()
                    self._stats["uniform"] += 1
                    nid = self.lastGeneratedNodeNumber
                    self.graph.add_node(
                        nid, pos=pos, color=self._mode_color(),
                        sample_mode=mode, sample_source="uniform_fallback"
                    )
                    self._sample_log.append({
                        "node_id": nid, "mode": mode, "source": "uniform_fallback",
                        "seed": None, "seed_key": None
                    })
                    addedNodes.append(nid)
                    self.lastGeneratedNodeNumber += 1
            else:
                for _ in range(int(numNodes)):
                    pos = self._sample_start_goal_corridor(lims, start, goal)
                    nid = self.lastGeneratedNodeNumber
                    self.graph.add_node(
                        nid,
                        pos=pos,
                        color=self._mode_color(),
                        sample_mode=mode,
                        sample_source="start_goal_corridor",
                    )
                    self._sample_log.append({
                        "node_id": nid, "mode": mode, "source": "start_goal_corridor",
                        "seed": None, "seed_key": None
                    })
                    addedNodes.append(nid)
                    self.lastGeneratedNodeNumber += 1


        else:
            for _ in range(int(numNodes)):
                pos = self._getRandomPosition()
                self._stats["uniform"] += 1
                nid = self.lastGeneratedNodeNumber
                self.graph.add_node(
                    nid,
                    pos=pos,
                    color=self._mode_color(),
                    sample_mode=mode,
                    sample_source="uniform",
                )
                self._sample_log.append({"node_id": nid, "mode": mode, "source": "uniform", "seed": None, "seed_key": None})
                addedNodes.append(nid)
                self.lastGeneratedNodeNumber += 1

        # k-Nearest connections (STRICT LAZY: ohne Collisioncheck)
        pos_items = list(nx.get_node_attributes(self.graph, "pos").items())
        node_ids = [nid for nid, _ in pos_items]
        pos_list = [p for _, p in pos_items]

        if len(pos_list) == 0:
            return

        kdTree = cKDTree(pos_list)
        k = min(int(kNearest), len(pos_list))

        for node in addedNodes:
            _, idxs = kdTree.query(self.graph.nodes[node]["pos"], k=k)
            if np.isscalar(idxs):
                idxs = [int(idxs)]
            else:
                idxs = [int(i) for i in idxs]

            for idx in idxs:
                c_node = node_ids[idx]
                if node == c_node:
                    continue

                if (node, c_node) in self.collidingEdges or (c_node, node) in self.collidingEdges:
                    continue

                self.graph.add_edge(node, c_node)