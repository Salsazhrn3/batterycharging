"""
pipeline2_analysis.py
─────────────────────
Standalone offline analysis for Pipeline 2 (Affinity Propagation) based on
Baras et al. 2023 "Improving the Efficiency of Modern Warehouses Using Smart
Battery Placement".

Goals
─────
1. Compare the current implementation of Pipeline 2 in
   model/charging_layout_generator.py against the paper's Algorithm 1 and
   Equations 1-2.
2. Generate a realistic traffic matrix for the current grid
   (generated_pod.csv) where traffic is higher near picking / replenishment
   stations — the places robots converge to.
3. Run Affinity Propagation using:
     (a) the CURRENT formulas in the repo, and
     (b) a PAPER-FAITHFUL variant (corrected self-similarity sign +
         TrafficDesirability inverted-bell curve).
4. Restrict candidates to floor (0) and pod (1) cells only — matching the
   user's explicit feedback for Pipeline 1.
5. Print proposed charger coordinates — does NOT write charging_config.json
   so the running Pipeline 1 simulation is untouched.

Run:  python pipeline2_analysis.py
"""

from __future__ import annotations

import csv
import math
from collections import deque
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
from sklearn.cluster import AffinityPropagation

Cell = Tuple[int, int]

# ── cell-value constants (match model/charging_layout_generator.py) ─────────
POD = 1
CHARGER = 2
PICKING = 11
REPLENISH = 21

TRAVERSABLE: FrozenSet[int] = frozenset({
    3, 4, 5, 6, 7,
    12, 13, 14, 16, 17, 18, 19,
    22, 23, 24, 26, 27, 28, 29,
    99,
})
BFS_TRAVERSABLE: FrozenSet[int] = TRAVERSABLE | frozenset({0, 1})
CHARGER_CANDIDATE_VALUES: FrozenSet[int] = frozenset({0, 1})


def load_grid(path: str = "generated_pod.csv") -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append([int(x) for x in r])
    return np.array(rows, dtype=int)


def bfs_distance_to_stations(grid: np.ndarray) -> np.ndarray:
    """
    Multi-source BFS: distance (in hops) from every BFS_TRAVERSABLE cell to
    the nearest picking (11) or replenishment (21) station cell, travelling
    only through BFS_TRAVERSABLE cells. Unreachable cells get +inf.
    """
    rows, cols = grid.shape
    dist = np.full_like(grid, fill_value=10_000, dtype=np.int32)
    q: deque[Cell] = deque()
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] in (PICKING, REPLENISH):
                dist[r, c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows and 0 <= nc < cols
                and grid[nr, nc] in BFS_TRAVERSABLE
                and dist[nr, nc] > dist[r, c] + 1
            ):
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def build_traffic_matrix(grid: np.ndarray) -> np.ndarray:
    """
    Heuristic traffic model:
      traffic(c) = exp(-d(c) / tau)
    where d(c) = shortest path (hops) from c to nearest station.
    Cells adjacent to stations see ≈ all pod-delivery flows, so they get
    the highest traffic; remote pod cells get very low traffic. tau
    controls how quickly traffic decays with distance.
    """
    dist = bfs_distance_to_stations(grid)
    tau = 8.0
    traffic = np.where(
        dist < 10_000,
        np.exp(-dist.astype(np.float64) / tau),
        0.0,
    )
    return traffic


def candidate_cells(grid: np.ndarray) -> List[Cell]:
    """Floor (0) or pod (1) cells only — matches Pipeline 1 restriction."""
    rows, cols = grid.shape
    return [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if grid[r, c] in CHARGER_CANDIDATE_VALUES
    ]


def run_ap(
    candidates: List[Cell],
    traffic: np.ndarray,
    variant: str,
    c_reg: float = 1.0,
    random_state: int = 42,
) -> Tuple[Dict[int, List[Cell]], np.ndarray]:
    """
    Run Affinity Propagation on the candidate cells.

    variant = "current": S[i,i] = -log(c + traffic[i])  (as in repo)
    variant = "paper"  : S[i,i] = +log(c + traffic[i])  (corrected sign)
    """
    n = len(candidates)
    coords = np.asarray(candidates, dtype=np.float64)

    # Vectorised off-diagonal  S[i,j] = -Manhattan(i,j)^2
    S = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        d = np.abs(coords - coords[i])
        md = d[:, 0] + d[:, 1]
        S[i] = -(md ** 2)

    # Diagonal: preference (self-similarity) driven by traffic
    sign = -1.0 if variant == "current" else +1.0
    for i, (r, c) in enumerate(candidates):
        S[i, i] = sign * math.log(c_reg + float(traffic[r, c]))

    ap = AffinityPropagation(
        affinity="precomputed",
        random_state=random_state,
        max_iter=400,
        convergence_iter=25,
        damping=0.9,
    )
    ap.fit(S)

    clusters: Dict[int, List[Cell]] = {}
    for idx, lbl in enumerate(ap.labels_):
        clusters.setdefault(int(lbl), []).append(candidates[idx])

    return clusters, ap.cluster_centers_indices_


def traffic_desirability(
    cluster: List[Cell], traffic: np.ndarray
) -> Dict[Cell, float]:
    """
    Paper-faithful Equation 2 term:
        TrafficDesirability(c) = 1 - (Traffic(c) - mean(Traffic))^2
    Rewards cells whose traffic is near the cluster mean (inverted bell
    around the mean) — the paper's argument is that chargers should serve
    the *typical* robot, not the single busiest cell in the cluster.
    """
    if not cluster:
        return {}
    vals = np.array([traffic[r, c] for r, c in cluster], dtype=np.float64)
    mean = float(vals.mean())
    return {rc: 1.0 - (float(traffic[rc[0], rc[1]]) - mean) ** 2 for rc in cluster}


def score_cluster(
    cluster: List[Cell],
    grid: np.ndarray,
    traffic: np.ndarray,
    variant: str,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.5,
) -> Dict[Cell, float]:
    """
    Compute placement score for every cell in one AP cluster.

    "current" : alpha * traffic - beta * prox_pod_penalty - gamma * dist_hot
    "paper"   : alpha * TrafficDesirability - beta * prox_pod_penalty
                (no distance-from-hot term; paper prefers cluster mean)
    """
    rows, cols = grid.shape
    hot: Cell = max(cluster, key=lambda rc: traffic[rc[0], rc[1]])
    td = traffic_desirability(cluster, traffic) if variant == "paper" else {}

    def adj_pod(r: int, c: int) -> float:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == POD:
                return 1.0
        return 0.0

    out: Dict[Cell, float] = {}
    for r, c in cluster:
        prox = adj_pod(r, c)
        if variant == "current":
            out[(r, c)] = (
                alpha * float(traffic[r, c])
                - beta * prox
                - gamma * (abs(r - hot[0]) + abs(c - hot[1]))
            )
        else:
            out[(r, c)] = alpha * td[(r, c)] - beta * prox
    return out


def best_per_cluster(
    clusters: Dict[int, List[Cell]],
    grid: np.ndarray,
    traffic: np.ndarray,
    variant: str,
) -> List[Cell]:
    picks: List[Cell] = []
    for _, cells in clusters.items():
        sc = score_cluster(cells, grid, traffic, variant)
        if not sc:
            continue
        picks.append(max(sc, key=sc.__getitem__))
    return picks


def dump_positions(title: str, picks: List[Cell]) -> None:
    print(f"\n── {title} ({len(picks)} chargers) ──")
    for r, c in sorted(picks):
        print(f"    [{r:2d}, {c:2d}]")


def main() -> None:
    grid = load_grid()
    print(f"Grid shape: {grid.shape}")
    print(f"Unique values: {sorted(set(grid.flatten().tolist()))}")

    traffic = build_traffic_matrix(grid)
    nonzero = traffic[traffic > 0]
    print(
        f"Traffic — nonzero cells: {nonzero.size}, "
        f"min={nonzero.min():.4f}, max={nonzero.max():.4f}, mean={nonzero.mean():.4f}"
    )

    cands = candidate_cells(grid)
    print(f"Candidates (floor+pod only): {len(cands)}")

    # Subsample to keep N×N tractable if needed — paper used real data but
    # the algorithm is O(N^2) per iteration. Keep it deterministic.
    MAX = 900
    if len(cands) > MAX:
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(cands), size=MAX, replace=False)
        cands = [cands[i] for i in sorted(idxs)]
        print(f"  subsampled to {len(cands)} for tractability")

    for variant in ("current", "paper"):
        print(f"\n━━━━━━ Variant: {variant.upper()} ━━━━━━")
        clusters, _ = run_ap(cands, traffic, variant=variant)
        print(f"AP produced {len(clusters)} clusters")
        picks = best_per_cluster(clusters, grid, traffic, variant)
        dump_positions(f"Proposed chargers ({variant})", picks)


if __name__ == "__main__":
    main()
