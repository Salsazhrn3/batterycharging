"""
charging_layout_generator.py
─────────────────────────────────────────────────────────────────────────────
Four strategic pipelines for placing Kiva-robot charging stations in an
RMFS warehouse simulation.

Grid cell legend (matches data_matrix convention)
─────────────────────────────────────────────────
  0  Navigable aisle / floor
  1  Storage pod (obstacle)
  2  Charging station  ← written by this module
  3  Picking / delivery station
  4  Replenishment station

Any other positive integer is treated as a non-traversable obstacle during
BFS pathfinding unless ``TRAVERSABLE`` is overridden at the module level.

Pipelines
─────────
  1  Set Cover Approximation  — worst-case reachability guarantee
  2  Affinity Propagation     — data-driven traffic clustering (requires scikit-learn)
  3  Picking-Station Heuristic — opportunity charging in queue cells
  4  Perimeter Wall Strategy  — isolation / control baseline

Usage
─────
    from model.charging_layout_generator import ChargingLayoutGenerator
    import numpy as np

    grid = np.array(...)   # 2-D int array using the legend above

    # Pipeline 1 — every navigable cell is ≤ 8 hops from a charger
    gen = ChargingLayoutGenerator(grid, {"pipeline": 1, "d": 8})
    new_grid = gen.generate()

    # Pipeline 2 — traffic-aware clustering
    traffic = np.random.rand(*grid.shape)
    gen = ChargingLayoutGenerator(
        grid,
        {"pipeline": 2, "traffic_matrix": traffic, "c": 1.0,
         "alpha": 1.0, "beta": 1.5, "gamma": 0.5},
    )
    new_grid = gen.generate()

    # Pipeline 3 — opportunity charging near picking stations
    gen = ChargingLayoutGenerator(grid, {"pipeline": 3, "num_chargers": 12})
    new_grid = gen.generate()

    # Pipeline 4 — perimeter isolation
    gen = ChargingLayoutGenerator(grid, {"pipeline": 4, "num_chargers": 10})
    new_grid = gen.generate()
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── cell-value constants ──────────────────────────────────────────────────────
POD: int = 1
CHARGER: int = 2

# Di sistem Rika, Picking station = 11, Replenishment = 21
PICKING_STATIONS: FrozenSet[int] = frozenset({11, 21})

# Di sistem Rika, jalanan/rel Kiva punya banyak angka (3-7, 12-29, 99)
TRAVERSABLE: FrozenSet[int] = frozenset({
    3, 4, 5, 6, 7,                # Aisle & intersections
    12, 13, 14, 16, 17, 18, 19,   # Rails & corners (kiri/picking)
    22, 23, 24, 26, 27, 28, 29,   # Rails & corners (kanan/replenishment)
    99                            # Blank space / safe zone
})

# ── type aliases ──────────────────────────────────────────────────────────────
Cell = Tuple[int, int]
Matrix = np.ndarray


# ═════════════════════════════════════════════════════════════════════════════
#  Module-level helper utilities
# ═════════════════════════════════════════════════════════════════════════════

def manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
    """Manhattan (L¹) distance between two grid cells."""
    return abs(r1 - r2) + abs(c1 - c2)


def _evenly_spaced_sample(items: list, n: int) -> list:
    """
    Select *n* items from *items* at approximately equal index spacing.

    Fully deterministic — no randomness involved.  Useful for distributing
    chargers evenly along a list of candidate cells without clustering them
    at one end.

    Examples
    --------
    >>> _evenly_spaced_sample(list(range(10)), 3)
    [0, 3, 6]
    >>> _evenly_spaced_sample(list(range(10)), 10)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def bfs_on_grid(
    matrix: Matrix,
    start: Cell,
    max_depth: int,
    traversable: FrozenSet[int] = TRAVERSABLE,
) -> Set[Cell]:
    """
    General-purpose 4-connected BFS over a 2-D integer grid.

    Explores neighbours whose cell value is in *traversable*, up to
    *max_depth* hops from *start*.  The start cell itself is included in
    the returned set only if its value is also in *traversable*.

    Parameters
    ----------
    matrix : np.ndarray of int
    start : (row, col)
    max_depth : int
        Maximum number of hops (edges) from start.
    traversable : frozenset of int
        Cell values a robot can enter.

    Returns
    -------
    Set of (row, col) cells reachable from *start* within *max_depth* hops.
    """
    rows, cols = matrix.shape
    visited: Set[Cell] = {start}
    queue: deque[Tuple[Cell, int]] = deque([(start, 0)])
    reachable: Set[Cell] = set()

    while queue:
        (r, c), depth = queue.popleft()
        if matrix[r][c] in traversable:
            reachable.add((r, c))
        if depth < max_depth:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and matrix[nr][nc] in traversable
                ):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), depth + 1))

    return reachable


# ═════════════════════════════════════════════════════════════════════════════
#  Main class
# ═════════════════════════════════════════════════════════════════════════════

class ChargingLayoutGenerator:
    """
    Generate charging-station layouts for an RMFS warehouse grid.

    The class wraps four strategic placement pipelines behind a single
    ``generate()`` call.  The original *data_matrix* is never mutated;
    each call to ``generate()`` works on a fresh deep copy.

    Parameters
    ----------
    data_matrix : array-like of int, shape (R, C)
        Warehouse grid using the cell-value legend at the top of this module.
    config : dict
        Must contain ``'pipeline'`` (int 1–4) plus pipeline-specific keys
        described below.

        Pipeline 1 — Set Cover Approximation
            ``'d'`` (int, default 10)
                Maximum BFS hop-distance from any navigable cell to its
                nearest charger.  Smaller *d* guarantees shorter travel
                but produces more chargers.

        Pipeline 2 — Affinity Propagation
            ``'traffic_matrix'`` (array-like of float, **required**)
                Same shape as *data_matrix*.  Higher value means more
                robot traffic at that cell.
            ``'c'`` (float, default 1.0)
                Regularisation constant in self-similarity; prevents log(0).
            ``'alpha'`` (float, default 1.0)
                Weight of traffic term in placement score.
            ``'beta'`` (float, default 1.0)
                Weight of shelf-proximity penalty in placement score.
            ``'gamma'`` (float, default 1.0)
                Weight of distance-from-hotspot penalty in placement score.
            ``'max_ap_cells'`` (int, default 2 000)
                If the grid has more navigable cells than this, a random
                subsample of this size is used to keep the N×N similarity
                matrix tractable.  Set to 0 to disable subsampling.

        Pipeline 3 — Picking-Station Heuristic
            ``'num_chargers'`` (int, default 10)

        Pipeline 4 — Perimeter Wall Strategy
            ``'num_chargers'`` (int, default 10)

    Raises
    ------
    ValueError
        Unknown pipeline id, or required config key is missing / mismatched.
    ImportError
        scikit-learn not installed when Pipeline 2 is requested.
    """

    def __init__(self, data_matrix, config: dict) -> None:
        self.matrix: Matrix = np.array(data_matrix, dtype=int)
        self.config: dict = config
        self.rows, self.cols = self.matrix.shape

    def __repr__(self) -> str:
        return (
            f"ChargingLayoutGenerator("
            f"shape={self.matrix.shape}, "
            f"pipeline={self.config.get('pipeline', '?')})"
        )

    # ── public entry-point ────────────────────────────────────────────────────

    def generate(self) -> Matrix:
        """
        Execute the configured pipeline and return the modified grid.

        Returns
        -------
        np.ndarray of int, same shape as the input, with CHARGER (2) values
        stamped at positions chosen by the selected pipeline.
        """
        pipeline = int(self.config.get("pipeline", 1))
        work: Matrix = self.matrix.copy()

        dispatch = {
            1: self._pipeline_set_cover,
            2: self._pipeline_affinity,
            3: self._pipeline_picking_station,
            4: self._pipeline_perimeter,
        }

        handler = dispatch.get(pipeline)
        if handler is None:
            raise ValueError(
                f"Unknown pipeline '{pipeline}'.  Valid options are 1, 2, 3, 4."
            )
        return handler(work)

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE 1 — Set Cover Approximation (Worst-Case Reachability)
    # ═════════════════════════════════════════════════════════════════════════

    def get_reachability_subsets(
        self,
        matrix: Matrix,
        candidate_locations: List[Cell],
        d: int,
    ) -> Dict[Cell, Set[Cell]]:
        """
        Run BFS from every candidate location up to depth *d*.

        Each BFS traverses only TRAVERSABLE cells (value 0 by default),
        modelling a robot that cannot drive through pods or fixed stations.

        Parameters
        ----------
        matrix : Matrix
            Current warehouse grid (read-only inside this method).
        candidate_locations : list of (row, col)
            Cells under consideration for charger placement.  Typically the
            full set of navigable cells (value 0).
        d : int
            Maximum battery-distance in grid hops.

        Returns
        -------
        dict mapping each candidate (row, col) → set of navigable cells
        reachable within *d* hops from that candidate.
        """
        return {
            loc: bfs_on_grid(matrix, loc, d, TRAVERSABLE)
            for loc in candidate_locations
        }

    def greedy_set_cover(
        self,
        universe: Set[Cell],
        subsets: Dict[Cell, Set[Cell]],
    ) -> List[Cell]:
        """
        Greedy approximation of minimum set cover (Kundu & Saha 2012).

        At each iteration the candidate whose coverage set intersects the
        most currently uncovered navigable cells is selected.  The algorithm
        terminates when all cells are covered, or when no candidate covers
        any remaining uncovered cell (e.g., disconnected pocket — a warning
        is logged).

        The greedy approach guarantees a solution within O(log n) of the
        optimal minimum-cardinality cover.

        Parameters
        ----------
        universe : set of Cell
            All navigable cells that must be within *d* hops of a charger.
        subsets : dict of Cell → set of Cell
            Reachability subsets from :meth:`get_reachability_subsets`.

        Returns
        -------
        Ordered list of selected charger positions (greedy insertion order).
        """
        uncovered: Set[Cell] = set(universe)

        # Intersect each subset with uncovered upfront for O(|candidates|·|subset|)
        remaining: Dict[Cell, Set[Cell]] = {
            loc: s & uncovered for loc, s in subsets.items()
        }
        selected: List[Cell] = []

        while uncovered and remaining:
            best = max(remaining, key=lambda loc: len(remaining[loc]))
            gain: Set[Cell] = remaining[best]

            if not gain:
                logger.warning(
                    "Set cover: %d navigable cell(s) remain unreachable "
                    "within the maximum BFS distance. "
                    "The grid may contain disconnected pockets.",
                    len(uncovered),
                )
                break

            selected.append(best)
            uncovered -= gain

            # Remove selected candidate and trim every remaining intersection
            del remaining[best]
            remaining = {loc: s - gain for loc, s in remaining.items()}

        return selected

    def apply_set_cover_layout(self, matrix: Matrix, d: int) -> Matrix:
        """
        Run the full set-cover pipeline and stamp CHARGER (2) values.

        Steps
        -----
        1. Collect all navigable cells as the *universe* to be covered.
        2. Treat every navigable cell as a candidate charger location.
        3. BFS from each candidate to build its reachability subset.
        4. Greedily pick candidates until every navigable cell is covered.
        5. Write CHARGER (2) at each selected position.

        Parameters
        ----------
        matrix : Matrix
            Working copy of the warehouse grid (mutated in place).
        d : int
            BFS depth limit (maximum robot travel distance to a charger).

        Returns
        -------
        The modified matrix.
        """
        nav: Set[Cell] = self._navigable_cells(matrix)
        logger.info(
            "Set Cover: %d navigable cells, BFS depth d=%d.", len(nav), d
        )

        candidates = list(nav)
        subsets = self.get_reachability_subsets(matrix, candidates, d)
        selected = self.greedy_set_cover(nav, subsets)

        logger.info("Set Cover: %d charger(s) selected.", len(selected))
        for r, c in selected:
            matrix[r][c] = CHARGER

        return matrix

    def _pipeline_set_cover(self, work: Matrix) -> Matrix:
        d: int = int(self.config.get("d", 10))
        return self.apply_set_cover_layout(work, d)

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE 2 — Affinity Propagation (Data-Driven Traffic Clustering)
    # ═════════════════════════════════════════════════════════════════════════

    def run_affinity_propagation(
        self,
        traffic_matrix: Matrix,
        c: float = 1.0,
    ) -> Dict[int, List[Cell]]:
        """
        Cluster navigable cells via Affinity Propagation (Baras et al. 2004).

        Similarity matrix construction
        ───────────────────────────────
        Off-diagonal  S[i, j] = −Manhattan(i, j)²
            Nearby cells in the grid are more similar; the squared penalty
            strongly discourages merging distant cells into one cluster.

        Diagonal (preference)  S[i, i] = −log(c + traffic[i])
            The preference controls how willing a cell is to become an
            exemplar (cluster centre).  With c = 1 and traffic ≥ 0:
              • traffic → 0   ⟹  S[i,i] → 0    (neutral)
              • traffic → ∞   ⟹  S[i,i] → −∞   (suppressed as exemplar)

            **Sign note:** as specified, this formula makes high-traffic
            cells *less* likely to become exemplars.  If you want high-traffic
            cells to attract cluster centres, use S[i,i] = +log(c+traffic[i]).

        Requires
        --------
        scikit-learn (``pip install scikit-learn``).

        Parameters
        ----------
        traffic_matrix : Matrix
            Robot-traffic density; same shape as the warehouse grid.
        c : float
            Regularisation constant to avoid log(0).  Default 1.0.

        Returns
        -------
        dict mapping integer cluster label → list of (row, col) cells.
        """
        try:
            from sklearn.cluster import AffinityPropagation as _AP
        except ImportError as exc:
            raise ImportError(
                "Pipeline 2 requires scikit-learn.\n"
                "Install it with:  pip install scikit-learn"
            ) from exc

        nav_cells: List[Cell] = sorted(self._navigable_cells(self.matrix))
        n = len(nav_cells)
        if n == 0:
            logger.warning("AP: no navigable cells found in the grid.")
            return {}

        # ── optional subsampling to keep the N×N matrix tractable ──────────
        max_cells: int = int(self.config.get("max_ap_cells", 2_000))
        if max_cells > 0 and n > max_cells:
            logger.warning(
                "AP: %d navigable cells exceeds max_ap_cells=%d; "
                "randomly subsampling to keep computation tractable.",
                n, max_cells,
            )
            nav_cells = random.sample(nav_cells, max_cells)
            n = max_cells

        # ── build N×N similarity matrix ─────────────────────────────────────
        S = np.empty((n, n), dtype=np.float64)
        coords = np.array(nav_cells, dtype=np.float64)  # shape (N, 2)

        # Vectorised off-diagonal: S[i,j] = −Manhattan(i,j)²
        for i in range(n):
            diff = np.abs(coords - coords[i])      # (N, 2)  row/col deltas
            mdist = diff[:, 0] + diff[:, 1]        # (N,)    Manhattan distance
            S[i] = -(mdist ** 2)

        # Override diagonal with per-cell preference scores
        for i, (r, col_idx) in enumerate(nav_cells):
            traffic_val = float(traffic_matrix[r][col_idx])
            S[i, i] = -math.log(c + traffic_val)

        # ── fit AP with precomputed similarity ──────────────────────────────
        ap = _AP(affinity="precomputed", random_state=42, max_iter=300)
        ap.fit(S)

        # ── group cells by cluster label ─────────────────────────────────────
        clusters: Dict[int, List[Cell]] = {}
        for idx, label in enumerate(ap.labels_):
            clusters.setdefault(int(label), []).append(nav_cells[idx])

        logger.info("AP: produced %d cluster(s) from %d cells.", len(clusters), n)
        return clusters

    def calculate_placement_score(
        self,
        cluster_cells: List[Cell],
        traffic_matrix: Matrix,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> Dict[Cell, float]:
        """
        Score every candidate cell in a cluster for charger placement.

        Scoring function
        ────────────────
        Score(c) = α · Traffic(c)
                 − β · ProximityToShelves(c)
                 − γ · DistanceFromHighTrafficCell(c)

        where:
          ``Traffic(c)``
              Raw traffic density value from *traffic_matrix* at cell *c*.

          ``ProximityToShelves(c)``
              Binary penalty: 1 if any 4-connected neighbour in the
              *original* warehouse grid (``self.matrix``) is a storage pod
              (value 1), else 0.  Penalises positions that would block
              pod retrieval lanes.

          ``DistanceFromHighTrafficCell(c)``
              Manhattan distance from *c* to the cell in *cluster_cells*
              with the maximum traffic value.  Rewards chargers near the
              busiest part of each cluster.

        Higher score ⟹ better placement.

        Parameters
        ----------
        cluster_cells : list of (row, col)
        traffic_matrix : Matrix
        alpha, beta, gamma : float
            Relative importance weights for each term.

        Returns
        -------
        dict mapping each cell to its float score.
        """
        if not cluster_cells:
            return {}

        # Identify the hotspot (highest-traffic cell) in this cluster
        hot_cell: Cell = max(
            cluster_cells, key=lambda rc: traffic_matrix[rc[0]][rc[1]]
        )

        scores: Dict[Cell, float] = {}
        for r, c in cluster_cells:
            traffic_val = float(traffic_matrix[r][c])

            # Binary penalty: direct neighbour of a storage pod?
            adj_to_pod = any(
                0 <= r + dr < self.rows
                and 0 <= c + dc < self.cols
                and self.matrix[r + dr][c + dc] == POD
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            prox_penalty = 1.0 if adj_to_pod else 0.0

            dist_from_hot = manhattan(r, c, hot_cell[0], hot_cell[1])

            scores[(r, c)] = (
                alpha * traffic_val
                - beta  * prox_penalty
                - gamma * dist_from_hot
            )

        return scores

    def apply_traffic_layout(
        self, matrix: Matrix, traffic_matrix: Matrix
    ) -> Matrix:
        """
        AP-cluster the grid, score candidates per cluster, stamp chargers.

        One charger is placed per AP cluster, at the cell with the highest
        placement score.

        Parameters
        ----------
        matrix : Matrix
            Working copy of the warehouse grid (mutated in place).
        traffic_matrix : Matrix
            Robot-traffic density grid (same shape as *matrix*).

        Returns
        -------
        The modified matrix.
        """
        c     = float(self.config.get("c",     1.0))
        alpha = float(self.config.get("alpha", 1.0))
        beta  = float(self.config.get("beta",  1.0))
        gamma = float(self.config.get("gamma", 1.0))

        clusters = self.run_affinity_propagation(traffic_matrix, c)
        placed = 0

        for label, cells in clusters.items():
            scores = self.calculate_placement_score(
                cells, traffic_matrix, alpha, beta, gamma
            )
            if not scores:
                continue

            best_cell = max(scores, key=scores.__getitem__)
            r, col_idx = best_cell
            matrix[r][col_idx] = CHARGER
            placed += 1
            logger.debug(
                "AP cluster %d → charger at (%d, %d)  score=%.3f",
                label, r, col_idx, scores[best_cell],
            )

        logger.info("AP traffic layout: %d charger(s) placed.", placed)
        return matrix

    def _pipeline_affinity(self, work: Matrix) -> Matrix:
        raw_traffic = self.config.get("traffic_matrix")
        if raw_traffic is None:
            raise ValueError(
                "Pipeline 2 requires 'traffic_matrix' in the config dict."
            )
        traffic: Matrix = np.array(raw_traffic, dtype=np.float64)
        if traffic.shape != self.matrix.shape:
            raise ValueError(
                f"traffic_matrix shape {traffic.shape} does not match "
                f"data_matrix shape {self.matrix.shape}."
            )
        return self.apply_traffic_layout(work, traffic)

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE 3 — Picking-Station Heuristic (Opportunity Charging)
    # ═════════════════════════════════════════════════════════════════════════

    def find_picking_queues(self, matrix: Matrix) -> List[Cell]:
        """
        Identify navigable queueing cells adjacent to picking stations.

        Scans every cell valued PICKING (3) and collects its 4-connected
        NAVIGABLE (0) neighbours.  A cell adjacent to two picking stations
        appears only once (deduplication via set).

        Parameters
        ----------
        matrix : Matrix

        Returns
        -------
        Sorted list of (row, col) queue cells, in row-major order.
        """
        rows, cols = matrix.shape
        queue_cells: Set[Cell] = set()

        for r in range(rows):
            for c in range(cols):
                # Cek apakah sel ini adalah stasiun (11 atau 21)
                if matrix[r][c] in PICKING_STATIONS:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            # Cek apakah sel sebelahnya adalah jalanan yang bisa dilewati
                            and matrix[nr][nc] in TRAVERSABLE
                        ):
                            queue_cells.add((nr, nc))

        return sorted(queue_cells)

    def apply_picking_station_layout(
        self, matrix: Matrix, num_chargers: int
    ) -> Matrix:
        """
        Distribute *num_chargers* chargers among picking-queue cells.

        Placement is deterministic (evenly spaced by sorted index) rather
        than random, ensuring reproducibility.  If *num_chargers* exceeds
        the number of queue cells, every queue cell receives a charger.

        Parameters
        ----------
        matrix : Matrix
            Working copy (mutated in place).
        num_chargers : int
            Target number of chargers to place.

        Returns
        -------
        The modified matrix.
        """
        queue_cells = self.find_picking_queues(matrix)
        if not queue_cells:
            logger.warning(
                "Pipeline 3: no navigable cells adjacent to picking stations (3) found."
            )
            return matrix

        n_place = min(num_chargers, len(queue_cells))
        selected = _evenly_spaced_sample(queue_cells, n_place)

        for r, c in selected:
            matrix[r][c] = CHARGER

        logger.info(
            "Picking-station layout: %d charger(s) placed "
            "from %d available queue cell(s).",
            len(selected), len(queue_cells),
        )
        return matrix

    def _pipeline_picking_station(self, work: Matrix) -> Matrix:
        num_chargers: int = int(self.config.get("num_chargers", 10))
        return self.apply_picking_station_layout(work, num_chargers)

    # ═════════════════════════════════════════════════════════════════════════
    #  PIPELINE 4 — Perimeter Wall Strategy (Isolation / Control)
    # ═════════════════════════════════════════════════════════════════════════

    def find_perimeter_cells(self, matrix: Matrix) -> List[Cell]:
        """
        Collect navigable cells on the outermost top and bottom rows.

        Strictly row 0 and row (R − 1) are scanned, consistent with the
        "perimeter wall" placement concept where chargers are placed near
        replenishment infrastructure at the warehouse edges.

        Parameters
        ----------
        matrix : Matrix

        Returns
        -------
        List of (row, col) navigable perimeter cells in row-major order
        (top row first, then bottom row).
        """
        rows, cols = matrix.shape
        perimeter: List[Cell] = []

        for row_idx in (0, rows - 1):
            for c in range(cols):
                if matrix[row_idx][c] in TRAVERSABLE:
                    perimeter.append((row_idx, c))

        return perimeter
    # ═════════════════════════════════════════════════════════════════════════
    #  Private helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _navigable_cells(self, matrix: Matrix) -> Set[Cell]:
        """Return the set of all (row, col) cells that Kiva can drive on."""
        return {
            (int(r), int(c))
            for r in range(matrix.shape[0])
            for c in range(matrix.shape[1])
            if matrix[r][c] in TRAVERSABLE
        }
    
    def apply_perimeter_layout(
        self, matrix: Matrix, num_chargers: int
    ) -> Matrix:
        """
        Spread *num_chargers* chargers evenly along the perimeter cells.

        Placement is deterministic (evenly spaced by index).  If
        *num_chargers* exceeds the number of perimeter cells, every
        perimeter cell receives a charger.

        Parameters
        ----------
        matrix : Matrix
            Working copy (mutated in place).
        num_chargers : int

        Returns
        -------
        The modified matrix.
        """
        perimeter = self.find_perimeter_cells(matrix)
        if not perimeter:
            logger.warning(
                "Pipeline 4: no navigable cells found on the top or bottom rows."
            )
            return matrix

        n_place = min(num_chargers, len(perimeter))
        selected = _evenly_spaced_sample(perimeter, n_place)

        for r, c in selected:
            matrix[r][c] = CHARGER

        logger.info(
            "Perimeter layout: %d charger(s) spread across %d perimeter cell(s).",
            len(selected), len(perimeter),
        )
        return matrix

    def _pipeline_perimeter(self, work: Matrix) -> Matrix:
        num_chargers: int = int(self.config.get("num_chargers", 10))
        return self.apply_perimeter_layout(work, num_chargers)

    # ═════════════════════════════════════════════════════════════════════════
    #  Private helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _navigable_cells(self, matrix: Matrix) -> Set[Cell]:
        """Return the set of all (row, col) cells whose value is NAVIGABLE (0)."""
        return {
            (int(r), int(c))
            for r in range(matrix.shape[0])
            for c in range(matrix.shape[1])
            if matrix[r][c] == NAVIGABLE
        }
