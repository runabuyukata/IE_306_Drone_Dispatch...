"""Self-contained no-fly-aware BFS routing for the Role B feature extractor.

Reimplements the simulator's shortest-path distance field locally so Role B
depends only on the public observation contract (``obs["grid"]``) and ``Config``,
not on the simulator-internal ``drone_dispatch_env.world`` module. The BFS is
identical to the simulator's router (same 4-/8-neighborhood, never enters NOFLY
cells, unit step cost), so the routed distances match exactly — existing trained
weights stay valid.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from drone_dispatch_env.config import NOFLY

_MOVES4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_MOVES8 = _MOVES4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class LocalRouter:
    """BFS shortest-path distances on a static grid that never enter no-fly cells.

    Mirrors ``drone_dispatch_env.world.Router`` (same neighborhood and passability
    rule); ``dist_field`` is the only routine the feature extractor needs.
    """

    def __init__(self, grid: np.ndarray, neighborhood: int = 4):
        self.grid = grid
        self.H, self.W = grid.shape
        self.moves = _MOVES8 if neighborhood == 8 else _MOVES4

    def passable(self, x: int, y: int) -> bool:
        return 0 <= x < self.H and 0 <= y < self.W and self.grid[x, y] != NOFLY

    def dist_field(self, source) -> np.ndarray:
        """Single-source BFS: routed distance from ``source`` to every reachable
        cell, as a float ``[H, W]`` array; unreachable / no-fly cells are ``inf``."""
        dist = np.full((self.H, self.W), np.inf, dtype=np.float64)
        if not self.passable(*source):
            return dist
        dist[source] = 0.0
        q = deque([source])
        while q:
            cx, cy = q.popleft()
            d0 = dist[cx, cy]
            for dx, dy in self.moves:
                nx, ny = cx + dx, cy + dy
                if self.passable(nx, ny) and dist[nx, ny] == np.inf:
                    dist[nx, ny] = d0 + 1.0
                    q.append((nx, ny))
        return dist
