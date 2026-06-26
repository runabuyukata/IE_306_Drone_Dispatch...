"""Feature extraction for the discrete dispatcher (DroneDispatch-v0).

Turns the gymnasium Dict observation into fixed-width per-entity feature tensors
for a *factored*, dimension-robust policy:

  - per-drone features          (n_drones, FD)
  - per-order features          (k_max,    FO)
  - per-(drone, order) features  (n_drones, k_max, FP)   <- routed distances
  - global features             (FG,)

The per-(drone, order) block carries the **routed, no-fly-aware distance** that
the GreedyNearest baseline uses, plus deadline-feasibility signals, so the policy
has the same information greedy does (computed only from obs["grid"], so we stay
inside the frozen Policy contract). The grid is static within an episode, so BFS
distance fields are memoised per source cell and reused across the whole episode.

Everything is config-agnostic: shapes depend on n_drones/k_max/H/W taken from the
runtime Config, never hard-coded, so a model trained on one config still runs on
the held-out grading config.
"""
from __future__ import annotations

import numpy as np

from drone_dispatch_env.config import Config, NOFLY, CHARGER
from .routing import LocalRouter

# drone row layout (env_dispatch._obs): [x, y, soc, alive, status_onehot(5), has_order]
_D_X, _D_Y, _D_SOC, _D_ALIVE = 0, 1, 2, 3
_D_STATUS0, _D_HASORDER = 4, 9
# order row layout: [ox, oy, dx, dy, age]
_O_OX, _O_OY, _O_DX, _O_DY, _O_AGE = 0, 1, 2, 3, 4

FD = 12   # per-drone feature width
FO = 8    # per-order feature width
FP = 5    # per-(drone, order) feature width
FG = 6    # global feature width

_BIG = 1.0e6


class RoutedCache:
    """Memoised no-fly-aware BFS distance fields for one episode.

    Keyed on the grid bytes, so it auto-resets when a new episode supplies a new
    grid. `field(src)` returns routed distances from `src` to every cell; the
    nearest-charger field is precomputed once per grid.
    """

    def __init__(self, neighborhood: int = 4):
        self.neighborhood = neighborhood
        self._grid_key = None
        self._router: LocalRouter | None = None
        self._fields: dict[tuple[int, int], np.ndarray] = {}
        self._charger_field: np.ndarray | None = None

    def _ensure(self, grid: np.ndarray) -> None:
        key = grid.tobytes()
        if key != self._grid_key:
            self._grid_key = key
            self._router = LocalRouter(grid, self.neighborhood)
            self._fields = {}
            chargers = np.argwhere(grid == CHARGER)
            if len(chargers) == 0:
                self._charger_field = np.full(grid.shape, np.inf)
            else:
                stacked = np.stack([self._router.dist_field((int(cx), int(cy)))
                                    for cx, cy in chargers], axis=0)
                self._charger_field = stacked.min(axis=0)

    def field(self, grid: np.ndarray, src: tuple[int, int]) -> np.ndarray:
        self._ensure(grid)
        f = self._fields.get(src)
        if f is None:
            f = self._router.dist_field(src)
            self._fields[src] = f
        return f

    def charger_field(self, grid: np.ndarray) -> np.ndarray:
        self._ensure(grid)
        return self._charger_field


def extract_features(obs: dict, cfg: Config, cache: RoutedCache) -> dict:
    """Return per-entity feature arrays + validity masks for one observation."""
    n_drones, k_max = cfg.n_drones, cfg.k_max
    H, W = cfg.H, cfg.W
    dscale = float(H + W)
    sla = float(max(cfg.sla_steps, 1))

    drones = np.asarray(obs["drones"], dtype=np.float32)        # (n_drones, 10)
    orders = np.asarray(obs["orders"], dtype=np.float32)        # (k_max, 5)
    grid = np.asarray(obs["grid"])
    mask = np.asarray(obs["action_mask"], dtype=np.int8)        # (n_actions,)
    t_norm = float(obs["time"][0])

    # --- which orders / drones are real ------------------------------------
    assign_mask = mask[: n_drones * k_max].reshape(n_drones, k_max).astype(bool)
    order_valid = assign_mask.any(axis=0)                       # (k_max,)
    drone_alive = drones[:, _D_ALIVE] > 0.5                     # (n_drones,)

    charger_f = cache.charger_field(grid)

    # integer drone positions (valid cells; clipped for safe fancy-indexing)
    dxs = np.clip(drones[:, _D_X].astype(int), 0, H - 1)
    dys = np.clip(drones[:, _D_Y].astype(int), 0, W - 1)
    soc_range = drones[:, _D_SOC] / max(cfg.e_move, 1e-6)        # reach in cells

    # --- per-drone features (vectorised) -----------------------------------
    hub = charger_f[dxs, dys].astype(np.float32)
    hub = np.where(np.isfinite(hub), hub, dscale * 2.0)
    df = np.zeros((n_drones, FD), dtype=np.float32)
    df[:, 0] = drones[:, _D_X] / H
    df[:, 1] = drones[:, _D_Y] / W
    df[:, 2] = drones[:, _D_SOC]
    df[:, 3] = drones[:, _D_ALIVE]
    df[:, 4:9] = drones[:, _D_STATUS0:_D_STATUS0 + 5]
    df[:, 9] = drones[:, _D_HASORDER]
    df[:, 10] = np.minimum(hub / dscale, 4.0)
    df[:, 11] = np.minimum(soc_range / dscale, 4.0)

    # --- per-order + per-(drone,order) routed features ----------------------
    #     (inner loop over drones is vectorised; one BFS field per order origin)
    of = np.zeros((k_max, FO), dtype=np.float32)
    pf = np.zeros((n_drones, k_max, FP), dtype=np.float32)
    alive_col = drone_alive[:, None].astype(np.float32)
    for s in range(k_max):
        if not order_valid[s]:
            continue
        ox, oy = int(orders[s, _O_OX]), int(orders[s, _O_OY])
        odx, ody = int(orders[s, _O_DX]), int(orders[s, _O_DY])
        age = float(orders[s, _O_AGE])
        ttd_steps = sla - age                                  # steps to deadline
        pick_field = cache.field(grid, (ox, oy))
        trip = pick_field[odx, ody] if (0 <= odx < H and 0 <= ody < W) else np.inf
        if not np.isfinite(trip):
            trip = dscale * 2.0

        of[s, 0] = orders[s, _O_OX] / H
        of[s, 1] = orders[s, _O_OY] / W
        of[s, 2] = orders[s, _O_DX] / H
        of[s, 3] = orders[s, _O_DY] / W
        of[s, 4] = age / sla
        of[s, 5] = ttd_steps / sla
        of[s, 6] = min(trip / dscale, 4.0)
        of[s, 7] = 1.0

        to_pick = pick_field[dxs, dys].astype(np.float32)
        to_pick = np.where(np.isfinite(to_pick), to_pick, dscale * 2.0)
        total = to_pick + trip
        pf[:, s, 0] = np.minimum(to_pick / dscale, 4.0)
        pf[:, s, 1] = np.minimum(total / dscale, 4.0)
        pf[:, s, 2] = (soc_range >= total).astype(np.float32)         # battery-feasible
        pf[:, s, 3] = np.clip((soc_range - total) / dscale, -4.0, 4.0)
        pf[:, s, 4] = (total <= ttd_steps).astype(np.float32)         # deadline-feasible
        pf[:, s, :] *= alive_col                                      # zero dead drones

    # --- global features ----------------------------------------------------
    n_alive = float(drone_alive.sum())
    status = drones[:, _D_STATUS0:_D_STATUS0 + 5]
    n_idle = float(status[:, 0].sum())                          # IDLE one-hot col
    soc_all = drones[:, _D_SOC]
    gf = np.array([
        t_norm,
        n_idle / n_drones,
        n_alive / n_drones,
        float((soc_all < cfg.charge_threshold).mean()),
        float(order_valid.sum()) / k_max,
        float(soc_all.mean()),
    ], dtype=np.float32)

    return {
        "drone": df,                       # (n_drones, FD)
        "order": of,                       # (k_max, FO)
        "pair": pf,                        # (n_drones, k_max, FP)
        "global": gf,                      # (FG,)
        "drone_alive": drone_alive.astype(np.float32),   # (n_drones,)
        "order_valid": order_valid.astype(np.float32),   # (k_max,)
        "mask": mask.astype(np.float32),                 # (n_actions,)
    }
