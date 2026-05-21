"""
Process-pool asynchronous RRT planner server.

Main idea:
  - Start this server once.
  - The controller sends robot positions every frame to /submit_positions.
  - /submit_positions returns immediately with the latest completed cached path.
  - A background planner thread recomputes paths from the newest submitted positions.
  - RRT computation is distributed across a persistent ProcessPoolExecutor.
  - Worker processes keep static grid/mask/environment data alive, avoiding per-frame pool creation.
  - The main simulation should call /initialize first and wait until a valid path exists for every robot.

Run:
    python rrt_persistent_pool_server.py --host 127.0.0.1 --port 8765

Useful endpoints:
    GET  /health
    GET  /latest
    POST /initialize        JSON: {"robot_positions": [[x,y], ...], "frame": 0, "timeout_s": 300}
    POST /submit_positions  JSON: {"robot_positions": [[x,y], ...], "frame": int}

Assumes these files are importable from the same folder / PYTHONPATH:
    rrt_path_cbf.py, rrt.py, trajectory.py
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import numpy as np
from scipy.ndimage import binary_dilation

from rrt_path_cbf import (
    Robot2D_SpaceCBF,
    dv_from_path_lookahead,
    path_length,
    plan_one_robot_rrt,
)

# =============================================================================
# USER SETTINGS
# =============================================================================

# Number of worker PROCESSES used for RRT computation.
# If None, defaults to max(1, os.cpu_count() - 1).
RRT_NUM_WORKERS: int | None = 10 # for 10 robots

# Optional multiprocessing start method.
# None = Python default. On macOS this is usually "spawn". On Linux, "fork" can be faster.
# Valid examples: None, "spawn", "fork", "forkserver" depending on OS support.
RRT_PROCESS_START_METHOD: str | None = None

# How long /initialize waits for the first complete all-robot RRT result.
DEFAULT_INITIALIZE_TIMEOUT_S = 300.0

# Used to inflate the time and control magnitude for coarse but much faster RRT planner.
speedup_t = 10
speedup_u = 2

# Planner/environment parameters copied from Controller_sim.py
L = 10.0
Nx, Ny = 101, 101
dt = 0.02 * speedup_t

u_max = 1.0
lap_coeff = 0.045 * u_max
epsilon = 0.01
alpha_h = 0.05
sigma_energy = 0.15
u_max = 1.0 * speedup_u

RRT_MAX_EXTEND_STEPS = 1000
RRT_STEER_TO_GOAL_EVERY = 1
RRT_K = 20
RRT_BASE_SEED = 12
RRT_SEGMENT_LEN = u_max * dt
RRT_USE_SPACE_CBF = True

# This is a very interesting parameter. Setting it too high can cause the averaged direction invalidate CBF checks
# but setting it too low can cause circling in place without actually moving. These 2 values seems to work well.
DV_LOOKAHEAD_STEPS = 5
DV_LOOKAHEAD_BETA = 0.25

# Circles: (center_x, center_y, radius)
circle_regions = [
    (-1.5, 3.0, 0.6),
    (-2.5, -1.5, 0.5),
]

# Rectangles: (center_x, center_y, half_width, half_height)
rectangle_regions = [
    (2.5, 1.0, 0.5, 0.5),   # square
    (2.1, -1.2, 0.5, 0.8),
]

# Charger region.
charger_x, charger_y = 0,0
charger_radius = 1.0

# Inflated hard-check mask used inside the RRT collision/CBF filter.
MASK_INFLATION_ITERS = 3

# Path-length stabilization for RRT value-function smoothing.
# A new RRT path is accepted only if its length is no worse than
# previous_path_length + distance(current_position, previous_path_start).
# Otherwise, the server returns [current_position -> previous_path_start] + previous_path.
PATH_LENGTH_SWITCH_TOL = 1e-9

# If robot is within this margin outside the charger boundary,
# skip RRT and return a direct straight-line path to the closest point on the charger.
# This is added as rrt tends to 'circle around' when starting near the goal region.
# Also, physically this can mean that robot sees the charging station at very close distance, so no more RRT.
RRT_DIRECT_GOAL_MARGIN = 0.2

# =============================================================================
# JSON helpers
# =============================================================================
def _finite_or_none(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _finite_or_none(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    return value


def _paths_to_json(paths: list[Any]) -> list[Any]:
    out: list[Any] = []
    for p in paths:
        if p is None:
            out.append(None)
        else:
            out.append(_finite_or_none(np.asarray(p, dtype=float)))
    return out


# =============================================================================
# Path stabilization helpers
# =============================================================================
def _valid_path_array(path: Any) -> np.ndarray | None:
    """Return a finite (N,2) path array, or None if invalid."""
    if path is None:
        return None
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr

def _prepend_current_start_to_old_path(current_start: np.ndarray, old_path: np.ndarray) -> np.ndarray:
    """
    Construct [b' -> a'] + a, where:
      b' = current_start
      a' = old_path[0]
      a  = old_path

    If b' and a' are numerically identical, avoid adding a duplicate zero-length first segment.
    """
    current_start = np.asarray(current_start, dtype=float).reshape(1, 2)
    old_path = np.asarray(old_path, dtype=float)
    if np.linalg.norm(current_start[0] - old_path[0]) <= 1e-12:
        return old_path.copy()
    return np.vstack([current_start, old_path])

def _select_length_stable_path(
    *,
    robot_index: int,
    current_start: np.ndarray,
    candidate_path: np.ndarray,
    candidate_length: float,
    candidate_dv: np.ndarray,
    candidate_step: np.ndarray,
    previous_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Stabilize the RRT path value to avoid random path-switch jumps.

    Let:
      a  = previous accepted path
      a' = previous path start
      b  = newly planned RRT path
      b' = current robot position / candidate path start

    Accept b only if:
      len(b) <= len(a) + ||b' - a'||.

    Otherwise return:
      [b' -> a'] + a

    This makes the returned path length non-increasing up to the distance the robot moved
    relative to the previous path start, preventing sudden energy-CBF reserve jumps caused
    by RRT switching between different homotopy paths.
    """
    i = int(robot_index)
    current_start = np.asarray(current_start, dtype=float).reshape(2)
    candidate_path = np.asarray(candidate_path, dtype=float)
    candidate_length = float(candidate_length)

    out = {
        "path": candidate_path,
        "path_length": candidate_length,
        "dv_first": np.asarray(candidate_dv, dtype=float),
        "first_step_vec": np.asarray(candidate_step, dtype=float),
        "used_previous_splice": False,
        "accepted_new_path": True,
        "allowed_length": np.nan,
        "candidate_length": candidate_length,
    }

    if previous_result is None:
        return out

    old_paths = previous_result.get("paths", [])
    if i >= len(old_paths):
        return out

    old_path = _valid_path_array(old_paths[i])
    if old_path is None:
        return out

    old_lengths = previous_result.get("path_lengths", None)
    if old_lengths is not None and i < len(old_lengths) and np.isfinite(old_lengths[i]):
        old_length = float(old_lengths[i])
    else:
        old_length = float(path_length(old_path))

    if not np.isfinite(old_length):
        return out

    bridge_length = float(np.linalg.norm(current_start - old_path[0]))
    allowed_length = old_length + bridge_length
    out["allowed_length"] = allowed_length

    if candidate_length <= allowed_length + float(PATH_LENGTH_SWITCH_TOL):
        return out

    spliced_path = _prepend_current_start_to_old_path(current_start, old_path)
    spliced_length = float(path_length(spliced_path))

    # Important:
    # Use the stabilized/spliced path only for the returned path length.
    # Keep the candidate RRT direction for dV_E so the robot does not get pulled
    # backward toward the previous path start.
    # This is valid since the 'averaged' direction dV_E is generally safe&efficient even if b is slightly longer.
    out.update(
        {
            "path": spliced_path,
            "path_length": spliced_length,
            "dv_first": np.asarray(candidate_dv, dtype=float),
            "first_step_vec": np.asarray(candidate_step, dtype=float),
            "used_previous_splice": True,
            "accepted_new_path": False,
        }
    )
    return out


# =============================================================================
# Worker-process globals and functions
# =============================================================================

_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_X: np.ndarray | None = None
_WORKER_Y: np.ndarray | None = None
_WORKER_MASK_A: np.ndarray | None = None
_WORKER_MASK_A_INFLATED: np.ndarray | None = None
_WORKER_MASK_G: np.ndarray | None = None
_WORKER_BOUNDS: tuple[float, float, float, float] | None = None
_WORKER_CHARGE_CENTER: np.ndarray | None = None
_WORKER_POSITIONS_ALL: np.ndarray | None = None
_WORKER_ROBOT_CACHE: dict[tuple[int, int], Robot2D_SpaceCBF] = {}


def _make_static_config() -> dict[str, Any]:
    return {
        "L": float(L),
        "Nx": int(Nx),
        "Ny": int(Ny),
        "dt": float(dt),
        "lap_coeff": float(lap_coeff),
        "epsilon": float(epsilon),
        "alpha_h": float(alpha_h),
        "sigma_energy": float(sigma_energy),
        "u_max": float(u_max),
        "RRT_MAX_EXTEND_STEPS": int(RRT_MAX_EXTEND_STEPS),
        "RRT_STEER_TO_GOAL_EVERY": int(RRT_STEER_TO_GOAL_EVERY),
        "RRT_K": int(RRT_K),
        "RRT_BASE_SEED": int(RRT_BASE_SEED),
        "RRT_SEGMENT_LEN": float(RRT_SEGMENT_LEN),
        "RRT_USE_SPACE_CBF": bool(RRT_USE_SPACE_CBF),
        "DV_LOOKAHEAD_STEPS": int(DV_LOOKAHEAD_STEPS),
        "DV_LOOKAHEAD_BETA": float(DV_LOOKAHEAD_BETA),
        "circle_regions": [(float(cx), float(cy), float(r)) for cx, cy, r in circle_regions],
        "rectangle_regions": [(float(cx), float(cy), float(half_w), float(half_h)) for cx, cy, half_w, half_h in rectangle_regions],
        "charger_x": float(charger_x),
        "charger_y": float(charger_y),
        "charger_radius": float(charger_radius),
        "MASK_INFLATION_ITERS": int(MASK_INFLATION_ITERS),
        "RRT_DIRECT_GOAL_MARGIN": float(RRT_DIRECT_GOAL_MARGIN),
    }


def _build_masks_from_config(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Lc = float(cfg["L"])
    Nxc = int(cfg["Nx"])
    Nyc = int(cfg["Ny"])

    x_grid = np.linspace(-Lc / 2, Lc / 2, Nxc)
    y_grid = np.linspace(-Lc / 2, Lc / 2, Nyc)
    X, Y = np.meshgrid(x_grid, y_grid)

    mask_a = np.zeros_like(X, dtype=bool)

    # Add circular unsafe/invariance regions
    for cx, cy, r in cfg["circle_regions"]:
        mask_a |= ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2

    # Add rectangular unsafe/invariance regions
    for cx, cy, half_w, half_h in cfg["rectangle_regions"]:
        mask_a |= (
            (np.abs(X - cx) <= half_w) &
            (np.abs(Y - cy) <= half_h)
        )

    mask_g = (
        ((X - cfg["charger_x"]) ** 2 + (Y - cfg["charger_y"]) ** 2)
        <= cfg["charger_radius"] ** 2
    )

    mask_a_inflated = binary_dilation(
        mask_a,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        iterations=int(cfg["MASK_INFLATION_ITERS"]),
    )

    return X, Y, mask_a, mask_a_inflated, mask_g

def _worker_initializer(config: dict[str, Any]) -> None:
    """Runs once inside each worker process."""
    global _WORKER_CONFIG, _WORKER_X, _WORKER_Y, _WORKER_MASK_A, _WORKER_MASK_A_INFLATED
    global _WORKER_MASK_G, _WORKER_BOUNDS, _WORKER_CHARGE_CENTER, _WORKER_ROBOT_CACHE

    _WORKER_CONFIG = dict(config)
    _WORKER_X, _WORKER_Y, _WORKER_MASK_A, _WORKER_MASK_A_INFLATED, _WORKER_MASK_G = _build_masks_from_config(_WORKER_CONFIG)

    Lc = float(_WORKER_CONFIG["L"])
    _WORKER_BOUNDS = (-Lc / 2, Lc / 2, -Lc / 2, Lc / 2)
    _WORKER_CHARGE_CENTER = np.array([_WORKER_CONFIG["charger_x"], _WORKER_CONFIG["charger_y"]], dtype=float)
    _WORKER_ROBOT_CACHE = {}


def _positions_all_fn() -> np.ndarray:
    if _WORKER_POSITIONS_ALL is None:
        raise RuntimeError("Worker positions are not set yet.")
    return _WORKER_POSITIONS_ALL


def _get_cached_worker_robot(robot_index: int, num_robots: int) -> Robot2D_SpaceCBF:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker was not initialized.")
    if _WORKER_X is None or _WORKER_Y is None or _WORKER_MASK_A is None or _WORKER_MASK_A_INFLATED is None:
        raise RuntimeError("Worker static arrays were not initialized.")
    if _WORKER_MASK_G is None or _WORKER_BOUNDS is None or _WORKER_CHARGE_CENTER is None:
        raise RuntimeError("Worker charger/bounds were not initialized.")

    key = (int(num_robots), int(robot_index))
    cached = _WORKER_ROBOT_CACHE.get(key)
    if cached is not None:
        return cached

    robot = Robot2D_SpaceCBF(
        bounds=_WORKER_BOUNDS,
        X=_WORKER_X,
        Y=_WORKER_Y,
        invariance_mask=_WORKER_MASK_A,
        invariance_mask_inflated=_WORKER_MASK_A_INFLATED,
        charge_mask=_WORKER_MASK_G,
        u_max=float(_WORKER_CONFIG["u_max"]),
        use_charge_mask=True,
        charge_center=_WORKER_CHARGE_CENTER,
        charge_radius=float(_WORKER_CONFIG["charger_radius"]),
        robot_index=int(robot_index),
        positions_all_fn=_positions_all_fn,
        use_space_cbf=bool(_WORKER_CONFIG["RRT_USE_SPACE_CBF"]),
        sigma=float(_WORKER_CONFIG["sigma_energy"]),
        lap_coeff=float(_WORKER_CONFIG["lap_coeff"]),
        epsilon=float(_WORKER_CONFIG["epsilon"]),
        alpha_h=float(_WORKER_CONFIG["alpha_h"]),
        space_cbf=None,
    )
    _WORKER_ROBOT_CACHE[key] = robot
    return robot

def _direct_goal_path_if_close(init_xy: np.ndarray) -> dict[str, Any] | None:
    """
    If the robot is close enough to the charger, bypass RRT and return a straight
    path to the closest point on the charger boundary.

    Returns None if the robot is not close enough and RRT should be used.
    """
    if _WORKER_CONFIG is None or _WORKER_CHARGE_CENTER is None:
        raise RuntimeError("Worker charger config was not initialized.")

    x0 = np.asarray(init_xy, dtype=float)
    c = np.asarray(_WORKER_CHARGE_CENTER, dtype=float)

    charger_r = float(_WORKER_CONFIG["charger_radius"])
    margin = float(_WORKER_CONFIG["RRT_DIRECT_GOAL_MARGIN"])

    vec_to_center = c - x0
    d_center = float(np.linalg.norm(vec_to_center))

    # Only activate when close to the charger boundary.
    if d_center > charger_r + margin:
        return None

    # Already inside the charger.
    if d_center <= charger_r:
        path = np.vstack([x0, x0])
        return {
            "path": path,
            "path_length": 0.0,
            "dv_first": np.zeros(2, dtype=float),
            "first_step_vec": np.zeros(2, dtype=float),
        }

    # Unit direction from robot to charger center.
    dv = vec_to_center / max(d_center, 1e-12)

    # Closest point on the charger boundary, not the center.
    closest_goal_point = c - charger_r * dv

    path = np.vstack([x0, closest_goal_point])
    L_i = max(0.0, d_center - charger_r)

    first_step_mag = min(
        L_i,
        float(_WORKER_CONFIG["RRT_SEGMENT_LEN"]),
    )
    first_step_vec = dv * first_step_mag

    return {
        "path": path,
        "path_length": float(L_i),
        "dv_first": dv.astype(float),
        "first_step_vec": first_step_vec.astype(float),
    }

def _worker_plan_one(job: tuple[int, np.ndarray, int]) -> dict[str, Any]:
    """
    Compute one robot's RRT path in a worker process.

    job = (robot_index, positions_all, plan_counter)
    """
    global _WORKER_POSITIONS_ALL

    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker was not initialized.")

    robot_index, positions_all, plan_counter = job
    positions_all = np.asarray(positions_all, dtype=float)
    _WORKER_POSITIONS_ALL = positions_all

    J = int(positions_all.shape[0])
    i = int(robot_index)
    robot = _get_cached_worker_robot(i, J)

    # Near-goal override:
    # If the robot is close to the charger, do not call RRT. Return the direct
    # shortest path to the charger boundary instead.
    direct_goal = _direct_goal_path_if_close(positions_all[i])
    if direct_goal is not None:
        return {
            "robot_index": i,
            "success": True,
            "path": direct_goal["path"],
            "path_length": direct_goal["path_length"],
            "dv_first": direct_goal["dv_first"],
            "first_step_vec": direct_goal["first_step_vec"],
            "direct_goal": True,
        }

    seed = int(_WORKER_CONFIG["RRT_BASE_SEED"] + 1000 * int(plan_counter) + i)
    path_i, _rrt_i = plan_one_robot_rrt(
        init_xy=positions_all[i],
        robot=robot,
        u_max=float(_WORKER_CONFIG["u_max"]),
        segment_len=float(_WORKER_CONFIG["RRT_SEGMENT_LEN"]),
        max_extend_steps=int(_WORKER_CONFIG["RRT_MAX_EXTEND_STEPS"]),
        steer_to_goal_every=int(_WORKER_CONFIG["RRT_STEER_TO_GOAL_EVERY"]),
        k=int(_WORKER_CONFIG["RRT_K"]),
        rng_seed=seed,
        dt=float(_WORKER_CONFIG["dt"]),
    )

    if path_i is None or len(path_i) < 2:
        return {
            "robot_index": i,
            "success": False,
            "path": None,
            "path_length": None,
            "dv_first": [0.0, 0.0],
            "first_step_vec": [0.0, 0.0],
        }

    L_i = path_length(path_i)
    dv_i, step_i = dv_from_path_lookahead(
        path_i,
        k_steps=int(_WORKER_CONFIG["DV_LOOKAHEAD_STEPS"]),
        beta=float(_WORKER_CONFIG["DV_LOOKAHEAD_BETA"]),
    )

    success = bool(np.isfinite(L_i))
    return {
        "robot_index": i,
        "success": success,
        "path": np.asarray(path_i, dtype=float),
        "path_length": float(L_i) if success else None,
        "dv_first": np.asarray(dv_i, dtype=float),
        "first_step_vec": np.asarray(step_i, dtype=float),
    }


# =============================================================================
# Async manager in the server process
# =============================================================================


class PersistentPoolPlannerManager:
    def __init__(self, num_workers: int | None) -> None:
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)

        self.latest_positions: np.ndarray | None = None
        self.latest_frame: int | None = None
        self.submit_seq = 0
        self.planned_seq = 0

        self.latest_result: dict[str, Any] | None = None
        self.latest_result_seq = 0
        self.latest_result_frame: int | None = None

        self.num_robots: int | None = None
        self.planning = False
        self.stop_flag = False
        self.plan_counter = 0

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 2) - 1)
        self.num_workers = int(max(1, num_workers))

        config = _make_static_config()
        if RRT_PROCESS_START_METHOD is None:
            mp_context = None
        else:
            mp_context = mp.get_context(RRT_PROCESS_START_METHOD)

        pool_kwargs: dict[str, Any] = {
            "max_workers": self.num_workers,
            "initializer": _worker_initializer,
            "initargs": (config,),
        }
        if mp_context is not None:
            pool_kwargs["mp_context"] = mp_context

        self.pool = ProcessPoolExecutor(**pool_kwargs)
        self.thread = threading.Thread(target=self._worker_loop, name="rrt-persistent-pool-manager", daemon=True)
        self.thread.start()

        print(f"[RRT pool] persistent process pool started with {self.num_workers} worker(s)")

    def submit_positions(self, robot_positions: np.ndarray, frame: int | None = None) -> dict[str, Any]:
        robot_positions = np.asarray(robot_positions, dtype=float)
        if robot_positions.ndim != 2 or robot_positions.shape[1] != 2:
            raise ValueError(f"robot_positions must have shape (J,2), got {robot_positions.shape}")

        with self.cond:
            J = int(robot_positions.shape[0])
            if self.num_robots != J:
                self.num_robots = J
                self.latest_result = None
                self.latest_result_seq = 0
                self.latest_result_frame = None
                self.planned_seq = 0
                print(f"[RRT pool] team size set/reset to J={J}")

            self.latest_positions = robot_positions.copy()
            self.latest_frame = None if frame is None else int(frame)
            self.submit_seq += 1
            target_seq = int(self.submit_seq)
            self.cond.notify_all()
            response = self._make_response_locked()
            response["submitted_now_seq"] = target_seq
            return response

    def latest(self) -> dict[str, Any]:
        with self.lock:
            return self._make_response_locked()

    def wait_until_ready(self, target_seq: int, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        with self.cond:
            while True:
                response = self._make_response_locked()
                ready = bool(
                    response.get("has_plan", False)
                    and response.get("all_paths_valid", False)
                    and int(response.get("latest_seq", 0)) >= int(target_seq)
                )
                if ready:
                    response["ready"] = True
                    return response

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    response["ready"] = False
                    response["message"] = "Timed out waiting for initial all-robot RRT plan."
                    return response

                self.cond.wait(timeout=min(0.2, remaining))

    def _make_response_locked(self) -> dict[str, Any]:
        J = int(self.num_robots or 0)
        if self.latest_result is None:
            return {
                "has_plan": False,
                "all_paths_valid": False,
                "planning": bool(self.planning),
                "latest_seq": int(self.latest_result_seq),
                "submitted_seq": int(self.submit_seq),
                "latest_frame": self.latest_result_frame,
                "num_robots": J,
                "source": "none_yet",
                "elapsed_s": None,
                "paths": [],
                "path_lengths": [],
                "dv_first": [],
                "first_step_vecs": [],
                "ok": [],
                "fresh_ok": [],
            }

        result = self.latest_result
        ok = np.asarray(result.get("ok", []), dtype=bool)
        return {
            "has_plan": True,
            "all_paths_valid": bool(ok.size == J and np.all(ok)),
            "planning": bool(self.planning),
            "latest_seq": int(self.latest_result_seq),
            "submitted_seq": int(self.submit_seq),
            "latest_frame": self.latest_result_frame,
            "num_robots": J,
            "source": result.get("source", "rrt_persistent_pool_cache"),
            "elapsed_s": _finite_or_none(result.get("elapsed_s", None)),
            "paths": _paths_to_json(result["paths"]),
            "path_lengths": _finite_or_none(np.asarray(result["path_lengths"], dtype=float)),
            "dv_first": _finite_or_none(np.asarray(result["dv_first"], dtype=float)),
            "first_step_vecs": _finite_or_none(np.asarray(result["first_step_vecs"], dtype=float)),
            "ok": _finite_or_none(ok),
            "fresh_ok": _finite_or_none(np.asarray(result.get("fresh_ok", []), dtype=bool)),
            "smoothed": _finite_or_none(np.asarray(result.get("smoothed", []), dtype=bool)),
            "accepted_new_path": _finite_or_none(np.asarray(result.get("accepted_new_path", []), dtype=bool)),
        }

    def _worker_loop(self) -> None:
        while True:
            with self.cond:
                while not self.stop_flag and (self.latest_positions is None or self.submit_seq == self.planned_seq):
                    self.cond.wait(timeout=0.1)
                if self.stop_flag:
                    return

                positions = np.asarray(self.latest_positions, dtype=float).copy()
                frame = self.latest_frame
                seq = int(self.submit_seq)
                self.planning = True

            start = time.perf_counter()
            try:
                result = self._compute_plan(positions)
                result["elapsed_s"] = time.perf_counter() - start
                result["source"] = "rrt_persistent_process_pool"
                result["frame"] = frame
                result["seq"] = seq

                with self.cond:
                    self.latest_result = result
                    self.latest_result_seq = seq
                    self.latest_result_frame = frame
                    self.planned_seq = seq
                    self.planning = False
                    self.cond.notify_all()

                ok = np.asarray(result["ok"], dtype=bool)
                fresh_ok = np.asarray(result["fresh_ok"], dtype=bool)
                smoothed = np.asarray(result.get("smoothed", []), dtype=bool)
                print(
                    f"[RRT pool] planned seq={seq}, frame={frame}, "
                    f"fresh={int(fresh_ok.sum())}/{len(fresh_ok)}, "
                    f"smoothed={int(smoothed.sum())}/{len(smoothed)}, "
                    f"valid_cache={int(ok.sum())}/{len(ok)}, "
                    f"elapsed={result['elapsed_s']:.3f}s"
                )

            except Exception as exc:
                with self.cond:
                    self.planned_seq = seq
                    self.planning = False
                    self.cond.notify_all()
                print(f"[RRT pool] planning failed at seq={seq}, frame={frame}: {exc!r}")

    def _compute_plan(self, positions: np.ndarray) -> dict[str, Any]:
        positions = np.asarray(positions, dtype=float)
        J = int(positions.shape[0])

        with self.lock:
            previous = self.latest_result
            self.plan_counter += 1
            plan_counter = int(self.plan_counter)

        paths: list[Any] = [None] * J
        path_lengths = np.full(J, np.nan, dtype=float)
        dv_first = np.zeros((J, 2), dtype=float)
        first_step_vecs = np.zeros((J, 2), dtype=float)
        ok = np.zeros(J, dtype=bool)          # returned cache is valid/finite
        fresh_ok = np.zeros(J, dtype=bool)   # this cycle accepted a fresh candidate path
        smoothed = np.zeros(J, dtype=bool)   # candidate was rejected and [b' -> a'] + a was used
        accepted_new_path = np.zeros(J, dtype=bool)

        jobs = [(i, positions, plan_counter) for i in range(J)]
        future_to_i = {self.pool.submit(_worker_plan_one, job): job[0] for job in jobs}

        for fut in as_completed(future_to_i):
            i_default = int(future_to_i[fut])
            try:
                out = fut.result()
                i = int(out["robot_index"])
                if bool(out["success"]):
                    selected = _select_length_stable_path(
                        robot_index=i,
                        current_start=positions[i],
                        candidate_path=np.asarray(out["path"], dtype=float),
                        candidate_length=float(out["path_length"]),
                        candidate_dv=np.asarray(out["dv_first"], dtype=float),
                        candidate_step=np.asarray(out["first_step_vec"], dtype=float),
                        previous_result=previous,
                    )

                    paths[i] = selected["path"]
                    path_lengths[i] = float(selected["path_length"])
                    dv_first[i, :] = np.asarray(selected["dv_first"], dtype=float)
                    first_step_vecs[i, :] = np.asarray(selected["first_step_vec"], dtype=float)
                    ok[i] = bool(np.isfinite(path_lengths[i]))
                    accepted_new_path[i] = bool(selected["accepted_new_path"])
                    smoothed[i] = bool(selected["used_previous_splice"])
                    fresh_ok[i] = bool(selected["accepted_new_path"]) and ok[i]

                    if smoothed[i]:
                        print(
                            f"[RRT pool] robot {i}: rejected path-length jump "
                            f"candidate={selected['candidate_length']:.3f} > "
                            f"allowed={selected['allowed_length']:.3f}; "
                            f"using [current -> previous_start] + previous_path"
                        )
                else:
                    raise RuntimeError("RRT returned no solution")
            except Exception as exc:
                i = i_default
                # Runtime fail-safe: keep previous valid output for this robot.
                # During startup there is no previous output, so /initialize will not become ready.
                if previous is not None and i < len(previous["paths"]) and previous["paths"][i] is not None:
                    paths[i] = previous["paths"][i]
                    path_lengths[i] = previous["path_lengths"][i]
                    dv_first[i, :] = previous["dv_first"][i]
                    first_step_vecs[i, :] = previous["first_step_vecs"][i]
                    ok[i] = bool(np.isfinite(path_lengths[i]))
                    fresh_ok[i] = False
                    accepted_new_path[i] = False
                    smoothed[i] = False
                else:
                    ok[i] = False
                    fresh_ok[i] = False
                    accepted_new_path[i] = False
                    smoothed[i] = False
                print(f"[RRT pool] robot {i} planning failed; using previous cache if available. Reason: {exc!r}")

        return {
            "paths": paths,
            "path_lengths": path_lengths,
            "dv_first": dv_first,
            "first_step_vecs": first_step_vecs,
            "ok": ok,
            "fresh_ok": fresh_ok,
            "smoothed": smoothed,
            "accepted_new_path": accepted_new_path,
        }

    def shutdown(self) -> None:
        with self.cond:
            self.stop_flag = True
            self.cond.notify_all()
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.pool.shutdown(wait=False)


MANAGER: PersistentPoolPlannerManager | None = None


# =============================================================================
# HTTP request handler
# =============================================================================
class PlannerRequestHandler(BaseHTTPRequestHandler):
    server_version = "RRTPersistentPoolServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid printing a line for every frame POST.
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self) -> None:  # noqa: N802
        if MANAGER is None:
            self._send_json(503, {"error": "server manager is not initialized"})
            return

        if self.path in {"/", "/health"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "mode": "async_cached_rrt_persistent_process_pool",
                    "num_workers": MANAGER.num_workers,
                    "process_start_method": RRT_PROCESS_START_METHOD,
                    "message": "POST /initialize before simulation; POST /submit_positions every frame during simulation.",
                    "params": {
                        "L": L,
                        "Nx": Nx,
                        "Ny": Ny,
                        "dt": dt,
                        "u_max": u_max,
                        "RRT_MAX_EXTEND_STEPS": RRT_MAX_EXTEND_STEPS,
                        "RRT_K": RRT_K,
                        "RRT_SEGMENT_LEN": RRT_SEGMENT_LEN,
                    },
                },
            )
        elif self.path == "/latest":
            self._send_json(200, MANAGER.latest())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if MANAGER is None:
            self._send_json(503, {"error": "server manager is not initialized"})
            return

        if self.path not in {"/initialize", "/submit_positions", "/plan"}:
            self._send_json(404, {"error": "not found"})
            return

        try:
            payload = self._read_json()
            robot_positions = np.asarray(payload["robot_positions"], dtype=float)
            frame = payload.get("frame", None)

            if self.path == "/initialize":
                timeout_s = float(payload.get("timeout_s", DEFAULT_INITIALIZE_TIMEOUT_S))
                submit_response = MANAGER.submit_positions(robot_positions, frame=frame)
                target_seq = int(submit_response["submitted_now_seq"])
                ready_response = MANAGER.wait_until_ready(target_seq=target_seq, timeout_s=timeout_s)
                self._send_json(200, ready_response)
            else:
                response_payload = MANAGER.submit_positions(robot_positions, frame=frame)
                self._send_json(200, response_payload)

        except Exception as exc:
            self._send_json(
                500,
                {
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                },
            )


# =============================================================================
# Entrypoint
# =============================================================================


def main() -> None:
    global MANAGER

    parser = argparse.ArgumentParser(description="Persistent-process-pool async RRT planner server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--workers",
        type=int,
        default=RRT_NUM_WORKERS,
        help="Number of persistent worker processes for RRT computation.",
    )
    args = parser.parse_args()

    MANAGER = PersistentPoolPlannerManager(num_workers=args.workers)
    server = ThreadingHTTPServer((args.host, args.port), PlannerRequestHandler)

    print(f"[RRT pool] serving on http://{args.host}:{args.port}")
    print("[RRT pool] First call /initialize, then send /submit_positions every frame.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[RRT pool] shutting down")
    finally:
        server.server_close()
        if MANAGER is not None:
            MANAGER.shutdown()


if __name__ == "__main__":
    main()
