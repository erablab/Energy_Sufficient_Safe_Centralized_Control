# rrt_space_cbf_planner.py
import numpy as np
from trajectory import Trajectory
from rrt import RRT

# -----------------------------
# Utilities
# -----------------------------
class NullEnv:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

def inside_mask_xy(pos_xy, xmin, ymin, h, Nx, Ny, mask):
    ix = int(round((pos_xy[0] - xmin) / h))
    iy = int(round((pos_xy[1] - ymin) / h))
    if ix < 0 or ix >= Ny or iy < 0 or iy >= Nx:
        return True  # outside bounds treated as invalid
    return bool(mask[iy, ix])

def dist_to_charge_set_circle(pos_xy, c, rC) -> float:
    d = float(np.linalg.norm(pos_xy - c))
    return max(0.0, d - rC)

# -----------------------------
# Interpolator for point robot
# -----------------------------
def point2d_interpolator(s0, s1, *, u_max: float, segment_len: float, dt:float):
    """
    Return a Trajectory object (q, qd, qdd, duration) as expected by rrt.py.
    - state is [x,y]
    - q(t) is position
    - qd(t) is constant control u
    - qdd(t) is zero
    """
    s0 = np.asarray(s0, float).ravel()
    s1 = np.asarray(s1, float).ravel()
    p0 = s0[:2]
    p1 = s1[:2]

    v = p1 - p0
    dist = float(np.linalg.norm(v))
    if dist < 1e-12:
        # dummy trajectory of zero duration
        return Trajectory(
            q=lambda t: p0.copy(),
            qd=lambda t: np.zeros(2),
            qdd=lambda t: np.zeros(2),
            duration=0.0
        )

    # only go up to segment_len (local extension)
    pT = p0 + (min(segment_len, dist) / dist) * v

    # constant u to go from p0 to pT in duration T
    # choose T so ||u|| <= u_max, with u = (pT-p0)/T
    dp = pT - p0

    u = dp / dt
    un = float(np.linalg.norm(u))
    if un > u_max:
        u = (u_max / un) * u

    def q(t):
        tt = min(max(t, 0.0), dt)
        return p0 + u * tt

    def qd(t):
        return u

    def qdd(t):
        return np.zeros(2)

    return Trajectory(q=q, qd=qd, qdd=qdd, duration=dt)


# =============================
# Shared Space-CBF (FD5) module
# =============================

_SPACE_CBF_CACHE = {}   # key -> SpaceCBF_FD5 instance


def _periodic_delta(coord, center, L):
    d = np.abs(coord - center)
    return np.minimum(d, L - d)


class SpaceCBF_FD5:
    """
    Shared fast Space-CBF evaluator using FD5 stencil.

    Constraint:
      alpha_h*(epsilon - h^2*sum( (sum rho)^2 )) - 2*h^2*sum( (sum rho)*(sum rho_dot) ) >= 0

    Notes:
      - Mask is shared, not per-robot.
      - Grid neighbor indices are shared, precomputed once.
    """
    def __init__(self, *, X, Y, Nx, Ny, L, h, sigma, lap_coeff, Mask_bool, epsilon, alpha_h):
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.L = float(L)
        self.h = float(h)
        self.sigma = float(sigma)
        self.lap_coeff = float(lap_coeff)
        self.epsilon = float(epsilon)
        self.alpha_h = float(alpha_h)

        self.Xf = X.ravel().astype(float)
        self.Yf = Y.ravel().astype(float)
        self.Mask_f = Mask_bool.ravel().astype(float)

        N = self.Nx * self.Ny
        inds = np.arange(N, dtype=np.int32).reshape(self.Nx, self.Ny)
        self.iR = np.roll(inds, -1, axis=1).ravel()
        self.iL = np.roll(inds,  1, axis=1).ravel()
        self.iU = np.roll(inds, -1, axis=0).ravel()
        self.iD = np.roll(inds,  1, axis=0).ravel()

    def _rho_all(self, positions_all):
        positions_all = np.asarray(positions_all, float)
        cx = positions_all[:, 0][:, None]   # (J,1)
        cy = positions_all[:, 1][:, None]

        Xr = self.Xf[None, :]
        Yr = self.Yf[None, :]

        dx = _periodic_delta(Xr, cx, self.L)
        dy = _periodic_delta(Yr, cy, self.L)
        return np.exp(-0.5 * (dx*dx + dy*dy) / (self.sigma*self.sigma))  # (J,N)

    def cbf_lhs(self, *, positions_all, u_all):
        """
        positions_all: (J,2)
        u_all: (J,2) controls for all robots
        returns scalar LHS
        """
        positions_all = np.asarray(positions_all, float)
        u_all = np.asarray(u_all, float)

        rhoC = self._rho_all(positions_all)  # (J,N)
        rhoR = rhoC[:, self.iR]
        rhoL = rhoC[:, self.iL]
        rhoU = rhoC[:, self.iU]
        rhoD = rhoC[:, self.iD]

        gx = (rhoR - rhoL) / (2.0 * self.h)
        gy = (rhoU - rhoD) / (2.0 * self.h)
        gx *= -1.0
        gy *= -1.0

        lap = (rhoR + rhoL + rhoU + rhoD - 4.0 * rhoC) / (self.h*self.h)
        b   = self.lap_coeff * lap

        sum_rho = np.sum(rhoC, axis=0)  # (N,)

        # sum rho_dot = sum b  - sum_i (gx_i*ux_i + gy_i*uy_i)
        sum_rho_dot = np.sum(b, axis=0)
        sum_rho_dot += np.sum(gx * u_all[:, 0:1], axis=0)
        sum_rho_dot += np.sum(gy * u_all[:, 1:2], axis=0)

        # masked signals
        s = sum_rho * self.Mask_f
        s_dot = sum_rho_dot * self.Mask_f

        hh = self.h * self.h
        lhs = (self.alpha_h * (self.epsilon - hh * float(np.sum(s*s)))
               - 2.0 * hh * float(np.sum(s * s_dot)))
        return lhs

    def feasible(self, *, positions_all, u_all):
        flag = self.cbf_lhs(positions_all=positions_all, u_all=u_all) >= 0.0
        return flag


def get_shared_space_cbf(
    *,
    X, Y, invariance_mask,
    L, sigma, lap_coeff, epsilon, alpha_h
):
    """
    Shared builder: returns the cached SpaceCBF_FD5 for this exact grid+mask config.
    No per-robot masks. Built once.
    """
    Nx, Ny = X.shape[0], X.shape[1]
    h = float(L) / float(Nx - 1)

    key = (
        int(Nx), int(Ny), float(L), float(h),
        float(sigma), float(lap_coeff), float(epsilon), float(alpha_h),
        id(X), id(Y), id(invariance_mask)
    )
    obj = _SPACE_CBF_CACHE.get(key, None)
    if obj is None:
        obj = SpaceCBF_FD5(
            X=X, Y=Y, Nx=Nx, Ny=Ny, L=L, h=h,
            sigma=sigma, lap_coeff=lap_coeff,
            Mask_bool=invariance_mask,
            epsilon=epsilon, alpha_h=alpha_h
        )
        _SPACE_CBF_CACHE[key] = obj
    return obj



# -----------------------------
# Robot wrapper for their RRT
# -----------------------------
class Robot2D_SpaceCBF:
    """
    Implements the interface rrt.py expects:
      - sample_state()
      - state_dist()
      - check_torque_limits(q, qd, qdd)
      - env context
    """

    def __init__(
        self,
        *,
        bounds,
        X, Y,
        invariance_mask,
        invariance_mask_inflated, # inflated mask for faster computation
        charge_mask,
        u_max: float,
        use_charge_mask: bool = True,
        charge_center=None,
        charge_radius=None,
        space_cbf=None,
        robot_index,
        positions_all_fn=None,
        use_space_cbf: bool = True,
        sigma,
        lap_coeff,
        epsilon,
        alpha_h,
    ):
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)
        self.X = X
        self.Y = Y
        self.inv_mask = invariance_mask
        self.inv_mask_inflated = invariance_mask_inflated
        self.charge_mask = charge_mask
        # Precompute charge points once (shared mask)
        if self.charge_mask is not None:
            ys, xs = np.where(self.charge_mask)
            # X, Y are meshgrid; pick coordinates of True cells
            self.charge_pts = np.column_stack((self.X[ys, xs], self.Y[ys, xs])).astype(float)
        else:
            self.charge_pts = None
        self.u_max = float(u_max)
        self.env = NullEnv()

        self.use_charge_mask = use_charge_mask
        self.charge_center = None if charge_center is None else np.asarray(charge_center, float).reshape(2)
        self.charge_radius = float(charge_radius) if charge_radius is not None else None
        ys, xs = np.where(self.charge_mask)
        self.charge_pts = np.column_stack((self.X[ys, xs], self.Y[ys, xs])).astype(float)

        self.space_cbf = space_cbf
        self.robot_index = int(robot_index)
        self.positions_all_fn = positions_all_fn
        self.use_space_cbf = bool(use_space_cbf)
        self.robot_index = int(robot_index)
        self.positions_all_fn = positions_all_fn
        xmin, xmax, ymin, ymax = self.bounds
        self._xmin = xmin
        self._ymin = ymin
        self._h = float(self.X[0, 1] - self.X[0, 0])
        self._Nx, self._Ny = self.X.shape

        self.space_cbf = None
        if self.use_space_cbf:
            self.space_cbf = get_shared_space_cbf(
                X=self.X, Y=self.Y,
                invariance_mask=self.inv_mask,
                L=(self.bounds[1] - self.bounds[0]),  # assumes square domain [-L/2,L/2]
                sigma=float(sigma),
                lap_coeff=float(lap_coeff),
                epsilon=float(epsilon),
                alpha_h=float(alpha_h),
            )

    def sample_state(self):
        xmin, xmax, ymin, ymax = self.bounds
        xy = np.array([np.random.uniform(xmin, xmax),
                       np.random.uniform(ymin, ymax)], float)
        return np.hstack([xy, np.zeros(2)])  # [x,y,vx,vy]

    def _dist_to_charge_pts(self, q):
        if self.charge_pts is None or self.charge_pts.size == 0:
            return float("inf")
        # vectorized distance to all charge points
        d = self.charge_pts - q[None, :]
        return float(np.min(np.sqrt(np.sum(d * d, axis=1))))

    def state_dist(self, s0, s1):
        s0 = np.asarray(s0, float).ravel()
        s1 = np.asarray(s1, float).ravel()
        p0 = s0[:2]
        p1 = s1[:2]

        # We detect goal by comparing p1 to charge_center (since we set goal_state that way).
        if self.charge_center is not None and np.allclose(p1, self.charge_center, atol=1e-12):
            # success if inside charge_mask
            if inside_mask_xy(p0, self._xmin, self._ymin, self._h, self._Nx, self._Ny, self.charge_mask):
                return 0.0
            # distance-to-charge-set (approx): distance to nearest "True" pixel center
            q = np.asarray(p0, float).reshape(2)

            if self.use_charge_mask:
                return self._dist_to_charge_pts(q)
            else:
                return float(np.linalg.norm(q - self.charge_center))

        return float(np.linalg.norm(p0 - p1))

    def check_torque_limits(self, q, qd, qdd):
        """
        rrt.py calls this to check "CBF feasibility at this point".
        """
        p = np.asarray(q, float).reshape(2)
        u = np.asarray(qd, float).reshape(2)

        # bounds
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
            return False

        # limit on control magnitude
        if float(np.linalg.norm(u)) > self.u_max + 1e-9:
            return False

        # only check cbf is already in inflated zone, saves computing speed in most regions
        if inside_mask_xy(p, self._xmin, self._ymin, self._h, self._Nx, self._Ny, self.inv_mask_inflated):

            # Default Space-CBF check
            if self.use_space_cbf and (self.space_cbf is not None) and (self.positions_all_fn is not None):
                pos_all = np.asarray(self.positions_all_fn(), float).copy()  # (J,2)
                pos_all[self.robot_index, :] = p  # overwrite with current edge sample

                # Build u_all: only this robot uses u, others 0 (unless you later want something else)
                u_all = np.zeros((pos_all.shape[0], 2), dtype=float)
                u_all[self.robot_index, :] = u

                passflag = bool(self.space_cbf.feasible(positions_all=pos_all, u_all=u_all))
                return passflag

        return True


# -----------------------------
# Planner callable
# -----------------------------
def plan_one_robot_rrt(
    *,
    init_xy,
    robot: Robot2D_SpaceCBF,
    u_max: float,
    segment_len: float,
    max_extend_steps: int,
    steer_to_goal_every: int,
    k: int,
    rng_seed: int = 0,
    dt: float,
):
    """
    Returns:
      - path_points: (N,2) array of waypoints (including init and final)
      - rrt_obj: the RRT instance (optional for debugging)
    """
    np.random.seed(rng_seed)

    def interpolator(s0, s1):
        return point2d_interpolator(s0, s1, u_max=u_max, segment_len=segment_len, dt=dt)

    rrt = RRT(
        robot,
        init_state=np.hstack([np.asarray(init_xy, float).reshape(2), np.zeros(2)]),
        goal_state = np.hstack([robot.charge_center, np.zeros(2)]),
        state_interpolator=interpolator,
        steer_to_goal_every=steer_to_goal_every,
        k=k,
    )

    for _ in range(max_extend_steps):
        x_rand = robot.sample_state()
        rrt.step(x_rand)
        if rrt.solution is not None:
            break

    if rrt.solution is None:
        return None, rrt

    pts = [init_xy[:2]]
    for seg in rrt.solution:
        pts.append(seg.q(seg.duration))
    path_points = np.vstack(pts)

    return path_points, rrt

# -----------------------------
# Path metrics + DV helpers
# -----------------------------
def path_length(path_points):
    if path_points is None or len(path_points) < 2:
        return 0.0
    dif = np.diff(path_points, axis=0)
    seg_len = np.linalg.norm(dif, axis=1)
    return float(np.sum(seg_len))

def dv_from_path_lookahead(
    path_points,
    k_steps=5,
    beta=0,
    min_norm=1e-12
):
    """
    Weighted average of first k segment vectors.
    Returns unit DV and raw averaged vector.
    """
    if path_points is None:
        z = np.zeros(2, dtype=float)
        return z, z

    p = np.asarray(path_points, float)
    if p.shape[0] < 2:
        z = np.zeros(2, dtype=float)
        return z, z

    segs = p[1:] - p[:-1]               # (M,2)
    m = segs.shape[0]
    k = int(max(1, min(k_steps, m)))
    s = segs[:k]

    idx = np.arange(k, dtype=float)
    w = np.exp(-beta * idx)             # (k,)
    v = (w[:, None] * s).sum(axis=0)    # raw averaged vector

    n = float(np.linalg.norm(v))
    if n < min_norm:
        z = np.zeros(2, dtype=float)
        return z, v
    return v / n, v

