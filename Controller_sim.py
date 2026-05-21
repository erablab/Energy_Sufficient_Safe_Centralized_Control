import json
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mosek.fusion as mf
from matplotlib.patches import Circle
import time
import numpy as np
import urllib.error
import urllib.request


total_time = 0
total_time_count = 0
# ----------------------- Parameters -----------------------
L = 10.0
Nx, Ny = 101, 101
Ngrid = Nx*Ny
h = L / (Nx - 1)

alpha_v = 10
u_max = 1
motion_c = 1
lap_coeff = 0.045 * u_max
gemma = 100
epsilon = 0.01
alpha_h = 0.05
sigma_energy = 0.15

MEASURE_NOISE = True
MOTION_NOISE = True
# ----------------------- Grid Setup -----------------------
x_grid = np.linspace(-L / 2, L / 2, Nx)
y_grid = np.linspace(-L / 2, L / 2, Ny)

X, Y = np.meshgrid(x_grid, y_grid)

env = np.array([
    [-2.0, -2.0],
    [ 2.0, -2.0],
    [ 2.0,  2.0],
    [-2.0,  2.0],
    [-2.0, -2.0],
], float)
# ----------------------- Invariance Mask -----------------------

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

def invariance_mask(X, Y):
    mask = np.zeros_like(X, dtype=bool)

    # Add circular regions
    for cx, cy, r in circle_regions:
        circle_region = ((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2
        mask |= circle_region

    # Add rectangular regions
    for cx, cy, half_w, half_h in rectangle_regions:
        rectangle_region = (
            (np.abs(X - cx) <= half_w) &
            (np.abs(Y - cy) <= half_h)
        )
        mask |= rectangle_region

    return mask
# Target PDF
target_pdf_centers = np.asarray([(2.5, -2.5),(2.5, 2.5),(-2.5, -2.5),(-2.5, 2.5)])
# Precompute the mask on the grid
Mask = invariance_mask(X, Y)
Mask_A = invariance_mask(X, Y).astype(bool)

# Charger region
charger_x, charger_y = 0, 0
charger_radius = 1.0
charging_margin = 0.4

Mask_G = (((X - charger_x)**2 + (Y - charger_y)**2) <= charger_radius**2)

# ----------------------- Load Initial Data -----------------------
with open("initial_particles.json", "r") as file:
    particle_data = json.load(file)

# ----------------------- PDF Class -----------------------
class PDF:
    def __init__(self, position, sigma):
        self.position = np.array(position)  # position: [x, y]
        self.sigma = sigma
        self.field = self.compute_pdf(X, Y)
        self.fake_position = []

    def compute_pdf(self, x, y):
        px, py = self.position
        # Compute periodic distance (wrap-around)
        dx = np.abs(x - px)
        dy = np.abs(y - py)
        dx = np.minimum(dx, L - dx)
        dy = np.minimum(dy, L - dy)
        return np.exp(-((dx) ** 2 + (dy) ** 2) / (2 * self.sigma ** 2))

    def update(self, u_opt_i, dt, x_grid, y_grid, charging_flag):
        local_u = u_opt_i
        #motion noise
        if MOTION_NOISE and (not charging_flag):
            unit_noise = np.random.multivariate_normal([0, 0], [[0.3 ** 2, 0], [0, 0.3 ** 2]])
            while np.linalg.norm(unit_noise) >= 1.0:
                unit_noise = np.random.multivariate_normal([0, 0], [[0.3 ** 2, 0], [0, 0.3 ** 2]])
            px, py = unit_noise * np.linalg.norm(local_u) * motion_c
            self.position = self.position + dt * (local_u + np.array([px, py]))
        else:
            self.position = self.position + dt * local_u

        # Clip the position to ensure it stays within the grid boundaries.
        self.position[0] = np.clip(self.position[0], x_grid[0], x_grid[-1])
        self.position[1] = np.clip(self.position[1], y_grid[0], y_grid[-1])

        # Update the PDF based on the new position.
        self.field = self.compute_pdf(X, Y)
        return local_u

    def compute_pdf_random(self, x=X, y=Y):
        # a basic position sampling, in practice, can improve with kalman filter or other methods.
        n_samples = 10
        if MEASURE_NOISE:
            cov = np.array([[self.sigma ** 2, 0.0], [0.0, self.sigma ** 2]], dtype=float)
            samples = np.random.multivariate_normal(self.position, cov, size=n_samples)
            samples[:, 0] = np.clip(samples[:, 0], x_grid[0], x_grid[-1])
            samples[:, 1] = np.clip(samples[:, 1], y_grid[0], y_grid[-1])
            px, py = samples.mean(axis=0)
        else:
            px, py = self.position
        px = np.clip(px, x_grid[0], x_grid[-1])
        py = np.clip(py, y_grid[0], y_grid[-1])

        self.fake_position = [px, py]

        # Compute periodic distance (wrap-around)
        dx = np.abs(x - px)
        dy = np.abs(y - py)
        dx = np.minimum(dx, L - dx)
        dy = np.minimum(dy, L - dy)
        return np.exp(-((dx) ** 2 + (dy) ** 2) / (2 * self.sigma ** 2)), self.fake_position


# Initialize list of PDFs.
pdfs = [PDF(p["position"], sigma_energy) for p in particle_data]
num_pdfs = len(pdfs)

# ----------------------- Energy State -----------------------
alpha_e = 0.1
dt = 0.02
c_e = 0.05 # motion cost
E_min = 0.1
c_0 = 0.009 # constant operation cost
gain = 0.15 # gain on DV, loosely representing the effects of noise + const cost on expected energy decrease.
# Since it is very unlikely that the next step steps exactly in the direction of the path-to-charge.
# Half the motion noise (unit circle pointing away from DV) will cause increase of energy-to-charge,
# with the rest not lining up with the perfect path either.

Charging_flag = np.full(num_pdfs, False, dtype=bool)

E_init = 0.65
E = np.full(num_pdfs, E_init, dtype=float)

energy_history = [E.copy()]   # list of arrays length J
energy_time = [0.0]
energy_min_history = [float(E.min())]
energy_avg_history = [float(E.mean())]


# ----------------------- RRT planner config -----------------------
USE_RRT_PLANNER = True
RRT_UPDATE_PERIOD = 3

# ----------------------- Persistent-pool async RRT planner server config -----------------------
# Start the server first:
#   python rrt_persistent_pool_server.py --host 127.0.0.1 --port 8765 --workers 8
#
# This script calls /initialize once before the simulation starts and waits until
# the server has a valid RRT path for every robot. During the simulation, it sends
# positions every frame to /submit_positions, which returns immediately with the
# latest completed cached path while the server computes newer paths in the background.
RRT_SERVER_BASE_URL = "http://127.0.0.1:8765"
RRT_SERVER_SUBMIT_URL = f"{RRT_SERVER_BASE_URL}/submit_positions"
RRT_SERVER_INITIALIZE_URL = f"{RRT_SERVER_BASE_URL}/initialize"
RRT_SERVER_TIMEOUT_SEC = 0.05          # short timeout during simulation; reuse local cache if missed
RRT_SERVER_INIT_TIMEOUT_SEC = 300.0    # long blocking timeout before frame 0 starts

# Planner caches
rrt_plan_frame = -999
rrt_paths = None
rrt_path_lengths = None
rrt_dv_first = None            # (J,2), unit vector
rrt_first_step_vecs = None     # (J,2), raw vector

def planner_outputs_to_energy_terms(dv_first, path_lengths, *, u_max, c_e, c_0):
    dv_first = np.asarray(dv_first, float)
    path_lengths = np.asarray(path_lengths, float)

    Ve_dist = path_lengths.copy()
    scaleE = (c_e * u_max + c_0) / (0.5 * u_max)
    # 0.5 represents a conservative how much energy to overcome negative effects of motion noise.
    # ie. doubling means due to motion noise, let's conserve twice the energy to be safe. If no noise, 0.5 -> 1.0.
    Ve_energy = Ve_dist * scaleE
    dVex_energy = -dv_first[:, 0] * gain
    dVey_energy = -dv_first[:, 1] * gain

    bad = ~np.isfinite(Ve_energy)
    Ve_energy[bad] = 1e6
    dVex_energy[~np.isfinite(dVex_energy)] = 0.0
    dVey_energy[~np.isfinite(dVey_energy)] = 0.0
    return Ve_energy, dVex_energy, dVey_energy

def target_pdf(points, x, y):
    sigma = 1.5
    points = np.asarray(points)

    pdf = np.zeros_like(x, dtype=float)
    for px, py in points:
        pdf += np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sigma ** 2))

    return pdf

rho_d = target_pdf(target_pdf_centers, X, Y)

class InvarianceControllerSmall:
    """
    Minimal Fusion model:
      min t + gemma*s
      s >= 0, t >= 0
      ||h*vec(u)||^2 <= 2t  (rotated QCone)
      CLF: alpha_V + v_dot - s <= 0
      CBF: alpha_h_psi + h_dot >= 0
      CBF: alpha_E + E_i_dot >= 0 \forall i
    where
      v_dot = c_v + dot(A_vx, ux) + dot(A_vy, uy)
      h_dot = c_h + dot(A_hx, ux) + dot(A_hy, uy)
    u is a [J,2] variable so we can take columns cleanly.
    """
    def __init__(self):
        J = int(num_pdfs)
        M = self.model = mf.Model("Main")
        # Control Variable
        self.u = M.variable("u", [J, 2], mf.Domain.inRange(-u_max, u_max))
        # CLF slack
        self.s = M.variable("s", 1, mf.Domain.greaterThan(0.0))

        # Per-robot control magnitude variables:
        # z_i >= ||u_i||_2
        self.z = M.variable("z", J, mf.Domain.greaterThan(0.0))
        # Take columns: ux = u[:,0], uy = u[:,1]
        ux = self.u.slice([0, 0], [J, 1]).reshape([J])
        uy = self.u.slice([0, 1], [J, 2]).reshape([J])

        # Enforce ||u_i||_2 <= z_i for each robot
        for i in range(J):
            M.constraint(
                f"u_norm_{i}",
                mf.Expr.vstack([
                    self.z.index(i),
                    ux.index(i),
                    uy.index(i),
                ]),
                mf.Domain.inQCone()
            )

        # Objective:
        # min sum_i ||u_i|| + gamma*s
        M.objective(
            "obj",
            mf.ObjectiveSense.Minimize,
                mf.Expr.add(
                    mf.Expr.sum(self.z),
                    mf.Expr.mul(float(gemma), self.s.index(0))
                ),
        )

        # Parameters
        self.a_vx = M.parameter("a_vx", J)
        self.a_vy = M.parameter("a_vy", J)
        self.a_hx = M.parameter("a_hx", J)
        self.a_hy = M.parameter("a_hy", J)
        self.c_v  = M.parameter("c_v",  1)
        self.c_h  = M.parameter("c_h",  1)
        self.alpha_V     = M.parameter("alpha_V", 1)
        self.alpha_h_psi = M.parameter("alpha_h_psi", 1)

        self.Ve = M.parameter("Ve", J)  # V_E at each robot position
        self.dVex = M.parameter("dVex", J)  # dV_E/dx at each robot
        self.dVey = M.parameter("dVey", J)  # dV_E/dy

        self.Ecur = M.parameter("Ecur", J)  # current energy per robot
        self.Emin = M.parameter("Emin", 1)
        self.alpha_e = M.parameter("alpha_e", 1)
        self.c_e = M.parameter("c_e", 1)

        # v_dot, h_dot (scalars)
        v_dot = mf.Expr.add(
            mf.Expr.add(mf.Expr.dot(self.a_vx, ux), mf.Expr.dot(self.a_vy, uy)),
            self.c_v.index(0)
        )
        h_dot = mf.Expr.add(
            mf.Expr.add(mf.Expr.dot(self.a_hx, ux), mf.Expr.dot(self.a_hy, uy)),
            self.c_h.index(0)
        )

        # CLF / CBF
        M.constraint("CLF",
            mf.Expr.sub(mf.Expr.add(self.alpha_V.index(0), v_dot), self.s.index(0)),
            mf.Domain.lessThan(0.0)
        )
        M.constraint("CBF",
            mf.Expr.add(self.alpha_h_psi.index(0), h_dot),
            mf.Domain.greaterThan(0.0)
        )


        for i in range(J):
            # lhs = E_i_dot
            lhs = mf.Expr.add(
                mf.Expr.add(mf.Expr.mul(self.c_e.index(0), self.z.index(i)), c_0),
                mf.Expr.mul(
                    0.5,
                    mf.Expr.add(
                        mf.Expr.mul(self.dVex.index(i), ux.index(i)),
                        mf.Expr.mul(self.dVey.index(i), uy.index(i)),
                    )
                )
            )

            # rhs = alpha_e * (E_i - E_min - V_E_i)
            rhs = mf.Expr.mul(
                    self.alpha_e.index(0),
                    mf.Expr.sub(
                        mf.Expr.sub(self.Ecur.index(i), self.Emin.index(0)),
                        self.Ve.index(i)
                    )
                )

            M.constraint(f"EnergyCBF_{i}",mf.Expr.sub(rhs, lhs),mf.Domain.greaterThan(0.0))

        M.setLogHandler(None)

    def set_params_from_coeffs(self, coeffs, alpha_V_value, alpha_h_psi_value):
        # Ensure ndarray types are float; scalars as Python float in 1-elem lists
        self.a_vx.setValue(np.asarray(coeffs["A_vx"], float))
        self.a_vy.setValue(np.asarray(coeffs["A_vy"], float))
        self.a_hx.setValue(np.asarray(coeffs["A_hx"], float))
        self.a_hy.setValue(np.asarray(coeffs["A_hy"], float))
        self.c_v.setValue([float(coeffs["c_v"])])
        self.c_h.setValue([float(coeffs["c_h"])])
        self.alpha_V.setValue([float(alpha_V_value)])
        self.alpha_h_psi.setValue([float(alpha_h_psi_value)])

    def set_energy_params(self, Ve, dVex, dVey, Ecur, Emin, alpha_e, c_e):
        self.Ve.setValue(np.asarray(Ve, float))
        self.dVex.setValue(np.asarray(dVex, float))
        self.dVey.setValue(np.asarray(dVey, float))
        self.Ecur.setValue(np.asarray(Ecur, float))
        self.Emin.setValue([float(Emin)])
        self.alpha_e.setValue([float(alpha_e)])
        self.c_e.setValue([float(c_e)])

    def solve(self):
        J = int(num_pdfs)

        try:
            self.model.solve()

            sol = self.model.getPrimalSolutionStatus()
            pro = self.model.getProblemStatus()

            # Optional debug print
            # print("MOSEK problem status:", pro)
            # print("MOSEK primal status:", sol)

            return (
                self.u.level().reshape(J, 2),
                self.s.level()[0],
                float(np.sum(self.z.level())),
            )

        except Exception as exc:
            print(f"[MOSEK] solve failed: {exc}")

            try:
                pro = self.model.getProblemStatus()
                sol = self.model.getPrimalSolutionStatus()
                print(f"[MOSEK] problem status: {pro}")
                print(f"[MOSEK] primal status: {sol}")
            except Exception:
                pass

            # Fail-safe output: no motion, large reported slack/cost
            return (
                np.zeros((J, 2), dtype=float),
                1e6,
                0.0,
            )

# ----------------------- Global Data for Plotting -----------------------
time_history = []
h_history = []
error_history = []

# initialize
rho_new = np.sum([pdf.field for pdf in pdfs], axis=0)
error_norm = h ** 2 * np.sum((rho_d - rho_new) ** 2)
error_history.append(error_norm)

h_val = (epsilon - np.sum(rho_new[Mask] ** 2) * h ** 2)
h_history.append(h_val)
time_history.append(0.0)

# ----------------------- Figure Setup with GridSpec -----------------------
font_size = 22
legend_font_size = 16
text_font_size = 16

fig,ax_scatter = plt.subplots(figsize=(8, 6))
ax_scatter.set_title(f"Density Field with Invariance Area \n at t={0.00:.2f}", fontsize=font_size)

ax_scatter.set_xlabel("X", fontsize=font_size, labelpad=-2)
ax_scatter.set_ylabel("Y", fontsize=font_size, labelpad=-6)

ax_scatter.tick_params(
    axis='both',
    which='major',
    labelsize=font_size,
    pad=1
)
ax_scatter.set_xlim(-L / 2, L / 2)
ax_scatter.set_ylim(-L / 2, L / 2)
scat = ax_scatter.scatter([pdf.position[0] for pdf in pdfs],
                          [pdf.position[1] for pdf in pdfs],
                          color='blue', s=50, label='Robot')
ax_scatter.contour(X, Y, Mask_A.astype(float), colors="red", levels=[0.5], linewidths=1.0)

# Charging area
charge_patch = Circle((charger_x, charger_y), charger_radius,
                      edgecolor='cyan', facecolor='cyan', alpha=0.25, linewidth=2)
ax_scatter.add_patch(charge_patch)

im_rho_real = ax_scatter.imshow(rho_d, extent=[-L / 2, L / 2, -L / 2, L / 2], origin='lower', cmap='YlGn', alpha=1)
ax_scatter.scatter(target_pdf_centers[:,0], target_pdf_centers[:,1], color='gold', marker='*', s=150, label='Target Center')
plt.tight_layout(pad=0.3)

energy_texts = []
for i in range(num_pdfs):
    txt = ax_scatter.text(
        pdfs[i].position[0], pdfs[i].position[1],
        f"{E[i]:.2f}",
        fontsize=10, color="black",
        ha="center", va="center"
    )
    energy_texts.append(txt)

# Initialize trajectory lists for each robot
traj_x_list = [[] for _ in range(num_pdfs)]
traj_y_list = [[] for _ in range(num_pdfs)]
traj_lines = []
for i in range(num_pdfs):
    line, = ax_scatter.plot([], [], 'b-', linewidth=1.5)
    traj_lines.append(line)

rho_test = np.sum([pdf.field for pdf in pdfs], axis=0)

contour_levels = [0.05, 0.25, 0.5, 0.7, 0.9]  # Different contour levels for multiple rings
contour = ax_scatter.contour(X, Y, rho_test, levels=contour_levels, colors='black', linewidths=1.5, linestyles='dashed',
               alpha=0.5)

# RRT path
path_lines = []
path_ends = []
for i in range(num_pdfs):
    ln, = ax_scatter.plot([], [], linewidth=1.8, alpha=0.9)   # uses default color cycle
    ed = ax_scatter.scatter([], [], s=35, marker='x')
    path_lines.append(ln)
    path_ends.append(ed)

def update_rrt_path_overlay(paths, path_lines, path_ends):
    J = len(path_lines)
    for i in range(J):
        p = None if (paths is None or i >= len(paths)) else paths[i]

        if p is None or len(p) == 0:
            path_lines[i].set_data([], [])
            path_ends[i].set_offsets(np.empty((0, 2)))
            continue

        p = np.asarray(p, float)
        path_lines[i].set_data(p[:, 0], p[:, 1])
        path_ends[i].set_offsets(p[-1:, :2])  # shape (1,2)

controller = InvarianceControllerSmall()

print("Controller built")

# Fast stencil precomputes
def _periodic_delta(coord, center, L):
    d = np.abs(coord - center)
    return np.minimum(d, L - d)

N = Nx * Ny
inds = np.arange(N, dtype=np.int32).reshape(Nx, Ny)
iR = np.roll(inds, -1, axis=1).ravel()
iL = np.roll(inds,  1, axis=1).ravel()
iU = np.roll(inds, -1, axis=0).ravel()
iD = np.roll(inds,  1, axis=0).ravel()

Xf = X.ravel().astype(float)
Yf = Y.ravel().astype(float)
Mask_f  = Mask.ravel().astype(float)
rho_d_f = rho_d.ravel().astype(float)

def sumfield_coeffs_fd5(
    positions_meas,   # (J,2) measurement centers to evaluate rho this step
    *,
    Xf, Yf, iR, iL, iU, iD,        # flattened grid + neighbor indices
    L, h, sigma, lap_coeff,
    rho_d_f, Mask_f,
    alpha_v, alpha_h, epsilon
):
    """
    Central diffs + 5-point Laplacian.
    Builds all robots' rho at centers ONCE (J*N exps),
    neighbors by indexing (no extra exp), then accumulates CLF/CBF coeffs.
    """
    # -- 1) rho at centers for ALL robots (vectorized) --
    cx = positions_meas[:, 0][:, None]   # (J,1)
    cy = positions_meas[:, 1][:, None]   # (J,1)
    Xr = Xf[None, :]                      # (1,N)
    Yr = Yf[None, :]                      # (1,N)

    dx = _periodic_delta(Xr, cx, L)       # (J,N)
    dy = _periodic_delta(Yr, cy, L)       # (J,N)
    rhoC = np.exp(-0.5 * (dx*dx + dy*dy) / (sigma*sigma))   # (J,N)

    # -- 2) neighbors by indexing centers (cheap) --
    rhoR = rhoC[:, iR]     # (J,N)
    rhoL = rhoC[:, iL]
    rhoU = rhoC[:, iU]
    rhoD = rhoC[:, iD]

    # -- 3) central-diff gradients (then FP minus sign) --
    gx = (rhoR - rhoL) / (2.0 * h)   # (J,N)
    gy = (rhoU - rhoD) / (2.0 * h)
    gx *= -1.0
    gy *= -1.0

    # -- 4) 5-point Laplacian per robot --
    lap = (rhoR + rhoL + rhoU + rhoD - 4.0 * rhoC) / (h*h)   # (J,N)
    b   = lap_coeff * lap

    # -- 5) global sums for CLF/CBF signals --
    sum_rho = np.sum(rhoC, axis=0)     # (N,)
    d_vec   = rho_d_f - sum_rho        # CLF residual at each pixel
    a_vec   = sum_rho * Mask_f         # CBF signal at each pixel

    # -- 6) per-robot dot products (matrix-vector) --
    scale = -2.0 * h*h

    # A_vx[j] = scale * <gx_j, d>
    A_vx = scale * (gx @ d_vec)               # (J,)
    A_vy = scale * (gy @ d_vec)               # (J,)
    c_v  = float(scale * np.sum(b * d_vec[None, :]))   # scalar

    # A_hx[j] = scale * <gx_j, a>, etc.
    A_hx = scale * (gx @ a_vec)               # (J,)
    A_hy = scale * (gy @ a_vec)               # (J,)
    c_h  = float(scale * np.sum(b * a_vec[None, :]))   # scalar

    coeffs = dict(A_vx=A_vx, A_vy=A_vy, A_hx=A_hx, A_hy=A_hy, c_v=c_v, c_h=c_h)

    # -- 7) alphas (full-domain; equal to sum over cells since no partition) --
    alpha_v_value = float(alpha_v * h*h * np.sum(d_vec*d_vec))
    alpha_h_value = float(alpha_h * (epsilon - h*h * np.sum((a_vec*a_vec))))

    return coeffs, alpha_v_value, alpha_h_value


# ----------------------- Persistent-pool async RRT planner server client -----------------------
def _none_to_nan_array(values, *, ndim=None):
    """Convert JSON lists containing None back to float arrays with np.nan."""
    arr = np.asarray(values, dtype=float)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"Expected {ndim}D array from planner server, got shape {arr.shape}")
    return arr


def parse_rrt_server_response(response_payload):
    if "error" in response_payload:
        raise RuntimeError(
            f"RRT planner server error: {response_payload.get('error')}: "
            f"{response_payload.get('message', '')}"
        )

    # Before initialization completes, the server may legitimately have no plan yet.
    if not response_payload.get("has_plan", False):
        return {
            "paths": [],
            "path_lengths": np.array([], dtype=float),
            "dv_first": np.empty((0, 2), dtype=float),
            "first_step_vecs": np.empty((0, 2), dtype=float),
            "ok": np.array([], dtype=bool),
            "fresh_ok": np.array([], dtype=bool),
            "elapsed_s": np.nan,
            "source": response_payload.get("source", "none_yet"),
            "has_plan": False,
            "all_paths_valid": False,
            "planning": bool(response_payload.get("planning", False)),
            "latest_seq": int(response_payload.get("latest_seq", 0)),
            "submitted_seq": int(response_payload.get("submitted_seq", 0)),
            "ready": bool(response_payload.get("ready", False)),
        }

    paths = []
    for path in response_payload["paths"]:
        if path is None:
            paths.append(None)
        else:
            paths.append(np.asarray(path, dtype=float))

    return {
        "paths": paths,
        "path_lengths": _none_to_nan_array(response_payload["path_lengths"], ndim=1),
        "dv_first": _none_to_nan_array(response_payload["dv_first"], ndim=2),
        "first_step_vecs": _none_to_nan_array(response_payload["first_step_vecs"], ndim=2),
        "ok": np.asarray(response_payload.get("ok", []), dtype=bool),
        "fresh_ok": np.asarray(response_payload.get("fresh_ok", []), dtype=bool),
        "elapsed_s": float(response_payload.get("elapsed_s", np.nan)),
        "source": response_payload.get("source", "unknown"),
        "has_plan": bool(response_payload.get("has_plan", False)),
        "all_paths_valid": bool(response_payload.get("all_paths_valid", False)),
        "planning": bool(response_payload.get("planning", False)),
        "latest_seq": int(response_payload.get("latest_seq", 0)),
        "submitted_seq": int(response_payload.get("submitted_seq", 0)),
        "ready": bool(response_payload.get("ready", False)),
    }


def _post_json_to_rrt_server(url, payload, *, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    return parse_rrt_server_response(response_payload)


def initialize_rrt_server(robot_positions, *, timeout_s=RRT_SERVER_INIT_TIMEOUT_SEC):
    """
    Blocking startup handshake.
    The simulation should not start until this returns a valid path for every robot.
    """
    payload = {
        "robot_positions": np.asarray(robot_positions, dtype=float).tolist(),
        "frame": 0,
        "timeout_s": float(timeout_s),
    }
    plan_out = _post_json_to_rrt_server(
        RRT_SERVER_INITIALIZE_URL,
        payload,
        timeout=float(timeout_s) + 5.0,
    )

    if not plan_out["has_plan"] or not plan_out["all_paths_valid"]:
        raise RuntimeError(
            "RRT server did not return a valid initial path for every robot. "
            f"has_plan={plan_out['has_plan']}, all_paths_valid={plan_out['all_paths_valid']}, "
            f"ok={plan_out['ok']}"
        )

    print(
        f"[RRT server init] ready: success={plan_out['ok'].sum()}/{num_pdfs}, "
        f"source={plan_out['source']}, elapsed={plan_out['elapsed_s']:.3f}s"
    )
    return plan_out


def submit_rrt_positions_to_server(robot_positions, frame=None):
    """
    Runtime call. It should return quickly with the latest completed cached path.
    If the server misses the short timeout, return None and keep the previous local cache.
    """
    payload = {
        "robot_positions": np.asarray(robot_positions, dtype=float).tolist(),
        "frame": None if frame is None else int(frame),
    }
    try:
        return _post_json_to_rrt_server(
            RRT_SERVER_SUBMIT_URL,
            payload,
            timeout=RRT_SERVER_TIMEOUT_SEC,
        )
    except Exception as exc:
        print(f"[RRT server] no fresh response; using previous local cache. Reason: {exc}")
        return None


# ----------------------- Animation -----------------------
def animate(frame):
    global time_history, h_history, error_history, slack_history, contour
    global traj_x_list, traj_y_list, traj_lines
    global total_time, total_time_count
    global V_eff_all, dVdx_all, dVdy_all, shell_all, energy_map_built_frame
    global E, Charging_flag
    global rrt_plan_frame, rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs

    if hasattr(animate, "primed") or frame != 0:
        current_time = (frame + 1) * dt
        time_history.append(current_time)
        # Get noised positions
        tmp = [pdf.compute_pdf_random() for pdf in pdfs]
        robot_positions = np.array([t[1] for t in tmp])

        start = time.perf_counter()

        # Always send the newest positions to the async server. This returns immediately
        # with the latest completed cached path, while the server computes the next path
        # in its background worker.
        plan_out = submit_rrt_positions_to_server(robot_positions, frame=frame)

        should_refresh_local_cache = (frame - rrt_plan_frame) >= RRT_UPDATE_PERIOD or (rrt_dv_first is None)
        if should_refresh_local_cache and plan_out is not None:
            rrt_paths = plan_out["paths"]
            rrt_path_lengths = plan_out["path_lengths"]
            rrt_dv_first = plan_out["dv_first"]
            rrt_first_step_vecs = plan_out["first_step_vecs"]
            rrt_plan_frame = frame
            update_rrt_path_overlay(rrt_paths, path_lines, path_ends)
            ok = plan_out.get("ok", np.isfinite(plan_out["path_lengths"]))
            fresh_ok = plan_out.get("fresh_ok", ok)
            print(
                f"[RRT server] source={plan_out.get('source')}, "
                f"has_plan={plan_out.get('has_plan')}, planning={plan_out.get('planning')}, "
                f"valid_cache={ok.sum()}/{num_pdfs}, fresh={fresh_ok.sum()}/{num_pdfs}, "
                f"fail={np.where(~ok)[0].tolist()}, "
                f"server_elapsed={plan_out['elapsed_s']:.3f}s"
            )
            print("path_lengths:", plan_out["path_lengths"])

        if rrt_dv_first is None or rrt_path_lengths is None:
            raise RuntimeError("RRT path cache is empty. Did initialize_rrt_server(...) run before animation?")

        # Convert planner outputs to energy terms
        Ve_energy, dVex_energy, dVey_energy = planner_outputs_to_energy_terms(
            dv_first=rrt_dv_first,
            path_lengths=rrt_path_lengths,
            u_max=u_max,
            c_e=c_e,
            c_0=c_0,
        )
        #print("V", Ve_energy, "dVex", dVex_energy, "dVey", dVey_energy)
        print("Path time ",  time.perf_counter() - start)
        coeffs, alpha_v_value, alpha_h_value = sumfield_coeffs_fd5(
            robot_positions,
            Xf=Xf, Yf=Yf, iR=iR, iL=iL, iU=iU, iD=iD,
            L=L, h=h, sigma=pdfs[0].sigma, lap_coeff=lap_coeff,
            rho_d_f=rho_d_f, Mask_f=Mask_f,
            alpha_v=alpha_v, alpha_h=alpha_h, epsilon=epsilon
        )
        controller.set_params_from_coeffs(coeffs, alpha_v_value, alpha_h_value)

        controller.set_energy_params(
            Ve=Ve_energy,
            dVex=dVex_energy,
            dVey=dVey_energy,
            Ecur=E,
            Emin=E_min,
            alpha_e=alpha_e,
            c_e=c_e
        )
        hE = E - E_min - Ve_energy
        print("alpha_h_value: ", alpha_h_value, "min hE:", hE.min(),
              "bad Ve:", np.sum(~np.isfinite(Ve_energy)),
              "bad dV:", np.sum(~np.isfinite(dVex_energy) | ~np.isfinite(dVey_energy)))

        u_opt, s_val, t_val = controller.solve()

        curr_time = time.perf_counter() - start
        total_time += curr_time
        total_time_count += 1
        print("Total time: ", curr_time)

        # Refresh scatter plot (and re-add the invariance circle)
        for patch in list(ax_scatter.patches):
            patch.remove()
        ax_scatter.contour(X, Y, Mask_A.astype(float), colors="red", levels=[0.5], linewidths=1.0)
        ax_scatter.add_patch(Circle((charger_x, charger_y), charger_radius, edgecolor='yellow', facecolor='yellow',
                                    alpha=0.25,linewidth=2))

        r_goal_switch = charger_radius + charging_margin
        charger_center = np.array([charger_x, charger_y], dtype=float)
        for i in range(num_pdfs):
            d = np.linalg.norm(robot_positions[i] - charger_center)
            if (hE[i] <= 0.1) or Charging_flag[i] or d <= charger_radius:
                if d <= r_goal_switch:
                    if E[i] <= 0.91:
                        Charging_flag[i] = True
                        if d <= charger_radius:
                            u_opt[i] = [0.0, 0.0]
                            E[i] = min(1.0, E[i] + 0.5 * dt)
                        else:
                            # once low energy and very close to charger, just go directly to charger.
                            v = charger_center - robot_positions[i]
                            n = np.linalg.norm(v)
                            u_opt[i] = (u_max * v / n) if n > 1e-12 else np.zeros(2, dtype=float)
                    else:
                        Charging_flag[i] = False

        u_norm = np.sqrt(u_opt[:, 0] ** 2 + u_opt[:, 1] ** 2)
        E = E - (c_e * u_norm + c_0 )* dt
        E = np.clip(E, 0.0, 1.0)

        energy_time.append(current_time)
        energy_history.append(E.copy())
        energy_min_history.append(float(E.min()))
        energy_avg_history.append(float(E.mean()))

        print("Charging status: ", Charging_flag)
        scatter_velocities = np.array([
            pdf.update(u_opt[i], dt, x_grid, y_grid, Charging_flag[i])
            for i, pdf in enumerate(pdfs)
        ])
        if contour is not None:
            contour.remove()
        contour = ax_scatter.contour(X, Y, np.sum([pdf.field for pdf in pdfs], axis=0), levels=contour_levels, colors='black', linewidths=1.5,
                              linestyles='dashed',
                              alpha=0.5)

        robot_positions = np.array([pdf.position for pdf in pdfs], dtype=float)
        scat.set_offsets(robot_positions)
        for i, txt in enumerate(energy_texts):
            txt.set_position((robot_positions[i, 0], robot_positions[i, 1] + 0.2))
            txt.set_text(f"{E[i]:.2f}, {hE[i]:.2f}")

        print(f"Iteration {frame} u: {u_opt} t: {t_val} -- Slack s: {s_val} -- alphaV: {alpha_v_value} -- alphaH: {alpha_h_value}")

        for i in range(num_pdfs):
            traj_x_list[i].append(pdfs[i].position[0])
            traj_y_list[i].append(pdfs[i].position[1])
            traj_lines[i].set_data(traj_x_list[i], traj_y_list[i])

        ax_scatter.set_title(f"Density Field with Invariance Area \n at t={frame * dt:.2f}", fontsize=font_size)

        # Record true position measurement & performance
        rho_new = np.sum([pdf.field for pdf in pdfs], axis=0)
        h_val = epsilon - np.sum(rho_new[Mask] ** 2) * h ** 2
        h_history.append(h_val)
        error_norm = h ** 2 * np.sum((rho_d - rho_new) ** 2)
        error_history.append(error_norm)
    animate.primed = True
    return [scat, *traj_lines, *path_lines, *path_ends]

# ----------------------- Initialize RRT server before simulation -----------------------
# Use the true initial positions here. After this returns, the simulation can start without any fallback.
if USE_RRT_PLANNER:
    initial_robot_positions = np.array([pdf.position for pdf in pdfs], dtype=float)
    initial_plan_out = initialize_rrt_server(initial_robot_positions, timeout_s=RRT_SERVER_INIT_TIMEOUT_SEC)
    rrt_paths = initial_plan_out["paths"]
    rrt_path_lengths = initial_plan_out["path_lengths"]
    rrt_dv_first = initial_plan_out["dv_first"]
    rrt_first_step_vecs = initial_plan_out["first_step_vecs"]
    rrt_plan_frame = 0
    update_rrt_path_overlay(rrt_paths, path_lines, path_ends)

ani = animation.FuncAnimation(fig, animate, frames=2400, interval=200,
                              blit=False, repeat=False)
plt.show()


fig, axes = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.5)
font_size = 20
legend_font_size = 16
text_font_size = 16
time_history_cut = time_history[:-1]

# Error vs Time subplot with convergence time marked.
axes[0].plot(time_history, error_history, 'r-', linewidth=2)
axes[0].set_title("Density Error History", fontsize=font_size)
axes[0].set_xlabel("Time (s)", fontsize=font_size)
axes[0].set_ylabel("Error Norm", fontsize=font_size)
axes[0].tick_params(axis='both', which='major', labelsize=font_size)

# ψ vs Time subplot with zero crossings annotation.
axes[1].plot(time_history, h_history, 'r-')
axes[1].set_title("h History", fontsize=font_size)
axes[1].set_xlabel("Time (s)", fontsize=font_size)
axes[1].set_ylabel("h Value", fontsize=font_size)
axes[1].set_ylim(epsilon * (-0.5), epsilon * 1.1)
axes[1].tick_params(axis='both', which='major', labelsize=font_size)

plt.show()

E_hist = np.stack(energy_history, axis=0)

plt.figure()
plt.title("Energy History", fontsize=font_size)
plt.plot(energy_time, E_hist.min(axis=1), "r-", label="min(E)")
plt.plot(energy_time, E_hist.mean(axis=1), "k--", label="avg(E)")
plt.axhline(0.1, linestyle="--", linewidth=1.5, label="E_min")
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.ylim(0, 1.05)
plt.legend()
plt.show()


print(f"average time per step: {total_time / total_time_count:.5f} s")


# ============================================================
# Batch runner: CSV logs for Step_log_visualize.py
# ============================================================
def run_batch_collect_step_logs(
        n_runs=100,
        n_frames=4000,
        output_dir="step_logs_001",
        min_frame_period_s=0.10,
        between_run_sleep_s=1.0,
        clear_existing=True,
        seed_base=None,
        quiet_inner_prints=True,
):
    """
    Run the current single-run simulation repeatedly without animation display.

    Outputs:
        output_dir/step_log_001.csv
        output_dir/step_log_002.csv
        ...

    CSV format matches Step_log_visualize.py:
        run_id, frame, time, CLF, spaceCBF, robot_id,
        E_i, hE_i, Ve_i, dVex_i, dVey_i, u_x, u_y, u_norm

    Notes:
      - Resets PDFs, E, Charging_flag, histories, controller, and RRT cache each run.
      - Calls initialize_rrt_server(...) at the start of every run to reset/reseed
        the persistent RRT server cache/model for the new initial state.
      - Does not call animate(...) and does not display animation.
      - Uses min_frame_period_s to give the async RRT server time between frames.
    """
    import csv
    import os
    import contextlib
    from pathlib import Path

    global pdfs, num_pdfs
    global E, Charging_flag
    global energy_history, energy_time, energy_min_history, energy_avg_history
    global time_history, h_history, error_history, slack_history
    global total_time, total_time_count
    global controller
    global rrt_plan_frame, rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs
    global traj_x_list, traj_y_list

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if clear_existing:
        for old in out_dir.glob("step_log_*.csv"):
            old.unlink()
        for old in out_dir.glob("step_log_*.csv.tmp"):
            old.unlink()

    header = [
        "run_id", "frame", "time", "CLF", "spaceCBF", "robot_id",
        "E_i", "hE_i", "Ve_i", "dVex_i", "dVey_i",
        "u_x", "u_y", "u_norm",
    ]

    @contextlib.contextmanager
    def _silence_stdout(enabled=True):
        if not enabled:
            yield
            return
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                yield

    def _reset_run_state(run_idx):
        global pdfs, num_pdfs
        global E, Charging_flag
        global energy_history, energy_time, energy_min_history, energy_avg_history
        global time_history, h_history, error_history, slack_history
        global total_time, total_time_count
        global controller
        global rrt_plan_frame, rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs
        global traj_x_list, traj_y_list

        if seed_base is not None:
            np.random.seed(int(seed_base) + int(run_idx))

        # Reset robot states from the original JSON-loaded particle_data.
        pdfs = [PDF(p["position"], sigma_energy) for p in particle_data]
        num_pdfs = len(pdfs)

        # Reset energy state.
        E = np.full(num_pdfs, E_init, dtype=float)
        Charging_flag = np.full(num_pdfs, False, dtype=bool)

        energy_history = [E.copy()]
        energy_time = [0.0]
        energy_min_history = [float(E.min())]
        energy_avg_history = [float(E.mean())]

        # Reset controller timing/history.
        total_time = 0.0
        total_time_count = 0

        rho0 = np.sum([pdf.field for pdf in pdfs], axis=0)
        error0 = h ** 2 * np.sum((rho_d - rho0) ** 2)
        h0 = epsilon - np.sum(rho0[Mask] ** 2) * h ** 2

        time_history = [0.0]
        error_history = [float(error0)]
        h_history = [float(h0)]
        slack_history = [0.0]

        traj_x_list = [[] for _ in range(num_pdfs)]
        traj_y_list = [[] for _ in range(num_pdfs)]

        # Rebuild MOSEK model per run so there is no solver-state carryover.
        try:
            controller.model.dispose()
        except Exception:
            pass
        controller = InvarianceControllerSmall()

        # Reset local RRT cache.
        rrt_plan_frame = -999
        rrt_paths = None
        rrt_path_lengths = None
        rrt_dv_first = None
        rrt_first_step_vecs = None

        # Reset animation priming flag just in case animate(...) was used before.
        if hasattr(animate, "primed"):
            delattr(animate, "primed")

    def _initialize_rrt_for_current_run():
        global rrt_plan_frame, rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs

        initial_robot_positions = np.array([pdf.position for pdf in pdfs], dtype=float)

        if USE_RRT_PLANNER:
            initial_plan_out = initialize_rrt_server(
                initial_robot_positions,
                timeout_s=RRT_SERVER_INIT_TIMEOUT_SEC,
            )
            rrt_paths = initial_plan_out["paths"]
            rrt_path_lengths = initial_plan_out["path_lengths"]
            rrt_dv_first = initial_plan_out["dv_first"]
            rrt_first_step_vecs = initial_plan_out["first_step_vecs"]
            rrt_plan_frame = 0

    def _run_one_frame(frame, writer, run_id):
        global E, Charging_flag
        global total_time, total_time_count
        global rrt_plan_frame, rrt_paths, rrt_path_lengths, rrt_dv_first, rrt_first_step_vecs

        current_time = (frame + 1) * dt
        time_history.append(current_time)

        # Measurement-noisy positions
        tmp = [pdf.compute_pdf_random() for pdf in pdfs]
        robot_positions_meas = np.array([t[1] for t in tmp], dtype=float)

        start = time.perf_counter()

        # Async RRT server update/cache refresh.
        plan_out = submit_rrt_positions_to_server(robot_positions_meas, frame=frame)

        should_refresh_local_cache = (
                (frame - rrt_plan_frame) >= RRT_UPDATE_PERIOD
                or (rrt_dv_first is None)
        )

        if should_refresh_local_cache and plan_out is not None:
            rrt_paths = plan_out["paths"]
            rrt_path_lengths = plan_out["path_lengths"]
            rrt_dv_first = plan_out["dv_first"]
            rrt_first_step_vecs = plan_out["first_step_vecs"]
            rrt_plan_frame = frame

        if rrt_dv_first is None or rrt_path_lengths is None:
            raise RuntimeError(
                "RRT path cache is empty. initialize_rrt_server(...) did not produce a valid cache."
            )

        Ve_energy, dVex_energy, dVey_energy = planner_outputs_to_energy_terms(
            dv_first=rrt_dv_first,
            path_lengths=rrt_path_lengths,
            u_max=u_max,
            c_e=c_e,
            c_0=c_0,
        )

        coeffs, alpha_v_value, alpha_h_value = sumfield_coeffs_fd5(
            robot_positions_meas,
            Xf=Xf, Yf=Yf, iR=iR, iL=iL, iU=iU, iD=iD,
            L=L, h=h,
            sigma=pdfs[0].sigma,
            lap_coeff=lap_coeff,
            rho_d_f=rho_d_f,
            Mask_f=Mask_f,
            alpha_v=alpha_v,
            alpha_h=alpha_h,
            epsilon=epsilon,
        )

        controller.set_params_from_coeffs(
            coeffs,
            alpha_v_value,
            alpha_h_value,
        )

        controller.set_energy_params(
            Ve=Ve_energy,
            dVex=dVex_energy,
            dVey=dVey_energy,
            Ecur=E,
            Emin=E_min,
            alpha_e=alpha_e,
            c_e=c_e,
        )
        # Log the energy/margin values used by the controller at this frame.
        E_logged = E.copy()
        hE_logged = E_logged - E_min - Ve_energy

        u_opt, s_val, t_val = controller.solve()

        curr_time = time.perf_counter() - start
        total_time += curr_time
        total_time_count += 1

        # Same charging override logic as animate
        r_goal_switch = charger_radius + charging_margin
        charger_center = np.array([charger_x, charger_y], dtype=float)
        for i in range(num_pdfs):
            d = np.linalg.norm(robot_positions_meas[i] - charger_center)
            if (hE_logged[i] <= 0.1) or Charging_flag[i] or d <= charger_radius:
                if d <= r_goal_switch:
                    if E[i] <= 0.91:
                        Charging_flag[i] = True
                        if d <= charger_radius:
                            u_opt[i] = [0.0, 0.0]
                            E[i] = min(1.0, E[i] + 0.5 * dt)
                        else:
                            v = charger_center - robot_positions_meas[i]
                            n = np.linalg.norm(v)
                            u_opt[i] = (u_max * v / n) if n > 1e-12 else np.zeros(2, dtype=float)
                    else:
                        Charging_flag[i] = False

        u_norm = np.sqrt(u_opt[:, 0] ** 2 + u_opt[:, 1] ** 2)

        # Advance energy and true robot states.
        E = E - (c_e * u_norm + c_0) * dt
        E = np.clip(E, 0.0, 1.0)

        energy_time.append(current_time)
        energy_history.append(E.copy())
        energy_min_history.append(float(E.min()))
        energy_avg_history.append(float(E.mean()))

        for i, pdf in enumerate(pdfs):
            pdf.update(u_opt[i], dt, x_grid, y_grid, Charging_flag[i])

        rho_new = np.sum([pdf.field for pdf in pdfs], axis=0)
        h_val = epsilon - np.sum(rho_new[Mask] ** 2) * h ** 2
        error_norm = h ** 2 * np.sum((rho_d - rho_new) ** 2)

        h_history.append(float(h_val))
        error_history.append(float(error_norm))
        slack_history.append(float(s_val))

        # Write one row per robot for Step_log_visualize.py.
        for rid in range(num_pdfs):
            writer.writerow([
                run_id,
                frame,
                current_time,
                float(error_norm),
                float(h_val),
                rid,
                float(E_logged[rid]),
                float(hE_logged[rid]),
                float(Ve_energy[rid]),
                float(dVex_energy[rid]),
                float(dVey_energy[rid]),
                float(u_opt[rid, 0]),
                float(u_opt[rid, 1]),
                float(u_norm[rid]),
            ])

    # Make sure batch itself never shows figures.
    plt.close("all")

    meta_path = out_dir / "batch_config.json"
    with meta_path.open("w") as f:
        json.dump(
            {
                "n_runs": int(n_runs),
                "n_frames": int(n_frames),
                "dt": float(dt),
                "min_frame_period_s": float(min_frame_period_s),
                "between_run_sleep_s": float(between_run_sleep_s),
                "num_pdfs": int(num_pdfs),
                "output_dir": str(out_dir),
            },
            f,
            indent=2,
        )

    completed = 0

    for run_idx in range(1, n_runs + 1):
        print(f"run [{run_idx}/{n_runs}]...")

        log_path = out_dir / f"step_log_{run_idx:03d}.csv"
        tmp_path = out_dir / f"step_log_{run_idx:03d}.csv.tmp"

        try:
            with _silence_stdout(quiet_inner_prints):
                _reset_run_state(run_idx)
                _initialize_rrt_for_current_run()

            with tmp_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for frame in range(n_frames):
                    frame_start_wall = time.perf_counter()

                    with _silence_stdout(quiet_inner_prints):
                        _run_one_frame(frame, writer, run_idx)

                    # Periodically flush so interrupted runs are not empty.
                    if frame % 50 == 0:
                        f.flush()

                    elapsed_wall = time.perf_counter() - frame_start_wall
                    sleep_s = max(0.0, float(min_frame_period_s) - elapsed_wall)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                f.flush()

            tmp_path.replace(log_path)
            completed += 1
            print(f"run [{run_idx}/{n_runs}] complete -> {log_path}")

        except Exception as exc:
            print(f"run [{run_idx}/{n_runs}] FAILED: {exc}")
            print(f"partial temp log kept at: {tmp_path}")
            raise

        if between_run_sleep_s > 0 and run_idx < n_runs:
            time.sleep(float(between_run_sleep_s))

    print(f"Done. Completed {completed}/{n_runs} runs.")
    print(f"Logs saved in: {out_dir.resolve()}")


# -----------------------
# Call block
# -----------------------
run_batch_collect_step_logs(
    n_runs=100,
    n_frames=5000,
    output_dir="step_logs_001",
    min_frame_period_s=0.05,
    between_run_sleep_s=1.0,
    clear_existing=True,
    seed_base=None,  # set e.g. 123 if you want reproducible stochastic runs
    quiet_inner_prints=True,
)