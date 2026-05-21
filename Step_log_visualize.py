"""
viz_step_logs.py

Collects *all* CSV files in ./step_logs/ (matching step_log_*.csv),
then plots:

1) CLF + spaceCBF on ONE plot with twin y-axes (left=CLF, right=spaceCBF)
2) hE_i (energy CBF margin) : single plot with 4 robot traces
3) E_i : single plot with 4 robot traces

Assumes per-step logger header is:
run_id, frame, time, CLF, spaceCBF, robot_id, E_i, hE_i, Ve_i, dVex_i, dVey_i, u_x, u_y, u_norm
"""

import csv
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
LOG_DIR = Path("step_logs_001")          # folder containing step_log_*.csv
GLOB_PATTERN = "step_log_*.csv"      # file pattern
EXPECTED_ROBOTS = 10
STRICT_SAME_FRAMES = True           # require all runs to have same frames
USE_TIME_COL = True                 # plot vs time (else vs frame)

COLS = {
    "run_id": "run_id",
    "frame": "frame",
    "time": "time",
    "clf": "CLF",
    "space": "spaceCBF",
    "rid": "robot_id",
    "E": "E_i",
    "hE": "hE_i",
    "Ve": "Ve_i",
    "ux": "u_x",
    "uy": "u_y",
    "umag": "u_norm",
}

# =========================
# Helpers
# =========================
def _to_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return float(default)
        s = str(x).strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)

def _to_int(x, default=-1) -> int:
    try:
        if x is None:
            return int(default)
        s = str(x).strip()
        if s == "":
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)

def _read_one_csv(path: Path) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Returns a nested dict:
      data[frame][robot_id] = {
        "time", "clf", "space", "E", "hE", "Ve", "umag"
      }
    Raises ValueError if header is missing OR required columns are missing.
    """
    # quick empty-file check
    try:
        if path.stat().st_size == 0:
            raise ValueError("empty file (0 bytes)")
    except FileNotFoundError:
        raise ValueError("file not found")

    data: Dict[int, Dict[int, Dict[str, float]]] = {}

    with path.open("r", newline="") as f:
        # ensure first non-empty line exists (header)
        # (DictReader doesn't skip blank lines reliably)
        first_nonempty = None
        pos0 = f.tell()
        for line in f:
            if line.strip():
                first_nonempty = line
                break
        if first_nonempty is None:
            raise ValueError("no non-empty lines (likely empty / not flushed)")
        # rewind and let DictReader read normally
        f.seek(pos0)

        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("no header found (fieldnames is None/empty)")

        missing = [v for v in COLS.values() if v not in set(reader.fieldnames)]
        if missing:
            raise ValueError(f"missing columns: {missing}. Found: {reader.fieldnames}")

        for row in reader:
            fr = _to_int(row[COLS["frame"]])
            rid = _to_int(row[COLS["rid"]])

            entry = {
                "time":  _to_float(row[COLS["time"]]),
                "clf":   _to_float(row[COLS["clf"]]),
                "space": _to_float(row[COLS["space"]]),
                "E":     _to_float(row[COLS["E"]]),
                "hE":    _to_float(row[COLS["hE"]]),
                "Ve":    _to_float(row[COLS["Ve"]]),
                "umag":  _to_float(row[COLS["umag"]]),
            }

            data.setdefault(fr, {})[rid] = entry

    if not data:
        raise ValueError("no data rows found (header exists but file has no rows)")
    return data


def _stack_run_to_arrays(run_data: Dict[int, Dict[int, Dict[str, float]]]):
    """
    Convert one run dict to arrays aligned by sorted frames and sorted robot_ids.
    Returns:
      frames, times, robot_ids,
      clf_ts, space_ts, E_ts, hE_ts, Ve_ts, umag_ts

    Shapes:
      times: (T,)
      clf_ts: (T,)   (one per frame; taken as mean across robots just in case)
      space_ts: (T,) (one per frame)
      E_ts etc: (T,J)
    """
    frames = np.array(sorted(run_data.keys()), dtype=int)
    if frames.size == 0:
        raise ValueError("Empty run_data")

    # Determine robot ids from first frame
    rids = sorted(run_data[frames[0]].keys())
    J = len(rids)

    times = np.full(frames.shape[0], np.nan, dtype=float)

    clf = np.full(frames.shape[0], np.nan, dtype=float)
    space = np.full(frames.shape[0], np.nan, dtype=float)

    E = np.full((frames.shape[0], J), np.nan, dtype=float)
    hE = np.full((frames.shape[0], J), np.nan, dtype=float)
    Ve = np.full((frames.shape[0], J), np.nan, dtype=float)
    umag = np.full((frames.shape[0], J), np.nan, dtype=float)

    for t_idx, fr in enumerate(frames):
        frame_dict = run_data[fr]

        # sanity check: same robot set per frame
        if sorted(frame_dict.keys()) != rids:
            raise ValueError(f"Robot ids mismatch at frame={fr}. Expected {rids}, got {sorted(frame_dict.keys())}")

        # time should be same for all robots; take robot0
        times[t_idx] = float(frame_dict[rids[0]]["time"])

        # clf / space also same across robots in your logger; robustly average
        clf[t_idx] = float(np.mean([frame_dict[r]["clf"] for r in rids]))
        space[t_idx] = float(np.mean([frame_dict[r]["space"] for r in rids]))

        for j, rid in enumerate(rids):
            E[t_idx, j] = frame_dict[rid]["E"]
            hE[t_idx, j] = frame_dict[rid]["hE"]
            Ve[t_idx, j] = frame_dict[rid]["Ve"]
            umag[t_idx, j] = frame_dict[rid]["umag"]

    return frames, times, np.array(rids, dtype=int), clf, space, E, hE, Ve, umag

def _mean_over_runs(arr_list: List[np.ndarray]) -> np.ndarray:
    """
    NaN-safe mean over runs with identical shapes.
    """
    stack = np.stack(arr_list, axis=0)  # (R, ...)
    return np.nanmean(stack, axis=0)

def _worst_over_runs(arr_list: List[np.ndarray], mode: str) -> np.ndarray:
    """
    NaN-safe worst-case over runs with identical shapes.
    mode:
      - "min": take nanmin over runs
      - "max": take nanmax over runs
    """
    stack = np.stack(arr_list, axis=0)  # (R, ...)
    if mode == "min":
        return np.nanmin(stack, axis=0)
    elif mode == "max":
        return np.nanmax(stack, axis=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
# =========================
# Load all logs and average
# =========================
paths = sorted(LOG_DIR.glob(GLOB_PATTERN))
if not paths:
    raise FileNotFoundError(f"No CSV logs found in {LOG_DIR.resolve()} matching {GLOB_PATTERN}")

print(f"[viz] found {len(paths)} logs:")
for p in paths[:10]:
    print("  -", p)
if len(paths) > 10:
    print(f"  ... (+{len(paths)-10} more)")

frames_list = []
times_list = []
rids_list = []
clf_list = []
space_list = []
E_list = []
hE_list = []
Ve_list = []
umag_list = []

skipped = []

for p in paths:
    try:
        run_dict = _read_one_csv(p)
        frames, times, rids, clf, space, E, hE, Ve, umag = _stack_run_to_arrays(run_dict)

        if EXPECTED_ROBOTS is not None and len(rids) != EXPECTED_ROBOTS:
            raise ValueError(f"has {len(rids)} robots (ids={rids}), expected {EXPECTED_ROBOTS}")

        frames_list.append(frames)
        times_list.append(times)
        rids_list.append(rids)
        clf_list.append(clf)
        space_list.append(space)
        E_list.append(E)
        hE_list.append(hE)
        Ve_list.append(Ve)
        umag_list.append(umag)

        print(f"[viz] loaded {p} (T={len(frames)}, J={len(rids)})")

    except Exception as e:
        skipped.append((p, str(e)))
        print(f"[viz] SKIP {p}: {e}")

if len(frames_list) == 0:
    msg = "\n".join([f"  - {p}: {err}" for p, err in skipped])
    raise RuntimeError("No valid logs loaded. Reasons:\n" + msg)

if skipped:
    print(f"[viz] skipped {len(skipped)} logs:")
    for p, err in skipped:
        print(f"  - {p}: {err}")

# --- alignment checks (unchanged) ---
base_frames = frames_list[0]
base_rids = rids_list[0]

n_runs = len(frames_list)
for k in range(1, n_runs):
    if not np.array_equal(rids_list[k], base_rids):
        raise ValueError(f"Robot ids differ between runs:\n  {paths[0]}: {base_rids}\n  {paths[k]}: {rids_list[k]}")

    if STRICT_SAME_FRAMES and not np.array_equal(frames_list[k], base_frames):
        raise ValueError(f"Frames differ between runs:\n  {paths[0]}: [{base_frames[0]}..{base_frames[-1]}] len={len(base_frames)}\n"
                         f"  {paths[k]}: [{frames_list[k][0]}..{frames_list[k][-1]}] len={len(frames_list[k])}\n"
                         f"Set STRICT_SAME_FRAMES=False if you want a different merging strategy.")

# Worst-case
frames_avg = base_frames
times_avg  = _mean_over_runs(times_list)                # leave as mean

clf_avg    = _worst_over_runs(clf_list,  mode="max")    # highest CLF (worst)
space_avg  = _worst_over_runs(space_list, mode="min")   # lowest space safety (worst)
E_worst    = _worst_over_runs(E_list,   mode="min")     # lowest E_i per (t,robot)
E_avg      = _mean_over_runs(E_list)

x = times_avg if USE_TIME_COL else frames_avg
xlabel = "time (s)" if USE_TIME_COL else "frame"

robot_ids = base_rids.tolist()
T, J = E_avg.shape
print(f"[viz] averaged runs: T={T}, J={J}, robot_ids={robot_ids}")

import matplotlib as mpl

scale = 1.55  # +25% (change this)
base = mpl.rcParams["font.size"]

mpl.rcParams.update({
    "font.size": base * scale,          # fallback / default text
    "axes.titlesize": base * (scale + 0.2),
    "axes.labelsize": base * scale,
    "xtick.labelsize": base * scale,
    "ytick.labelsize": base * scale,
    "legend.fontsize": base * (scale - 0.2),
})
# =========================
# Plot 1: CLF + spaceCBF with twin y-axis
# =========================
fig0, ax0 = plt.subplots(figsize=(5.5, 4.5))
ax0.plot(x, clf_avg, linewidth=2, label="CLF V")
ax0.set_xlabel(xlabel)
ax0.set_xlim([0,100])

ax0.set_ylabel("V")
ax0.grid(True, alpha=0.3)

ax0b = ax0.twinx()
ax0b.plot(x, space_avg, linewidth=2, linestyle="-", color='red', label="CBF h_s")
ax0b.set_ylim([0.00,0.0105])
ax0b.set_ylabel("h_s")

# combined legend
lines0, labels0 = ax0.get_legend_handles_labels()
lines1, labels1 = ax0b.get_legend_handles_labels()
ax0.legend(lines0 + lines1, labels0 + labels1, loc="lower right")
ax0.set_title("Coverage Task and Safety Constraint")

plt.tight_layout()
# =========================
# Plot 2: E per robot
# =========================
fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
for j, rid in enumerate(robot_ids):
    ax2.plot(x, E_worst[:, j], linewidth=2, linestyle="-" ,label=f"R{rid + 1}")
ax2.set_xlabel(xlabel)
ax2.set_ylabel("E")
ax2.axhline(0.1, linestyle="--", linewidth=1.5, label="E_min")
ax2.set_title("Energy E_i per robot")
ax2.grid(True, alpha=0.3)
ax2.legend(
    ncol=6,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    fontsize=base * (scale - 0.2),
    handlelength=1.1,
    handletextpad=0.3,
    columnspacing=0.5,
    labelspacing=0.2,
    borderpad=0.25,
    frameon=True,
    framealpha=0.85
)
ax2.set_ylim([0,1])
ax2.set_xlim([0,100])
plt.tight_layout()

plt.show()
