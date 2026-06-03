"""
Scans step_log_*.csv in ./step_logs/ and reports which files ever have
E_i for any robot below a threshold (default 0.1).

Assumes CSV header includes at least:
  frame, robot_id, E_i

Usage:
  python find_E_below_threshold_all_robots_min_only.py
  python find_E_below_threshold_all_robots_min_only.py --threshold 0.1
"""

import argparse
import csv
from pathlib import Path
import numpy as np

LOG_DIR = Path("step_logs_001")
GLOB_PATTERN = "step_log_*.csv"

COLS = {
    "frame": "frame",
    "time": "time",       # optional
    "rid": "robot_id",
    "E": "E_i",
}


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


def scan_file(path: Path, threshold: float):
    """
    Returns:
      hit (bool),
      min_E (float),
      min_hit (dict or None): {rid, frame, time, E}

    Notes:
      - min_E is the minimum E_i over all robots in the file.
    """
    # empty-file check
    if not path.exists():
        return False, np.nan, None
    if path.stat().st_size == 0:
        return False, np.nan, None

    min_E = np.inf
    min_hit = None
    hit = False

    with path.open("r", newline="") as f:
        # ensure header exists
        first_nonempty = None
        pos0 = f.tell()
        for line in f:
            if line.strip():
                first_nonempty = line
                break
        if first_nonempty is None:
            return False, np.nan, None
        f.seek(pos0)

        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return False, np.nan, None

        required = [COLS["frame"], COLS["rid"], COLS["E"]]
        missing = [c for c in required if c not in set(reader.fieldnames)]
        if missing:
            raise ValueError(f"missing columns: {missing}. Found: {reader.fieldnames}")

        has_time = COLS["time"] in set(reader.fieldnames)

        for row in reader:
            rid = _to_int(row.get(COLS["rid"]))
            E = _to_float(row.get(COLS["E"]))
            if np.isnan(E):
                continue

            fr = _to_int(row.get(COLS["frame"]))
            t = _to_float(row.get(COLS["time"])) if has_time else np.nan
            this_hit = {"rid": rid, "frame": fr, "time": t, "E": E}

            if E < min_E:
                min_E = E
                min_hit = this_hit

            if E < threshold:
                hit = True

    if min_E == np.inf:
        min_E = np.nan

    return hit, min_E, min_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default=str(LOG_DIR))
    ap.add_argument("--pattern", type=str, default=GLOB_PATTERN)
    ap.add_argument("--threshold", type=float, default=0.1, help="E threshold")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    paths = sorted(log_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV logs found in {log_dir.resolve()} matching {args.pattern}")

    hits = []
    errors = []

    for p in paths:
        try:
            hit, min_E, min_hit = scan_file(p, args.threshold)
            if hit:
                hits.append((p, min_E, min_hit))
        except Exception as e:
            errors.append((p, str(e)))

    print(f"[scan] checked {len(paths)} files in {log_dir.resolve()} pattern={args.pattern}")
    print(f"[scan] target robot_id=ALL, threshold={args.threshold}")

    if hits:
        print(f"\n[scan] FOUND {len(hits)} files where E_i for any robot dips below {args.threshold}:\n")
        # sort by worst min_E (smallest first)
        hits.sort(key=lambda x: (np.nan_to_num(x[1], nan=np.inf)))
        for p, min_E, mh in hits:
            min_t_str = "n/a" if np.isnan(mh["time"]) else f"{mh['time']:.6g}"
            print(f" - {p.name}")
            print(f"    min_E={min_E:.6g}  robot_id=R{mh['rid']}  frame={mh['frame']}  time={min_t_str}")
    else:
        print(f"\n[scan] No files had E_i for any robot below {args.threshold}.")

    if errors:
        print(f"\n[scan] {len(errors)} files had errors (skipped):")
        for p, err in errors:
            print(f" - {p.name}: {err}")


if __name__ == "__main__":
    main()
