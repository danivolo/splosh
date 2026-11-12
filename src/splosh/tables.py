"""Reader for the text tables a code writes next to its snapshots (one file per MPI task, one row per time step of a particle, columns that were appended over the code's history).

read_rows(path, ncols, widths)    one file -> (N, ncols) float64; lines of an older, narrower layout are kept with NaN in the missing columns, malformed / truncated lines dropped
stale_after_restart(times)        for each file the rows that a restart rewrote (the run resumed from an earlier restart file): they appear twice, the earlier copy is stale
load_table(files, id_col, time_col, min_cols)   all rows of all files, restart overlap dropped, sorted by (time, id)
Schema = named columns of a layout: see Schema."""
import os

import numpy as np


class Schema:
    """name -> column index (0-based), with the layouts (line widths) the code has written. `has(name, width)`: whether a layout of that width contains the column."""
    def __init__(self, columns, widths=None):
        self.columns = dict(columns)
        self.widths = tuple(sorted(widths)) if widths else None

    def __getitem__(self, name):
        return self.columns[name]

    def has(self, name, width):
        return name in self.columns and self.columns[name] < width


def _is_number_start(tok):
    return tok[:1] in "0123456789+-."


def n_columns(path, min_cols, scan=1000):
    """column count of the first numeric line with at least min_cols fields."""
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = line.replace("\x00", "").split()
            if len(parts) >= min_cols and _is_number_start(parts[0]):
                return len(parts)
            if i > scan:
                break
    return None


def last_line_columns(path, min_cols):
    """column count of the last complete numeric line (a file continued by a newer binary has wider lines at the end)."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 65536))
        lines = f.read().decode(errors="ignore").replace("\x00", "").split("\n")
    for line in reversed(lines[1:-1] if len(lines) > 2 else lines):
        parts = line.split()
        if len(parts) >= min_cols and _is_number_start(parts[0]):
            return len(parts)
    return None


def _to_float(tok):
    try:
        return float(tok)
    except ValueError:
        return np.nan


def read_rows(path, ncols, widths=None):
    """one file -> (N, ncols) float64. `widths`: accepted line widths (default: only ncols). A line of width w < ncols is padded with NaN.
    Fast path (numpy's C parser) when every line has ncols fields; otherwise a tolerant line-by-line parse."""
    widths = set(widths or {ncols})
    try:
        arr = np.loadtxt(path, comments="#", ndmin=2, dtype=np.float64)
        if arr.shape[1] == ncols:
            good = np.all(np.isfinite(arr[:, :2]), axis=1)
            return arr[good]
    except (ValueError, UnicodeDecodeError):
        pass
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.replace("\x00", "").split()
            if len(parts) not in widths or not parts or not _is_number_start(parts[0]):
                continue
            vals = [_to_float(t) for t in parts]
            if not (np.isfinite(vals[0]) and np.isfinite(vals[1])):
                continue
            rows.append(vals + [np.nan] * (ncols - len(vals)))
    return np.array(rows, dtype=np.float64).reshape(-1, ncols)


def stale_after_restart(times_per_file):
    """Masks, one per file, of the rows superseded by a restart.

    Within a file the time never decreases, except where a restart resumed from an earlier restart file and rewrote that interval. Restart k resumes at T_k;
    rows written before it with T_k < t <= Tmax_k are stale. A file whose task wrote no rows after a restart has no drop for it, so all its rows count as pre-restart."""
    drops = []                                        # (file index, row position, resume time, last time before the drop)
    for fi, t in enumerate(times_per_file):
        if len(t) == 0:
            continue
        runmax = np.maximum.accumulate(t)
        for p in np.where(t[1:] < t[:-1])[0] + 1:
            drops.append((fi, p, t[p], runmax[p - 1]))
    masks = [np.zeros(len(t), dtype=bool) for t in times_per_file]
    if not drops:
        return masks
    # drops in different files belong to the same restart when their stale windows (t_res, t_prev] overlap: the first post-restart row of a task can come well
    # after the common resume time, so compare windows, not times
    drops.sort(key=lambda d: d[2])
    restarts = []                                     # [resume time, max stale time, {file index: drop position}]
    for fi, p, t_res, t_prev in drops:
        r = restarts[-1] if restarts else None
        if r is not None and t_res < r[1] and fi not in r[2]:
            r[0] = min(r[0], t_res); r[1] = max(r[1], t_prev); r[2][fi] = p
        else:
            restarts.append([t_res, t_prev, {fi: p}])
    for t_res, t_stale, pos_by_file in restarts:
        for fi, t in enumerate(times_per_file):
            n_pre = pos_by_file.get(fi, len(t))
            pre = np.arange(len(t)) < n_pre
            masks[fi] |= pre & (t > t_res) & (t <= t_stale)
    return masks


def load_table(files, id_col, time_col, min_cols, return_info=False):
    """All rows of `files` merged, restart overlap dropped, sorted by (time, id). Rows are read at the widest layout found (first and last line of every file);
    narrower lines get NaN in the columns they lack. Returns None without data (or (arr, info) with return_info: widths, n_stale)."""
    files = [f for f in files if os.path.getsize(f) > 0]
    widths = {w for f in files for w in (n_columns(f, min_cols), last_line_columns(f, min_cols)) if w}
    if not widths:
        return (None, {}) if return_info else None
    ncols = max(widths)
    arrays = [read_rows(f, ncols, widths) for f in files]
    stale = stale_after_restart([a[:, time_col] for a in arrays])
    arr = np.vstack([a[~m] for a, m in zip(arrays, stale)])
    arr = arr[np.lexsort((arr[:, id_col], arr[:, time_col]))]
    info = {"widths": sorted(widths), "ncols": ncols, "n_stale": int(sum(m.sum() for m in stale)), "n_files": len(files)}
    return (arr, info) if return_info else arr
