"""Several runs in one viewer: how `splosh serve` groups its arguments into runs, the times of their snapshots, and the time alignment of runs with different snapshot spacing.

Grouping (`group_runs`): every `label=path` argument is a run (a directory, or one file); every other directory is a run (label = the directory's name; a directory without snapshots of its own
but with an `output` subdirectory uses that, like `sheet` / `movie`); loose snapshot files that are not in a directory argument form ONE run together (label = the name of the directory of the
first). Within a run the snapshots are in natural order (snap_2 before snap_10). Duplicate labels get a numeric suffix (`run`, `run-2`). A single directory (or only files) is one run, as before.

Time alignment (the rules of `splosh compare`; `nearest` adds a tie-break by index): an aligned coordinate per snapshot,
`time` (code time of the header), `time_myr` (time x the header's length / velocity unit in Myr; not available for comoving runs or headers without units), `index` (position in the run),
`offset` (= time; the per-run offsets are required in this mode), minus the run's offset. `nearest` picks, for a wanted aligned value, the snapshot of a run whose coordinate is closest, if it is within
`tol` (default: half of the run's median snapshot spacing in that coordinate), else none. Equal distances (several snapshots with the same time) go to the one nearest to the index it should
prefer (the pane's current snapshot)."""
import glob
import math
import os

import h5py
import numpy as np



def run_snapshots(path):
    """the snapshots of a run given its directory: those of <path>/output when it has any (a run directory may also hold its initial conditions or other
    snapshot files next to output/), else those found under `path` itself (find_snapshots)."""
    out = os.path.join(path, "output")
    if os.path.isdir(out):
        got = find_snapshots([out])
        if got:
            return got
    return find_snapshots([path])


def find_snapshots(paths):
    from .server import find_snapshots as f          # server imports this module: import it late
    return f(paths)

ALIGNS = ("time", "time_myr", "index", "offset", "orbits")
G_CGS = 6.6743e-8
MYR_S = 3.156e13          # the Myr of analyze.py (seconds)


def _find(path):
    path = os.path.expanduser(path)
    return run_snapshots(path) if os.path.isdir(path) else find_snapshots([path])


def group_runs(args):
    """[(label, [snapshot paths])] from the command line arguments of `splosh serve` (see the module doc)."""
    runs, loose, slot = [], [], None
    for a in args:
        a = str(a)
        if "=" in a and not os.path.exists(a):
            label, _, path = a.partition("=")
            runs.append((label, _find(path)))
        elif os.path.isdir(os.path.expanduser(a)):
            runs.append((os.path.basename(os.path.abspath(os.path.expanduser(a)).rstrip("/")) or "run", _find(a)))
        else:
            if slot is None:
                slot = len(runs)
                runs.append(None)
            loose.append(a)
    if loose:
        files = find_snapshots(loose)
        runs[slot] = (os.path.basename(os.path.dirname(files[0])) or "run" if files else "run", files)
    out, seen = [], {}
    for label, paths in runs:
        if not paths:
            raise SystemExit(f"no snapshots found for run {label!r}")
        label = "".join(c if c not in "/\\ " else "_" for c in label) or "run"
        n = seen.get(label, 0) + 1
        seen[label] = n
        out.append((label if n == 1 else f"{label}-{n}", paths))
    return out


def snapshot_times(path):
    """(time in code units or nan, time in Myr or None) from the header of a snapshot."""
    try:
        with h5py.File(path, "r") as f:
            h = {k: np.asarray(v).tolist() for k, v in f["Header"].attrs.items()}
        t = float(h["Time"])
    except Exception:
        return float("nan"), None
    try:
        from .units import Units
        u = Units.from_header(h)
        if u.comoving or "length_cm" in u.defaulted or "velocity_cm_s" in u.defaulted:      # Time is a scale factor, or the header has no length / velocity unit
            return t, None
        myr = t * u.time_s / MYR_S
        return t, (myr if math.isfinite(myr) else None)
    except Exception:
        return t, None


_PERIODS = {}


def binary_period(path):
    """Orbital period (code time units) of the black-hole binary in snapshot `path`: the two lowest-ID black holes (PartType5, dynamical Masses), vis-viva on their relative
    position and velocity (a = 1 / (2/r - v^2/(G M)), P = 2 pi sqrt(a^3 / (G M))), G in the code units of the header. None without two black holes or when they are unbound."""
    if path in _PERIODS:
        return _PERIODS[path]
    P = None
    try:
        with h5py.File(path, "r") as f:
            h = {k: np.asarray(v).tolist() for k, v in f["Header"].attrs.items()}
            if "PartType5" in f and len(f["PartType5/Masses"]) >= 2:
                g = f["PartType5"]
                ids = np.asarray(g["ParticleIDs"]) if "ParticleIDs" in g else np.arange(len(g["Masses"]))
                o = np.argsort(ids)[:2]
                x, v, m = np.asarray(g["Coordinates"], float)[o], np.asarray(g["Velocities"], float)[o], np.asarray(g["Masses"], float)[o]
                from .units import Units
                u = Units.from_header(h)
                if not u.comoving:
                    G = G_CGS * u.mass_g * u.time_s ** 2 / u.length_cm ** 3
                    r, v2, M = float(np.linalg.norm(x[1] - x[0])), float(((v[1] - v[0]) ** 2).sum()), float(m.sum())
                    inv_a = 2.0 / r - v2 / (G * M) if r > 0 and M > 0 else -1.0
                    if inv_a > 0:
                        P = 2.0 * math.pi * math.sqrt((1.0 / inv_a) ** 3 / (G * M))
    except Exception:
        P = None
    _PERIODS[path] = P
    return P


class Run:
    def __init__(self, label, paths):
        self.label, self.paths = label, list(paths)
        tt = [snapshot_times(p) for p in self.paths]
        self.times = [t for t, _ in tt]
        self.times_myr = [m for _, m in tt]

    @property
    def period(self):
        """the binary's orbital period P0 at the run's first snapshot (code time), or None (binary_period)."""
        if not hasattr(self, "_period"):
            self._period = binary_period(self.paths[0]) if self.paths else None
        return self._period

    @property
    def has_myr(self):
        return all(m is not None for m in self.times_myr)

    def coords(self, mode, offset=0.0):
        """aligned coordinate of every snapshot (nan where the unit is not available)."""
        if mode not in ALIGNS:
            raise ValueError(f"align must be one of {ALIGNS}")
        if mode == "index":
            c = np.arange(len(self.paths), dtype=float)
        elif mode == "orbits":
            P = self.period
            c = np.array(self.times, dtype=float) / P if P else np.full(len(self.paths), np.nan)
        elif mode == "time_myr":
            c = np.array([float("nan") if m is None else m for m in self.times_myr], dtype=float)
        else:
            c = np.array(self.times, dtype=float)
        return c - float(offset)

    def describe(self, i):
        P = self.period
        t = self.times[i] if math.isfinite(self.times[i]) else None          # NaN (unreadable header) is not valid JSON for the browser
        return {"i": i, "name": os.path.basename(self.paths[i]), "time": t, "time_myr": self.times_myr[i], "orbits": t / P if P and t is not None else None}


def spacing(coord):
    """median distance between consecutive snapshots in this coordinate (inf for fewer than two)."""
    c = np.asarray(coord, float)
    c = c[np.isfinite(c)]
    return float(np.median(np.abs(np.diff(c)))) if c.size > 1 else float("inf")


def nearest(coord, at, tol=None, prefer=None):
    """(index or None, reason) of the snapshot whose aligned coordinate is nearest to `at`, if within tol (default half of the median spacing). Ties go to the index nearest to `prefer`."""
    c = np.asarray(coord, float)
    ok = np.isfinite(c)
    if not ok.any():
        return None, "no snapshot has this time coordinate"
    d = np.where(ok, np.abs(c - at), np.inf)
    dmin = float(d.min())
    eps = 1e-9 * dmin + 1e-12 * (1.0 + abs(at))
    cand = np.nonzero(d <= dmin + eps)[0]
    j = int(cand[np.argmin(np.abs(cand - prefer))]) if prefer is not None else int(cand[0])
    lim = float(tol) if tol is not None else 0.5 * spacing(c)
    if dmin > lim + eps:
        return None, f"no snapshot within {lim:.3g} of {at:.6g} (nearest {c[j]:.6g})"
    return j, ""


def align(runs, mode, at, offsets=None, tol=None, prefer=None):
    """For the wanted aligned value `at`: one dict per run {run, label, snap (index or None), coord, distance, tol, reason}. `offsets`: per run shift (list in run order or dict by label),
    `tol`: None (auto per run) or a number, `prefer`: per run the index to prefer on ties."""
    out = []
    for r, run in enumerate(runs):
        off = (offsets.get(run.label, 0.0) if isinstance(offsets, dict) else (offsets[r] if offsets and r < len(offsets) else 0.0))
        if mode == "offset" and not offsets:
            raise ValueError("align = offset needs offsets")
        c = run.coords(mode, off)
        j, why = nearest(c, at, tol, None if prefer is None or r >= len(prefer) or prefer[r] is None else prefer[r])
        lim = float(tol) if tol is not None else 0.5 * spacing(c)
        out.append({"run": r, "label": run.label, "snap": j, "coord": None if j is None else float(c[j]), "distance": None if j is None else abs(float(c[j]) - at),
                    "tol": None if math.isinf(lim) else lim, "reason": why})
    return out
