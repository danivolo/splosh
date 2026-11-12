"""Colour-limit policies for sets of maps (contact sheets, movies).

    fixed    the given vmin / vmax, both required
    frame    each map on its own: percentiles `pct` (default 1, 99.9) of its finite pixels (positive ones for a log stretch), as `imaging.limits`
    global   one pair locked over the whole selection: the minimum / maximum of all maps (exact), or, with `pct`, the percentiles of all pixels of all maps together
             (exact up to `cap` values per map; above that a regular subsample of each map is used, so global percentiles of large maps are approximate, min / max never)
A `vmin` / `vmax` given together with `frame` or `global` replaces that bound only."""
import numpy as np

from . import imaging

POLICIES = ("global", "frame", "fixed")
SYM_PCT = 99.0          # default upper percentile of |value| for symmetric (contrast) limits: the noise tail of a residual map would otherwise set the range


def _usable(img, scale):
    v = np.asarray(img)[np.isfinite(img)]
    return v[v > 0] if scale == "log" else v


def _fix(lo, hi):
    return float(lo), float(hi if hi > lo else lo + abs(lo) * 1e-6 + 1e-30)


class Accumulator:
    """Collects what the global policy needs from maps one at a time (so a movie never holds all frames)."""
    def __init__(self, scale="log", cap=300_000, symmetric=False):
        self.scale, self.cap, self.symmetric = scale, cap, symmetric
        self.lo, self.hi, self.n = np.inf, -np.inf, 0
        self._samples = []

    def add(self, img):
        v = _usable(img, self.scale)
        if v.size == 0:
            return
        if self.symmetric:
            v = np.abs(v)
        self.lo, self.hi, self.n = min(self.lo, float(v.min())), max(self.hi, float(v.max())), self.n + v.size
        step = -(-v.size // self.cap)
        self._samples.append(v[::step].astype(np.float32))

    def result(self, pct=None):
        if not self._samples:
            return 0.0, 1.0
        if self.symmetric:           # (-m, m): m the pct[1] (default 99) percentile of |value| over all maps
            m = float(np.percentile(np.concatenate(self._samples), SYM_PCT if pct is None else pct[1]))
            m = m if m > 0 else 1e-30
            return -m, m
        if pct is None:
            return _fix(self.lo, self.hi)
        return _fix(*np.percentile(np.concatenate(self._samples), pct))


def resolve(policy, scale, pct, vmin, vmax, accumulator=None, img=None, symmetric=False):
    """(vmin, vmax) for one map under `policy`; `accumulator` (global) or `img` (frame) supplies the data. symmetric: (-v, v) around zero (contrast maps)."""
    if policy not in POLICIES:
        raise ValueError(f"limits policy must be one of {POLICIES}")
    if policy == "fixed":
        if vmin is None or vmax is None:
            raise ValueError("--limits fixed needs both --vmin and --vmax")
        return float(vmin), float(vmax)
    if policy == "frame":
        lo, hi = imaging.limits(img, pclip=pct or ((1.0, SYM_PCT) if symmetric else (1.0, 99.9)), scale=scale, symmetric=symmetric)
    else:
        lo, hi = accumulator.result(pct)
    return (lo if vmin is None else float(vmin)), (hi if vmax is None else float(vmax))


def parse_pct(text):
    if text is None:
        return None
    try:
        lo, hi = (float(x) for x in text.split(","))
    except ValueError:
        raise SystemExit("--pct must be 'lo,hi', e.g. 1,99.9")
    if not 0 <= lo < hi <= 100:
        raise SystemExit("--pct needs 0 <= lo < hi <= 100")
    return (lo, hi)
