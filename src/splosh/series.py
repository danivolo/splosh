"""Time series of a run from any source: the snapshots' scalars table, GIZMO's own logs (energy.txt, blackholes.txt), and sources that plugins register
(plugins.register_series: e.g. the per-BH logs of a modified GIZMO). What the viewer's time-series panes and the layouts' timeseries panels plot.

    ctx = context(run_dir, snapshots, analysis=None, period=None)
    available(ctx)                   -> [{"name", "label"}]: the sources with input files for this run
    table(name, ctx)                 -> Table (parsed once, cached as npz next to the run; re-read when an input file changes)
    curves(tab, columns, ...)        -> [{"column", "id", "t", "y", "style"}]: thinned (min/max per bucket, so spikes survive) and optionally smoothed

A Table is a set of rows: `time` (code units), named columns, and optionally `ids` (one curve per id, e.g. per black hole, ascending).
Column metadata (`meta[column]`): label, unit, style (line | dots | events: events are never thinned), group (the viewer's column menu groups them),
wrap (an angle's period: lines break at a wrap and get no running mean)."""
import hashlib
import json
import os

import numpy as np

from . import plugins, tables

CACHE_VERSION = 1


class Table:
    def __init__(self, time, columns, ids=None, meta=None, note=None):
        self.time = np.asarray(time, dtype=float)
        self.columns = {k: np.asarray(v, dtype=float) for k, v in columns.items()}
        self.ids = None if ids is None else np.asarray(ids, dtype=np.int64)
        self.meta = {k: dict(v) for k, v in (meta or {}).items()}
        self.note = note
        for k, v in self.columns.items():
            if len(v) != len(self.time):
                raise ValueError(f"column {k}: {len(v)} rows, time has {len(self.time)}")

    def __len__(self):
        return len(self.time)

    def id_list(self):
        return [] if self.ids is None else [int(i) for i in np.unique(self.ids)]

    def describe(self):
        """what the viewer needs to offer this source: columns with their metadata, ids, row count, time range."""
        cols = {k: {"label": self.meta.get(k, {}).get("label", k), "unit": self.meta.get(k, {}).get("unit", ""), "style": self.meta.get(k, {}).get("style", "line"),
                    "group": self.meta.get(k, {}).get("group", ""), **({"wrap": self.meta[k]["wrap"]} if self.meta.get(k, {}).get("wrap") else {})}
                for k, v in self.columns.items() if np.isfinite(v).any()}
        t = self.time[np.isfinite(self.time)]
        return {"columns": cols, "ids": self.id_list(), "rows": len(self), "t_range": [float(t.min()), float(t.max())] if t.size else None, "note": self.note}

    def save(self, path, key):
        data = {"time": self.time, **{"c_" + k: v for k, v in self.columns.items()}}
        if self.ids is not None:
            data["ids"] = self.ids
        tmp = path + ".tmp.npz"
        np.savez(tmp, **data)
        os.replace(tmp, path)
        with open(path + ".json.tmp", "w") as f:
            json.dump({"key": key, "meta": self.meta, "note": self.note}, f)
        os.replace(path + ".json.tmp", path + ".json")

    @classmethod
    def load(cls, path, key):
        try:
            with open(path + ".json") as f:
                info = json.load(f)
            if info.get("key") != key:
                return None
            with np.load(path) as z:
                cols = {k[2:]: z[k] for k in z.files if k.startswith("c_")}
                return cls(z["time"], cols, ids=z["ids"] if "ids" in z.files else None, meta=info.get("meta"), note=info.get("note"))
        except (OSError, ValueError, KeyError):
            return None


def context(run_dir, snapshots=(), analysis=None, period=None):
    """what a source gets: the run directory (the one holding output/), its snapshot paths, the analysis directory with scalars.csv (or None) and the binary period P0 (code time, or None)."""
    return {"run_dir": os.path.abspath(run_dir), "snapshots": list(snapshots), "analysis": analysis, "period": period}


def output_file(ctx, *names):
    """the first existing, non-empty file among <run>/output/NAME and <run>/NAME."""
    for base in (os.path.join(ctx["run_dir"], "output"), ctx["run_dir"]):
        for n in names:
            f = os.path.join(base, n)
            if os.path.isfile(f) and os.path.getsize(f) > 0:
                return f
    return None


# ----------------------------------------------------------------------------- built-in sources

def _snapshots_files(ctx):
    f = os.path.join(ctx["analysis"], "scalars.csv") if ctx.get("analysis") else None
    return [f] if f and os.path.isfile(f) else []


def _snapshots(ctx):
    from .layout_preview import read_scalars_table
    t = read_scalars_table(ctx["analysis"])
    t.pop("_snapshots", None); t.pop("index", None)
    time = t.pop("time")
    t.pop("time_myr", None)
    return Table(time, t, meta={k: {"style": "linepoints"} for k in t})


def _energy_files(ctx):
    f = output_file(ctx, "energy.txt")
    return [f] if f else []


def _energy(ctx):
    """GIZMO's energy.txt (run.c): Time, E_int, E_pot, E_kin, then (int, pot, kin) per particle type 0..5, then (newer GIZMO) the mass of each type.
    E_pot is 0 unless GIZMO computed it (EVALPOTENTIAL / COMPUTE_POTENTIAL_ENERGY). Rows a restart rewrote are dropped."""
    f = _energy_files(ctx)[0]
    arr = tables.read_rows(f, 28, widths={22, 28})
    stale = tables.stale_after_restart([arr[:, 0]])[0]
    arr = arr[~stale]
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    eint, epot, ekin = arr[:, 1], arr[:, 2], arr[:, 3]
    has_pot = bool(np.any(epot != 0))
    tot = eint + ekin + (epot if has_pot else 0.0)
    e0 = tot[0] if len(tot) and tot[0] != 0 else 1.0
    cols = {"E_int": eint, "E_kin": ekin, "E_tot": tot, "dE_rel": np.abs((tot - e0) / e0)}
    meta = {"E_int": {"label": "internal energy (gas)", "unit": "code energy", "group": "energy"}, "E_kin": {"label": "kinetic energy (all types)", "unit": "code energy", "group": "energy"},
            "E_tot": {"label": "E_int + E_kin" + (" + E_pot" if has_pot else ""), "unit": "code energy", "group": "energy"},
            "dE_rel": {"label": "|E - E0| / |E0| (not conserved with cooling, accretion, feedback)", "group": "conservation"}}
    if has_pot:
        cols["minus_E_pot"] = -epot
        meta["minus_E_pot"] = {"label": "-E_pot", "unit": "code energy", "group": "energy"}
    for k in range(6):
        for j, nm in enumerate(("int", "pot", "kin")):
            v = arr[:, 4 + 3 * k + j]
            if np.any(np.nan_to_num(v) != 0):
                cols[f"E_{nm}_{k}"] = v
                meta[f"E_{nm}_{k}"] = {"label": f"E_{nm} of type {k}", "unit": "code energy", "group": "by type"}
        v = arr[:, 22 + k]
        if np.isfinite(v).any() and np.any(np.nan_to_num(v) != 0):
            cols[f"mass_{k}"] = v
            meta[f"mass_{k}"] = {"label": f"mass of type {k}", "unit": "code mass", "group": "by type"}
    return Table(arr[:, 0], cols, meta=meta, note=None if has_pot else "E_pot is not in energy.txt (GIZMO writes 0 without EVALPOTENTIAL or COMPUTE_POTENTIAL_ENERGY)")


def _blackholes_files(ctx):
    f = output_file(ctx, "blackholes.txt")
    return [f] if f else []


BLACKHOLES_COLUMNS = (("n_bh", "number of BHs", ""), ("m_bh_total", "total BH mass", "code mass"), ("mdot_total", "total accretion rate", "code mass / code time"),
                      ("mdot_msun_yr", "total accretion rate", "Msun/yr"), ("m_dyn_total", "total dynamical BH mass", "code mass"),
                      ("mdot_edd_total", "sum of the Eddington ratios", ""))


def _blackholes(ctx):
    """GIZMO's blackholes.txt (blackhole_util.c): All.Time, TotBHs, sum of BH_Mass, sum of BH_Mdot, the latter in Msun/yr, sum of the dynamical masses, sum of Mdot/Mdot_Edd."""
    arr = tables.read_rows(_blackholes_files(ctx)[0], 7)
    stale = tables.stale_after_restart([arr[:, 0]])[0]
    arr = arr[~stale]
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    return Table(arr[:, 0], {k: arr[:, 1 + j] for j, (k, _, _) in enumerate(BLACKHOLES_COLUMNS)},
                 meta={k: {"label": lab, "unit": u, "group": "black holes"} for k, lab, u in BLACKHOLES_COLUMNS})


plugins.register_series("snapshots", _snapshots, _snapshots_files, label="snapshot scalars (scalars.csv)", defaults=["bh_sep", "gas_mass", "rho_max"], priority=1)
plugins.register_series("energy", _energy, _energy_files, label="GIZMO energy.txt", defaults=["dE_rel"])
plugins.register_series("blackholes", _blackholes, _blackholes_files, label="GIZMO blackholes.txt", defaults=["mdot_msun_yr"])


# ----------------------------------------------------------------------------- access, cache

def available(ctx):
    plugins.load_plugins()
    products.register_sources()
    out = []
    for name, src in plugins.SERIES.items():
        try:
            if src["files"](ctx):
                out.append({"name": name, "label": src["label"], "doc": src["doc"], "defaults": src.get("defaults", []), "priority": src.get("priority", 0)})
        except Exception:                                    # a broken plugin hides its source, never the others
            continue
    return sorted(out, key=lambda s: -s["priority"])


def _key(name, src, files):
    sig = [(os.path.abspath(f), os.path.getsize(f), int(os.path.getmtime(f))) for f in files]
    return hashlib.sha1(json.dumps([CACHE_VERSION, name, src["version"], sig]).encode()).hexdigest()


def cache_dir(ctx):
    """<run>/analysis_viewer/cache when writable, else ~/.cache/splosh/series/<hash of the run path>."""
    d = os.path.join(ctx["run_dir"], "analysis_viewer", "cache")
    try:
        os.makedirs(d, exist_ok=True)
        if os.access(d, os.W_OK):
            return d
    except OSError:
        pass
    d = os.path.join(os.path.expanduser("~"), ".cache", "splosh", "series", hashlib.sha1(ctx["run_dir"].encode()).hexdigest()[:16])
    os.makedirs(d, exist_ok=True)
    return d


_memo = {}
_locks, _locks_guard = {}, __import__("threading").Lock()


def _lock(key):
    """one parse of a source at a time (the viewer asks for several panes at once: parallel parses of the same big log would each hold it in memory)."""
    with _locks_guard:
        return _locks.setdefault(key, __import__("threading").Lock())


def table(name, ctx, use_cache=True):
    plugins.load_plugins()
    products.register_sources()
    if name not in plugins.SERIES:
        raise KeyError(f"no series source {name!r}; sources: {sorted(plugins.SERIES)}")
    src = plugins.SERIES[name]
    files = src["files"](ctx)
    if not files:
        raise LookupError(f"{name}: no input files for this run")
    mk = (ctx["run_dir"], name)
    with _lock(mk):
        return _table_locked(name, src, files, ctx, mk, use_cache)


def _table_locked(name, src, files, ctx, mk, use_cache):
    key = _key(name, src, files)
    got = _memo.get(mk)
    if got and got[0] == key:
        return got[1]
    path = os.path.join(cache_dir(ctx), f"series_{name.replace(':', '_')}.npz")
    tab = Table.load(path, key) if use_cache else None
    if tab is None:
        tab = src["fn"](ctx)
        try:
            tab.save(path, key)
        except OSError:
            pass
    _memo[mk] = (key, tab)
    return tab


# ----------------------------------------------------------------------------- curves

def running_mean(t, y, width, min_points=7, short_frac=None):
    """Centred running mean of y over a window `width` (units of t), NaN rows skipped. A series spanning less than two windows gets the window short_frac x span
    (short_frac given) or min_points neighbours (trend lines for short runs)."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) == 0:
        return t, y
    c = np.concatenate([[0.0], np.cumsum(y)])
    span = t[-1] - t[0]
    if span < 2.0 * width and short_frac is not None:
        width = short_frac * span
    if span >= 2.0 * width:
        lo = np.searchsorted(t, t - 0.5 * width, side="left")
        hi = np.searchsorted(t, t + 0.5 * width, side="right")
    else:
        k = min_points // 2
        idx = np.arange(len(t))
        lo, hi = np.clip(idx - k, 0, len(t)), np.clip(idx + k + 1, 0, len(t))
    return t, (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def thin_minmax(t, y, n):
    """at most ~2n points of (t, y) keeping each bucket's minimum and maximum (so spikes survive) and a NaN where a whole bucket has no value (so gaps stay gaps)."""
    t = np.asarray(t, dtype=float); y = np.asarray(y, dtype=float)
    if len(t) <= 2 * n:
        return t, y
    k = int(np.ceil(len(t) / n))
    m = (len(t) // k) * k
    Y = y[:m].reshape(-1, k)
    fin = np.isfinite(Y)
    any_ = fin.any(axis=1)
    lo = np.argmin(np.where(fin, Y, np.inf), axis=1)
    hi = np.argmax(np.where(fin, Y, -np.inf), axis=1)
    base = np.arange(Y.shape[0]) * k
    idx = np.sort(np.concatenate([(base + np.minimum(lo, hi))[any_], (base + np.maximum(lo, hi))[any_]]))
    idx = np.unique(idx)
    gaps = base[~any_]                                       # buckets without any value: one NaN point
    tt = np.concatenate([t[idx], t[gaps], t[m:]])
    yy = np.concatenate([y[idx], np.full(len(gaps), np.nan), y[m:]])
    o = np.argsort(tt, kind="stable")
    return tt[o], yy[o]


def curves(tab, columns, ids=None, n=4000, smooth=None, short_frac=0.02, tmin=None, tmax=None, max_events=200000):
    """[{"column", "id", "t", "y", "style", "smooth": bool}] for the asked columns (and ids; None = all), thinned to ~n buckets; `smooth` (code time) adds a running-mean curve
    per series (trend lines; short runs get short_frac of the span). Events (style 'events') are sent as they are, up to max_events."""
    out = []
    want = tab.id_list() if ids is None else [int(i) for i in ids]
    groups = [(None, np.ones(len(tab), dtype=bool))] if tab.ids is None else [(i, tab.ids == i) for i in want]
    win = np.ones(len(tab), dtype=bool)
    if tmin is not None:
        win &= tab.time >= tmin
    if tmax is not None:
        win &= tab.time <= tmax
    for c in columns:
        if c not in tab.columns:
            raise KeyError(f"no column {c!r}; columns: {sorted(tab.columns)}")
        style = tab.meta.get(c, {}).get("style", "line")
        for i, sel in groups:
            sel = sel & win
            t, y = tab.time[sel], tab.columns[c][sel]
            if style == "events":
                ok = np.isfinite(y)
                t, y = t[ok][:max_events], y[ok][:max_events]
                out.append({"column": c, "id": i, "t": t, "y": y, "style": style, "smooth": False})
                continue
            if smooth and not tab.meta.get(c, {}).get("wrap"):           # no mean of an angle that wraps (it would average 179 and -179 to 0)
                ts, ys = running_mean(t, y, float(smooth), short_frac=short_frac)
                ts, ys = thin_minmax(ts, ys, n)
                out.append({"column": c, "id": i, "t": ts, "y": ys, "style": "line", "smooth": True})
            tt, yy = thin_minmax(t, y, n)
            out.append({"column": c, "id": i, "t": tt, "y": yy, "style": style, "smooth": False})
    return out


def jsonable(curves_):
    """curves with plain lists (NaN -> None: browsers reject NaN in JSON)."""
    def lst(a):
        a = np.asarray(a, dtype=float)
        return [None if not np.isfinite(v) else float(v) for v in a]
    return [{**c, "t": lst(c["t"]), "y": lst(c["y"])} for c in curves_]


# ----------------------------------------------------------------------------- layout panels (layout_preview, both renderers)

_RUNCTX = {}


def context_for_snapshot(path, analysis=None):
    """the context of the run that holds snapshot `path` (cached): run directory, its snapshots, P0 of the first one, and Myr per code time unit (header units)."""
    from .analyze import run_root
    from .runs import find_snapshots, binary_period, snapshot_times
    folder = os.path.dirname(os.path.abspath(path))
    root = run_root(folder)
    key = (root, analysis)
    if key not in _RUNCTX:
        snaps = find_snapshots([folder]) or [path]          # the snapshots next to this one (not, e.g., initial conditions kept in the run directory)
        period = binary_period(snaps[0]) if snaps else None
        myr = None
        for s in reversed(snaps):
            t, tm = snapshot_times(s)
            if tm is not None and t and np.isfinite(t) and t > 0:
                myr = tm / t
                break
        ctx = context(root, snaps, analysis, period)
        ctx["myr_per_code"] = myr
        _RUNCTX[key] = ctx
    return _RUNCTX[key]


def auto_scale(values, want="auto"):
    """'lin' | 'log' | ('symlog', linthresh): log when all positive over more than 100x, symlog when both signs and |y| spans more than 3 decades
    (linthresh = 20th percentile of |y|); `want` forces one (log falls back to lin without positive values)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return "lin"
    pos, a = v[v > 0], np.sort(np.abs(v[v != 0]))
    lt = float(a[int(0.2 * (len(a) - 1))]) if a.size else 1.0
    if want == "log":
        return "log" if pos.size else "lin"
    if want == "symlog":
        return ("symlog", lt or 1.0)
    if want == "lin":
        return "lin"
    if (v >= 0).all() and pos.size and pos.max() > 100 * pos.min():
        return "log"
    if (v < 0).any() and a.size and a[-1] > 1e3 * lt:
        return ("symlog", lt)
    return "lin"


def _break_wraps(t, y, period):
    jumps = np.where(np.abs(np.diff(y)) > 0.5 * period)[0] + 1
    return np.insert(t, jumps, np.nan), np.insert(y, jumps, np.nan)


COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf")


def panel_tile(p, ctx, tnow=None, n=3000):
    """What a layout timeseries panel with a `source` draws: {'curves': [(label, xs, ys)], 'styles': [...], 'xnow', 'xlabel', 'ylabel', 'yscale', 'twin', 'hlines'}.
    x is code time, Myr or t/P0 (xaxis time | time_myr | orb | index is not available here); `smooth` is in units of P0 (or of the run's duration without a binary)."""
    tab = table(p["source"], ctx)
    period, myr = ctx.get("period"), ctx.get("myr_per_code")
    xa = p.get("xaxis", "time")
    if xa == "orb" and period:
        fx, xlabel = (lambda t: t / period), "t / P0"
    elif xa == "time_myr" and myr:
        fx, xlabel = (lambda t: t * myr), "t [Myr]"
    else:
        fx, xlabel = (lambda t: t), "t [code time]"
    span = (tab.time.max() - tab.time.min()) if len(tab) else 0.0
    sm = float(p.get("smooth") or 0.0)
    sm_code = sm * (period if period else span) if sm > 0 else None
    tmax = tnow if (p.get("series") == "past" and tnow is not None) else None
    left, right = list(p["columns"]), list(p.get("twin") or [])
    ids = p.get("ids")
    multi_col = len(left) + len(right) > 1
    out, styles, vals = [], [], {"left": [], "right": []}
    id_list = [i for i in tab.id_list() if ids is None or i in ids]          # the ids this panel shows
    multi_id = len(id_list) > 1
    for side, cols in (("left", left), ("right", right)):
        cv = curves(tab, cols, ids=ids, n=n, smooth=sm_code, tmax=tmax)
        has_mean = {(c["column"], c["id"]) for c in cv if c["smooth"]}
        if p.get("raw", True) is False:                  # raw = false: only the running means of the columns that have one
            cv = [c for c in cv if c["smooth"] or (c["column"], c["id"]) not in has_mean]
        for c in cv:
            meta = tab.meta.get(c["column"], {})
            t, y = np.asarray(c["t"], float), np.asarray(c["y"], float)
            if meta.get("wrap") and c["style"] != "events":
                t, y = _break_wraps(t, y, meta["wrap"])
            k_col = (left + right).index(c["column"])
            k_id = id_list.index(c["id"]) if c["id"] in id_list else 0
            color = COLORS[(k_col if multi_col else k_id) % len(COLORS)]
            dash = ("-", "--", ":", "-.")[k_id % 4] if (multi_col and multi_id) else "-"
            label = meta.get("label", c["column"]) + (f", BH {c['id']}" if c["id"] is not None and multi_id else "") + (" (mean)" if c["smooth"] else "")
            faint = (not c["smooth"]) and (c["column"], c["id"]) in has_mean
            out.append((label, fx(t), y))
            styles.append({"style": "line" if c["smooth"] else c["style"], "color": color, "dash": dash, "twin": side == "right", "alpha": 0.35 if faint else 1.0,
                           "lw": 1.8 if c["smooth"] else 1.0, "legend": not faint})
            vals[side].append(y)
    def axis_label(cols):                     # short: a long description goes to the legend and the panel title, the axis gets the column name
        u = {tab.meta.get(c, {}).get("unit", "") for c in cols} - {""}
        name = ""
        if len(cols) == 1:
            lab = tab.meta.get(cols[0], {}).get("label", cols[0])
            name = lab if len(lab) <= 22 else cols[0]
        return (name + (" " if name and u else "") + (f"[{', '.join(sorted(u))}]" if u else "")) or None
    ylabel, twin_label = axis_label(left), (axis_label(right) if right else None)
    cat = lambda side: np.concatenate(vals[side]) if vals[side] else np.array([])
    return {"curves": out, "styles": styles, "xnow": fx(tnow) if (tnow is not None and p.get("marker", True)) else None, "xlabel": xlabel,
            "ylabel": ylabel, "yscale": auto_scale(cat("left"), p.get("yscale") or ("log" if p.get("ylog") else "auto")),
            "twin_scale": auto_scale(cat("right"), p.get("twin_scale") or "auto") if right else None, "twin_label": twin_label,
            "hlines": [float(h) for h in (p.get("hlines") or [])], "xlog": bool(p.get("xlog")), "ylog": False, "note": tab.note}


from . import perf  # noqa: E402,F401  (registers the GIZMO performance sources: cpu, timebin, timings, balance)


# ----------------------------------------------------------------------------- heatmaps (a family of indexed columns: occ_0 .. occ_63, ...)

def heat_columns(tab, prefix):
    """[(y value, column)] of the columns named <prefix><number>, by number."""
    out = []
    for c in tab.columns:
        if c.startswith(prefix):
            try:
                out.append((float(c[len(prefix):]), c))
            except ValueError:
                pass
    return sorted(out)


def heat_grid(tab, prefix, ids=None, nx=600, tmin=None, tmax=None, norm="none"):
    """(x edges [code time], y values, Z [ny, nx]) of the columns <prefix><y> of a table, averaged into nx time buckets (rows of one id when `ids`);
    norm = column: every time column sums to 1 (a distribution over y at each time), none: the values."""
    cols = heat_columns(tab, prefix)
    if not cols:
        raise KeyError(f"no columns {prefix}<number> in this source")
    sel = np.ones(len(tab), dtype=bool)
    if tab.ids is not None and ids:
        sel &= np.isin(tab.ids, list(ids))
    if tmin is not None:
        sel &= tab.time >= tmin
    if tmax is not None:
        sel &= tab.time <= tmax
    t = tab.time[sel]
    if not len(t):
        raise ValueError("no rows in this window")
    Y = np.array([y for y, _ in cols])
    M = np.vstack([np.nan_to_num(tab.columns[c][sel]) for _, c in cols])      # [ny, nrows]
    t0, t1 = float(t.min()), float(t.max())
    if t1 <= t0:
        t1 = t0 + 1.0
    nx = int(max(2, min(nx, len(t))))
    edges = np.linspace(t0, t1, nx + 1)
    k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nx - 1)
    cnt = np.bincount(k, minlength=nx).astype(float)
    Z = np.vstack([np.bincount(k, weights=row, minlength=nx) for row in M]) / np.maximum(cnt, 1.0)
    Z[:, cnt == 0] = np.nan
    if norm == "column":
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = Z / np.nansum(Z, axis=0, keepdims=True)
    return edges, Y, Z


def heat_rows(Y, Z):
    """integer y (bins): a row per integer, the gaps as empty rows, without the edge rows that hold anything in under 0.5% of the time columns."""
    if len(Y) > 1 and np.all(Y == np.round(Y)):
        full = np.arange(Y.min(), Y.max() + 1)
        Zf = np.full((len(full), Z.shape[1]), np.nan)
        Zf[np.searchsorted(full, Y)] = Z
        used = np.nanmean(np.nan_to_num(Zf) > 0, axis=1) >= 0.005
        if used.any():
            k0, k1 = np.argmax(used), len(used) - np.argmax(used[::-1])
            full, Zf = full[k0:k1], Zf[k0:k1]
        return full.astype(float), Zf
    return Y, Z


def heat_tile(p, ctx, tnow=None):
    """what a layout heatmap panel draws: {'heat': {'x', 'y', 'z', 'zscale', 'cmap', 'norm'}, 'curves', 'styles', 'xnow', 'xlabel', 'ylabel'}; x in the panel's time axis."""
    tab = table(p["source"], ctx)
    period, myr = ctx.get("period"), ctx.get("myr_per_code")
    xa = p.get("xaxis", "time")
    fx, xlabel = ((lambda t: t / period), "t / P0") if (xa == "orb" and period) else (((lambda t: t * myr), "t [Myr]") if (xa == "time_myr" and myr) else ((lambda t: t), "t [code time]"))
    edges, Y, Z = heat_grid(tab, p["prefix"], ids=p.get("ids"), nx=int(p.get("nx") or 600), norm=p.get("norm", "none"))
    Y, Z = heat_rows(Y, Z)
    out = {"heat": {"x": fx(edges), "y": Y, "z": Z, "zscale": p.get("zscale") or "log", "cmap": p.get("cmap") or "magma", "norm": p.get("norm", "none"),
                    "label": p.get("zlabel") or ("fraction" if p.get("norm") == "column" else p["prefix"].rstrip("_"))},
           "curves": [], "styles": [], "xnow": fx(tnow) if (tnow is not None and p.get("marker", True)) else None, "xlabel": xlabel,
           "ylabel": p.get("ylabel") or p["prefix"].rstrip("_"), "yscale": p.get("yscale") or "lin", "note": tab.note}
    for c in p.get("overlay") or []:                     # curves on the same y axis (e.g. the lowest occupied bin over the occupancy)
        for cv in curves(tab, [c], ids=p.get("ids"), n=1500):
            out["curves"].append((tab.meta.get(c, {}).get("label", c), fx(np.asarray(cv["t"], float)), np.asarray(cv["y"], float)))
            out["styles"].append({"color": "#00e5ff", "lw": 1.0, "style": "line"})
    return out
from . import products  # noqa: E402  (snapshot products as series sources prod_<name>)
