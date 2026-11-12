"""Phase diagrams of the gas in a view's window, and selections drawn on them, shown back on the map ("where does this gas live").

The gas of a view: the particles whose position falls inside the window (horizontal and vertical window coordinates), all along the line of sight for a
projection, within their support H of the plane z0 for a slice.

Quantities (`quantities()`): `density` (the SPH Density dataset), every per-particle field of the map registry (u, T, cs, pressure, entropy, vx, vy, vz, vlos,
speed, vR, vphi, jz, H, ... and plugin fields; not `rho`, which is a map quantity), and the geometry of the frame: `x`, `y`, `z` (frame), `R` (cylindrical,
about the frame's z axis), `r` (spherical distance from the centre), `h`, `v`, `los` (the view's horizontal, vertical and line-of-sight coordinates).
Values are converted to the view's unit system (View.units), like the maps.

    histogram(snap, view, xq, yq, bins=256, xlog=True, ylog=True, xr=None, yr=None, weight="mass") -> (grid (bins, bins) float32, origin lower-left, meta)
    select(snap, view, xq, yq, poly, xlog=True, ylog=True) -> boolean mask over PartType0 rows: in the view's gas and inside the polygon (the axes' coordinates,
                                                                 log10 for a log axis), the polygon given as [[x, y], ...]
    subset(snap, mask) -> the snapshot dict with only those gas particles (black holes, header and other types untouched)
    weights: mass (the default: where the mass is), count (particles), volume (mass / density)."""
import numpy as np

from . import plugins
from .fields import resolve_unit
from .frames import make_frame
from .maps import AXES, FieldContext, _field_values, field_names, required_gas_fields
from .units import DENSITY, LENGTH, Units, quantity_meta

GEOMETRY = {"x": "frame x", "y": "frame y", "z": "frame z", "R": "cylindrical radius", "r": "distance from the centre",
            "h": "horizontal view coordinate", "v": "vertical view coordinate", "los": "line-of-sight coordinate"}
PER_SNAPSHOT_KEYS = ("header", "time", "bh_pos", "bh_vel", "bh_mass", "bh_mdyn", "bh_id", "pt")


def quantities():
    return ["density"] + [f for f in field_names() if f != "rho"] + list(GEOMETRY)


def needs(view, *names):
    """PartType0 datasets the view and these quantities read."""
    f = set(required_gas_fields(view)) | {"Density"}
    for n in names:
        if n in GEOMETRY or n == "density":
            continue
        f |= required_gas_fields(view.with_(field=n))
    return f


def _in_view(snap, view, frame):
    ia, ib, il = AXES[view.axes]
    pos = frame.pos(snap["Coordinates"])
    half = 0.5 * view.width
    keep = (np.abs(pos[:, ia] - view.cx) < half) & (np.abs(pos[:, ib] - view.cy) < half)
    if view.mode == "slice":
        keep &= np.abs(pos[:, il] - view.z0) < np.asarray(snap["SmoothingLength"]) * view.hscale
    return keep, pos


def values(snap, view, name, frame=None, pos=None):
    """(values (N,) float64 for every gas particle in the view's unit system, quantity meta)."""
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    units = Units.from_header(snap.get("header", {}))
    if name in GEOMETRY:
        pos = frame.pos(snap["Coordinates"]) if pos is None else pos
        ia, ib, il = AXES[view.axes]
        v = {"x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2], "R": np.hypot(pos[:, 0], pos[:, 1]), "r": np.sqrt((pos ** 2).sum(1)),
             "h": pos[:, ia], "v": pos[:, ib], "los": pos[:, il]}[name]
        dim, label = LENGTH, GEOMETRY[name]
    elif name == "density":
        if "Density" not in snap:
            raise ValueError("phase: 'density' needs the Density dataset")
        v, dim, label = np.asarray(snap["Density"], dtype=np.float64), DENSITY, "density"
    else:
        plugins.load_plugins()
        from . import datasets
        datasets.ensure(name)                                   # a snapshot dataset by its name
        if name not in field_names() or name == "rho":
            raise KeyError(f"unknown phase quantity {name!r}; available: {quantities()}")
        v = np.asarray(_field_values(name, FieldContext(snap, frame, view)), dtype=np.float64)
        dim, label = resolve_unit(view, name, "slice"), name
    qm = quantity_meta(units, dim, name, view.units)
    return v * qm["factor"], dict(qm, label=label)


def _axis(v, log):
    if not log:
        return v, np.isfinite(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log10(v)
    return out, np.isfinite(out)


def _weights(snap, idx, weight):
    if weight == "count":
        return np.ones(len(idx))
    m = np.asarray(snap["Masses"], dtype=np.float64)[idx]
    if weight == "mass":
        return m
    if weight == "volume":
        return m / np.asarray(snap["Density"], dtype=np.float64)[idx]
    raise ValueError("weight must be mass, count or volume")


def histogram(snap, view, xq, yq, bins=256, xlog=True, ylog=True, xr=None, yr=None, weight="mass"):
    view.validate()
    if not 8 <= int(bins) <= 1024:
        raise ValueError("bins must be in 8..1024")
    frame = make_frame(snap, view.center, view.align, view.tilt)
    keep, pos = _in_view(snap, view, frame)
    xv, xm = values(snap, view, xq, frame, pos)
    yv, ym = values(snap, view, yq, frame, pos)
    ax, okx = _axis(xv, xlog)
    ay, oky = _axis(yv, ylog)
    idx = np.nonzero(keep & okx & oky)[0]
    w = _weights(snap, idx, weight)
    a, b = ax[idx], ay[idx]

    def rng(arr, given):
        if given is not None:
            lo, hi = float(given[0]), float(given[1])
        elif len(arr):
            lo, hi = np.percentile(arr, [0.05, 99.95]) if len(arr) > 100 else (arr.min(), arr.max())
            lo, hi = float(lo), float(hi)
        else:
            lo, hi = 0.0, 1.0
        if not hi > lo:
            lo, hi = lo - 0.5 * (abs(lo) or 1.0), hi + 0.5 * (abs(hi) or 1.0)
        return lo, hi
    xr, yr = rng(a, xr), rng(b, yr)
    grid, _, _ = np.histogram2d(b, a, bins=int(bins), range=[yr, xr], weights=w)       # rows = y (origin lower-left), columns = x
    inside = (a >= xr[0]) & (a <= xr[1]) & (b >= yr[0]) & (b <= yr[1])
    meta = {"x": dict(xm, log=bool(xlog), range=list(xr)), "y": dict(ym, log=bool(ylog), range=list(yr)), "bins": int(bins), "weight": weight,
            "n_in_view": int(keep.sum()), "n_used": int(len(idx)), "n_in_range": int(inside.sum()), "total": float(w.sum()), "total_in_range": float(w[inside].sum()),
            "frame": frame.describe(), "view": view.as_dict(), "time": float(snap["time"])}
    return grid.astype(np.float32), meta


def _inside_polygon(px, py, poly):
    """even-odd rule, vectorised over the points."""
    poly = np.asarray(poly, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError("the selection polygon needs at least 3 [x, y] vertices")
    inside = np.zeros(len(px), bool)
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        crosses = (y1 > py) != (y0 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (px < xint)
        x0, y0 = x1, y1
    return inside


def select(snap, view, xq, yq, poly, xlog=True, ylog=True, window=True):
    """window=False: every gas particle inside the polygon, wherever it is (what a map of the selection splats: the kernels of particles just outside the
    window still reach into it)."""
    frame = make_frame(snap, view.center, view.align, view.tilt)
    keep, pos = _in_view(snap, view, frame)
    if not window:
        keep = np.ones(len(keep), bool)
    ax, okx = _axis(values(snap, view, xq, frame, pos)[0], xlog)
    ay, oky = _axis(values(snap, view, yq, frame, pos)[0], ylog)
    mask = keep & okx & oky
    idx = np.nonzero(mask)[0]
    mask[idx] = _inside_polygon(ax[idx], ay[idx], poly)
    return mask


def subset(snap, mask):
    """a shallow copy of the snapshot dict keeping only the masked gas particles (every PartType0 array of the same length)."""
    n = len(mask)
    out = {}
    for k, v in snap.items():
        if k not in PER_SNAPSHOT_KEYS and isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n:
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def parse_selection(text):
    """the `sel` request parameter: JSON {x, y, xlog, ylog, poly} -> dict (validated)."""
    import json
    d = json.loads(text)
    if not isinstance(d, dict) or not {"x", "y", "poly"} <= set(d):
        raise ValueError("sel needs x, y and poly")
    if d["x"] not in quantities() or d["y"] not in quantities():
        raise ValueError(f"sel: unknown quantity; available: {quantities()}")
    poly = [[float(a), float(b)] for a, b in d["poly"]]
    if not 3 <= len(poly) <= 2000:
        raise ValueError("sel: the polygon needs 3 to 2000 vertices")
    return {"x": d["x"], "y": d["y"], "xlog": bool(d.get("xlog", True)), "ylog": bool(d.get("ylog", True)), "poly": poly}


def selected_points(snap, view, mask, limit=20000, seed=0):
    """window coordinates of the selected particles inside the window (a random subsample of at most `limit`), for a point overlay."""
    frame = make_frame(snap, view.center, view.align, view.tilt)
    keep, pos = _in_view(snap, view, frame)
    idx = np.nonzero(keep & mask)[0]
    n = len(idx)
    if n > limit:
        idx = np.sort(np.random.default_rng(seed).choice(idx, limit, replace=False))
    ia, ib, _ = AXES[view.axes]
    m = np.asarray(snap["Masses"], dtype=np.float64)
    return {"n_selected": int(mask.sum()), "n_in_window": int(n), "mass_selected": float(m[mask].sum()), "mass_in_window": float(m[keep].sum()),
            "points": np.round(np.column_stack([pos[idx, ia], pos[idx, ib]]), 6).tolist(), "subsampled": n > limit}
