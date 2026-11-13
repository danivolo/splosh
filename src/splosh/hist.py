"""Distributions (histograms and cumulative distributions) of gas quantities: the layout panel type `hist` (a view window, like `phase`) and the snapshot
product `hist` (a region about the frame centre; per-snapshot histograms as families, so a distribution against time is a heatmap).

Quantities: those of the phase diagram (phase.quantities(): density, every per-particle field, the geometry x y z R r h v los), in the view's unit system.

    values_in_view(snap, view, name, weight="mass") -> (values, weights, quantity meta)    the gas of the view's window (phase._in_view)
    distribution(v, w, bins=128, xlog=True, xr=None) -> {"edges", "centres", "pdf", "cdf", "total", "in_range", "below", "above", "median", "p16", "p84"}
        pdf: weight per unit of the axis (per dex when xlog) / the total weight (so it integrates to the fraction in the range);
        cdf: the fraction of the total weight at or below each right edge (the weight below the range included); total: of the particles with a finite
        axis value (positive for a log axis); median, p16, p84 in the quantity's units (weighted quantiles).

Product `hist` (series source prod_hist), parameters: quantities (["density", "T"]), weight (mass | count | volume), per_dex (bins per decade of a log
quantity, 10: bins on a fixed grid of 1 / per_dex dex, so every snapshot has the same rows), bins (64, linear quantities), ranges ({name: [lo, hi]}, axis
units: log10 when log), log ({name: bool}; default: log when every value is positive), center (auto: the black holes' centre of mass, else the gas
centre of mass; or any frames.py centre rule), tilt (frames.py), r_min / r_max (the region, code length; r_max None = all gas), radius (r spherical |
R cylindrical), units (code | cgs | astro). Columns: <q>_<x> the PDF (x the bin centre, log10 for a log axis), <q>cdf_<x> the CDF at the bin's right
edge, <q>_med, <q>_p16, <q>_p84, <q>_frac (the fraction of the region's weight inside the range), and wtotal (the region's total weight)."""
import numpy as np

from . import plugins

SHOW = ("pdf", "cdf", "both")


def values_in_view(snap, view, name, weight="mass", frame=None):
    from . import phase
    from .frames import make_frame
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    keep, pos = phase._in_view(snap, view, frame)
    v, meta = phase.values(snap, view, name, frame, pos)
    idx = np.nonzero(keep)[0]
    return v[idx], phase._weights(snap, idx, weight), meta


def _quantile(a, w, q):
    if not len(a):
        return np.nan
    o = np.argsort(a, kind="stable")
    c = np.cumsum(w[o])
    c = (c - 0.5 * w[o]) / c[-1]
    return float(np.interp(q, c, a[o]))


def auto_range(a, given=None):
    """[lo, hi] of an axis: as given, else the 0.05 and 99.95 percentiles (the extremes for <= 100 values), widened when empty."""
    if given is not None:
        lo, hi = float(given[0]), float(given[1])
    elif len(a):
        lo, hi = (np.percentile(a, [0.05, 99.95]) if len(a) > 100 else (a.min(), a.max()))
        lo, hi = float(lo), float(hi)
    else:
        lo, hi = 0.0, 1.0
    if not hi > lo:
        lo, hi = lo - 0.5 * (abs(lo) or 1.0), hi + 0.5 * (abs(hi) or 1.0)
    return lo, hi


def distribution(v, w, bins=128, xlog=True, xr=None, edges=None):
    """see the module docstring. `edges` (axis units) replaces bins / xr."""
    v, w = np.asarray(v, np.float64), np.asarray(w, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log10(v) if xlog else v.copy()
    ok = np.isfinite(a) & np.isfinite(w)
    a, w = a[ok], w[ok]
    if edges is None:
        lo, hi = auto_range(a, xr)
        edges = np.linspace(lo, hi, int(bins) + 1)
    edges = np.asarray(edges, np.float64)
    total = float(w.sum())
    h, _ = np.histogram(a, bins=edges, weights=w)
    below, above = float(w[a < edges[0]].sum()), float(w[a > edges[-1]].sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = h / (total * np.diff(edges)) if total > 0 else np.full(len(h), np.nan)
        cdf = (below + np.cumsum(h)) / total if total > 0 else np.full(len(h), np.nan)
    back = (lambda x: 10.0 ** x) if xlog else (lambda x: x)
    return {"edges": edges, "centres": 0.5 * (edges[1:] + edges[:-1]), "pdf": pdf, "cdf": cdf, "total": total, "in_range": float(h.sum()),
            "below": below, "above": above, "xlog": bool(xlog), "median": back(_quantile(a, w, 0.5)) if len(a) else np.nan,
            "p16": back(_quantile(a, w, 0.16)) if len(a) else np.nan, "p84": back(_quantile(a, w, 0.84)) if len(a) else np.nan}


def curves_of(d, label, show="pdf", ylog=False, color=None, dash="-"):
    """the curves and styles of one distribution in a curve tile: the PDF against the bin centres, the CDF against the right edges (on a twin axis
    when both are shown). x in the quantity's units (10^x for a log axis: the renderers draw a log x axis)."""
    back = (lambda x: 10.0 ** x) if d["xlog"] else (lambda x: x)
    out, st = [], []
    if show in ("pdf", "both"):
        out.append((label, back(d["centres"]), d["pdf"]))
        st.append({"style": "line", "color": color, "dash": dash, "lw": 1.4, "legend": True})
    if show in ("cdf", "both"):
        out.append((label + (" (cumulative)" if show == "both" else ""), back(d["edges"][1:]), d["cdf"]))
        st.append({"style": "line", "color": color, "dash": ":" if show == "both" else dash, "lw": 1.2, "legend": True, "twin": show == "both"})
    return out, st


# ----------------------------------------------------------------------------- the snapshot product `hist`

DEFAULT_QUANTITIES = ["density", "T"]


def _center(spec, s):
    if spec in (None, "", "auto"):
        return "bh" if ("bh_pos" in s and len(s["bh_pos"])) else "mass-weighted"
    return spec


def hist(path, params):
    from . import phase
    from .frames import make_frame
    from .io import load_snapshot
    from .maps import View
    qs = params.get("quantities") or DEFAULT_QUANTITIES
    qs = [qs] if isinstance(qs, str) else list(qs)
    weight = params.get("weight") or "mass"
    probe = load_snapshot(path, gas_fields=("Masses",))
    view = View(center=_center(params.get("center"), probe), tilt=params.get("tilt") or "none", units=params.get("units") or "code",
                **({"gamma": float(params["gamma"])} if params.get("gamma") else {}))
    base = {"Coordinates", "Masses", "Density", "Velocities"}
    opt = set()
    for q in qs:                                        # a quantity whose datasets this snapshot lacks is left out (its columns are missing)
        try:
            opt |= {f.lstrip("?") for f in phase.needs(view, q)}
        except (KeyError, ValueError):
            pass
    s = load_snapshot(path, gas_fields=tuple(sorted(base | {"?" + f for f in opt - base})))
    frame = make_frame(s, view.center, view.align, view.tilt)
    pos = frame.pos(s["Coordinates"])
    rad = np.hypot(pos[:, 0], pos[:, 1]) if params.get("radius") == "R" else np.sqrt((pos ** 2).sum(1))
    sel = rad >= float(params.get("r_min") or 0.0)
    if params.get("r_max") is not None:
        sel &= rad < float(params["r_max"])
    idx = np.nonzero(sel)[0]
    w = phase._weights(s, idx, weight)
    out = {"wtotal": float(w.sum())}
    ranges, logs = params.get("ranges") or {}, params.get("log") or {}
    per_dex, nb = float(params.get("per_dex") or 10), int(params.get("bins") or 64)
    for q in qs:
        try:
            v = phase.values(s, view, q, frame, pos)[0][idx]
        except (KeyError, ValueError):
            continue
        lg = bool(logs[q]) if q in logs else bool(len(v) and np.all(v[np.isfinite(v)] > 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.log10(v) if lg else v
        a = a[np.isfinite(a)]
        if q in ranges:
            lo, hi = float(ranges[q][0]), float(ranges[q][1])
        else:
            lo, hi = auto_range(a)
        if lg:                                          # a fixed grid of 1 / per_dex dex: the same rows in every snapshot
            k0, k1 = int(np.floor(lo * per_dex + 1e-9)), int(np.ceil(hi * per_dex - 1e-9))
            edges = np.arange(k0, max(k1, k0 + 1) + 1) / per_dex
        else:
            edges = np.linspace(lo, hi, nb + 1)
        d = distribution(v, w, xlog=lg, edges=edges)
        for c, pv in zip(d["centres"], d["pdf"]):
            out[f"{q}_{c:.4g}"] = pv
        for e, cv in zip(d["edges"][1:], d["cdf"]):
            out[f"{q}cdf_{e:.4g}"] = cv
        out[f"{q}_med"], out[f"{q}_p16"], out[f"{q}_p84"] = d["median"], d["p16"], d["p84"]
        out[f"{q}_frac"] = d["in_range"] / d["total"] if d["total"] > 0 else np.nan
    return {"scalars": out}


plugins.register_product("hist", hist, needs=("Coordinates", "Masses", "Density", "Velocities", "(the quantities' datasets)"), version=1,
                         label="hist: distributions (PDF, CDF) of gas quantities in a region", applies=None,
                         params={"quantities": DEFAULT_QUANTITIES, "weight": "mass", "per_dex": 10, "bins": 64, "ranges": {}, "log": {}, "center": "auto",
                                 "tilt": "none", "r_min": 0.0, "r_max": None, "radius": "r", "units": "code", "gamma": None},
                         defaults=["density_med"],
                         meta={"wtotal": {"label": "total weight in the region", "group": "hist"},
                               **{f"{q}_{s}": {"label": f"{q}: {t}", "group": "hist"} for q in DEFAULT_QUANTITIES
                                  for s, t in (("med", "weighted median"), ("p16", "16th percentile"), ("p84", "84th percentile"), ("frac", "fraction in range"))},
                               **{f"{q}_": {"label": f"d(weight)/d log {q} / total", "group": "distributions"} for q in DEFAULT_QUANTITIES},
                               **{f"{q}cdf_": {"label": f"fraction with {q} below", "group": "distributions"} for q in DEFAULT_QUANTITIES}})
