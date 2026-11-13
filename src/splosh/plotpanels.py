"""The layout panel types `hist`, `pk` and `dmd` (layout.py registers their keys and checks, layout_preview.py computes their tiles; both renderers
draw them as curves, or as a heatmap for a DMD mode). See docs/PLOTS.md.

    hist   the distribution (PDF and / or CDF) of quantities of the gas in the panel's view window (hist.py), one curve per quantity and per snapshot
    pk     the isotropic power spectrum of the panel's map (pk.py), one curve per snapshot, optional reference slopes
    dmd    a saved DMD result of the run (dmd.py: `splosh dmd RUN`): its spectrum, eigenvalues, singular values or one mode as a map

`snaps` (hist, pk): None = the current snapshot (a dashboard), or with scope = run four snapshots spread over the run; an integer N = N snapshots
spread over the run; a list = these snapshot indices (negative from the end)."""
import os

import numpy as np

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf")
NAMES = ("hist", "pk", "dmd")


def types(view):
    """{type: {key: default}} without the common keys (layout.panel_types adds them)."""
    return {"hist": {"type": "hist", "quantities": None, "hweight": "mass", "bins": 128, "xr": None, "xlog": True, "ylog": False, "show": "pdf",
                     "snaps": None, **view},
            "pk": {"type": "pk", "window": "hann", "subtract": "delta", "nk": None, "slopes": [], "compensate": 0.0, "snaps": None, **view},
            "dmd": {"type": "dmd", "name": "dmd", "show": "spectrum", "mode": 0, "part": "real", "fmax": None, "cmap": "RdBu_r"}}


def _snaps_ok(v):
    return v is None or (isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 32) or \
        (isinstance(v, (list, tuple)) and 1 <= len(v) <= 32 and all(isinstance(i, int) and not isinstance(i, bool) for i in v))


def validate(t, p, raw, where, err):
    """check and complete panel `p` of type t (one of NAMES); err(message) raises the layout's error."""
    from .maps import View
    if t in ("hist", "pk"):
        from . import layout as lay
        try:
            View(**{k: p[k] for k in lay._view_keys()}).validate()
        except ValueError as e:
            err(f"{where}: {e}")
        if not _snaps_ok(p["snaps"]):
            err(f"{where}: snaps = N (1-32) or a list of snapshot indices")
    if t == "hist":
        q = p["quantities"]
        q = [q] if isinstance(q, str) else list(q or [])
        if not q or not all(isinstance(x, str) and x for x in q):
            err(f"{where}: a hist panel needs `quantities` (phase-diagram quantities: density, T, u, R, ...)")
        p["quantities"] = q
        if not (isinstance(p["bins"], int) and 4 <= p["bins"] <= 2048) or p["hweight"] not in ("mass", "count", "volume") or p["show"] not in ("pdf", "cdf", "both"):
            err(f"{where}: bins 4-2048, hweight mass|count|volume, show pdf|cdf|both")
        if p["xr"] is not None and not (isinstance(p["xr"], (list, tuple)) and len(p["xr"]) == 2):
            err(f"{where}: xr = [lo, hi] (log10 when xlog)")
    elif t == "pk":
        from .pk import SUBTRACT, WINDOWS
        if p["window"] not in WINDOWS or p["subtract"] not in SUBTRACT:
            err(f"{where}: window {'|'.join(WINDOWS)}, subtract {'|'.join(SUBTRACT)}")
        if p["nk"] is not None and not (isinstance(p["nk"], int) and 4 <= p["nk"] <= 200):
            err(f"{where}: nk (bins in k) an integer 4-200, or unset")
        try:
            p["slopes"] = [float(s) for s in ([p["slopes"]] if isinstance(p["slopes"], (int, float)) else (p["slopes"] or []))]
            p["compensate"] = float(p["compensate"] or 0.0)
        except (TypeError, ValueError):
            err(f"{where}: slopes a list of numbers (e.g. [-1.6667, -3.6667]), compensate a number")
    elif t == "dmd":
        if not p["name"] or not isinstance(p["name"], str) or "/" in p["name"]:
            err(f"{where}: name = the result's name (`splosh dmd RUN --name NAME`, default dmd)")
        if p["show"] not in ("spectrum", "eigs", "sv", "mode") or p["part"] not in ("real", "imag", "abs", "phase"):
            err(f"{where}: show spectrum|eigs|sv|mode, part real|imag|abs|phase")
        if not (isinstance(p["mode"], int) and p["mode"] >= 0):
            err(f"{where}: mode = an index >= 0 (0: the strongest mode with f >= 0)")
    return p


def snapshot_list(p, scope, path):
    """[(snapshot path, label)] a hist / pk panel draws (see the module docstring)."""
    from . import series as SR
    from .runs import snapshot_times
    sp = p.get("snaps")
    if sp is None and scope != "run":
        return [(path, None)]
    ctx = SR.context_for_snapshot(path)
    snaps = ctx["snapshots"] or [path]
    n = len(snaps)
    if sp is None or isinstance(sp, int):
        k = min(4 if sp is None else sp, n)
        idx = sorted(set(int(round(x)) for x in np.linspace(0, n - 1, k)))
    else:
        idx = [i if i >= 0 else n + i for i in sp]
        idx = [i for i in idx if 0 <= i < n]
    P = ctx.get("period")
    out = []
    for i in idx:
        t = snapshot_times(snaps[i])[0]
        out.append((snaps[i], f"t = {t / P:.3g} P0" if P else f"t = {t:.4g}"))
    return out


def _loader(path, cur_path, load):
    if os.path.abspath(path) == os.path.abspath(cur_path):
        return load
    from .io import load_snapshot
    return lambda f: load_snapshot(path, gas_fields=tuple(sorted(f)))


def _view(p):
    from . import layout as lay
    from .maps import View
    return View(**{k: p[k] for k in lay._view_keys()})


def tile(p, lo, path, load, store):
    """the tile of a hist / pk / dmd panel for snapshot `path` (load(fields): that snapshot loaded; store: the map store)."""
    t = p["type"]
    if t == "hist":
        return hist_tile(p, lo, path, load)
    if t == "pk":
        return pk_tile(p, lo, path, load, store)
    return dmd_tile(p, path)


def hist_tile(p, lo, path, load):
    from . import hist as H, phase as PH
    view = _view(p)
    snaps = snapshot_list(p, lo["scope"], path)
    qs = p["quantities"]
    curves, styles, meta0 = [], [], None
    for ks, (sp, slab) in enumerate(snaps):
        ld = _loader(sp, path, load)
        snap = ld(tuple(sorted(PH.needs(view, *qs))))
        for kq, q in enumerate(qs):
            v, w, meta = H.values_in_view(snap, view, q, p["hweight"])
            meta0 = meta0 or meta
            d = H.distribution(v, w, p["bins"], p["xlog"], p["xr"])
            lab = (meta.get("label") or q) if len(qs) > 1 or not slab else ""
            lab = ", ".join(x for x in (lab, slab) if x) or (meta.get("label") or q)
            col = COLORS[(kq if len(qs) > 1 else ks) % len(COLORS)]
            dash = ("-", "--", ":", "-.")[ks % 4] if (len(qs) > 1 and len(snaps) > 1) else "-"
            c, s = H.curves_of(d, lab, p["show"], color=col, dash=dash)
            curves += c
            styles += s
    unit = (meta0 or {}).get("unit")
    xl = (qs[0] if len(qs) == 1 else "value") + (f" [{unit}]" if unit and len(qs) == 1 else "")
    per = "per dex" if p["xlog"] else "per unit"
    w = {"mass": "mass", "count": "particle", "volume": "volume"}[p["hweight"]]
    ylab = {"pdf": f"{w} fraction {per}", "cdf": f"cumulative {w} fraction", "both": f"{w} fraction {per}"}[p["show"]]
    return {"curves": curves, "styles": styles, "xlog": bool(p["xlog"]), "ylog": bool(p["ylog"]) and p["show"] != "cdf", "xlabel": xl, "ylabel": ylab,
            "yscale": "log" if (p["ylog"] and p["show"] != "cdf") else "lin", "twin_label": f"cumulative {w} fraction" if p["show"] == "both" else None,
            "twin_scale": "lin" if p["show"] == "both" else None, "xnow": None}


def pk_tile(p, lo, path, load, store):
    from . import pk as PK
    from .maps import required_gas_fields
    view = _view(p)
    snaps = snapshot_list(p, lo["scope"], path)
    curves, styles, unit = [], [], "code length"
    first = None
    for ks, (sp, slab) in enumerate(snaps):
        ld = _loader(sp, path, load)
        img, meta, _ = store.get_or_render(sp, view, snap=ld(tuple(sorted(required_gas_fields(view)))))
        lu = meta.get("length_unit") or {}
        f = lu.get("factor") or 1.0
        unit = lu.get("unit") or unit
        r = PK.power_spectrum(img, view.width * f, p["window"], p["subtract"], p["nk"])
        y = r["P"] * r["k"] ** p["compensate"] if p["compensate"] else r["P"]
        curves.append((slab or f"{view.field} ({p['subtract']})", r["k"], y))
        styles.append({"style": "line", "color": COLORS[ks % len(COLORS)], "lw": 1.4, "legend": True})
        first = first or (r["k"], y)
    for lab, k, y in PK.reference_lines(first[0], first[1], p["slopes"], p["compensate"]) if first else []:
        curves.append((lab, k, y))
        styles.append({"style": "line", "color": "#808080", "dash": "--", "lw": 0.9, "legend": True})
    comp = f"k^{PK.power_text(p['compensate'])} " if p["compensate"] else ""
    return {"curves": curves, "styles": styles, "xlog": True, "ylog": True, "yscale": "log", "xlabel": f"k = 2 pi / wavelength [1/{unit}]",
            "ylabel": comp + "P(k)", "xnow": None}


def dmd_tile(p, path):
    from . import dmd as D, series as SR
    ctx = SR.context_for_snapshot(path)
    f = os.path.join(D.result_dir(ctx["run_dir"]), p["name"] + ".npz")
    if not os.path.isfile(f):
        return {"blank": f"no DMD result {p['name']!r}: run `splosh dmd RUN --name {p['name']}`"}
    r = D.load(f)
    u = "f_orb" if r["unit"] == "P0" else "1/t"
    fr, amp, g = np.asarray(r["freq"]), np.asarray(r["amp"]), np.asarray(r["growth"])
    if p["show"] == "spectrum":
        sel = fr >= -1e-12
        if p.get("fmax") is not None:
            sel &= fr <= float(p["fmax"])
        o = np.argsort(fr[sel])
        return {"curves": [("amplitude", fr[sel][o], amp[sel][o])], "styles": [{"style": "linepoints", "color": COLORS[0], "lw": 0.6, "legend": True}],
                "xlog": False, "ylog": True, "yscale": "log", "xlabel": f"f [{u}]", "ylabel": "amplitude", "xnow": None}
    if p["show"] == "eigs":
        th = np.linspace(0, 2 * np.pi, 361)
        lam = np.asarray(r["lam"])
        return {"curves": [("unit circle", np.cos(th), np.sin(th)), ("eigenvalues", lam.real, lam.imag)],
                "styles": [{"style": "line", "color": "#808080", "lw": 0.8, "legend": True}, {"style": "events", "color": COLORS[3], "legend": True}],
                "xlog": False, "ylog": False, "xlabel": "Re lambda", "ylabel": "Im lambda", "xnow": None, "aspect": "equal"}
    if p["show"] == "sv":
        sv = np.asarray(r["sv"])
        return {"curves": [("singular values / s_0", np.arange(len(sv), dtype=float), sv / sv[0])],
                "styles": [{"style": "linepoints", "color": COLORS[0], "legend": True}], "xlog": False, "ylog": True, "yscale": "log", "xlabel": "index",
                "ylabel": "s / s_0", "vlines": [float(r["rank"]) - 0.5], "xnow": None}
    lead = D.leading(r, p["mode"] + 1)
    if p["mode"] >= len(lead):
        return {"blank": f"mode {p['mode']}: the result has {len(lead)} modes with f >= 0"}
    i = lead[p["mode"]]
    m = np.asarray(r["modes"][:, :, i])
    z = {"real": m.real, "imag": m.imag, "abs": np.abs(m), "phase": np.degrees(np.angle(m))}[p["part"]].astype(float)
    e = np.asarray(r["extent"], float)
    n = z.shape[0]
    v = float(np.nanmax(np.abs(z))) or 1.0
    zl = (-180.0, 180.0) if p["part"] == "phase" else ((0.0, v) if p["part"] == "abs" else (-v, v))
    return {"heat": {"x": np.linspace(e[0], e[1], n + 1), "y": e[2] + (np.arange(n) + 0.5) * (e[3] - e[2]) / n, "z": z, "zscale": "lin", "zlim": zl,
                     "cmap": p.get("cmap") or "RdBu_r", "norm": "none", "aspect": "equal",
                     "label": f"mode {i} ({p['part']})"},
            "curves": [], "styles": [], "xnow": None, "xlabel": f"x   [f = {fr[i]:.3g} {u}, growth {g[i]:+.3g} / {'P0' if u == 'f_orb' else 't'}]", "ylabel": "y",
            "yscale": "lin"}
