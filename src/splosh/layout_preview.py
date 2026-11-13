"""One picture of a layout for one snapshot, for the layout editor (`splosh serve`, page /layout): the same drawing code as an analyze dashboard (`layout.render_layout`), fed directly instead of from the
analysis output.

    preview(layout, path, load, store, products=None, table=None, time=None, cache=None) -> (rgb, notes)

- map panels with inline settings are rendered from the snapshot (through the map store, so a second preview is instant); a panel with `product` is resolved from the analyze config's
  products (`products`, from `analyze.normalize_config`), else drawn blank with a note;
- limits: `fixed` as given; `global` cannot be known from one snapshot, so the preview uses the frame's percentiles (noted); `frame` / `viewport` as in a pipeline run;
- profile panels need their `[[profile]]` product from the config (computed on this snapshot);
- time-series panels with `stack` (parts of a total) are turned into stacked zones by `layout.stack_tile`, for both renderers;
- time-series panels use `table` (`read_scalars_table` of the run's analysis: column -> array over the snapshots, and the snapshot names) with the marker at the row of this
  snapshot (by file name; no marker, and a note, when the analysis does not have it); without a table they are blank with a note;
- `notes` lists, per panel id, anything the editor should tell the user (a blank panel, an approximation);
- `cache`: a dict the caller keeps between previews (profiles and the `auto` radius of a snapshot, keyed by file signature and settings; bounded)."""
import copy
import os
import re

import numpy as np

from . import layout as lay


def read_scalars_table(path):
    """{column: float array} (plus `index`, and `_snapshots`: the snapshot names of the rows) from an analysis directory's scalars.csv, or None."""
    f = os.path.join(path, "scalars.csv") if os.path.isdir(path) else path
    if not os.path.isfile(f):
        return None
    with open(f) as fh:
        head = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    if not rows:
        return None
    out = {"_snapshots": [r[0] for r in rows] if head[:1] == ["snapshot"] else []}
    for k, c in enumerate(head):
        if c == "snapshot":
            continue
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, IndexError):
                vals.append(np.nan)
        out[c] = np.array(vals)
    out["index"] = np.arange(len(rows), dtype=float)
    return out


def map_profile(img, meta, nbins=48, band=True, ylog=True):
    """azimuthal statistics of a map in log-spaced annuli about the frame's origin (the centre rule's point): mean, 16th and 84th percentiles; what the
    viewer's radial-profile pane draws (viewer.html profileOf), as the curves of a profile tile. R is in the map's length unit."""
    n = img.shape[0]
    e = meta["extent"]
    d = (e[1] - e[0]) / n
    xs = e[0] + (np.arange(n) + 0.5) * d
    ys = e[2] + (np.arange(n) + 0.5) * d
    R = np.hypot(*np.meshgrid(xs, ys))
    v = np.asarray(img, dtype=float)
    ok = np.isfinite(v)
    near = float(np.hypot(max(e[0], 0, -e[1]), max(e[2], 0, -e[3])))
    far = float(max(np.hypot(a, b) for a in (e[0], e[1]) for b in (e[2], e[3])))
    r0 = max(d, near)
    lu = (meta.get("length_unit") or {})
    f = lu.get("factor") or 1.0
    unit = lu.get("unit") or "code length"
    if not far > r0:
        return {"curves": [], "xlog": True, "ylog": ylog, "xlabel": f"R [{unit}]"}
    edges = np.geomspace(r0, far, nbins + 1)
    k = np.digitize(R[ok], edges) - 1
    vals = v[ok]
    rc, mean, lo, hi = [], [], [], []
    for b in range(nbins):
        s = vals[k == b]
        if s.size < 3:
            continue
        rc.append(np.sqrt(edges[b] * edges[b + 1]) * f)
        mean.append(float(s.mean())); lo.append(float(np.percentile(s, 16))); hi.append(float(np.percentile(s, 84)))
    field = (meta.get("view") or {}).get("field", "")
    curves = [(f"{field} (azimuthal mean)", rc, mean)]
    styles = [{"color": "#1f77b4", "lw": 1.6}]
    if band:
        curves += [("16th / 84th percentiles", rc, lo), ("", rc, hi)]
        styles += [{"color": "#1f77b4", "alpha": 0.4, "dash": ":", "lw": 1.0}, {"color": "#1f77b4", "alpha": 0.4, "dash": ":", "lw": 1.0, "legend": False}]
    qu = (meta.get("quantity_unit") or {}).get("unit")
    return {"curves": curves, "styles": styles, "xlog": True, "ylog": ylog, "xlabel": f"R [{unit}] about the centre", "ylabel": field + (f" [{qu}]" if qu else "")}


def _view_of(p, auto):
    from .maps import View
    d = {k: p[k] for k in lay._view_keys() if k in p and p[k] != "auto"}
    if p.get("width") == "auto":
        d["width"] = auto()
    return View(**d).validate()


def _limits(p, img):
    """(vmin, vmax), note: the pipeline's rule for one frame (analyze._png_limits); `global` needs every frame of the run, so the preview uses the frame policy and says so."""
    from . import analyze
    if p["limits"] == "fixed":
        return (float(p["vmin"]), float(p["vmax"])), None
    if p["limits"] == "global":
        return analyze._png_limits(dict(p, limits="frame"), img, None), "global limits need the whole run: the preview uses this frame's percentiles"
    return analyze._png_limits(p, img, None), None


_CACHE_MAX = 32


def _cached(cache, key, make):
    if cache is None:
        return make()
    if key in cache:
        cache[key] = cache.pop(key)                      # most recent last
        return cache[key]
    v = cache[key] = make()
    while len(cache) > _CACHE_MAX:
        cache.pop(next(iter(cache)))
    return v


def preview(layout, path, load, store, products=None, table=None, time=None, cache=None):
    """See the module docstring. `load(fields)` returns the loaded snapshot dict with at least those gas fields (the caller caches it)."""
    lo, tiles, notes, header = tiles_for(layout, path, load, store, products, table, time, cache)
    return lay.render_layout(lo, tiles, header + "   [preview]"), notes


def preview_mpl(layout, path, load, store, products=None, table=None, time=None, cache=None):
    """The same layout drawn by the publication renderer (layout_mpl.render): (matplotlib Figure, notes)."""
    from . import layout_mpl
    lo, tiles, notes, header = tiles_for(layout, path, load, store, products, table, time, cache)
    return layout_mpl.render(lo, tiles, header), notes


def tiles_for(layout, path, load, store, products=None, table=None, time=None, cache=None, limits=None):
    """(validated layout, tiles, notes, header) of one snapshot: what both renderers draw. `limits`: {panel id: (vmin, vmax)} that replace a map panel's
    own (a movie's global limits)."""
    import json
    from . import analyze, figures
    from .maps import View, required_gas_fields
    from .mapstore import snapshot_hash
    lo = lay.validate_layout(layout)
    sig = snapshot_hash(path)
    products = {p["name"]: p for p in (products or [])}
    notes, tiles = {}, {}
    def auto():                                          # width / rmax = 'auto' as in a pipeline run, but from this snapshot
        return _cached(cache, ("auto", sig), lambda: float(f"{2.2 * analyze._auto_radius(load(())):.2g}"))

    meta0 = None
    stem = os.path.splitext(os.path.basename(path))[0]
    for p in lo["panel"]:
        pid = p["id"]
        try:
            if p["type"] == "map":
                q = p
                if p["product"]:
                    q = products.get(p["product"])
                    if q is None or q["kind"] != "map":
                        notes[pid] = f"product {p['product']!r}: not a [[map]] of the analyze config (serve with --config)"
                        tiles[pid] = {"blank": "product not found"}
                        continue
                view = _view_of(q, auto)
                snap = load(tuple(sorted(required_gas_fields(view))))
                img, meta, _ = store.get_or_render(path, view, snap=snap)
                meta0 = meta0 or meta
                lim, note = _limits(q, img)
                if limits and pid in limits:
                    lim, note = tuple(limits[pid]), None
                if note:
                    notes[pid] = note
                tiles[pid] = {"img": img, "meta": meta, "cmap": q["cmap"], "scale": q["scale"], "lim": lim, "scalebar": q.get("scalebar", True),
                              "colorbar": q.get("colorbar", True), "label": p["label"]}
                specs = list(q.get("overlays") or [])
                if specs:                                        # overlays as plain items: both renderers draw them (bitmap: rasterised; matplotlib: vector graphics)
                    from . import overlays as ovl
                    gas, types = ovl.required_data(specs, view)
                    osnap = ovl.load_for(path, view, specs) if types else load(tuple(sorted(required_gas_fields(view) | {"?" + g for g in gas})))
                    res = ovl.compute(osnap, view, specs, get_map=lambda v: store.get_or_render(path, v, snap=osnap)[:2])
                    tiles[pid]["overlays"] = res["items"]
                    if res["notes"]:
                        notes[pid] = "; ".join(filter(None, [notes.get(pid)] + res["notes"]))
            elif p["type"] == "profile" and not p["product"]:     # the azimuthal statistics of the panel's own map (the viewer's radial-profile pane)
                view = _view_of(p, auto)
                img, meta, _ = store.get_or_render(path, view, snap=load(tuple(sorted(required_gas_fields(view)))))
                meta0 = meta0 or meta
                tiles[pid] = map_profile(img, meta, p["bins"], band=p.get("band", True), ylog=p.get("ylog", True))
            elif p["type"] == "profile":
                q = products.get(p["product"])
                if q is None or q["kind"] != "profile":
                    notes[pid] = f"product {p['product']!r}: not a [[profile]] of the analyze config (serve with --config)"
                    tiles[pid] = {"blank": "product not found"}
                    continue
                q = copy.deepcopy(q)
                if "auto" in (q["rmin"], q["rmax"]):
                    q["rmax"] = auto() / 2 if q["rmax"] == "auto" else q["rmax"]
                    q["rmin"] = float(q["rmax"]) * 1e-3 if q["rmin"] == "auto" else q["rmin"]
                fields = {"Coordinates", "Masses", "Density"}
                for name in q["quantities"]:
                    if name != "density" and not re.fullmatch(r"(?:A|phi)\d+", name):
                        fields |= required_gas_fields(View(center=q["center"], align=q["align"], field=name))
                prof = _cached(cache, ("profile", sig, json.dumps(q, sort_keys=True)), lambda: analyze._profile(load(tuple(sorted(fields))), q))
                qs = p["quantities"] or [k for k in prof if k not in ("edges", "rbins", "rgeo", "n", "mass", "sigma", "density") and not k.endswith("_std")]
                tiles[pid] = {"curves": [(n, prof["rbins"], prof[n]) for n in qs if n in prof], "xlog": p["xlog"], "ylog": p["ylog"]}
            elif p["type"] == "heatmap":
                from . import series as SR
                from .runs import snapshot_times
                ctx = _cached(cache, ("seriesctx", os.path.dirname(os.path.abspath(path))), lambda: SR.context_for_snapshot(path))
                tnow = None if lo["scope"] == "run" else (time if time is not None else snapshot_times(path)[0])
                tiles[pid] = SR.heat_tile(p, ctx, tnow=tnow)
                if p.get("yscale", "lin") != "lin":                  # the panel's y scale (log-spaced radii), whatever the source passes on
                    tiles[pid]["yscale"] = p["yscale"]
            elif p["type"] == "phase":                          # drawn like a heatmap: x = the bin edges, y = the row centres, z = the weighted counts
                from . import phase as PH
                from .maps import View
                view = View(**{k: p[k] for k in lay._view_keys()})
                snap = load(tuple(sorted(PH.needs(view, p["x"], p["y"]))))
                grid, meta = PH.histogram(snap, view, p["x"], p["y"], p["bins"], p["xlog"], p["ylog"], p["xr"], p["yr"], p["hweight"])
                (x0, x1), (y0, y1) = meta["x"]["range"], meta["y"]["range"]
                nb = int(p["bins"])
                lab = lambda m, q, lg: ("log10 " if lg else "") + (m.get("label") or q) + (f" [{m['unit']}]" if m.get("unit") else "")
                tiles[pid] = {"heat": {"x": np.linspace(x0, x1, nb + 1), "y": y0 + (np.arange(nb) + 0.5) * (y1 - y0) / nb, "z": np.asarray(grid, float),
                                       "zscale": "log", "cmap": p.get("cmap") or "magma", "norm": "none", "label": p["hweight"]},
                              "curves": [], "styles": [], "xnow": None, "xlabel": lab(meta["x"], p["x"], p["xlog"]), "ylabel": lab(meta["y"], p["y"], p["ylog"]),
                              "yscale": "lin"}
            elif p["type"] in ("hist", "pk", "dmd"):           # distributions, map power spectra, DMD results (plotpanels.py)
                from . import plotpanels
                tiles[pid] = plotpanels.tile(p, lo, path, load, store)
                if tiles[pid].get("blank"):
                    notes[pid] = tiles[pid]["blank"]
            elif p["type"] in ("radial", "spectrum"):          # time-averaged radial profiles / power spectra of a series source
                from . import series as SR
                ctx = _cached(cache, ("seriesctx", os.path.dirname(os.path.abspath(path))), lambda: SR.context_for_snapshot(path))
                tiles[pid] = (SR.radial_tile if p["type"] == "radial" else SR.spectrum_tile)(p, ctx)
                if tiles[pid].get("note"):
                    notes[pid] = tiles[pid]["note"]
            elif p["type"] == "timeseries" and p.get("source", "snapshots") != "snapshots":     # a series source (series.py): GIZMO logs, plugin sources
                from . import series as SR
                from .runs import snapshot_times
                ctx = _cached(cache, ("seriesctx", os.path.dirname(os.path.abspath(path))), lambda: SR.context_for_snapshot(path))
                tnow = None if lo["scope"] == "run" else (time if time is not None else snapshot_times(path)[0])
                tiles[pid] = lay.stack_tile(SR.panel_tile(p, ctx, tnow=tnow), p.get("stack"))
                if tiles[pid].get("note"):
                    notes[pid] = tiles[pid]["note"]
            elif p["type"] == "timeseries":
                if table is None:
                    notes[pid] = "this run has no analysis table yet (scalars.csv): make one with \"analyze these panes\" or `splosh analyze RUN`"
                    tiles[pid] = {"blank": "no scalars table"}
                    continue
                stems = table.get("_snapshots", [])
                cur = stems.index(stem) if stem in stems else None
                if lo["scope"] == "run":
                    cur = None                                   # a run-wide figure has no current snapshot
                elif cur is None:
                    notes[pid] = f"{stem} is not in scalars.csv: no marker"
                full = lay.with_derived(lo, {k: v for k, v in table.items() if k != "_snapshots"})
                if p["xaxis"] == "orb":                          # t / P0 of the snapshot table
                    from . import series as SR
                    P0 = _cached(cache, ("seriesctx", os.path.dirname(os.path.abspath(path))), lambda: SR.context_for_snapshot(path)).get("period")
                    full["orb"] = np.asarray(full["time"], float) / P0 if P0 else np.asarray(full["time"], float)
                tiles[pid] = lay.timeseries_data(p, full, cur=cur)
            else:
                tiles[pid] = {"text": p["text"].replace("{time}", f"{time:.4g}" if time is not None else "?").replace("{snapshot}", stem)}
        except Exception as e:                                 # one broken panel must not blank the whole preview
            notes[pid] = f"{type(e).__name__}: {e}"
            tiles[pid] = {"blank": "error (see message)"}
    n = figures._snap_number(path)
    tstr = figures.time_text(meta0["time"], figures.unit_names(meta0.get("units"))) if meta0 else (f"t = {time:.4g}" if time is not None else "")
    header = (f"snap {n}   " if n is not None else "") + tstr
    w0 = next((t["meta"]["view"]["width"] for t in tiles.values() if isinstance(t, dict) and "meta" in t), None)
    tiles["_info"] = {"path": str(path), "snap": n, "time": meta0["time"] if meta0 else time, "width": w0,
                      "L": figures.unit_names(meta0.get("units"))["L"] if meta0 else ""}           # for the publication header (layout_mpl)
    return lo, tiles, notes, header
