"""Maps of splosh on matplotlib Axes: the connector between splosh (what is shown) and a figure script or a layout (how it looks).

The script owns the figure, the axes, the style (e.g. a `matplotlib.style` or a plotting package applied before), sizes and labels' fonts; splosh
draws into the Axes it is given, with the same maps, colour tables and stretches as the viewer:

    import matplotlib.pyplot as plt
    from splosh import View
    from splosh import mpl as avm
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    a = avm.draw_map(axes[0], "run/output/snapshot_100.hdf5", View(field="rho", width=8), cmap="inferno", overlays=["contours:n=6", "particles:bh"])
    b = avm.draw_map(axes[1], "run/output/snapshot_200.hdf5", View(field="rho", width=8), cmap="inferno", vmin=a.limits[0], vmax=a.limits[1])
    avm.add_colorbar(b, ax=axes, label="column density")       # one bar for both panels (same limits)
    avm.add_time(a); avm.add_time(b); avm.add_scalebar(a)
    fig.savefig("pair.pdf")

    draw_map(ax, snapshot, view=None, *, cmap, scale, vmin, vmax, pct, overlays, store, use_cache, axis_labels, units_label) -> MapPanel
    add_colorbar(panel, ax=None, cax=None, label=None, **kw) -> matplotlib Colorbar (ticks in data values, as the viewer's)
    add_scalebar(panel, length=None, loc="lower left", color="w")      add_time(panel, loc="upper left", color="w", period=None)
    draw_overlays(panel, specs)  (draw_map does it for `overlays=`): contours, vectors, streamlines, particles, circles as vector graphics

Colours: the image is drawn as the stretched values in [0, 1] (imaging.stretch) with the viewer's colour table, so the PDF shows the same colours as
the viewer; the colour bar's ticks are placed at the stretched positions of round data values. NaN pixels (no data) get the viewer's background.
Coordinates: the axes are in the frame's window coordinates (code length unless the view's units say otherwise), x horizontal, y vertical.
matplotlib is needed only by this module (splosh itself does not depend on it).
"""
from dataclasses import dataclass, field

import numpy as np

from . import colormaps, imaging, limits as limits_mod, overlays as overlays_mod
from .figures import NAN_COLOR, _get_map, field_unit, fmt_tick, make_ticks, resolve_style, time_text, unit_names
from .maps import View

PX_TO_PT = 0.75      # overlay line widths are given in screen pixels (96 dpi); matplotlib wants points (72 per inch)


@dataclass
class MapPanel:
    ax: object
    img: np.ndarray
    meta: dict
    limits: tuple
    cmap: str
    scale: object
    image: object = None                              # the AxesImage
    names: dict = field(default_factory=dict)          # unit names (L, M, V) for labels
    notes: list = field(default_factory=list)          # overlay notes (e.g. "no level crosses the map")
    snapshot: str = ""
    view: object = None

    @property
    def extent(self):
        return self.meta["extent"]

    @property
    def unit(self):
        q = self.meta.get("quantity_unit") or {}
        u = q.get("unit") if isinstance(q, dict) else None
        return u if u and u != "1" else field_unit(self.meta, self.names)


def colormap(name, nodata="dark"):
    """the viewer's colour table as a matplotlib colormap (NAME_r reversed); NaN in the viewer's background colour (nodata = white: white)."""
    from matplotlib.colors import ListedColormap
    bad = (1.0, 1.0, 1.0) if nodata == "white" else tuple(np.array(NAN_COLOR) / 255.0)
    return ListedColormap(colormaps.lut(name) / 255.0, name=name).with_extremes(bad=bad)


def _native_norm(scale, lo, hi):
    """matplotlib's own norm for a stretch: LogNorm / Normalize / AsinhNorm (so that colour bars get matplotlib's ticks, e.g. 10^n)."""
    from matplotlib import colors as mc
    s = str(scale)
    if s.startswith("log") and lo > 0:
        return mc.LogNorm(vmin=lo, vmax=hi)
    if s.startswith("asinh"):
        f = float(s.split(":")[1]) if ":" in s else 0.1
        return mc.AsinhNorm(linear_width=max(1e-300, f * 0.5 * (abs(lo) + abs(hi))), vmin=lo, vmax=hi)
    return mc.Normalize(vmin=lo, vmax=hi)


def draw_map(ax, snapshot, view=None, *, cmap=None, scale=None, vmin=None, vmax=None, pct=None, overlays=(), store=None, use_cache=True,
             axis_labels=True, interpolation="nearest"):
    """Render (or take from the map store) `view` of `snapshot` and draw it on `ax`. Limits: vmin and vmax both given -> fixed; otherwise the percentiles
    `pct` (default 1, 99.9) of this map, a given bound replacing its side (as `splosh shot`). Returns a MapPanel (its `limits` can be passed to
    other panels so that they share a colour bar)."""
    view = (view or View()).validate()
    cmap, scale, sym = resolve_style(view, cmap, scale)
    img, meta, _ = _get_map(snapshot, view, store, use_cache)
    policy = "fixed" if (vmin is not None and vmax is not None) else "frame"
    lo, hi = limits_mod.resolve(policy, scale, pct, vmin, vmax, img=img, symmetric=sym)
    panel = draw_image(ax, img, meta, cmap, scale, (lo, hi), axis_labels=axis_labels, interpolation=interpolation)
    panel.snapshot, panel.view = str(snapshot), view
    specs = [s for s in (overlays or ()) if s]
    if specs:
        draw_overlays(panel, specs)
    return panel


def draw_image(ax, img, meta, cmap, scale, limits, axis_labels=True, interpolation="nearest", native=False, nodata="dark"):
    """draw an already rendered map (img, meta as render_map / the map store give them) with the viewer's colours: what draw_map and the layout
    renderer (layout_mpl.py) share. Returns a MapPanel."""
    lo, hi = limits
    scale = imaging.fit_scale(img, lo, hi, scale)
    names = unit_names(meta.get("units"))
    if native:                                   # the data with matplotlib's norm: its colour bar ticks (the colours are the same stretch)
        im = ax.imshow(np.asarray(img, dtype=np.float64), origin="lower", extent=meta["extent"], cmap=colormap(cmap, nodata),
                       norm=_native_norm(scale, lo, hi), interpolation=interpolation)
    else:
        t = imaging.stretch(img, lo, hi, scale)
        im = ax.imshow(t, origin="lower", extent=meta["extent"], cmap=colormap(cmap, nodata), vmin=0.0, vmax=1.0, interpolation=interpolation)
    ax.set_xlim(meta["extent"][0], meta["extent"][1])
    ax.set_ylim(meta["extent"][2], meta["extent"][3])
    if axis_labels:
        ax.set_xlabel(f"{meta.get('xlabel', 'x')} [{names['L']}]")
        ax.set_ylabel(f"{meta.get('ylabel', 'y')} [{names['L']}]")
    pan = MapPanel(ax, img, meta, (lo, hi), cmap, scale, im, names, [])
    pan.native = native
    return pan


def add_colorbar(panel, ax=None, cax=None, label=None, **kw):
    """a colour bar for `panel` (or for several panels drawn with the same limits: pass them all as `ax`), ticks at round data values."""
    fig = panel.ax.figure
    cb = fig.colorbar(panel.image, ax=ax if ax is not None else panel.ax, cax=cax, **kw)
    if getattr(panel, "native", False):          # matplotlib's own ticks (10^n on a log bar)
        if label is None:
            u = panel.unit
            label = panel.meta.get("label", "") + (f" [{'code units' if u and 'code_' in u else u}]" if u else "")
        if label:
            cb.set_label(label)
        return cb
    lo, hi = panel.limits
    ticks = make_ticks(lo, hi, panel.scale)
    pos = [float(imaging.stretch(np.array([v]), lo, hi, panel.scale)[0]) for v in ticks]
    keep = [(p, v) for p, v in zip(pos, ticks) if 0.0 <= p <= 1.0]
    cb.set_ticks([p for p, _ in keep])
    cb.set_ticklabels([fmt_tick(v) for _, v in keep])
    cb.minorticks_off()
    if label is None:
        u = panel.unit
        u = "code units" if u and "code_" in u else u
        label = panel.meta.get("label", "") + (f" [{u}]" if u else "")
    cb.set_label(label)
    return cb


def add_scalebar(panel, length=None, loc="lower left", color="w", lw=2.0, fontsize=None):
    """a bar of a round length (about a fifth of the window) with its label."""
    ax, (x0, x1, y0, y1) = panel.ax, panel.extent
    width = x1 - x0
    if length is None:
        mag = 10.0 ** np.floor(np.log10(0.2 * width))
        length = max(m * mag for m in (1, 2, 5, 10) if m * mag <= 0.2 * width * (1 + 1e-9))
    left = "left" in loc
    xs = x0 + 0.05 * width if left else x1 - 0.05 * width - length
    ys = y0 + 0.07 * (y1 - y0) if "lower" in loc else y1 - 0.09 * (y1 - y0)
    ax.plot([xs, xs + length], [ys, ys], color=color, lw=lw, solid_capstyle="butt")
    ax.text(xs + 0.5 * length, ys + 0.015 * width, f"{length:.4g} {panel.names.get('L', '')}", color=color, ha="center", va="bottom", fontsize=fontsize)
    return length


def add_time(panel, loc="upper left", color="w", period=None, fontsize=None, text=None):
    """the snapshot time (in physical units when the header has them; `period`: also in orbits) in a corner of the panel."""
    t = text if text is not None else time_text(panel.meta["time"], panel.names, period=period)
    x = 0.03 if "left" in loc else 0.97
    y = 0.96 if "upper" in loc else 0.04
    return panel.ax.text(x, y, t, transform=panel.ax.transAxes, color=color, ha="left" if "left" in loc else "right",
                         va="top" if "upper" in loc else "bottom", fontsize=fontsize)


def _rgb(c, alpha=1.0):
    r, g, b = overlays_mod.color_rgb(c)
    return (r / 255.0, g / 255.0, b / 255.0, float(alpha))


def _halo(lw_pt, halo_px):
    if not halo_px:
        return None
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=lw_pt + 2 * halo_px * PX_TO_PT, foreground=(0, 0, 0, 0.6)), pe.Normal()]


def draw_items(ax, items, extent):
    """draw overlay items (overlays.py's plain data) as matplotlib artists; returns the notes of the legend items (for a caption)."""
    legends, arrows = [], []
    for it in items:
        k = it.get("kind")
        col = _rgb(it.get("color", "white"), it.get("alpha", 1.0))
        lw = float(it.get("linewidth", 1.0)) * PX_TO_PT
        if k == "arrow" and (it.get("dash") or it.get("thin")):   # a dashed or thin arrow: its own line and head (the quiver draws solid, wide arrows)
            x0, y0, x1, y1 = it["x"], it["y"], it["x"] + it["dx"], it["y"] + it["dy"]
            ls = (0, [float(v) * PX_TO_PT / max(lw, 0.1) for v in it["dash"]]) if it.get("dash") else "-"
            ax.plot([x0, x1], [y0, y1], color=col, lw=lw, linestyle=ls, zorder=4)
            ax.annotate("", xy=(x1, y1), xytext=(x0 + 0.9 * (x1 - x0), y0 + 0.9 * (y1 - y0)),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=8 + 2 * lw, shrinkA=0, shrinkB=0), zorder=4)
        elif k == "arrow":
            arrows.append(it)
        elif k == "polyline":
            pts = np.asarray(it["points"], dtype=float)
            if len(pts) < 2:
                continue
            ls = (0, [float(v) * PX_TO_PT / max(lw, 0.1) for v in it["dash"]]) if it.get("dash") else "-"
            (ln,) = ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, solid_capstyle="round", linestyle=ls)
            pe = _halo(lw, it.get("halo"))
            if pe:
                ln.set_path_effects(pe)
            h = it.get("head")
            if h is not None and 0 < int(h) < len(pts):
                i = int(h)
                ax.annotate("", xy=tuple(pts[i]), xytext=tuple(pts[i - 1]), arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=col, lw=lw, shrinkA=0, shrinkB=0))
        elif k == "points":
            size = float(it.get("size_px", 4)) * PX_TO_PT
            if it.get("style") == "ring":
                ax.scatter(it["x"], it["y"], s=(2 * size) ** 2, facecolors="none", edgecolors=[col], linewidths=max(0.6, lw))
            else:
                ax.scatter(it["x"], it["y"], s=size ** 2, c=[col], linewidths=0)
        elif k == "circle":
            from matplotlib.patches import Circle
            ax.add_patch(Circle((it["x"], it["y"]), it["r"], fill=bool(it.get("fill")), facecolor=col if it.get("fill") else "none", edgecolor=col, lw=lw))
        elif k == "legend":
            legends.append(it.get("text", ""))
        elif k == "image":         # a grey texture with alpha (the LIC overlay), over the map and under the lines
            g, a = overlays_mod.image_la(it)
            ax.imshow(np.dstack([g, g, g, a]), extent=it["extent"], origin="upper", interpolation="nearest", zorder=1.5)
        elif k == "polygon":
            from matplotlib.patches import Polygon
            ax.add_patch(Polygon(np.asarray(it["points"], dtype=float), closed=True, facecolor=col, edgecolor="none", zorder=2))
        elif k == "text":
            ax.annotate(str(it.get("text", "")), (it["x"], it["y"]), xytext=(float(it.get("dx_px", 0)) * PX_TO_PT, float(it.get("dy_px", 0)) * PX_TO_PT),
                        textcoords="offset points", color=col, fontsize=float(it.get("size_px", 11.0)) * PX_TO_PT, ha="left", va="bottom", zorder=5)
        elif k is None and "x" in it and "y" in it:
            ax.scatter([it["x"]], [it["y"]], s=60, facecolors="none", edgecolors="w", linewidths=1.0)
    if arrows:                     # one quiver per colour: arrow lengths in data units, heads in proportion to the shaft
        by = {}
        for a in arrows:
            by.setdefault((tuple(_rgb(a.get("color", "white"), a.get("alpha", 1.0))), float(a.get("linewidth", 1.0)), float(a.get("halo", 0) or 0)), []).append(a)
        for (col, lwpx, halo), group in by.items():
            X = np.array([[a["x"], a["y"], a["dx"], a["dy"]] for a in group], dtype=float)
            q = ax.quiver(X[:, 0], X[:, 1], X[:, 2], X[:, 3], angles="xy", scale_units="xy", scale=1.0, color=[col], width=0.0025 * max(1.0, lwpx),
                          headwidth=3.5, headlength=4.0, headaxislength=3.5, pivot="tail", zorder=3)
            if halo:
                import matplotlib.patheffects as pe
                q.set_path_effects([pe.Stroke(linewidth=halo * PX_TO_PT, foreground=(0, 0, 0, 0.6)), pe.Normal()])
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return legends


def draw_overlays(panel, specs):
    """compute the overlays of `specs` (see overlays.py: contours, vectors:v|B, streamlines:v|B, particles:bh,star,...) for the panel's snapshot and view
    and draw them; their notes and legend texts go to panel.notes."""
    res = overlays_mod.compute_for_path(panel.snapshot, panel.view, list(specs))
    panel.notes += list(res.get("notes", [])) + draw_items(panel.ax, res["items"], res["extent"])
    return res
