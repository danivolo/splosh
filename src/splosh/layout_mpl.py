"""The publication renderer of a layout: the same layout file and the same panel data as the bitmap dashboards (layout.render_layout), drawn with
matplotlib (vector PDF / SVG, or PNG) through the connector splosh.mpl.

    render(layout, tiles, header="") -> matplotlib Figure          (layout: validated; tiles: layout_preview.tiles_for or an analyze dashboard's)
    render_bytes(layout, tiles, header="", fmt="png", dpi=None) -> bytes
    save(layout, tiles, path, header="")                           (the format from the extension: .pdf .svg .png)

The figure's look comes from the layout's [figure] options (layout.FIGURE, fixed lists): size single (3.4 in wide) | wide (7.1 in) | page (7.1 in,
taller rows allowed); colorbar panel | row | column | none (one bar for the maps of a row / column when they share colour map, stretch and
limits, else one each); letters (a), (b), ...; time in every map | the first | none; style default | astroplot (its rcParams, e.g. the pgplot
look, when the package is installed; applied only while this figure is drawn). Maps are square; rows of curves only are shorter. Axis labels are
written on the outer panels only. matplotlib's constrained layout keeps labels and colour bars from overlapping.
"""
import contextlib
import io

import numpy as np

from . import layout as lay
from . import mpl as avm

WIDTH_IN = {"single": 3.4, "wide": 7.1, "page": 7.1}
MAX_HEIGHT_IN = {"single": 6.0, "wide": 6.5, "page": 9.6}


def style_problem(name):
    """why the style `name` cannot be drawn as asked on this machine (astroplot not installed, its font not found), or None."""
    if name in (None, "default"):
        return None
    try:
        import astroplot.style as aps
    except ImportError:
        return f"figure style {name!r} needs the astroplot package, which is not installed in this environment: the default look is used"
    if name in getattr(aps, "STYLE_FONTS", {}):
        try:
            found = aps._find_font_dir(aps.STYLE_FONTS[name][0])
        except Exception:                        # noqa: BLE001
            found = True
        if not found:
            return (f"figure style {name!r}: its font ({aps.STYLE_FONTS[name][0]}) is not in astroplot's font folders (see ASTROPLOT_FONT_DIR): "
                    "the default sans font is used")
    return None


@contextlib.contextmanager
def _style(name):
    import matplotlib
    with matplotlib.rc_context():
        try:
            if name == "astroplot":
                import astroplot
                astroplot.init()                 # its style and its big sizes
            elif name in ("journal", "pgplot", "journal_mono"):
                from astroplot.style import apply_style
                apply_style(name)                # only the style (fonts, ticks), matplotlib's sizes
        except Exception:                        # noqa: BLE001  (not installed / no font: the default look)
            pass
        yield


def _grid(layout):
    panels = layout["panel"]
    nr = max((p["row"] + p["rowspan"] for p in panels), default=1)
    curve_only = []
    for r in range(nr):
        here = [p for p in panels if p["row"] <= r < p["row"] + p["rowspan"]]
        curve_only.append(bool(here) and not any(p["type"] == "map" for p in here))
    return nr, curve_only


def _letter(k):
    return "(" + "abcdefghijklmnopqrstuvwxyz"[k % 26] + ")"


def _widths(style):
    """figure widths (in) for size single / wide / page, and the tallest figure: the journal widths, or with style = astroplot that package's own
    (big) figure widths, which its font sizes are made for (its figures are scaled down when placed in a paper)."""
    if style == "astroplot":
        try:
            from astroplot import layout as apl
            w1, w2 = float(apl.SINGLE[0]), float(apl.WIDE[0])
            return {"single": w1, "wide": w2, "page": w2}, {"single": 2.0 * w1, "wide": 1.3 * w2, "page": 2.0 * w2}
        except Exception:                        # noqa: BLE001
            pass
    return WIDTH_IN, MAX_HEIGHT_IN


def render(layout, tiles, header=""):
    """a matplotlib Figure of the layout (see the module docstring); call inside `with _style(...)` to keep a style while saving (render_bytes / save do)."""
    from matplotlib.figure import Figure
    fo = layout.get("figure") or lay.figure_options(layout)
    nc = layout["ncols"]
    nr, curve_only = _grid(layout)
    ratios = [0.5 if c else 1.0 for c in curve_only]
    if fo.get("panel_in"):                       # inches per map panel (and its colour bar)
        cell = float(fo["panel_in"])
        W, H = nc * cell, cell * sum(ratios) + (0.08 * cell if (layout.get("title") or fo.get("header") == "run") else 0.0)
    else:
        widths, tallest = _widths(fo["style"])
        W = widths[fo["size"]]
        cell = W / nc
        H = min(tallest[fo["size"]], cell * sum(ratios) * 1.08 + (0.05 * W if layout.get("title") else 0.0))
    touch = fo.get("gap") == "none"
    head0 = bool(layout.get("title") or fo.get("header") == "run")
    if touch and fo.get("panel_in"):                 # touching square maps: the figure is the panels plus fixed margins, so no cell is wider than its map
        cbw = 0.0 if fo["colorbar"] in ("inside", "none") else 0.09 * cell + 0.6
        M = {"l": 0.11 * cell, "r": 0.03 * cell + cbw, "b": 0.09 * cell, "t": (0.12 if head0 else 0.03) * cell}
        W, H = M["l"] + M["r"] + nc * cell, M["b"] + M["t"] + cell * sum(ratios)
    fig = Figure(figsize=(W, H), layout=None if fo.get("panel_in") else "constrained")   # panel_in: tight_layout at the end
    gs = fig.add_gridspec(nr, nc, height_ratios=ratios)
    maps, k_letter, first_time = {}, 0, True
    for p in layout["panel"]:
        ax = fig.add_subplot(gs[p["row"]:p["row"] + p["rowspan"], p["col"]:p["col"] + p["colspan"]])
        t = tiles.get(p["id"]) or {"blank": "no data for this panel"}
        label = p.get("label") or p["id"]
        if "blank" in t:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{label}\n{t['blank']}", ha="center", va="center", transform=ax.transAxes, color="0.5", fontsize="small")
            continue
        if p["type"] == "map":
            end = p["row"] + p["rowspan"]
            outer_left, outer_bottom = p["col"] == 0, (end == nr or curve_only[end])        # the last row of maps keeps its x labels
            pan = avm.draw_image(ax, t["img"], t["meta"], t["cmap"], t["scale"], t["lim"], native=fo.get("colorbar_ticks") == "matplotlib",
                                 nodata=fo.get("nodata", "dark"))
            if not outer_left:
                ax.set_ylabel("")
                if touch:
                    ax.tick_params(labelleft=False)
            if not outer_bottom:
                ax.set_xlabel("")
                if touch:
                    ax.tick_params(labelbottom=False)
            _sizes(ax, fo)
            if t.get("overlays"):
                avm.draw_items(ax, t["overlays"], t["meta"]["extent"])
                _legends(ax, t["overlays"])
            if t.get("scalebar", True):
                avm.add_scalebar(pan, fontsize="small")
            if fo["time"] == "panel" or (fo["time"] == "first" and first_time):
                avm.add_time(pan, loc="lower right", fontsize="small")
                first_time = False
            if fo.get("titles", True):
                ax.set_title(label, fontsize=fo.get("title_pt") or "medium")
            maps[p["id"]] = (p, pan, t)
        elif p["type"] == "text":
            ax.set_axis_off()
            ax.text(0.5, 0.5, t["text"], ha="center", va="center", transform=ax.transAxes, wrap=True)
            continue
        elif "heat" in t:                            # heatmap: the grid, overlay curves, the current-time marker
            _draw_heat(fig, ax, t)
            if fo.get("titles", True):
                ax.set_title(label, fontsize="medium")
            _sizes(ax, fo)
        else:                                        # profile / timeseries: curves
            _draw_curves(ax, t)
            ax.set_xlabel(t.get("xlabel") or ("r" if p["type"] == "profile" else "time"))
            if fo.get("titles", True):
                ax.set_title(label, fontsize="medium")
            _sizes(ax, fo)
        if fo["letters"]:
            ax.text(0.03, 0.96, _letter(k_letter), transform=ax.transAxes, ha="left", va="top", fontweight="bold",
                    color="w" if p["type"] == "map" else "k")
            k_letter += 1
    _colorbars(fig, fo["colorbar"], maps, label=None if fo.get("colorbar_label", True) else "", fo=fo)
    head = _header(layout, tiles) if fo.get("header") == "run" else layout.get("title")
    if head:
        fig.suptitle(head, fontsize=fo.get("header_pt") or "large", y=0.96 if fo.get("panel_in") else None)
    if fo.get("panel_in") and touch:                 # the maps touch: no space between the panels, margins as computed above
        fig.subplots_adjust(left=M["l"] / W, right=1 - M["r"] / W, bottom=M["b"] / H, top=1 - M["t"] / H, wspace=0.0, hspace=0.0)
        if head:
            fig.suptitle(head, fontsize=fo.get("header_pt") or "large", y=1 - 0.35 * M["t"] / H)
    elif fo.get("panel_in"):
        fig.tight_layout(rect=[0, 0.03, 1, 0.95] if head else None)
    elif touch:
        fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
    return fig


def _scale(ax, sc, axis="y"):
    """apply 'lin' | 'log' | ('symlog', linthresh) to an axis."""
    setter = ax.set_yscale if axis == "y" else ax.set_xscale
    if sc == "log":
        setter("log")
    elif isinstance(sc, (list, tuple)) and sc and sc[0] == "symlog":
        setter("symlog", linthresh=float(sc[1]) or 1.0)


def _draw_heat(fig, ax, t):
    """a heatmap tile: pcolormesh of z (y rows, x columns) with a log or linear colour scale and its colour bar, the overlay curves, the marker."""
    import matplotlib.colors as mc
    hm = t["heat"]
    x, y, z = np.asarray(hm["x"], float), np.asarray(hm["y"], float), np.ma.masked_invalid(np.asarray(hm["z"], float))
    if len(y) > 1:                                    # y values are centres: edges halfway between them
        ye = np.concatenate([[y[0] - 0.5 * (y[1] - y[0])], 0.5 * (y[1:] + y[:-1]), [y[-1] + 0.5 * (y[-1] - y[-2])]])
    else:
        ye = np.array([y[0] - 0.5, y[0] + 0.5])
    pos = z.compressed()[z.compressed() > 0]
    if hm.get("zscale") == "log" and pos.size:
        norm = mc.LogNorm(vmin=max(pos.min(), pos.max() * 1e-6), vmax=pos.max())
        z = np.ma.masked_less_equal(z, 0)
    else:
        norm = None
    im = ax.pcolormesh(x, ye, z, cmap=hm.get("cmap", "magma"), norm=norm, shading="flat", rasterized=True)
    fig.colorbar(im, ax=ax, label=hm.get("label") or "", pad=0.01)
    for (name, xs, ys), s in zip(t.get("curves") or [], t.get("styles") or []):
        ax.plot(np.asarray(xs, float), np.asarray(ys, float), color=s.get("color", "c"), lw=s.get("lw", 1.0), label=name)
    if t.get("curves"):
        ax.legend(fontsize="x-small", frameon=True, framealpha=0.7, edgecolor="none")
    if t.get("xnow") is not None:
        ax.axvline(t["xnow"], color="#ffd23c", lw=1.0)
    ax.set_ylim(ye[0], ye[-1])                         # the grid's rows (an overlay curve does not stretch the axis)
    ax.set_xlabel(t.get("xlabel") or "time")
    ax.set_ylabel(t.get("ylabel") or "")


def _draw_curves(ax, t):
    """the curves of a profile / timeseries tile: plain lines, or with `styles` (series sources): colour, dash, dots / events, faint raw curves under
    their running means, a twin y axis, horizontal reference lines, the current-time marker; the y label from the tile or the single curve."""
    styles = t.get("styles") or [{} for _ in t["curves"]]
    axes = {False: ax}
    if any(s.get("twin") for s in styles):
        axes[True] = ax.twinx()
    for (name, xs, ys), s in zip(t["curves"], styles):
        a = axes[bool(s.get("twin"))]
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        kw = {"color": s["color"]} if s.get("color") else {}
        lab = name if s.get("legend", True) else None
        if s.get("style") in ("dots", "events"):
            a.plot(xs, ys, ".", ms=1.5 if s["style"] == "dots" else 3.0, alpha=s.get("alpha", 1.0), label=lab, **kw)
        else:
            a.plot(xs, ys, ls=s.get("dash", "-"), lw=s.get("lw", 1.2), alpha=s.get("alpha", 1.0), label=lab, **kw)
            if s.get("style") == "linepoints":
                a.plot(xs, ys, ".", ms=3, alpha=s.get("alpha", 1.0), **kw)
    if t.get("xlog"):
        ax.set_xscale("log")
    _scale(ax, "log" if t.get("ylog") else t.get("yscale"))
    if True in axes:
        _scale(axes[True], t.get("twin_scale"))
        if t.get("twin_label"):
            axes[True].set_ylabel(t["twin_label"])
    for h in t.get("hlines") or []:
        ax.axhline(h, color="0.5", lw=0.8, ls="--")
    if t.get("xnow") is not None:
        ax.axvline(t["xnow"], color="0.4", lw=0.8, ls="--")
    labelled = [c for c, s in zip(t["curves"], styles) if s.get("legend", True)]
    if t.get("ylabel"):
        ax.set_ylabel(t["ylabel"])
    elif len(labelled) == 1:
        ax.set_ylabel(labelled[0][0])
    if len(labelled) > 1:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axes[True].get_legend_handles_labels() if True in axes else ([], [])
        ax.legend(h1 + h2, l1 + l2, fontsize="x-small", frameon=True, framealpha=0.7, edgecolor="none", loc="best")


def _sizes(ax, fo):
    if fo.get("label_pt"):
        ax.xaxis.label.set_size(fo["label_pt"]); ax.yaxis.label.set_size(fo["label_pt"])
    if fo.get("tick_pt"):
        ax.tick_params(labelsize=fo["tick_pt"])


def _header(layout, tiles):
    """[run] Snap: N | t = T | Width = W unit, a dashboard header (run: the folder above output/, or the snapshot's folder)."""
    import os
    i = tiles.get("_info") or {}
    path = i.get("path") or ""
    d = os.path.dirname(os.path.abspath(path)) if path else ""
    run = os.path.basename(os.path.dirname(d)) if os.path.basename(d) == "output" else os.path.basename(d)
    n = i.get("snap")
    parts = [f"[{run}]" + (f" Snap: {n:03d}" if isinstance(n, int) else "")]
    if i.get("time") is not None:
        parts.append(f"t = {float(i['time']):.4f}")
    if i.get("width") is not None:
        parts.append(f"Width = {float(i['width']):.1f} {i.get('L') or ''}".rstrip())
    return " | ".join(parts) if layout.get("title") is None else layout["title"] + "   " + " | ".join(parts)


def _legends(ax, items):
    """the overlays' legend texts in the panel: lower left stacked upwards (loc = lower right for those that ask), in their colours, small."""
    rows = {"lower left": 0, "lower right": 0}
    for it in items:
        if it.get("kind") != "legend" or not it.get("text"):
            continue
        loc = it.get("loc", "lower left")
        loc = loc if loc in rows else "lower left"
        c = it.get("color", [255, 255, 255])
        col = tuple(v / 255.0 for v in (c if isinstance(c, (list, tuple)) else [255, 255, 255]))
        x = 0.02 if loc == "lower left" else 0.98
        ax.text(x, 0.02 + 0.035 * rows[loc], it.get("text_tex") or it["text"], transform=ax.transAxes, ha="left" if loc == "lower left" else "right", va="bottom",
                color=col, fontsize="x-small", zorder=6)
        rows[loc] += 1


def _same(a, b):
    return a["cmap"] == b["cmap"] and str(a["scale"]) == str(b["scale"]) and np.allclose(a["lim"], b["lim"], rtol=1e-9)


def _colorbars(fig, mode, maps, label=None, fo=None):
    fo = fo or {}
    if mode == "none":
        return
    if mode == "inside":                         # a short horizontal bar in the map's top right corner, its numbers outlined to read on any colour
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=2.5, foreground="black")]
        for p, pan, t in maps.values():
            if not t.get("colorbar", True):
                continue
            cax = pan.ax.inset_axes([0.55, 0.91, 0.4, 0.035])
            cb = avm.add_colorbar(pan, cax=cax, orientation="horizontal", label=label)
            if getattr(pan, "native", False) and str(pan.scale).startswith("log"):    # a short bar: label only 1, 2, 5 x 10^n, never every minor tick
                from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
                cax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=6))
                cax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
                cax.xaxis.set_minor_formatter(NullFormatter())
            cax.tick_params(colors="w", labelsize=fo.get("tick_pt") or "small", direction="in")
            for tl in cax.get_xticklabels():
                tl.set_path_effects(halo)
            cax.xaxis.label.set_color("w"); cax.xaxis.label.set_path_effects(halo)
            if fo.get("label_pt"):
                cax.xaxis.label.set_size(fo["label_pt"])
            for sp in cax.spines.values():
                sp.set_edgecolor("w")
        return
    wanted = [(p, pan, t) for p, pan, t in maps.values() if t.get("colorbar", True)]
    if mode == "panel":
        for p, pan, t in wanted:
            cb = avm.add_colorbar(pan, ax=pan.ax, label=label, fraction=0.046, pad=0.04)      # the bar as tall as the map
            if fo.get("tick_pt"):
                cb.ax.tick_params(labelsize=fo["tick_pt"])
        return
    key = (lambda e: e[0]["row"]) if mode == "row" else (lambda e: e[0]["col"])
    groups = {}
    for e in wanted:
        groups.setdefault(key(e), []).append(e)
    for members in groups.values():
        if all(_same(members[0][2], m[2]) for m in members[1:]):
            avm.add_colorbar(members[-1][1], ax=[m[1].ax for m in members], shrink=0.9, label=label,
                             location="right" if mode == "row" else "bottom")
        else:                                        # different limits or colours: one bar each
            for _, pan, _ in members:
                avm.add_colorbar(pan, ax=pan.ax, shrink=0.9, label=label)


def render_bytes(layout, tiles, header="", fmt="png", dpi=None):
    fo = layout.get("figure") or lay.figure_options(layout)
    with _style(fo["style"]):
        fig = render(layout, tiles, header)
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi or fo["dpi"])
    return buf.getvalue()


def save(layout, tiles, path, header=""):
    import os
    fmt = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    if fmt not in ("png", "pdf", "svg"):
        raise ValueError("figure formats: .png .pdf .svg")
    data = render_bytes(layout, tiles, header, fmt=fmt)
    with open(path, "wb") as f:
        f.write(data)
    return path
