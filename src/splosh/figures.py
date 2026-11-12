"""Finished pictures without a GUI: `shot` (one map + colour bar, scale bar, time, title), `contact_sheet` (many snapshots, shared colour limits) and `movie`
(frames with the same policies, encoded with ffmpeg when it is on PATH). Everything is drawn with numpy and the built-in bitmap font (`bitmapfont.py`) and written
with the existing PNG writer; matplotlib is optional (PDF / `backend="mpl"`) and imported only when asked for. Every written image gets a JSON sidecar
(`provenance.py`) from which `replay` redoes the picture.

Python API:
    pic = splosh.shot(path, View(width=5, field="u"), cmap="inferno", scale="linear", title="...")      # Picture: .rgba (ny, nx, 4), .png() bytes, .meta, .limits, .sidecar
    pic.save("a.png")                                                                                       # writes a.png and a.json
    splosh.contact_sheet(["run/output"], View(...), every=5, limits="global", pct=(1, 99.9)).save("sheet.png")
    splosh.movie(["run/output"], "m.mp4", View(...), fps=12)                                              # dict: output, frames_dir, encoded, command, ...
Overlays: `overlays=["vectors:v", "streamlines:B", "particles:bh,star"]` (spec strings, see overlays.py) are drawn onto every map tile before the labels; the specs are recorded in the sidecar."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import colormaps, imaging, limits as limits_mod, overlays as overlays_mod, provenance, transforms
from .bitmapfont import CELL_H, CELL_W, text_mask
from .maps import View, render_map, required_gas_fields
from .mapstore import MapStore, snapshot_hash

BG = (18, 18, 22)
FG = (236, 236, 236)
DIM = (165, 165, 172)
OUTLINE = (0, 0, 0)
NAN_COLOR = (40, 40, 40)

_LENGTH = [("pc", 3.0857e18), ("kpc", 3.0857e21), ("Mpc", 3.0857e24), ("AU", 1.496e13), ("cm", 1.0)]
_MASS = [("Msun", 1.989e33), ("1e10 Msun", 1.989e43), ("g", 1.0)]
_VEL = [("km/s", 1e5), ("cm/s", 1.0)]
_TIME = [("Myr", 3.156e13), ("kyr", 3.156e10), ("yr", 3.156e7), ("s", 1.0)]


# ----------------------------------------------------------------------------- text and units

def _blit(canvas, mask, x, y, color):
    h, w = mask.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    sub = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    canvas[y0:y1, x0:x1][sub] = color


def draw_text(canvas, x, y, text, s=1, color=FG, outline=None, rotate=False):
    """Draw one line of text with its top-left corner at (x, y) (rotate: reads bottom to top, (x, y) is the top-left of the rotated box). Returns the mask shape."""
    mask = text_mask(text, s)
    if rotate:
        mask = np.rot90(mask)
    if outline is not None:
        d = max(1, (s + 1) // 2)
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                _blit(canvas, mask, x + dx, y + dy, outline)
    _blit(canvas, mask, x, y, color)
    return mask.shape


def _near(value, table, default):
    if value:
        for name, ref in table:
            if abs(value / ref - 1) < 0.01:
                return name, None
    return default, (f"{default} = {value:.4g} {table[-1][0]}" if value else None)


def unit_names(header):
    """Names of the code units from the snapshot header's cgs factors: {'L','M','V','T' (names), 'notes' (definitions of the names that are not standard units), 'tcgs'}."""
    header = header or {}
    L, nl = _near(header.get("UnitLength_In_CGS"), _LENGTH, "L")
    M, nm = _near(header.get("UnitMass_In_CGS"), _MASS, "M")
    V, nv = _near(header.get("UnitVelocity_In_CGS"), _VEL, "V")
    ul, uv = header.get("UnitLength_In_CGS"), header.get("UnitVelocity_In_CGS")
    return {"L": L, "M": M, "V": V, "notes": [n for n in (nl, nm, nv) if n], "tcgs": (ul / uv) if ul and uv else None}


def time_text(t, names, physical=True, period=None):
    """'t = 1.23 (4.5 Myr)'; with `period` (the binary's P0, code time) also ', 2.47 P0'."""
    s = f"t = {t:.4g}"
    orb = f", {t / period:.3g} P0" if period else ""
    if physical and names["tcgs"]:
        sec = t * names["tcgs"]
        for nm, ref in _TIME:
            if abs(sec) >= ref or nm == "s":
                return s + f" ({sec / ref:.3g} {nm}{orb})"
    return s + (f" ({orb[2:]})" if orb else "")


def orbit_period(path):
    """P0 for time labels in orbits: the binary period at the first snapshot (natural order) of the folder holding `path` (runs.binary_period), or None."""
    from .runs import binary_period, find_snapshots
    files = find_snapshots([os.path.dirname(os.path.abspath(path))])
    return binary_period(files[0]) if files else None


def field_unit(meta, names):
    if "unit_override" in meta:
        return meta["unit_override"]
    v = meta["view"]
    f, L, M, V = v["field"], names["L"], names["M"], names["V"]
    if f == "rho":
        return f"{M}/{L}^{2 if v['mode'] == 'proj' else 3}"
    if f == "u":
        return f"({V})^2" if "/" in V else f"{V}^2"
    if f in ("vx", "vy", "vz", "vlos", "speed"):
        return V
    if f == "H":
        return L
    return ""


def fmt_tick(v):
    if v == 0:
        return "0"
    s = f"{v:.4g}" if 1e-3 <= abs(v) < 1e5 else f"{v:.2g}"
    m = re.fullmatch(r"(-?[\d.]+)e([+-])0*(\d+)", s)
    return f"{m.group(1)}e{'-' if m.group(2) == '-' else ''}{m.group(3)}" if m else s


def _nice_step(raw):
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw * (1 - 1e-9):
            return m * mag


def make_ticks(vmin, vmax, scale):
    """Tick values for a colour bar: decades (log, thinned to at most 6; 1-2-5 steps when under 3 decades), 1-2-5 steps (linear), five equally spaced stretch positions (asinh)."""
    if scale == "log":
        lo, hi = np.log10(max(vmin, 1e-300)), np.log10(vmax)
        decs = [10.0 ** k for k in range(int(np.ceil(lo - 1e-9)), int(np.floor(hi + 1e-9)) + 1)]
        if len(decs) >= 3:
            return decs[::-(-len(decs) // 6)]
        c = [m * 10.0 ** k for k in range(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 1) for m in (1, 2, 5)]
        c = [v for v in c if vmin * (1 - 1e-9) <= v <= vmax * (1 + 1e-9)]
        if len(c) < 2:
            return [vmin, vmax]
        return c[::2] if len(c) > 7 else c
    if scale == "asinh":
        grid = np.linspace(vmin, vmax, 4001)
        t = imaging.stretch(grid, vmin, vmax, "asinh")
        return [float(float(f"{v:.3g}")) for v in np.interp(np.linspace(0, 1, 5), t, grid)]
    if scale == "histeq" and getattr(scale, "table", None) is not None:
        t = np.asarray(scale.table)
        vals = []
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            v = float(float(f"{np.interp(q, np.linspace(0, 1, len(t)), t):.3g}"))
            if not vals or v != vals[-1]:
                vals.append(v)
        return vals
    step = _nice_step((vmax - vmin) / 5)
    k = np.arange(np.ceil(vmin / step - 1e-9), np.floor(vmax / step + 1e-9) + 1)
    return [round(float(v), 12) for v in k * step] or [vmin, vmax]


def view_summary(meta):
    v = meta["view"]
    names = unit_names(meta.get("units"))
    return f"{v['mode']} {v['field']} | axes {v['axes']} | centre {v['center']} | align {v['align']}{' | tilt ' + v['tilt'] if v.get('tilt', 'none') != 'none' else ''} | width {v['width']:.6g} {names['L']}"


def footer_lines(meta, source=None):
    v = meta["view"]
    first = view_summary(meta)
    lines = [f"{source} | {first}" if source else first]
    extra = []
    if v.get("hscale", 1.0) != 1.0:
        extra.append(f"kernel: H x {v['hscale']:g} (legacy convention, not the code's kernel)")
    notes = unit_names(meta.get("units"))["notes"]
    if notes:
        extra.append("code units: " + ", ".join(notes))
    return lines + extra


# ----------------------------------------------------------------------------- drawing pieces

def resolve_style(view, cmap, scale, sheet=False):
    """(cmap, scale, symmetric): None means the default of the view's transform (contrast: RdBu_r, linear, limits symmetric about 0; gradient: magma, linear), else viridis / log."""
    st = transforms.style(view.post) or {}
    cmap = cmap or st.get("cmap", "viridis")
    scale = imaging.parse_scale(scale or st.get("scale", "log"))
    if sheet and scale == "histeq":
        raise ValueError("histeq ranks the pixels of one map: use it with shot (or the viewer), not with a sheet or a movie")
    return cmap, scale, bool(st.get("symmetric", False))


def _scale_for(nx, font_scale):
    return font_scale or max(1, round(nx / 300))


def draw_colorbar(canvas, x, y, h, w, cmap, scale, vmin, vmax, label, s, label_chars=None):
    """Colour bar with tick marks and labels at the right and a rotated label; returns its total width."""
    cw, ch = CELL_W * s, CELL_H * s
    lut = colormaps.lut(cmap)
    t = np.linspace(1, 0, h)
    canvas[y:y + h, x:x + w] = lut[(t * 255).round().astype(int)][:, None, :]
    canvas[y - 1, x - 1:x + w + 1] = canvas[y + h, x - 1:x + w + 1] = DIM
    canvas[y - 1:y + h + 1, x - 1] = canvas[y - 1:y + h + 1, x + w] = DIM
    ticks = make_ticks(vmin, vmax, scale)
    labels = [fmt_tick(v) for v in ticks]
    labw = (label_chars or max(len(l) for l in labels)) * cw - s
    tick_len, gap = 2 * s, 2 * s
    for v, lab in zip(ticks, labels):
        tt = float(imaging.stretch(np.array([v]), vmin, vmax, scale)[0])
        if not np.isfinite(tt):
            continue
        py = y + int(round((1 - tt) * (h - 1)))
        canvas[max(py - s // 2, 0):py + (s + 1) // 2 + (0 if s > 1 else 1), x + w:x + w + tick_len] = DIM
        draw_text(canvas, x + w + tick_len + gap, int(np.clip(py - ch // 2, y - ch // 4, y + h - ch + ch // 4)), lab, s, FG)
    lx = x + w + tick_len + gap + labw + 3 * s
    if label:
        lh = len(label) * cw - s
        draw_text(canvas, lx, max(0, y + (h - lh) // 2), label, s, FG, rotate=True)
    return w + tick_len + gap + labw + 3 * s + ch


def _bar_block_width(vmin, vmax, scale, s, label_chars=None):
    cw, ch = CELL_W * s, CELL_H * s
    labs = [fmt_tick(v) for v in make_ticks(vmin, vmax, scale)]
    return 6 * s + 2 * s + 2 * s + (label_chars or max(len(l) for l in labs)) * cw - s + 3 * s + ch


def overlay_tile(rgb, meta, s, label=None, scalebar=True, range_text=None):
    """Time label (top left), scale bar (bottom left) and optional limits text (bottom right) drawn on a coloured map."""
    ny, nx = rgb.shape[:2]
    cw, ch = CELL_W * s, CELL_H * s
    if label:
        draw_text(rgb, 3 * s, 3 * s, label, s, FG, OUTLINE)
    if range_text:
        draw_text(rgb, nx - 3 * s - len(range_text) * cw + s, ny - 3 * s - ch, range_text, s, FG, OUTLINE)
    if scalebar:
        names = unit_names(meta.get("units"))
        ext = meta["extent"]
        width = ext[1] - ext[0]
        target = 0.2 * width
        mag = 10.0 ** np.floor(np.log10(target))
        length = max(m * mag for m in (1, 2, 5, 10) if m * mag <= target * (1 + 1e-9))
        px = max(2, int(round(length / width * nx)))
        x0, y0, th = 3 * s, ny - 3 * s - ch - 3 * s, max(2, s)
        if y0 > 0 and x0 + px < nx:
            rgb[y0 - 1:y0 + th + 1, x0 - 1:x0 + px + 1] = OUTLINE
            rgb[y0:y0 + th, x0:x0 + px] = FG
            draw_text(rgb, x0, y0 + th + 2 * s, f"{length:.4g} {names['L']}", s, FG, OUTLINE)
    return rgb


def _wrap(lines, nchars):
    out = []
    for ln in lines:
        out += textwrap.wrap(ln, width=max(nchars, 12), break_long_words=True) or [""]
    return out


def frame_around(content, meta, s, *, cmap, scale, vmin, vmax, title=None, colorbar=True, footer=(), label_chars=None):
    """Put a coloured content image (a map or a grid of tiles) on a canvas with title, colour bar (label + unit from the map metadata) and footer."""
    H0, W0 = content.shape[:2]
    cw, ch = CELL_W * s, CELL_H * s
    m = 5 * s
    bar = _bar_block_width(vmin, vmax, scale, s, label_chars) if colorbar else 0
    W = m + W0 + ((m + bar) if colorbar else 0) + m
    tlines = _wrap([title], (W - 2 * m) // cw) if title else []
    flines = _wrap(list(footer), (W - 2 * m) // cw)
    ty = m + (len(tlines) * (ch + s) + m - s if tlines else 0)
    H = ty + H0 + m + (len(flines) * (ch + s) if flines else 0) + (0 if flines else 0)
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = BG
    canvas[ty:ty + H0, m:m + W0] = content
    for k, ln in enumerate(tlines):
        draw_text(canvas, m, m + k * (ch + s), ln, s, FG)
    if colorbar:
        names = unit_names(meta.get("units"))
        unit = field_unit(meta, names)
        label = meta.get("label", meta["view"]["field"]) + (f" [{unit}]" if unit else "")
        draw_colorbar(canvas, m + W0 + m, ty, H0, 6 * s, cmap, scale, vmin, vmax, label, s, label_chars)
    for k, ln in enumerate(flines):
        draw_text(canvas, m, ty + H0 + m + k * (ch + s), ln, s, DIM)
    return canvas


def _fit(rgb, size):
    """pad (background) or crop to size = (height, width)."""
    h, w = size
    out = np.empty((h, w, 3), np.uint8)
    out[:] = BG
    hh, ww = min(h, rgb.shape[0]), min(w, rgb.shape[1])
    out[:hh, :ww] = rgb[:hh, :ww]
    return out


def _rgba(rgb):
    return np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)


# ----------------------------------------------------------------------------- maps, selection

def _get_map(path, view, store, use_cache=True):
    if use_cache:
        return (store if store is not None else MapStore()).get_or_render(path, view)
    from .io import load_snapshot
    t0 = time.perf_counter()
    img, meta = render_map(load_snapshot(path, gas_fields=tuple(sorted(required_gas_fields(view)))), view)
    meta["snapshot"] = os.path.basename(path)
    meta["render_seconds"] = round(time.perf_counter() - t0, 4)
    return img, meta, False


def _overlay_specs(overlays):
    """list of canonical spec strings (validated) or []."""
    if not overlays:
        return []
    if isinstance(overlays, (str, dict)):
        overlays = [overlays]
    out = []
    for sp in overlays:
        if callable(sp):
            raise ValueError("overlays: pass spec strings (plugin overlays by their registered name); callables cannot be recorded in the sidecar")
        out.append(overlays_mod.spec_string(sp))
    return out


def _draw_overlays(rgb, path, view, specs, s, legend=True):
    """draw the overlays of `specs` for snapshot `path` onto the colour map `rgb` (row 0 = top); returns the compute() notes."""
    res = overlays_mod.compute_for_path(path, view, specs)
    overlays_mod.draw_items(rgb, res["items"], res["extent"], s, legend=legend)
    return res["notes"]


def _snap_number(path):
    m = re.search(r"(\d+)(?=\D*$)", os.path.splitext(os.path.basename(path))[0])
    return int(m.group(1)) if m else None


def select_snapshots(paths, every=None, snaps=None):
    """Snapshot files from files and directories (natural order). A directory with an `output` subdirectory that holds snapshots uses that (a run directory may keep its initial conditions next to output/).
    `every=N` keeps every Nth file (the first, the N+1th, ...); `snaps` = list of snapshot numbers (the last number in the file name)."""
    from .server import find_snapshots
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    expanded = []
    for p in paths:
        p = os.path.expanduser(str(p))
        if not os.path.exists(p):
            raise SystemExit(f"{p}: no such file or directory")
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "output")) and find_snapshots([os.path.join(p, "output")]):
            p = os.path.join(p, "output")                    # a run directory: its output/ (not initial conditions or other files kept next to it)
        expanded.append(p)
    files = find_snapshots(expanded)
    if not files:
        raise SystemExit("no snapshots found in " + ", ".join(map(str, paths)))
    if snaps:
        by = {}
        for f in files:
            by.setdefault(_snap_number(f), f)
        want = [int(x) for x in (snaps.split(",") if isinstance(snaps, str) else snaps)]
        missing = [w for w in want if w not in by]
        if missing:
            raise SystemExit(f"snapshot numbers not found: {missing}; available: {sorted(k for k in by if k is not None)[:12]}...")
        files = [by[w] for w in want]
    if every is not None:
        if int(every) < 1:
            raise SystemExit("--every must be >= 1")
        files = files[::int(every)]
    return files


def _snap_record(path, meta):
    return {"path": os.path.abspath(path), "name": os.path.basename(path), "time": meta["time"], "hash": snapshot_hash(path)}


def _view_record(view, meta):
    d = view.as_dict()
    return {"view": d, "view_summary": {"center": d["center"], "align": d["align"], "width": d["width"], "field": d["field"], "mode": d["mode"], "axes": d["axes"],
                                          "hscale": d["hscale"], "kernel": meta.get("kernel")}, "frame": meta.get("frame")}


def _tile_label(path, meta, s, nx, with_number=True):
    t = f"t = {meta['time']:.4g}"
    n = _snap_number(path)
    full = f"snap {n}  {t}" if (with_number and n is not None) else t
    return full if len(full) * CELL_W * s <= nx - 6 * s else t


# ----------------------------------------------------------------------------- Picture

@dataclass
class Picture:
    """A finished picture: `rgba` (ny, nx, 4) uint8, `meta` of the (first) map, `limits` actually used, `sidecar` dict (what `save` writes next to the image)."""
    rgba: np.ndarray
    meta: dict
    limits: object
    sidecar: dict
    _fig: object = field(default=None, repr=False)

    def __array__(self, dtype=None, copy=None):
        return self.rgba if dtype is None else self.rgba.astype(dtype)

    @property
    def shape(self):
        return self.rgba.shape

    def png(self):
        return imaging.png_bytes(np.ascontiguousarray(self.rgba[..., :3]))

    def save(self, path, sidecar=True):
        """Write the image (.png; .pdf / .svg need matplotlib and the mpl backend) and its sidecar. Returns the list of files written."""
        path = str(path)
        ext = os.path.splitext(path)[1].lower()
        if self._fig is not None:
            self._fig.savefig(path)
        elif ext == ".png":
            with open(path, "wb") as f:
                f.write(self.png())
        else:
            raise SystemExit(f"cannot write {ext or 'this'} format with the built-in renderer: use .png, or backend='mpl' (matplotlib) for .pdf / .svg")
        written = [path]
        if sidecar:
            written.append(provenance.write_sidecar(path, self.sidecar))
        return written


# ----------------------------------------------------------------------------- shot

def _mpl_figure(img, meta, cmap, scale, vmin, vmax, title, scalebar, period=None):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import ListedColormap
        from matplotlib.figure import Figure
    except ImportError:
        raise SystemExit("matplotlib is not installed: PDF output and --backend mpl need it (the built-in renderer writes .png without it)")
    names = unit_names(meta.get("units"))
    fig = Figure(figsize=(6.6, 5.6), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    cm = ListedColormap(colormaps.lut(cmap) / 255.0).with_extremes(bad=tuple(np.array(NAN_COLOR) / 255.0))
    t = imaging.stretch(img, vmin, vmax, scale)
    ext = meta["extent"]
    im = ax.imshow(t, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    ticks = make_ticks(vmin, vmax, scale)
    cb = fig.colorbar(im, ax=ax)
    cb.set_ticks([float(imaging.stretch(np.array([v]), vmin, vmax, scale)[0]) for v in ticks])
    cb.set_ticklabels([fmt_tick(v) for v in ticks])
    unit = field_unit(meta, names)
    cb.set_label(meta.get("label", "") + (f" [{unit}]" if unit else ""))
    ax.set_xlabel(f"{meta['xlabel']} [{names['L']}]")
    ax.set_ylabel(f"{meta['ylabel']} [{names['L']}]")
    ax.set_title(title or time_text(meta["time"], names, period=period), fontsize=10)
    if title:
        ax.text(0.03, 0.96, time_text(meta["time"], names, period=period), transform=ax.transAxes, color="w", va="top", fontsize=8)
    if scalebar:
        width = ext[1] - ext[0]
        mag = 10.0 ** np.floor(np.log10(0.2 * width))
        length = max(m * mag for m in (1, 2, 5, 10) if m * mag <= 0.2 * width * (1 + 1e-9))
        x0, y0 = ext[0] + 0.04 * width, ext[2] + 0.07 * width
        ax.plot([x0, x0 + length], [y0, y0], color="w", lw=2)
        ax.text(x0, y0 + 0.015 * width, f"{length:.4g} {names['L']}", color="w", fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    return fig, np.asarray(fig.canvas.buffer_rgba()).copy()


def shot(snapshot, view=None, *, cmap=None, scale=None, vmin=None, vmax=None, pct=None, title=None, scalebar=True, colorbar=True,
         backend="png", store=None, use_cache=True, font_scale=None, overlays=None, orbits=False):
    """One finished picture of `snapshot` (orbits=True: the time label also in units of the binary's initial period, see `orbit_period`). Limits: both `vmin` and `vmax` -> fixed; otherwise percentiles `pct` (default 1, 99.9) of this map, with a given bound replacing its side.
    Returns a Picture (RGBA array, `.png()` bytes, `.save(path)` writes image + sidecar). backend 'mpl' uses matplotlib (if installed) for a publication-style figure.
    `overlays`: list of overlay specs drawn on the map (built-in renderer only)."""
    t0 = time.perf_counter()
    view = (view or View()).validate()
    cmap, scale, sym = resolve_style(view, cmap, scale)
    ospecs = _overlay_specs(overlays)
    if ospecs and backend == "mpl":
        raise ValueError("overlays are drawn by the built-in renderer: use backend='png' (the default) with overlays")
    if backend not in ("png", "mpl"):
        raise ValueError("backend must be png or mpl")
    img, meta, hit = _get_map(snapshot, view, store, use_cache)
    policy = "fixed" if (vmin is not None and vmax is not None) else "frame"
    lo, hi = limits_mod.resolve(policy, scale, pct, vmin, vmax, img=img, symmetric=sym)
    scale = imaging.fit_scale(img, lo, hi, scale)
    names = unit_names(meta.get("units"))
    period = orbit_period(snapshot) if orbits else None
    fig = None
    s = None
    onotes = []
    if backend == "mpl":
        fig, rgba = _mpl_figure(img, meta, cmap, scale, lo, hi, title, scalebar, period=period)
    else:
        rgb, _ = imaging.colorize(img, cmap, scale, lo, hi, nan_color=NAN_COLOR)
        s = _scale_for(rgb.shape[1], font_scale)
        onotes = _draw_overlays(rgb, snapshot, view, ospecs, s) if ospecs else []
        overlay_tile(rgb, meta, s, time_text(meta["time"], names, period=period), scalebar)
        rgba = _rgba(frame_around(rgb, meta, s, cmap=cmap, scale=scale, vmin=lo, vmax=hi, title=title, colorbar=colorbar, footer=footer_lines(meta, os.path.basename(snapshot))))
    sidecar = {"command": "shot", "snapshots": [_snap_record(snapshot, meta)], **_view_record(view, meta),
               "limits": {"policy": policy, "pct": list(pct) if pct else None, "vmin": lo, "vmax": hi}, "colormap": cmap, "stretch": str(scale), "title": title,
               "scalebar": bool(scalebar), "colorbar": bool(colorbar), "overlays": ospecs, "backend": backend, "font_scale": s, "image_size": [int(rgba.shape[1]), int(rgba.shape[0])],
               "map_cache_hit": bool(hit), "wall_seconds": round(time.perf_counter() - t0, 3)}
    if onotes:
        sidecar["overlay_notes"] = onotes
    return Picture(rgba, meta, (lo, hi), sidecar, fig)


# ----------------------------------------------------------------------------- sheet and movie

def _policy_limits(policy, scale, pct, vmin, vmax, maps_iter_factory, n, symmetric=False):
    """Limits per frame: returns a function(img) -> (vmin, vmax); for `global` runs one pass over the maps to lock them."""
    if policy == "global":
        acc = limits_mod.Accumulator(scale, cap=max(20_000, 20_000_000 // max(n, 1)), symmetric=symmetric)
        for img in maps_iter_factory():
            acc.add(img)
        fixed = limits_mod.resolve("global", scale, pct, vmin, vmax, accumulator=acc)
        return lambda img: fixed, fixed
    if policy == "fixed":
        fixed = limits_mod.resolve("fixed", scale, pct, vmin, vmax)
        return lambda img: fixed, fixed
    return (lambda img: limits_mod.resolve("frame", scale, pct, vmin, vmax, img=img, symmetric=symmetric)), None


def _common(snapshots, every, snaps, view, policy):
    if policy not in limits_mod.POLICIES:
        raise ValueError(f"limits must be one of {limits_mod.POLICIES}")
    return select_snapshots(snapshots, every, snaps), (view or View()).validate()


def _source_name(paths):
    d = os.path.basename(os.path.dirname(os.path.abspath(paths[0])))
    return f"{d}: {len(paths)} snapshots" if len(paths) > 1 else os.path.basename(paths[0])


def contact_sheet(snapshots, view=None, *, every=None, snaps=None, ncols=None, limits="global", pct=None, vmin=None, vmax=None, cmap=None, scale=None,
                  title=None, scalebar=True, store=None, font_scale=None, overlays=None, _per_frame=None):
    """Contact sheet of many snapshots (files and/or directories; `every` / `snaps` select) with a time label per tile. `limits`: 'global' (locked over the selection:
    min/max, or percentiles `pct` of all maps together), 'frame' (each tile its own percentiles; the tile shows its range, there is no shared colour bar) or 'fixed' (vmin, vmax).
    Maps come from the mapstore, so a repeated sheet re-renders nothing. `overlays` are drawn on every tile (their legends on the first). Returns a Picture."""
    t0 = time.perf_counter()
    paths, view = _common(snapshots, every, snaps, view, limits)
    cmap, scale, sym = resolve_style(view, cmap, scale, sheet=True)
    ospecs = _overlay_specs(overlays)
    store = store if store is not None else MapStore()
    maps = [_get_map(p, view, store) for p in paths]
    n = len(paths)
    if _per_frame is not None:
        lims = list(map(tuple, _per_frame))
        shared = None
    else:
        fn, shared = _policy_limits(limits, scale, pct, vmin, vmax, lambda: (m[0] for m in maps), n, sym)
        lims = [fn(m[0]) for m in maps]
    ncols = int(ncols or int(np.ceil(np.sqrt(n))))
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    nx = maps[0][0].shape[1]
    ny = maps[0][0].shape[0]
    s = _scale_for(nx, font_scale)
    gap = 2 * s
    grid = np.empty((nrows * ny + (nrows - 1) * gap, ncols * nx + (ncols - 1) * gap, 3), np.uint8)
    grid[:] = BG
    for k, ((img, meta, _), p, (lo, hi)) in enumerate(zip(maps, paths, lims)):
        rgb, _ = imaging.colorize(img, cmap, scale, lo, hi, nan_color=NAN_COLOR)
        rng = None if shared is not None else f"{fmt_tick(lo)} .. {fmt_tick(hi)}"
        if ospecs:
            _draw_overlays(rgb, p, view, ospecs, s, legend=(k == 0))
        overlay_tile(rgb, meta, s, _tile_label(p, meta, s, nx), scalebar and k == 0, rng)
        r, c = divmod(k, ncols)
        grid[r * (ny + gap):r * (ny + gap) + ny, c * (nx + gap):c * (nx + gap) + nx] = rgb
    meta0 = maps[0][1]
    lo0, hi0 = shared if shared is not None else lims[0]
    rgb = frame_around(grid, meta0, s, cmap=cmap, scale=scale, vmin=lo0, vmax=hi0, title=title, colorbar=shared is not None,
                       footer=footer_lines(meta0, _source_name(paths)) + ([] if shared is not None else ["limits: per frame (range printed in each tile)"]))
    lim_rec = {"policy": limits, "pct": list(pct) if pct else None, "vmin_arg": vmin, "vmax_arg": vmax}
    if shared is not None:
        lim_rec.update(vmin=shared[0], vmax=shared[1])
    else:
        lim_rec["per_frame"] = [list(l) for l in lims]
    sidecar = {"command": "sheet", "snapshots": [_snap_record(p, m[1]) for p, m in zip(paths, maps)], **_view_record(view, meta0), "limits": lim_rec, "colormap": cmap,
               "stretch": str(scale), "title": title, "scalebar": bool(scalebar), "overlays": ospecs, "ncols": ncols, "every": every, "snaps": list(snaps.split(",")) if isinstance(snaps, str) else snaps,
               "font_scale": s, "image_size": [int(rgb.shape[1]), int(rgb.shape[0])], "map_cache_hits": int(sum(m[2] for m in maps)), "wall_seconds": round(time.perf_counter() - t0, 3)}
    return Picture(_rgba(rgb), meta0, shared if shared is not None else lims, sidecar)


EXPLORE_PANELS = (
    {"title": "log, 1-99.9%", "post": "none", "scale": "log"},
    {"title": "asinh F=0.02", "post": "none", "scale": "asinh:0.02"},
    {"title": "histogram equalised", "post": "none", "scale": "histeq"},
    {"title": "contrast vs azimuthal mean", "post": "resid:azim"},
    {"title": "contrast vs smoothed (W/25)", "post": "resid:smooth:{w25}"},
    {"title": "|grad log f| (W/150)", "post": "grad:{w150}"},
)


def explore(snapshot, view=None, *, panels=None, ncols=3, title=None, store=None, use_cache=True, font_scale=None):
    """The discovery sheet: one map of the snapshot shown the ways that make faint structure visible, each tile with its own colours (the range is printed in the tile): a log stretch
    with percentile limits, a stronger asinh, histogram equalisation, the contrast against the azimuthal mean, against a smoothed copy (width W/25) and the gradient of the logarithm (smoothing W/150).
    The map is rendered once; the transforms are applied to it. `panels`: a list of dicts {title, post, scale, cmap} to replace the default six ('{w25}' / '{w150}' in a post stand for W/25, W/150;
    scale and cmap default to the transform's own, else log / viridis). Signed fields (velocities) get a linear first panel. Returns a Picture; the sidecar records the panels, so `replay` redoes it."""
    t0 = time.perf_counter()
    base = (view or View()).with_(post="none").validate()
    ncols = max(1, int(ncols))
    panels = [dict(p) for p in (panels if panels is not None else EXPLORE_PANELS)]
    if not panels:
        raise ValueError("explore: no panels")
    fill = {"w25": f"{base.width / 25:.3g}", "w150": f"{base.width / 150:.3g}"}
    signed = base.field != "rho" and base.field not in transforms.POSITIVE_FIELDS
    raw, meta_raw, hit = _get_map(snapshot, base, store, use_cache)
    s = None
    tiles, used, lims = [], [], []
    for pn in panels:
        post = str(pn.get("post", "none")).format(**fill)
        cmap, scale, sym = resolve_style(base.with_(post=post), pn.get("cmap"),
                                          "linear" if signed and post == "none" and pn.get("scale") in (None, "log") else pn.get("scale"), sheet=False)
        img, meta = transforms.finish(raw, meta_raw, post)
        lo, hi = limits_mod.resolve("frame", scale, None, None, None, img=img, symmetric=sym)
        scale = imaging.fit_scale(img, lo, hi, scale)
        rgb, _ = imaging.colorize(img, cmap, scale, lo, hi, nan_color=NAN_COLOR)
        if s is None:
            s = _scale_for(rgb.shape[1], font_scale)
        overlay_tile(rgb, meta, s, pn.get("title") or post, False, f"{fmt_tick(lo)} .. {fmt_tick(hi)}")
        tiles.append(rgb)
        used.append({"title": pn.get("title"), "post": post, "cmap": cmap, "scale": str(scale), "limits": [lo, hi]})
        lims.append((lo, hi))
    n = len(tiles)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    ny, nx = tiles[0].shape[:2]
    gap = 2 * s
    grid = np.empty((nrows * ny + (nrows - 1) * gap, ncols * nx + (ncols - 1) * gap, 3), np.uint8)
    grid[:] = BG
    for k, rgb in enumerate(tiles):
        r, c = divmod(k, ncols)
        grid[r * (ny + gap):r * (ny + gap) + ny, c * (nx + gap):c * (nx + gap) + nx] = rgb
    foot = footer_lines(meta_raw, os.path.basename(snapshot)) + ["every tile has its own colours: the range is printed in the tile (the normalisation is part of what you see)"]
    rgb = frame_around(grid, meta_raw, s, cmap="viridis", scale="log", vmin=1.0, vmax=10.0, title=title, colorbar=False, footer=foot)
    sidecar = {"command": "explore", "snapshots": [_snap_record(snapshot, meta_raw)], **_view_record(base, meta_raw), "panels": used, "panels_spec": [{k: v for k, v in p.items()} for p in panels],
               "title": title, "ncols": ncols, "font_scale": s, "image_size": [int(rgb.shape[1]), int(rgb.shape[0])], "map_cache_hit": bool(hit), "wall_seconds": round(time.perf_counter() - t0, 3)}
    return Picture(_rgba(rgb), meta_raw, lims, sidecar)


def _even(rgb):
    h, w = rgb.shape[:2]
    return rgb if (h % 2 == 0 and w % 2 == 0) else _fit(rgb, (h + h % 2, w + w % 2))


def ffmpeg_command(ffmpeg, fps, pattern, out):
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern]
    if os.path.splitext(out)[1].lower() in (".mp4", ".mov", ".mkv"):
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    return cmd + [out]


def movie_workers(threads=None):
    """frames at the same time for a movie: about one per 4 rendering threads (NUMBA_NUM_THREADS, at most the physical cores), at most 6, at least 1."""
    from .cpus import worker_threads
    t = worker_threads(threads)
    return max(1, min(6, t // 4))


def _pool_map(fn, items, workers):
    """fn over items in `workers` processes (spawned, each with NUMBA_NUM_THREADS = its share of the threads), results in completion order."""
    import concurrent.futures as cf
    import multiprocessing as mp
    from .cpus import worker_threads
    total = worker_threads()                                 # physical cores: hyperthreads do not speed up the kernels
    per = str(max(1, total // workers))
    old = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = per                       # read by the spawned children when they import numba
    try:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(fn, it) for it in items]
            for f in cf.as_completed(futs):
                yield f.result()
    finally:
        if old is None:
            os.environ.pop("NUMBA_NUM_THREADS", None)
        else:
            os.environ["NUMBA_NUM_THREADS"] = old


def _prerender_star(a):
    p, view, root = a
    _get_map(p, view, MapStore(root))
    return p


def _movie_frame_star(a):
    return _movie_frame(*a)


def _movie_frame(k, p, ctx, size):
    """one movie frame: the map (from the store), its colours, overlays and labels, written as frame_k.png; returns what the sidecar needs."""
    view, cmap, scale = ctx["view"], ctx["cmap"], ctx["scale"]
    img, meta, hit = _get_map(p, view, MapStore(ctx["store_root"]))
    if ctx["per_frame"] is not None:
        lo, hi = tuple(ctx["per_frame"][k])
    elif ctx["shared"] is not None:
        lo, hi = ctx["shared"]
    else:
        q = ctx["policy"]
        lo, hi = limits_mod.resolve("frame", q["scale"], q["pct"], q["vmin"], q["vmax"], img=img, symmetric=q["symmetric"])
    names = unit_names(meta.get("units"))
    rgb, _ = imaging.colorize(img, cmap, scale, lo, hi, nan_color=NAN_COLOR)
    s = _scale_for(rgb.shape[1], ctx["font_scale"])
    if ctx["ospecs"]:
        _draw_overlays(rgb, p, view, ctx["ospecs"], s)
    overlay_tile(rgb, meta, s, time_text(meta["time"], names, period=ctx["period"]), ctx["scalebar"])
    rgb = frame_around(rgb, meta, s, cmap=cmap, scale=scale, vmin=lo, vmax=hi, title=ctx["title"], colorbar=True, footer=footer_lines(meta, ctx["source"]), label_chars=8)
    rgb = _even(rgb)
    size = size or rgb.shape[:2]
    with open(os.path.join(ctx["fdir"], f"frame_{k:05d}.png"), "wb") as f:
        f.write(imaging.png_bytes(_fit(rgb, size)))
    return {"k": k, "lim": (lo, hi), "snap": _snap_record(p, meta), "hit": bool(hit), "time": meta["time"], "meta": meta, "size": size}


def movie(snapshots, out, view=None, *, fps=12, every=None, snaps=None, limits="global", pct=None, vmin=None, vmax=None, cmap=None, scale=None, title=None,
          scalebar=True, store=None, font_scale=None, keep_frames=False, ffmpeg=None, _per_frame=None, log=None, overlays=None, orbits=False, workers=None):
    """Frames of the snapshots with the same limit policies as the contact sheet, encoded to `out` with ffmpeg when it is on PATH (or `ffmpeg` = its path). Without ffmpeg
    (or if it fails) the numbered PNGs `<out stem>_frames/frame_00000.png ...` are kept, with the command that would encode them. A sidecar is written either way.
    workers: frames rendered at the same time, each in its own process with its share of the cores (NUMBA_NUM_THREADS / workers); default
    `movie_workers()` (about one per 4 cores, at most 6: reading a snapshot, the overlays and the PNG are single-threaded, the splat is not).
    Returns a dict (output, frames_dir, encoded, command, n_frames, limits, sidecar, message)."""
    t0 = time.perf_counter()
    log = log or (lambda msg: None)
    paths, view = _common(snapshots, every, snaps, view, limits)
    workers = movie_workers() if workers is None else max(1, int(workers))
    cmap, scale, sym = resolve_style(view, cmap, scale, sheet=True)
    ospecs = _overlay_specs(overlays)
    store = store if store is not None else MapStore()
    n = len(paths)
    fn = shared = None
    if _per_frame is None:
        if limits == "global" and workers > 1 and n > 1:        # the maps of the limits pass, rendered in parallel into the store (then read back quickly)
            todo = [(p, view, store.root) for p in paths]
            for k, _ in enumerate(_pool_map(_prerender_star, todo, workers), start=1):
                log(f"pass {k}/{n}")
        fn, shared = _policy_limits(limits, scale, pct, vmin, vmax, lambda: (_get_map(p, view, store)[0] for p in paths), n, sym)
    exe = ffmpeg or os.environ.get("SPLOSH_FFMPEG") or shutil.which("ffmpeg")         # SPLOSH_FFMPEG: an ffmpeg that is not on PATH
    stem = os.path.splitext(out)[0]
    outdir = os.path.dirname(os.path.abspath(out))
    os.makedirs(outdir, exist_ok=True)
    if exe and not keep_frames:
        fdir = tempfile.mkdtemp(prefix=".frames_", dir=outdir)
    else:
        fdir = stem + "_frames"
        os.makedirs(fdir, exist_ok=True)
    period = orbit_period(paths[0]) if orbits and paths else None
    ctx = dict(view=view, cmap=cmap, scale=scale, ospecs=ospecs, font_scale=font_scale, scalebar=scalebar, title=title, period=period, store_root=store.root,
               source=_source_name(paths), fdir=fdir, per_frame=_per_frame, shared=shared, policy=dict(scale=scale, pct=pct, vmin=vmin, vmax=vmax, symmetric=sym))
    first = _movie_frame(0, paths[0], ctx, None)                # the first frame fixes the size of all
    size = first["size"]
    results = [first]
    log(f"frame 1/{n}  t={first['time']:.4g}")
    rest = [(k, p) for k, p in enumerate(paths)][1:]
    if workers > 1 and len(rest) > 1:
        for k, r in enumerate(_pool_map(_movie_frame_star, [(k, p, ctx, size) for k, p in rest], workers), start=2):
            results.append(r)
            log(f"frame {k}/{n}  t={r['time']:.4g}")
    else:
        for k, p in rest:
            results.append(_movie_frame(k, p, ctx, size))
            log(f"frame {k + 1}/{n}  t={results[-1]['time']:.4g}")
    results.sort(key=lambda r: r["k"])
    used = [r["lim"] for r in results]
    snaprec = [r["snap"] for r in results]
    hits = sum(r["hit"] for r in results)
    meta0 = first["meta"]
    pattern = os.path.join(fdir, "frame_%05d.png")
    cmd = ffmpeg_command(exe or "ffmpeg", fps, pattern, out)
    encoded, msg = False, ""
    if exe:
        r = subprocess.run(cmd, capture_output=True, text=True)
        encoded = r.returncode == 0
        if encoded:
            msg = f"encoded {n} frames to {out}"
            if not keep_frames:
                shutil.rmtree(fdir, ignore_errors=True)
                fdir = None
        else:
            if os.path.basename(fdir).startswith(".frames_"):
                keep = stem + "_frames"
                shutil.rmtree(keep, ignore_errors=True)
                os.replace(fdir, keep)
                fdir = keep
            msg = f"ffmpeg failed (exit {r.returncode}): {r.stderr.strip()[:300]}; the frames are kept in {fdir}"
    else:
        msg = (f"ffmpeg not found on PATH: wrote {n} numbered PNGs to {fdir} (frame_00000.png ...), no video. To encode them yourself:\n  " + " ".join(cmd))
    lim_rec = {"policy": limits, "pct": list(pct) if pct else None, "vmin_arg": vmin, "vmax_arg": vmax}
    if shared is not None:
        lim_rec.update(vmin=shared[0], vmax=shared[1])
    else:
        lim_rec["per_frame"] = [list(l) for l in used]
    sidecar = {"command": "movie", "snapshots": snaprec, **_view_record(view, meta0), "limits": lim_rec, "colormap": cmap, "stretch": str(scale), "title": title, "scalebar": bool(scalebar), "overlays": ospecs,
               "fps": fps, "every": every, "snaps": list(snaps.split(",")) if isinstance(snaps, str) else snaps, "encoded": encoded, "ffmpeg_command": cmd if exe else None,
               "frames_dir": os.path.abspath(fdir) if fdir else None, "image_size": [int(size[1]), int(size[0])], "message": msg.split("\n")[0],
               "map_cache_hits": int(hits), "wall_seconds": round(time.perf_counter() - t0, 3)}
    side = provenance.write_sidecar(out, sidecar)
    return {"output": out if encoded else None, "frames_dir": fdir, "encoded": encoded, "command": cmd, "n_frames": n, "limits": shared or used, "sidecar": side, "message": msg}


# ----------------------------------------------------------------------------- replay

def replay(sidecar_path, out=None, store=None, use_cache=True, log=None):
    """Redo the picture described by a sidecar. The limits used originally are applied as they were (also for the per-frame policy), so with the same snapshots the
    image is the same. Writes `out` (default: `<stem>_repro.<ext>` next to the sidecar, never the original) and returns (command, out)."""
    doc = provenance.read_sidecar(sidecar_path)
    cmd = doc["command"]
    orig = doc["outputs"][0]
    if out is None:
        stem, ext = os.path.splitext(orig)
        out = stem + "_repro" + ext
    view = View(**doc["view"])
    for rec in doc["snapshots"]:
        if not os.path.exists(rec["path"]):
            raise SystemExit(f"snapshot {rec['path']} no longer exists")
        if snapshot_hash(rec["path"]) != rec.get("hash"):
            warnings.warn(f"{rec['path']} differs from the snapshot recorded in the sidecar (content hash): the picture may not be identical", stacklevel=2)
    if cmd == "explore":
        explore(doc["snapshots"][0]["path"], view, panels=doc.get("panels_spec"), ncols=doc.get("ncols", 3), title=doc.get("title"), font_scale=doc.get("font_scale"), store=store, use_cache=use_cache).save(out)
        return cmd, out
    lim = doc["limits"]
    common = dict(cmap=doc["colormap"], scale=doc["stretch"], title=doc.get("title"), scalebar=doc.get("scalebar", True), font_scale=doc.get("font_scale"), store=store, overlays=doc.get("overlays") or None)
    paths = [r["path"] for r in doc["snapshots"]]
    if cmd == "shot":
        pic = shot(paths[0], view, vmin=lim["vmin"], vmax=lim["vmax"], colorbar=doc.get("colorbar", True), backend=doc.get("backend", "png"), use_cache=use_cache, **common)
        pic.save(out)
        return cmd, out
    per = lim.get("per_frame")
    kw = dict(limits=lim["policy"], pct=tuple(lim["pct"]) if lim.get("pct") else None, **common)
    if per is None:
        kw.update(limits="fixed", vmin=lim["vmin"], vmax=lim["vmax"])
    else:
        kw.update(_per_frame=per)
    if cmd == "sheet":
        contact_sheet(paths, view, ncols=doc.get("ncols"), **kw).save(out)
        return cmd, out
    if cmd == "movie":
        movie(paths, out, view, fps=doc.get("fps", 12), log=log, **kw)
        return cmd, out
    raise SystemExit(f"unknown sidecar command {cmd!r}")
