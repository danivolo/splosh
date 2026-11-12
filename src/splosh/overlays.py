"""Overlays: vector and streamline fields, particles by type, click-to-inspect, and the small numpy rasteriser that draws them into headless pictures.

Everything that COMPUTES an overlay returns plain data (lists / dicts of Python numbers and strings, JSON-serialisable) in WINDOW coordinates: the coordinates of the
view's frame (centred, aligned), horizontal and vertical axes as in the map, in code length units; the window is `extent = [x0, x1, y0, y1]` (the map's `meta['extent']`).
Map pixel i, j (j counted from the bottom, like the map array) is  x = x0 + (i + 0.5) * (x1 - x0) / npix.  The rasteriser (`draw_items`) and the browser viewer both turn
the same data into pixels.

Specs.  An overlay is requested with a spec string  `name[:arg,arg,k=v,...]`  (CLI `--overlay`), or a dict {"name": ..., "args": [...], "opts": {...}}:
    vectors:v                      arrow glyphs of the projected velocity on a coarse regular grid (v: frame velocity; B: magnetic field, needs MagneticField)
    streamlines:B                  RK4 streamlines of the same gridded field
    contours:rho,n=8,step=0.5      lines of constant value of a field (log-spaced levels by default; see HELP['contours'])
    particles:bh,star,dm           points by type (bh = PartType5, dm = 1, dm2 = 2, type3 = 3, star = 4, or pt1 .. pt5)
    inspect-free: see `inspect()` for the click query.
Options are listed with their defaults in `DEFAULTS`; `splosh.overlays.describe()` prints them.  Registered overlays (`plugins.OVERLAYS`, signature
`function(snap, frame, view) -> list of dict`, optionally `**opts`) are used by name too.

Item schema (each overlay yields a list of dicts; `kind` selects the primitive; colours are [r, g, b] 0..255; lengths are window units unless the name ends in _px):
    {"kind": "arrow",    "x", "y"  (tail), "dx", "dy" (offset to the tip), "magnitude", "head" (head length / arrow length), "color", "linewidth" (px), "alpha", "halo" (px), "overlay"}
    {"kind": "polyline", "points": [[x, y], ...] (in flow order), "magnitude": [per point], "head": index of the point where an arrow head is drawn or null,
                         "color", "linewidth", "alpha", "halo", "overlay"}
    {"kind": "points",   "type": name, "x": [...], "y": [...], "depth": [...], "id": [...] (when known), "mass": [...] (when known), "size_px", "style": "dot" | "ring",
                         "color", "alpha", "n_total" (in the window before the cap), "n_drawn", "overlay"}
    {"kind": "circle",   "x", "y", "r" (window units), "fill": bool, "color", "linewidth", "alpha"}
    {"kind": "legend",   "overlay", "text", "length" (arrow length in window units or 0), "color"}
    arrow and polyline items may carry "dash": [on, off] in pixels (a dashed line; the arrow head stays solid); an arrow with "thin": true is drawn
    as a line with a head (not a filled shaft) where the renderer distinguishes (matplotlib). A legend may carry "text_tex" (matplotlib mathtext) and
    "loc" ("lower left" | "lower right": where a figure puts it).
    {"kind": "text",     "x", "y", "text", "color", "size_px" (letter height, default 11), "dx_px", "dy_px" (offset of the text's lower left corner from x, y)}
    {"kind": "polygon",  "points": [[x, y], ...] (closed), "color", "alpha" (of the fill), "overlay"}            a filled shape (e.g. the disc overlay's midplane)
    no "kind" (old plugin markers, e.g. maps.bh_markers):  a dict with "x", "y" is drawn as a ring marker.
`compute` wraps a list of specs: {"extent", "npix", "items": [...], "notes": [str, ...]}."""
import inspect as _inspect
import math
from collections import OrderedDict

import numpy as np
from numba import njit as _njit

from . import plugins
from .frames import make_frame
from .kernel import tables, w3
from .maps import AXES, FieldContext, View, render_map, required_gas_fields, resolve_weight, bh_markers

# ----------------------------------------------------------------------------- colours and specs

COLORS = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (235, 60, 60), "green": (80, 220, 90), "blue": (70, 110, 255), "yellow": (255, 230, 60),
          "gold": (255, 200, 40), "orange": (255, 150, 40), "cyan": (60, 220, 235), "magenta": (235, 70, 220), "deepskyblue": (0, 170, 255), "violet": (190, 110, 255),
          "gray": (170, 170, 170), "grey": (170, 170, 170), "lime": (170, 255, 60), "pink": (255, 140, 190)}

TYPES = {"bh": 5, "star": 4, "stars": 4, "dm": 1, "dm1": 1, "dm2": 2, "type3": 3, "type4": 4, "pt1": 1, "pt2": 2, "pt3": 3, "pt4": 4, "pt5": 5}
TYPE_STYLE = {5: ("white", 6.0, "ring"), 4: ("gold", 2.0, "dot"), 1: ("deepskyblue", 1.5, "dot"), 2: ("violet", 1.5, "dot"), 3: ("orange", 1.5, "dot")}
TYPE_NAME = {5: "bh", 4: "star", 1: "dm", 2: "dm2", 3: "type3"}

DEFAULTS = {
    "vectors": {"grid": 24, "arrows": "prop", "ref": 0.0, "len": 0.9, "min": 0.0, "max": 0.0, "color": "white", "lw": 1.0, "halo": 1.0, "alpha": 1.0, "head": 0.35, "legend": True},
    "streamlines": {"grid": 32, "seeds": 14, "density": 28, "maxlen": 0.6, "step": 0.25, "stag": 0.02, "min": 0.0, "minlen": 0.05, "color": "white", "lw": 1.0, "halo": 1.0,
                    "alpha": 0.9, "heads": True},
    "particles": {"max": 2000, "size": 0.0, "color": "", "style": "", "seed": 0, "depth": 0.0},
    "contours": {"levels": "log", "n": 8, "step": 0.5, "top": 0.0, "smooth": 2.0, "post": "none", "color": "white", "lw": 1.0, "halo": 1.0, "alpha": 0.9, "legend": True},
    "disc": {"R": 0.0, "frac": 0.8, "color": "white", "fill": 0.22, "lw": 1.2, "shape": True, "legend": True},
}
HELP = {
    "vectors": "grid cells per axis; arrows = prop (length ~ |v|, `ref` = the magnitude drawn as `len` cell spacings, default the 99th percentile of the drawn cells) | unit (all the same length `len`); "
               "min: cells below this magnitude are skipped; max: lengths saturate there (0 = off); lw in px; halo = dark outline width in px; head = head length / arrow length; legend: reference arrow",
    "streamlines": "grid = data grid cells per axis; seeds = seed grid per axis; density = occupancy cells per axis (one line per cell); maxlen = maximum length / window width; step = RK4 step in grid cells; "
                   "stag = stop where |v| < stag * p99(|v|) (min overrides with an absolute magnitude); minlen = shortest line kept / window width",
    "particles": "types are the bare arguments; max = cap per type (random, deterministic given `seed`); size in px (0 = per type default); color name or #rrggbb; style dot | ring; "
                 "depth = only particles within +- depth (window units) of the slice plane z0 along the line of sight (0 = all); per type: `star.color=gold,bh.size=8`",
    "contours": "the argument is the field to contour (default: the view's field), of the same window; levels = log (n levels `step` dex apart, the highest at `top` or, when 0, at the 99.5th percentile of the map) | lin "
                "(n levels evenly between the 5th and 99.5th percentile) | explicit values joined by ';' (levels=1e-3;1e-2); smooth = Gaussian in pixels applied first (0 = none; SPH noise makes raw contours ragged); "
                "post = a transform of the contoured map (default none: the raw field, whatever the view's own post); lw / halo in px",
    "disc": "the gas disc around the frame's centre: R (0 = the radius holding frac of the window's gas mass), its plane from the angular momentum within R "
            "(semi-transparent, alpha fill), a constant-aspect-ratio fit H/R (rims at +-H dashed, outline; shape=0 hides them), legend R, H/R, inclination",
}


def parse_spec(spec):
    """str or dict -> {"name": str, "args": [str], "opts": {k: str}}. `name:a,b,k=v`; a dict may carry typed option values."""
    if isinstance(spec, dict):
        return {"name": spec["name"], "args": [str(a) for a in spec.get("args", [])], "opts": dict(spec.get("opts", {}))}
    name, _, rest = str(spec).strip().partition(":")
    args, opts = [], {}
    for tok in filter(None, (t.strip() for t in rest.split(","))):
        k, eq, v = tok.partition("=")
        if eq:
            opts[k.strip()] = v.strip()
        else:
            args.append(tok)
    if not name:
        raise ValueError(f"empty overlay spec {spec!r}")
    return {"name": name.strip(), "args": args, "opts": opts}


def spec_string(spec):
    s = parse_spec(spec)
    toks = s["args"] + [f"{k}={v}" for k, v in s["opts"].items()]
    return s["name"] + (":" + ",".join(toks) if toks else "")


def _coerce(value, default, key):
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return str(value)


def _opts(name, opts, extra=()):
    d = DEFAULTS[name]
    out = dict(d)
    for k, v in opts.items():
        if k not in d and k not in extra:
            raise ValueError(f"overlay {name!r}: unknown option {k!r}; options: {sorted(d)}")
        if k in d:
            out[k] = _coerce(v, d[k], k)
    return out


def color_rgb(c):
    if isinstance(c, (tuple, list)) and len(c) == 3:
        return [int(v) for v in c]
    c = str(c).strip().lower()
    if c in COLORS:
        return list(COLORS[c])
    if c.startswith("#") and len(c) == 7:
        return [int(c[k:k + 2], 16) for k in (1, 3, 5)]
    raise ValueError(f"unknown colour {c!r} (names: {sorted(COLORS)} or #rrggbb)")


def _rl(a, nd=6):
    """list of floats with `nd` significant digits (compact JSON)."""
    return [float(f"{v:.{nd}g}") for v in np.asarray(a, dtype=np.float64).ravel()]


def _f(v, nd=7):
    return float(f"{v:.{nd}g}")


# ----------------------------------------------------------------------------- magnetic field components as fields (frame axes)

def _bfield(ctx):
    B = np.asarray(ctx.snap["MagneticField"], dtype=np.float64) @ ctx.frame.rot.T       # a vector: rotated with the frame, no centre offset
    return B


def _register_B():
    for k, ax in enumerate("xyz"):
        plugins.register_field("B" + ax, (lambda c, k=k: _bfield(c)[:, k]), label=f"magnetic field {ax} (frame)", needs=("MagneticField",), unit=None, builtin=True,
                               doc="B component along the frame axis, code units of the snapshot (B_gauss = B * UnitMagneticField_in_gauss, see `beta`)")


_register_B()


# ----------------------------------------------------------------------------- loading what overlays need

def required_data(specs, view):
    """(PartType0 datasets, particle types) a list of overlay specs needs on top of the view's own (the loader's gas_fields and types)."""
    gas, types = set(), set()
    for s in map(parse_spec, specs or ()):
        if s["name"] in ("vectors", "streamlines"):
            q = (s["args"] or ["v"])[0]
            gas |= required_gas_fields(view.with_(field="Bx" if q == "B" else "vx", mode=view.mode))
        elif s["name"] == "contours":
            if len(s["args"]) > 1:
                raise ValueError(f"contours: one field name, got {s['args']}")
            gas |= required_gas_fields(view.with_(field=(s["args"] or [view.field])[0], post=s["opts"].get("post", "none")))
        elif s["name"] == "disc":
            gas |= {"Velocities"}
        elif s["name"] == "particles":
            for a in s["args"]:
                t = TYPES.get(a.lower())
                if t is None:
                    raise ValueError(f"unknown particle type {a!r}; known: {sorted(TYPES)}")
                if t != 5:
                    types.add(t)
    return gas, types


_CACHE = OrderedDict()


def load_for(path, view, specs=(), keep=2):
    """Snapshot dict with what the view and the overlays need (PartType0 datasets, black holes, particle types); a small LRU keeps the last loaded ones.
    What only an overlay needs (Velocities, MagneticField, ...) is read as optional, so that a missing dataset gives the overlay's own clear error."""
    import os
    from .io import load_snapshot
    gas_o, types = required_data(specs, view)
    gas_r = required_gas_fields(view)
    gas_o = (gas_o | {"Velocities", "Density", "InternalEnergy", "ParticleIDs", "ElectronAbundance", "MagneticField", "Metallicity"}) - gas_r
    key = (os.path.abspath(path), os.path.getmtime(path))
    got = _CACHE.get(key)
    if got is not None and gas_r | gas_o <= got[0] and types <= got[1]:
        _CACHE.move_to_end(key)
        return got[2]
    if got is not None:
        gas_o, types = (gas_o | got[0]) - gas_r, types | got[1]
    snap = load_snapshot(path, gas_fields=tuple(sorted(gas_r)) + tuple("?" + k for k in sorted(gas_o)), types=tuple(sorted(types)))
    _CACHE[key] = (gas_r | gas_o, types, snap)
    while len(_CACHE) > keep:
        _CACHE.popitem(last=False)
    return snap


# ----------------------------------------------------------------------------- gridded vector fields

def _extent(view):
    h = 0.5 * view.width
    return [view.cx - h, view.cx + h, view.cy - h, view.cy + h]


def vector_grid(snap, view, quantity="v", grid=24, frame=None):
    """The two components of the projected vector on a regular grid, as mass-weighted (or volume-weighted in a slice) maps of the view (`render_map` with field vx/vy or Bx/By
    along the view's horizontal and vertical axes, npix = grid, code units). Returns (U, W) float64 arrays [grid, grid] (row 0 = bottom, NaN where undefined), extent, field names."""
    if quantity not in ("v", "B"):
        raise ValueError("quantity must be v (velocity) or B (magnetic field)")
    if not 8 <= int(grid) <= 512:
        raise ValueError("grid must be between 8 and 512")
    if quantity == "B" and "MagneticField" not in snap:
        raise ValueError("the snapshot has no MagneticField dataset (PartType0), so there is no B overlay")
    if quantity == "v" and "Velocities" not in snap:
        raise ValueError("the snapshot was loaded without Velocities")
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    ia, ib, _ = AXES[view.axes]
    names = [quantity + "xyz"[ia], quantity + "xyz"[ib]]
    vg = view.with_(npix=int(grid), units="code", field=names[0])
    if vg.mode in ("proj", "slice"):
        from .maps import render_fields
        U, W = render_fields(snap, vg, names, frame)            # both components in one splat pass
    else:
        U, W = (render_map(snap, vg.with_(field=n), frame)[0].astype(np.float64) for n in names)
    return U, W, _extent(view), names


def _quantity_label(quantity, snap, view, value):
    from .figures import unit_names
    if quantity == "B":
        from .fields import magnetic_unit_gauss
        b, _ = magnetic_unit_gauss(snap, view)
        g = value * b
        return f"{g * 1e6:.3g} uG" if g < 1e-3 else f"{g:.3g} G"
    v = unit_names(snap.get("header")).get("V", "V")
    return f"{value:.3g} {v}"


def _head_pack(o):
    return {"color": color_rgb(o["color"]), "linewidth": o["lw"], "alpha": o["alpha"], "halo": o["halo"]}


def vectors(snap, frame, view, quantity="v", **opts):
    """Arrow glyphs: one per grid cell of the gridded field (see `vector_grid`), centred on the cell centre. Arrow length: `arrows=prop`: |v| / ref * len * cell, with ref the
    magnitude drawn as `len` cell spacings (option `ref`, default the 99th percentile of the magnitudes drawn); `arrows=unit`: always `len` * cell. Returns arrow items + a legend."""
    o = _opts("vectors", opts)
    if o["arrows"] not in ("prop", "unit"):
        raise ValueError("arrows must be prop or unit")
    U, W, ext, names = vector_grid(snap, view, quantity, o["grid"], frame)
    n = U.shape[0]
    cell = (ext[1] - ext[0]) / n
    mag = np.hypot(U, W)
    ok = np.isfinite(mag) & (mag > 0) & (mag >= o["min"])
    if not ok.any():
        return [{"kind": "note", "text": f"vectors:{quantity}: no cell has a defined field"}]
    ref = o["ref"] if o["ref"] > 0 else float(np.percentile(mag[ok], 99))
    jj, ii = np.nonzero(ok)
    items = []
    for j, i in zip(jj, ii):
        m = mag[j, i]
        mm = min(m, o["max"]) if o["max"] > 0 else m
        L = o["len"] * cell * (mm / ref if o["arrows"] == "prop" else 1.0)
        ux, uy = U[j, i] / m, W[j, i] / m
        items.append({"kind": "arrow", "x": _f(ext[0] + (i + 0.5) * cell - 0.5 * L * ux), "y": _f(ext[2] + (j + 0.5) * cell - 0.5 * L * uy), "dx": _f(L * ux), "dy": _f(L * uy),
                      "magnitude": _f(m), "head": o["head"], **_head_pack(o), "overlay": "vectors:" + quantity})
    if o["legend"]:
        L = o["len"] * cell
        txt = (f"{_quantity_label(quantity, snap, view, ref)}" if o["arrows"] == "prop" else f"{quantity}: unit length")
        items.append({"kind": "legend", "overlay": "vectors:" + quantity, "text": txt, "length": _f(L), "value": _f(ref) if o["arrows"] == "prop" else None, "color": color_rgb(o["color"]),
                      "head": o["head"], "linewidth": o["lw"]})
    return items


def _sampler(U, W, ext):
    """bilinear sampler on cell centres, clamped at the window edge; returns None where any of the four cells is undefined. Python lists: the integrator calls it a lot."""
    n = U.shape[0]
    cell = (ext[1] - ext[0]) / n
    Ul, Wl = U.tolist(), W.tolist()
    x0, y0 = ext[0], ext[2]

    def sample(x, y):
        fx, fy = (x - x0) / cell - 0.5, (y - y0) / cell - 0.5
        i, j = int(math.floor(fx)), int(math.floor(fy))
        tx, ty = fx - i, fy - j
        i0, i1 = min(max(i, 0), n - 1), min(max(i + 1, 0), n - 1)
        j0, j1 = min(max(j, 0), n - 1), min(max(j + 1, 0), n - 1)
        a, b, c, d = Ul[j0][i0], Ul[j0][i1], Ul[j1][i0], Ul[j1][i1]
        e, f, g, h = Wl[j0][i0], Wl[j0][i1], Wl[j1][i0], Wl[j1][i1]
        u = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
        w = (e * (1 - tx) + f * tx) * (1 - ty) + (g * (1 - tx) + h * tx) * ty
        if u != u or w != w:
            return None
        return u, w
    return sample


def integrate_streamline(U, W, ext, seed, step, maxlen, stag, occupied=None, cell_of=None, own=None, direction=1.0):
    """One direction of an RK4 integration of dx/ds = direction * v / |v| (arc length s) from `seed`. Stops: outside the window, undefined cell, |v| < stag (stagnation),
    length > maxlen, entering a cell of `occupied` that this line has not marked itself, or closing the loop on the seed. Returns (points [[x, y]...] without the seed,
    closed flag)."""
    sample = _sampler(U, W, ext)

    def unit(x, y):
        v = sample(x, y)
        if v is None:
            return None
        m = math.hypot(*v)
        if m <= stag or m == 0.0:
            return None
        return direction * v[0] / m, direction * v[1] / m
    x, y = seed
    pts, length, closed = [], 0.0, False
    while length < maxlen:
        k1 = unit(x, y)
        if k1 is None:
            break
        k2 = unit(x + 0.5 * step * k1[0], y + 0.5 * step * k1[1])
        k3 = k2 and unit(x + 0.5 * step * k2[0], y + 0.5 * step * k2[1])
        k4 = k3 and unit(x + step * k3[0], y + step * k3[1])
        if k2 is None or k3 is None or k4 is None:
            break
        nx_ = x + step / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ny_ = y + step / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        if not (ext[0] <= nx_ <= ext[1] and ext[2] <= ny_ <= ext[3]):
            break
        if cell_of is not None:
            c = cell_of(nx_, ny_)
            if occupied is not None and c in occupied and c not in own:
                break
            own.add(c)
        x, y = nx_, ny_
        pts.append([x, y])
        length += step
        if len(pts) > 8 and math.hypot(x - seed[0], y - seed[1]) < 0.6 * step:
            closed = True
            break
    return pts, closed


def streamline_paths(U, W, ext, seeds=14, density=28, maxlen=0.6, step=0.25, stag=0.02, min_mag=0.0, minlen=0.05):
    """Streamlines of the gridded field (U, W): seeds on a regular `seeds` x `seeds` grid (visited from the window centre outwards); a seed is skipped when its occupancy cell
    (`density` cells per axis) already holds a line, and a line stops where it meets another line's cells (even spacing). Returns a list of (points, magnitudes)."""
    n = U.shape[0]
    width = ext[1] - ext[0]
    cell = width / n
    mag = np.hypot(U, W)
    fin = mag[np.isfinite(mag)]
    if fin.size == 0:
        return []
    p99 = float(np.percentile(fin, 99))
    thr = min_mag if min_mag > 0 else stag * p99
    ds = step * cell
    sample = _sampler(U, W, ext)
    dens = max(int(density), 2)

    def cell_of(x, y):
        return (min(int((x - ext[0]) / width * dens), dens - 1), min(int((y - ext[2]) / width * dens), dens - 1))
    gx = ext[0] + (np.arange(seeds) + 0.5) * width / seeds
    gy = ext[2] + (np.arange(seeds) + 0.5) * width / seeds
    sx, sy = np.meshgrid(gx, gy)
    sx, sy = sx.ravel(), sy.ravel()
    order = np.argsort(np.hypot(sx - 0.5 * (ext[0] + ext[1]), sy - 0.5 * (ext[2] + ext[3])), kind="stable")
    occupied, lines = set(), []
    for k in order:
        seed = (float(sx[k]), float(sy[k]))
        c0 = cell_of(*seed)
        if c0 in occupied or sample(*seed) is None:
            continue
        own = {c0}
        fwd, closed = integrate_streamline(U, W, ext, seed, ds, maxlen * width, thr, occupied, cell_of, own, 1.0)
        bwd = [] if closed else integrate_streamline(U, W, ext, seed, ds, maxlen * width, thr, occupied, cell_of, own, -1.0)[0]
        pts = bwd[::-1] + [list(seed)] + fwd
        if len(pts) * ds < minlen * width or len(pts) < 3:
            continue
        occupied |= own
        lines.append(pts)
    out = []
    for pts in lines:
        m = [math.hypot(*sample(x, y)) if sample(x, y) else 0.0 for x, y in pts]
        out.append((pts, m))
    return out


def streamlines(snap, frame, view, quantity="v", **opts):
    """Streamline polylines of the gridded projected field (`vector_grid`), see `streamline_paths`. One polyline item per line, in flow order, with an arrow head at the middle point."""
    o = _opts("streamlines", opts)
    U, W, ext, names = vector_grid(snap, view, quantity, o["grid"], frame)
    paths = streamline_paths(U, W, ext, o["seeds"], o["density"], o["maxlen"], o["step"], o["stag"], o["min"], o["minlen"])
    items = []
    for pts, m in paths:
        items.append({"kind": "polyline", "points": [[_f(x), _f(y)] for x, y in pts], "magnitude": _rl(m, 4), "head": (len(pts) // 2) if o["heads"] else None,
                      **_head_pack(o), "overlay": "streamlines:" + quantity})
    return items or [{"kind": "note", "text": f"streamlines:{quantity}: no line could be drawn (field undefined or stagnant)"}]


# ----------------------------------------------------------------------------- particles by type

def type_points(snap, t):
    """(positions (N,3) in the snapshot frame, ids or None, masses or None) of particle type t; type 5 = the black holes."""
    if t == 5:
        if "bh_pos" not in snap or len(snap["bh_pos"]) == 0:
            return None
        m = snap["bh_mdyn"] if "bh_mdyn" in snap else snap["bh_mass"]
        return np.asarray(snap["bh_pos"], dtype=np.float64), np.asarray(snap["bh_id"]), np.asarray(m)
    d = (snap.get("pt") or {}).get(t)
    if d is None:
        return None
    return d["pos"], d.get("id"), d.get("mass")


def particles(snap, frame, view, *types, **opts):
    """Points by type (bare arguments: bh, star, dm, ...; default bh). Only the particles inside the window (and within +- depth of z0 if given) count; more than `max` of them
    -> a deterministic random subsample (numpy default_rng(seed), sorted by index). Per-type options: `star.color=gold`. One `points` item per type present."""
    names = list(types) or ["bh"]
    base = _opts("particles", {k: v for k, v in opts.items() if "." not in k})
    items = []
    ia, ib, il = AXES[view.axes]
    ext = _extent(view)
    for nm in names:
        t = TYPES.get(nm.lower())
        if t is None:
            raise ValueError(f"unknown particle type {nm!r}; known: {sorted(TYPES)}")
        po = {k.split(".", 1)[1]: v for k, v in opts.items() if k.split(".", 1)[0].lower() == nm.lower() and "." in k}
        for k in po:
            if k not in ("color", "size", "style", "max"):
                raise ValueError(f"particles: unknown per-type option {nm}.{k}")
        dcol, dsize, dstyle = TYPE_STYLE[t]
        color = po.get("color") or base["color"] or dcol
        size = float(po.get("size") or base["size"] or dsize)
        style = po.get("style") or base["style"] or dstyle
        cap = int(po.get("max", base["max"]))
        got = type_points(snap, t)
        if got is None:
            items.append({"kind": "note", "text": f"particles:{nm}: the snapshot has no particles of this type (PartType{t}) or they were not loaded"})
            continue
        pos, ids, mass = got
        p = frame.pos(pos)
        keep = (p[:, ia] >= ext[0]) & (p[:, ia] <= ext[1]) & (p[:, ib] >= ext[2]) & (p[:, ib] <= ext[3])
        if base["depth"] > 0:
            keep &= np.abs(p[:, il] - view.z0) <= base["depth"]
        idx = np.nonzero(keep)[0]
        n_total = len(idx)
        if n_total > cap > 0:
            idx = np.sort(np.random.default_rng(int(base["seed"])).choice(idx, size=cap, replace=False))
        it = {"kind": "points", "type": TYPE_NAME.get(t, nm), "x": _rl(p[idx, ia]), "y": _rl(p[idx, ib]), "depth": _rl(p[idx, il]), "size_px": size, "style": style,
              "color": color_rgb(color), "alpha": 1.0, "n_total": int(n_total), "n_drawn": int(len(idx)), "overlay": "particles:" + nm}
        if ids is not None:
            it["id"] = [int(v) for v in np.asarray(ids)[idx]]
        if mass is not None:
            it["mass"] = _rl(np.asarray(mass)[idx])
        items.append(it)
    return items


# ----------------------------------------------------------------------------- contours

# marching squares: corner order a (i, j) b (i+1, j) c (i+1, j+1) d (i, j+1); edges 0 bottom a-b, 1 right b-c, 2 top d-c, 3 left a-d
_MS = {1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),), 6: ((0, 2),), 7: ((3, 2),), 8: ((3, 2),), 9: ((0, 2),), 11: ((1, 2),), 12: ((3, 1),), 13: ((0, 1),), 14: ((3, 0),)}
_MS_SADDLE = {5: (((0, 1), (3, 2)), ((3, 0), (1, 2))), 10: (((3, 0), (1, 2)), ((0, 1), (3, 2)))}      # (centre above, centre below)


def contour_paths(Z, level, x0, dx, y0, dy):
    """Polylines of the level set Z = level of a 2D array (Z[j, i], i horizontal) on pixel centres x = x0 + i dx, y = y0 + j dy, by marching squares with linear interpolation along
    the cell edges. Cells with a NaN corner are skipped. Returns a list of point lists [(x, y), ...] (closed lines repeat their first point)."""
    ny, nx = Z.shape
    ok = np.isfinite(Z)
    A = np.where(ok, Z > level, False)
    cellok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, 1:] & ok[1:, :-1]
    case = (A[:-1, :-1] * 1 + A[:-1, 1:] * 2 + A[1:, 1:] * 4 + A[1:, :-1] * 8).astype(np.int64)
    case = np.where(cellok, case, 0)
    jj, ii = np.nonzero((case > 0) & (case < 15))

    def node(e, j, i):          # edge e of cell (j, i) -> key of the shared edge: H(j, i) horizontal edge at row j from column i; V(j, i) vertical edge at column i from row j
        return ("H", j, i) if e == 0 else (("V", j, i + 1) if e == 1 else (("H", j + 1, i) if e == 2 else ("V", j, i)))

    adj = {}
    for j, i in zip(jj.tolist(), ii.tolist()):
        c = int(case[j, i])
        if c in _MS_SADDLE:
            centre = 0.25 * (Z[j, i] + Z[j, i + 1] + Z[j + 1, i + 1] + Z[j + 1, i]) > level
            segs = _MS_SADDLE[c][0 if centre else 1]
        else:
            segs = _MS[c]
        for e1, e2 in segs:
            a, b = node(e1, j, i), node(e2, j, i)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

    def point(k):
        t, j, i = k
        if t == "H":
            v0, v1 = Z[j, i], Z[j, i + 1]
            f = (level - v0) / (v1 - v0)
            return (x0 + (i + f) * dx, y0 + j * dy)
        v0, v1 = Z[j, i], Z[j + 1, i]
        f = (level - v0) / (v1 - v0)
        return (x0 + i * dx, y0 + (j + f) * dy)

    paths, seen = [], set()
    ends = [k for k, nb in adj.items() if len(nb) == 1]          # open lines start at their ends, then the remaining closed loops
    for start in ends + [k for k in adj if k not in ends]:
        if start in seen:
            continue
        line, cur, prev = [start], start, None
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and (n not in seen or (n == start and len(line) > 2))]
            if not nxt:
                break
            n = nxt[0]
            line.append(n)
            if n == start:
                break
            seen.add(n)
            prev, cur = cur, n
        if len(line) >= 2:
            paths.append([point(k) for k in line])
    return paths


def _levels(o, Z):
    v = Z[np.isfinite(Z)]
    spec = str(o["levels"])
    if spec not in ("log", "lin"):
        try:
            return sorted(float(t) for t in spec.split(";") if t.strip())
        except ValueError:
            raise ValueError(f"contours: levels must be log, lin or numbers joined by ';', not {spec!r}") from None
    if v.size == 0 or o["n"] < 1:
        return []
    if spec == "log":
        vp = v[v > 0]
        if vp.size == 0:
            raise ValueError("contours: levels=log needs positive values (use levels=lin for a signed map)")
        top = o["top"] if o["top"] > 0 else float(np.percentile(vp, 99.5))
        return sorted(top * 10.0 ** (-o["step"] * k) for k in range(int(o["n"])))
    lo, hi = np.percentile(v, [5, 99.5])
    return [float(x) for x in np.linspace(lo, hi, int(o["n"]) + 2)[1:-1]] if hi > lo else []


def contours(snap, frame, view, field=None, **opts):
    """Contour polylines of the map of `field` (default the view's field) in the view's window: one polyline item per line. The map is the raw one (View.post = none) unless `post=` is given."""
    from . import transforms
    o = _opts("contours", opts)
    v = view.with_(field=field or view.field, post=o["post"])
    img, meta = _MAP_SOURCE(v) if _MAP_SOURCE is not None else render_map(snap, v, frame)
    Z = np.asarray(img, dtype=np.float64)
    if o["smooth"] > 0:
        Z = transforms.gaussian_nan(Z, o["smooth"])
    ext = meta["extent"]
    n = Z.shape[1]
    dx, dy = (ext[1] - ext[0]) / n, (ext[3] - ext[2]) / Z.shape[0]
    levels = _levels(o, Z)
    items, used = [], []
    for L in levels:
        paths = contour_paths(Z, L, ext[0] + 0.5 * dx, dx, ext[2] + 0.5 * dy, dy)
        for pts in paths:
            if len(pts) >= 3:
                items.append({"kind": "polyline", "points": [[_f(x), _f(y)] for x, y in pts], "magnitude": [float(f"{L:.4g}")] * len(pts), "head": None, **_head_pack(o), "overlay": "contours"})
        if paths:
            used.append(L)
    if not items:
        return [{"kind": "note", "text": f"contours:{v.field}: no level crosses the map"}]
    if o["legend"]:
        what = {"log": f", {o['step']:g} dex", "lin": ", linear"}.get(str(o["levels"]), "")
        txt = f"contours {v.field}: {used[0]:.2g} .. {used[-1]:.2g}{what}"
        items.append({"kind": "legend", "overlay": "contours", "text": txt, "length": 0.0, "value": None, "color": color_rgb(o["color"]), "head": 0.0, "linewidth": o["lw"]})
    return items


# ----------------------------------------------------------------------------- the gas disc: its plane and a fitted constant-aspect-ratio shape

def _hull(pts):
    """convex hull (monotone chain) of 2D points, counter-clockwise, closed."""
    pts = sorted(set(map(tuple, np.round(np.asarray(pts, dtype=np.float64), 12))))
    if len(pts) < 3:
        return [list(p) for p in pts]
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(pts), half(pts[::-1])
    h = lo[:-1] + up[:-1]
    return [list(p) for p in h + [h[0]]]


def disc(snap, frame, view, **opts):
    """The gas disc around the frame's centre: its plane (from the angular momentum of the gas within R) as a semi-transparent disc, and a
    constant-aspect-ratio fit (thickness growing linearly with radius: a cylinder minus two cones, H/R = the mass-weighted rms of z/R_cyl)
    drawn as its two rims at +-H and its outline; the legend gives R, H/R and the inclination to the line of sight.
    Options: R (0 = the radius holding `frac` of the gas mass within the window), frac, color, fill (alpha), lw, shape, legend."""
    o = _opts("disc", opts)
    if "Velocities" not in snap:
        raise ValueError("disc: the snapshot was loaded without Velocities")
    P = frame.pos(snap["Coordinates"])
    V = frame.vel(snap["Velocities"])
    m = np.asarray(snap["Masses"], dtype=np.float64)
    r = np.sqrt((P ** 2).sum(1))
    R = float(o["R"])
    if not R > 0:
        inw = r < 0.5 * view.width
        if not inw.any():
            return [{"kind": "note", "text": "disc: no gas in the window"}]
        order = np.argsort(r[inw])
        cm = np.cumsum(m[inw][order])
        R = float(r[inw][order][min(len(cm) - 1, np.searchsorted(cm, o["frac"] * cm[-1]))])
    sel = r < R
    if sel.sum() < 10:
        return [{"kind": "note", "text": f"disc: fewer than 10 gas particles within R = {R:.3g}"}]
    L = (m[sel, None] * np.cross(P[sel], V[sel])).sum(0)
    if not np.linalg.norm(L) > 0:
        return [{"kind": "note", "text": "disc: the gas within R has no net angular momentum"}]
    n = L / np.linalg.norm(L)
    z = P[sel] @ n
    rc = np.sqrt(np.maximum((P[sel] ** 2).sum(1) - z ** 2, 0.0))
    ok = rc > 0.05 * R
    a = float(np.sqrt((m[sel][ok] * (z[ok] / rc[ok]) ** 2).sum() / m[sel][ok].sum())) if ok.any() else 0.0      # H/R: constant aspect ratio
    e1 = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    th = np.linspace(0.0, 2 * np.pi, 97)
    ring = R * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
    ia, ib, il = AXES[view.axes]
    col = color_rgb(o["color"])
    proj = lambda Q: [[float(q[ia]), float(q[ib])] for q in Q]
    items = [{"kind": "polygon", "points": proj(ring), "color": col, "alpha": float(o["fill"]), "overlay": "disc"},
             {"kind": "polyline", "points": proj(ring), "magnitude": [0.0] * len(ring), "head": None, "color": col, "linewidth": o["lw"], "alpha": 0.9, "halo": 0.0,
              "overlay": "disc"}]
    if o["shape"] and a > 0:
        top, bot = ring + a * R * n, ring - a * R * n
        for rim in (top, bot):
            items.append({"kind": "polyline", "points": proj(rim), "magnitude": [0.0] * len(rim), "head": None, "color": col, "linewidth": o["lw"], "alpha": 0.8,
                          "halo": 0.0, "dash": [5, 4], "overlay": "disc"})
        hull = _hull(proj(np.concatenate([top, bot])))
        items.append({"kind": "polyline", "points": hull, "magnitude": [0.0] * len(hull), "head": None, "color": col, "linewidth": o["lw"] + 0.4, "alpha": 0.9,
                      "halo": 0.0, "overlay": "disc"})
    if o["legend"]:
        from .units import LENGTH, Units
        lu, lf = Units.from_header(snap.get("header", {})).quantity_unit(LENGTH, view.units)
        inc = float(np.degrees(np.arccos(min(1.0, abs(n[il])))))
        items.append({"kind": "legend", "overlay": "disc", "text": f"disc: R = {R * lf:.3g} {lu}, H/R = {a:.2f}, inclination {inc:.0f} deg", "length": 0.0,
                      "color": col, "loc": "lower left"})
    return items


# ----------------------------------------------------------------------------- registry and compute

for _n, _fn in (("vectors", vectors), ("streamlines", streamlines), ("particles", particles), ("contours", contours), ("disc", disc)):
    plugins.register_overlay(_n, _fn)
BUILTIN_OVERLAYS = ("vectors", "streamlines", "particles", "contours", "disc")


def _call(fn, snap, frame, view, args, opts):
    """Call a registered overlay: the core ones take (quantity | types, **opts); a plugin function of the documented 3-argument signature takes no options."""
    sig = _inspect.signature(fn)
    ps = list(sig.parameters.values())
    var_kw = any(p.kind == p.VAR_KEYWORD for p in ps)
    var_pos = any(p.kind == p.VAR_POSITIONAL for p in ps)
    named = {p.name for p in ps}
    if (args and not var_pos and len(ps) <= 3) or (opts and not var_kw and not (set(opts) <= named)):
        raise ValueError(f"overlay function {getattr(fn, '__name__', fn)!r} takes (snap, frame, view) only; arguments {args} / options {sorted(opts)} are not accepted")
    return list(fn(snap, frame, view, *args, **opts))


_MAP_SOURCE = None     # set by compute(get_map=...): where contours take their map (e.g. the server's map store, which already has the pane's map)


def compute(snap, view, specs, frame=None, get_map=None):
    """Items of all overlay specs for the loaded snapshot and the view: {"extent", "npix", "items", "notes"} (plain data, see the module docstring).
    get_map: optional callable(view) -> (img, meta) that contours use instead of rendering the map again (the server passes its map store)."""
    global _MAP_SOURCE
    _MAP_SOURCE = get_map
    try:
        return _compute(snap, view, specs, frame)
    finally:
        _MAP_SOURCE = None


def _compute(snap, view, specs, frame=None):
    plugins.load_plugins()
    view.validate()
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    items, notes = [], []
    for sp in map(parse_spec, specs or ()):
        fn = plugins.OVERLAYS.get(sp["name"])
        if fn is None:
            raise ValueError(f"unknown overlay {sp['name']!r}; known: {sorted(plugins.OVERLAYS)}")
        if sp["name"] in ("vectors", "streamlines"):
            args = sp["args"][:1] or ["v"]
            if len(sp["args"]) > 1:
                raise ValueError(f"{sp['name']}: one quantity (v or B), got {sp['args']}")
        else:
            args = sp["args"]
        for d in _call(fn, snap, frame, view, args, sp["opts"]):
            if d.get("kind") == "note":
                notes.append(d["text"])
                continue
            d.setdefault("overlay", sp["name"])
            items.append(d)
    return {"extent": _extent(view), "npix": view.npix, "items": items, "notes": notes}


def compute_for_path(path, view, specs):
    return compute(load_for(path, view, specs), view, specs)


def describe():
    lines = []
    for n, d in DEFAULTS.items():
        lines.append(f"{n}: " + ", ".join(f"{k}={v}" for k, v in d.items()) + f"\n    {HELP[n]}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- click-to-inspect

def inspect(snap, view, x, y, frame=None, limit=20):
    """Gas particles contributing at the map position (x, y) (window coordinates). Slice: the particles whose kernel reaches the point (x, y, z0), i.e. 3D distance < H, nearest
    first; contribution = w W(r, H) normalised over them. Projection: the particles whose projected kernel covers (x, y) (distance < H in the plane), ranked by their
    contribution w f(s, H) (column integral of W) normalised over all of them. w = mass, or mass / density for the 'volume' weight (slice default); for the plain `rho` field w = mass.
    Each entry: index (row in PartType0), id (if loaded), pos (frame coordinates), H, rho, u, T (if derivable), vel (frame), mass, weight, kernel, contribution (fraction),
    circle {x, y, r} (smoothing circle in window coordinates; in a slice the cross-section radius sqrt(H^2 - dz^2) at the plane, `H` is the full support radius).
    Returns {"x", "y", "mode", "z0", "pixel": [i, j], "n_total", "particles": [first `limit`], "contribution_listed"}."""
    view.validate()
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    ia, ib, il = AXES[view.axes]
    pos = frame.pos(snap["Coordinates"])
    h = np.asarray(snap["SmoothingLength"], dtype=np.float64) * view.hscale
    a, b, l = pos[:, ia], pos[:, ib], pos[:, il]
    dx, dy = a - x, b - y
    if view.mode == "slice":
        dz = l - view.z0
        r2 = dx * dx + dy * dy + dz * dz
    else:
        dz = np.zeros_like(a)
        r2 = dx * dx + dy * dy
    idx = np.nonzero(r2 < h * h)[0]
    weight_kind = resolve_weight(view)
    mass = np.asarray(snap["Masses"], dtype=np.float64)[idx]
    if view.field == "rho" or weight_kind == "mass":
        w = mass
    else:
        if "Density" not in snap:
            raise ValueError("inspect: volume weighting needs the Density dataset")
        w = mass / np.asarray(snap["Density"], dtype=np.float64)[idx]
    hh = h[idx]
    q2 = r2[idx] / (hh * hh)
    t2d, t3d = tables()
    grid = np.linspace(0.0, 1.0, len(t2d))
    if view.mode == "slice":
        kern = np.interp(q2, grid, t3d) / hh ** 3
    else:
        kern = np.interp(q2, grid, t2d) / hh ** 2
    c = w * kern
    tot = c.sum()
    order = np.argsort(-c if view.mode == "proj" else r2[idx], kind="stable")
    sel = idx[order][:max(int(limit), 0)]
    sub_i = order[:len(sel)]
    half = 0.5 * view.width
    npix = view.npix
    out = {"x": float(x), "y": float(y), "mode": view.mode, "z0": float(view.z0), "n_total": int(len(idx)),
           "pixel": [int(math.floor((x - (view.cx - half)) / view.width * npix)), int(math.floor((y - (view.cy - half)) / view.width * npix))],
           "weight": weight_kind if view.field != "rho" else "mass", "particles": []}
    if len(sel) == 0:
        out["contribution_listed"] = 0.0
        return out
    # per-particle quantities of the listed particles (a small snapshot dict so that the derived fields run unchanged)
    n_gas = len(snap["Coordinates"])
    sub = {k: (v[sel] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n_gas else v) for k, v in snap.items()}
    vel = frame.vel(snap["Velocities"][sel]) if "Velocities" in snap else None
    T = None
    try:
        from .fields import f_temperature
        T = np.asarray(f_temperature(FieldContext(sub, frame, view)), dtype=np.float64)
    except (KeyError, ValueError, TypeError):
        T = None
    for k, (g, p) in enumerate(zip(sel, sub_i)):
        e = {"index": int(g), "id": int(snap["ParticleIDs"][g]) if "ParticleIDs" in snap else None, "pos": _rl(pos[g], 7), "H": _f(h[g]),
             "rho": _f(float(snap["Density"][g])) if "Density" in snap else None, "u": _f(float(snap["InternalEnergy"][g])) if "InternalEnergy" in snap else None,
             "T": _f(float(T[k])) if T is not None and np.isfinite(T[k]) else None, "vel": _rl(vel[k], 7) if vel is not None else None, "mass": _f(float(snap["Masses"][g])),
             "weight": _f(float(w[p])), "kernel": _f(float(kern[p])), "contribution": float(c[p] / tot) if tot > 0 else 0.0, "rank": k + 1}
        rr = math.sqrt(max(h[g] ** 2 - dz[g] ** 2, 0.0))
        e["circle"] = {"x": _f(float(a[g])), "y": _f(float(b[g])), "r": _f(rr)}
        out["particles"].append(e)
    out["contribution_listed"] = float(sum(e["contribution"] for e in out["particles"]))
    return out


def inspect_items(result, color="white"):
    """Overlay items (circles) for the smoothing circles of an `inspect` result, to draw with `draw_items`."""
    col = color_rgb(color)
    return [{"kind": "circle", "x": e["circle"]["x"], "y": e["circle"]["y"], "r": e["circle"]["r"], "fill": False, "color": col, "linewidth": 1.0, "alpha": 0.9,
             "overlay": "inspect"} for e in result["particles"]]


# ----------------------------------------------------------------------------- rasteriser

@_njit(cache=True)
def _segs_cov(cov, bx0, by0, segs, lw):
    """max-accumulate the coverage of the segments segs[i] = (x0, y0, x1, y1) (pixels) into cov (the array of the box starting at bx0, by0):
    the per-segment formula of _Canvas.seg_cov, compiled (a streamline overlay has thousands of segments per frame)."""
    ny, nx = cov.shape
    pad = 0.5 * lw + 1.5
    for i in range(segs.shape[0]):
        x0, y0, x1, y1 = segs[i, 0], segs[i, 1], segs[i, 2], segs[i, 3]
        sx0 = max(int(math.floor(min(x0, x1) - pad)), bx0)
        sy0 = max(int(math.floor(min(y0, y1) - pad)), by0)
        sx1 = min(int(math.ceil(max(x0, x1) + pad)) + 1, bx0 + nx)
        sy1 = min(int(math.ceil(max(y0, y1) + pad)) + 1, by0 + ny)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        for yy in range(sy0, sy1):
            Y = yy + 0.5
            for xx in range(sx0, sx1):
                X = xx + 0.5
                t = 0.0
                if L2 > 0:
                    t = ((X - x0) * dx + (Y - y0) * dy) / L2
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                ex, ey = X - x0 - t * dx, Y - y0 - t * dy
                c = 0.5 * lw + 0.5 - math.hypot(ex, ey)
                c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
                if c > cov[yy - by0, xx - bx0]:
                    cov[yy - by0, xx - bx0] = c


class _Canvas:
    """Draws coverage-based anti-aliased primitives into an (ny, nx, 3) uint8 array (row 0 at the top) from window coordinates."""
    def __init__(self, rgb, extent):
        self.rgb = rgb
        self.ny, self.nx = rgb.shape[:2]
        self.x0, self.y0 = extent[0], extent[2]
        self.sx, self.sy = self.nx / (extent[1] - extent[0]), self.ny / (extent[3] - extent[2])

    def px(self, x, y):
        return (np.asarray(x, dtype=np.float64) - self.x0) * self.sx, self.ny - (np.asarray(y, dtype=np.float64) - self.y0) * self.sy

    def _box(self, xs, ys, pad):
        x0, x1 = int(math.floor(min(xs) - pad)), int(math.ceil(max(xs) + pad)) + 1
        y0, y1 = int(math.floor(min(ys) - pad)), int(math.ceil(max(ys) + pad)) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, self.nx), min(y1, self.ny)
        return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None

    def blend(self, box, cov, color, alpha):
        x0, y0, x1, y1 = box
        a = (np.clip(cov, 0.0, 1.0) * alpha)[..., None]
        sub = self.rgb[y0:y1, x0:x1]
        sub[...] = (sub * (1 - a) + np.asarray(color, dtype=np.float64) * a + 0.5).astype(np.uint8)

    @staticmethod
    def _grid(box):
        x0, y0, x1, y1 = box
        return np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)

    def seg_cov(self, box, cov, p0, p1, lw):
        """max-accumulate the coverage of one segment into `cov` (the array of `box`), looking only at the segment's own bounding box."""
        pad = 0.5 * lw + 1.5
        sx0, sy0 = max(int(math.floor(min(p0[0], p1[0]) - pad)), box[0]), max(int(math.floor(min(p0[1], p1[1]) - pad)), box[1])
        sx1, sy1 = min(int(math.ceil(max(p0[0], p1[0]) + pad)) + 1, box[2]), min(int(math.ceil(max(p0[1], p1[1]) + pad)) + 1, box[3])
        if sx0 >= sx1 or sy0 >= sy1:
            return
        X, Y = self._grid((sx0, sy0, sx1, sy1))
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L2 = dx * dx + dy * dy
        t = np.clip(((X - p0[0]) * dx + (Y - p0[1]) * dy) / L2, 0, 1) if L2 > 0 else np.zeros_like(X)
        d = np.hypot(X - p0[0] - t * dx, Y - p0[1] - t * dy)
        sub = cov[sy0 - box[1]:sy1 - box[1], sx0 - box[0]:sx1 - box[0]]
        np.maximum(sub, np.clip(0.5 * lw + 0.5 - d, 0, 1), out=sub)

    def stroke(self, segs, lw, color, alpha, halo):
        """segs: list of ((x0, y0), (x1, y1)) in pixels, drawn as ONE shape (coverage max, so joints do not darken)."""
        if not segs:
            return
        xs = [p[0] for s in segs for p in s]
        ys = [p[1] for s in segs for p in s]
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            return
        for width, col, al in ((lw + 2 * halo, (0, 0, 0), 0.55 * alpha), (lw, color, alpha)) if halo > 0 else ((lw, color, alpha),):
            box = self._box(xs, ys, width / 2 + 1.5)
            if box is None:
                return
            cov = np.zeros((box[3] - box[1], box[2] - box[0]))
            _segs_cov(cov, box[0], box[1], np.array([(a[0], a[1], b[0], b[1]) for a, b in segs], dtype=np.float64).reshape(-1, 4), float(width))
            self.blend(box, cov, col, al)

    def head_segs(self, tip, direction, length_px):
        ang = math.radians(25.0)
        ux, uy = direction
        segs = []
        for sgn in (-1, 1):
            c, s = math.cos(sgn * ang), math.sin(sgn * ang)
            bx, by = -(ux * c - uy * s), -(ux * s + uy * c)
            segs.append((tip, (tip[0] + bx * length_px, tip[1] + by * length_px)))
        return segs

    def disc(self, cx, cy, r, color, alpha, fill=True, lw=1.0, halo=0.0):
        box = self._box([cx], [cy], r + lw + 1.5 + halo)
        if box is None:
            return
        X, Y = self._grid(box)
        d = np.hypot(X - cx, Y - cy)
        if halo > 0:
            self.blend(box, (np.clip(r + 0.5 * lw + halo + 0.5 - d, 0, 1) if fill else np.clip(0.5 * (lw + 2 * halo) + 0.5 - np.abs(d - r), 0, 1)), (0, 0, 0), 0.5 * alpha)
        cov = np.clip(r + 0.5 - d, 0, 1) if fill else np.clip(0.5 * lw + 0.5 - np.abs(d - r), 0, 1)
        self.blend(box, cov, color, alpha)


def _dashed(segs, dash):
    """the segments cut into dashes of `dash` = [on, off] pixels along the whole line (no dash: unchanged)."""
    if not dash:
        return segs
    on, off = max(1.0, float(dash[0])), max(1.0, float(dash[1]))
    out, phase = [], 0.0                    # phase: distance into the current on+off period
    for (ax, ay), (bx, by) in segs:
        L = math.hypot(bx - ax, by - ay)
        t = 0.0
        while t < L:
            if phase < on:
                step = min(on - phase, L - t)
                f0, f1 = t / L, (t + step) / L
                out.append(((ax + f0 * (bx - ax), ay + f0 * (by - ay)), (ax + f1 * (bx - ax), ay + f1 * (by - ay))))
            else:
                step = min(on + off - phase, L - t)
            t += step
            phase = (phase + step) % (on + off)
    return out


LINE_REF_PX = 768.0       # the viewer's canvas: overlay line widths, halos, dashes and point sizes are in pixels of a picture this wide


def _scaled(it, g):
    out = dict(it)
    for k in ("linewidth", "halo", "size_px", "dx_px", "dy_px"):
        if out.get(k) is not None:
            out[k] = float(out[k]) * g
    if out.get("dash"):
        out["dash"] = [float(v) * g for v in out["dash"]]
    return out


def draw_items(rgb, items, extent, font_scale=1, legend=True):
    """Draw overlay items (see the module docstring) into the (ny, nx, 3) uint8 picture `rgb` (row 0 = top, the window `extent` fills it). Anti-aliased lines, arrow heads,
    discs and rings; legends (reference arrows) go to the top-right corner. Returns rgb (modified in place)."""
    cv = _Canvas(rgb, extent)
    legends = []
    g = rgb.shape[1] / LINE_REF_PX                     # widths are given for a picture of the viewer's size: thicker on bigger pictures (movies, figures)
    if g > 1.0:
        items = [_scaled(it, g) for it in items]
    for it in items:
        k = it.get("kind")
        col, al = it.get("color", [255, 255, 255]), float(it.get("alpha", 1.0))
        if k == "arrow":
            (ax, ay), (bx, by) = [tuple(map(float, cv.px(a, b))) for a, b in ((it["x"], it["y"]), (it["x"] + it["dx"], it["y"] + it["dy"]))]
            L = math.hypot(bx - ax, by - ay)
            if L < 2.0:                                  # too short to show a direction
                continue
            u = ((bx - ax) / L, (by - ay) / L)
            segs = _dashed([((ax, ay), (bx, by))], it.get("dash")) + cv.head_segs((bx, by), u, max(3.0, it.get("head", 0.35) * L) if L > 2 else 0.0)
            cv.stroke(segs, float(it.get("linewidth", 1.0)), col, al, float(it.get("halo", 0.0)))
        elif k == "polyline":
            pts = it["points"]
            if len(pts) < 2:
                continue
            px, py = cv.px([p[0] for p in pts], [p[1] for p in pts])
            P = list(zip(px.tolist(), py.tolist()))
            segs = _dashed(list(zip(P[:-1], P[1:])), it.get("dash"))
            h = it.get("head")
            if h is not None and 0 < h < len(P):
                a, b = P[h - 1], P[h]
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                if L > 0:
                    segs += cv.head_segs(b, ((b[0] - a[0]) / L, (b[1] - a[1]) / L), 5.0 * max(1.0, float(it.get("linewidth", 1.0))))
            cv.stroke(segs, float(it.get("linewidth", 1.0)), col, al, float(it.get("halo", 0.0)))
        elif k == "points":
            px, py = cv.px(it["x"], it["y"])
            r = 0.5 * float(it.get("size_px", 2.0))
            ring = it.get("style") == "ring"
            for x, y in zip(px.tolist(), py.tolist()):
                cv.disc(x, y, r, col, al, fill=not ring, lw=max(1.0, 0.35 * r), halo=1.0 if ring or r > 1.2 else 0.0)
        elif k == "circle":
            x, y = map(float, cv.px(it["x"], it["y"]))
            cv.disc(x, y, float(it["r"]) * cv.sx, col, al, fill=bool(it.get("fill", False)), lw=float(it.get("linewidth", 1.0)), halo=0.0)
        elif k == "legend":
            legends.append(it)
        elif k == "polygon":                             # filled (even-odd), alpha-blended
            px, py = cv.px([p[0] for p in it["points"]], [p[1] for p in it["points"]])
            box = cv._box(px, py, 1.0)
            if box is not None and len(px) >= 3:
                X, Y = cv._grid(box)
                inside = np.zeros(X.shape, bool)
                xs, ys = np.asarray(px), np.asarray(py)
                for i in range(len(xs)):
                    x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[i], ys[i]
                    cross = ((y0 > Y) != (y1 > Y)) & (X < (x1 - x0) * (Y - y0) / ((y1 - y0) if y1 != y0 else 1e-300) + x0)
                    inside ^= cross
                cv.blend(box, inside.astype(np.float64), col, al)
        elif k == "text":
            from .figures import draw_text
            from .bitmapfont import CELL_H
            x, y = map(float, cv.px(it["x"], it["y"]))
            s = max(1, int(round(float(it.get("size_px", 11.0)) / CELL_H)))
            tx, ty = int(round(x + float(it.get("dx_px", 0.0)))), int(round(y - float(it.get("dy_px", 0.0)) - CELL_H * s))
            if 0 <= ty < rgb.shape[0] and 0 <= tx < rgb.shape[1]:
                draw_text(rgb, tx, ty, str(it.get("text", "")), s, tuple(int(v) for v in color_rgb(it.get("color", "white"))))
        elif k is None and "x" in it and "y" in it:
            x, y = map(float, cv.px(it["x"], it["y"]))
            cv.disc(x, y, 4.0, col, al, fill=False, lw=1.5, halo=1.0)
    if legend and legends:
        from .figures import draw_text, OUTLINE, CELL_H, CELL_W
        s = max(1, int(font_scale))
        y = 3 * s
        for it in legends:
            txt = it["text"]
            tw = len(txt) * CELL_W * s
            L = float(it.get("length", 0.0)) * cv.sx
            xr = cv.nx - 3 * s
            if L > 0:
                tip = (xr - tw - 3 * s, y + CELL_H * s * 0.5)
                tail = (tip[0] - L, tip[1])
                segs = [(tail, tip)] + cv.head_segs(tip, (1.0, 0.0), max(3.0, float(it.get("head", 0.35)) * L))
                cv.stroke(segs, float(it.get("linewidth", 1.0)), it.get("color", [255, 255, 255]), 1.0, 1.0)
            draw_text(cv.rgb, xr - tw, y, txt, s, tuple(it.get("color", [255, 255, 255])), OUTLINE)
            y += CELL_H * s + 2 * s
    return rgb
