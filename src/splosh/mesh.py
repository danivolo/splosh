"""Block-structured mesh data (adaptive or fixed mesh refinement, a uniform grid as one level): the data model a kind's open() returns, and its slice maps.

Data model (duck-typed: any object with these attributes works; `Level` is a ready-made container):
    obj.time                       float, the snapshot's time
    obj.levels                     [Level, ...], coarsest first or in any order (the renderer sorts by dx)
    obj.variables                  [names] of the stored variables
    obj.read(var, level, patch, lo, hi) -> ndarray     the values of `var` on patch `patch` of level `level` at the index box lo..hi (inclusive, level
                                   index space, inside the patch), axes ordered (x, y, z): shape hi - lo + 1. A reader slices its file here (h5py
                                   partial read), so that a slice reads only the layers it needs.
    optional: obj.header (dict, kept in snap["header"]); obj.points() -> {"pos" (N, 3), "mass" (N), "vel" (N, 3), "id" (N)} or None: point masses the frame
    rules see (centre bh / align bh / tilt Lbh, the viewer's markers), like black holes of a particle snapshot; obj.markers() -> [{"name", "pos" (N, 3),
    "labels" [str] (optional), "color" (optional)}]: points drawn by the overlay `mesh_markers`; obj.close().
Level: dx (spacing), origin (position of index (0, 0, 0)), patches [(lo, hi)] (inclusive integer index ranges: what read() reads), centring "vertex"
(values at origin + i dx) or "cell" (at origin + (i + 1/2) dx), periodic (the level wraps; the period is its index box: N points of spacing dx),
boxes [(lo, hi)] (the refinement boxes drawn as outlines; default: the patches merged where they tile a larger box).

Slices (render_mesh, called by maps.render_map for a snapshot dict holding "mesh"): every pixel of the view's plane (the frame rules give its origin and
axes: centre + a e_a + b e_b + z0 e_los) takes the FINEST level that covers it: vertex-centred, a level covers a point when the corners of its cell are
all stored (the union of the level's patches, also across patch seams); cell-centred, when its cell is stored. Values are interpolated trilinearly
(View.interp = "linear", exact for linear functions) or taken from the nearest point ("nearest"). A level's data are read only for the patches that reach
the plane (and only the layers around it when the plane is thin in an index direction); patch arrays are kept in an LRU (MESH_CACHE_MB, default 512).
Derived fields are computed from the interpolated variables. NaN stored in a file reads as "not covered" (the next coarser level is used)."""
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import numba as nb
import numpy as np

from . import kinds, plugins, transforms
from .frames import make_frame

MESH_CACHE_MB = float(os.environ.get("SPLOSH_MESH_CACHE_MB", "512"))
THIN = 4                          # an index direction in which the needed range is at most this many points is read as that range only (a slab)


@dataclass
class Level:
    dx: float
    origin: tuple = (0.0, 0.0, 0.0)
    patches: list = field(default_factory=list)
    centring: str = "vertex"
    periodic: bool = False
    boxes: list = None

    def __post_init__(self):
        self.dx = float(self.dx)
        self.origin = tuple(float(x) for x in np.broadcast_to(np.asarray(self.origin, float), (3,)))
        self.patches = [(tuple(int(x) for x in lo), tuple(int(x) for x in hi)) for lo, hi in self.patches]
        if self.centring not in ("vertex", "cell"):
            raise ValueError("centring must be 'vertex' or 'cell'")
        if self.boxes is None:
            self.boxes = merge_boxes(self.patches)
        else:
            self.boxes = [(tuple(int(x) for x in lo), tuple(int(x) for x in hi)) for lo, hi in self.boxes]


def merge_boxes(boxes):
    """the boxes (inclusive index ranges) merged pairwise while two of them form a larger box (same extent in two directions, touching or overlapping in the third)."""
    bs = [(list(lo), list(hi)) for lo, hi in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                (l1, h1), (l2, h2) = bs[i], bs[j]
                for a in range(3):
                    o = [b for b in range(3) if b != a]
                    if all(l1[b] == l2[b] and h1[b] == h2[b] for b in o) and l2[a] <= h1[a] + 1 and l1[a] <= h2[a] + 1:
                        bs[i] = ([min(l1[b], l2[b]) for b in range(3)], [max(h1[b], h2[b]) for b in range(3)])
                        del bs[j]
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return [(tuple(lo), tuple(hi)) for lo, hi in bs]


def _shift(level):
    return 0.0 if level.centring == "vertex" else 0.5


def _level_bbox(level):
    lo = np.min([p[0] for p in level.patches], axis=0)
    hi = np.max([p[1] for p in level.patches], axis=0)
    return lo.astype(np.int64), hi.astype(np.int64)


def box_world(level, lo, hi):
    """(lower corner, upper corner) in world coordinates of the region an index box covers (vertex: its points; cell: its cells' faces)."""
    o, dx = np.asarray(level.origin), level.dx
    if level.centring == "vertex":
        return o + np.asarray(lo) * dx, o + np.asarray(hi) * dx
    return o + np.asarray(lo) * dx, o + (np.asarray(hi) + 1) * dx


# ----------------------------------------------------------------------------- patch cache
_CACHE = OrderedDict()
_CACHE_BYTES = [0]


def _file_sig(path):
    try:
        st = os.stat(path)
        return (os.path.abspath(str(path)), st.st_mtime_ns, st.st_size)
    except (OSError, TypeError):
        return (str(path), None, None)


def _read(obj, sig, var, li, pi, lo, hi):
    key = (sig, var, li, pi, tuple(int(x) for x in lo), tuple(int(x) for x in hi))
    got = _CACHE.get(key)
    if got is not None:
        _CACHE.move_to_end(key)
        return got
    a = np.asarray(obj.read(var, li, pi, tuple(int(x) for x in lo), tuple(int(x) for x in hi)), dtype=np.float64)
    want = tuple(int(h - l + 1) for l, h in zip(lo, hi))
    if a.shape != want:
        raise ValueError(f"mesh read({var!r}, level {li}, patch {pi}, {tuple(lo)}, {tuple(hi)}) returned shape {a.shape}, expected {want} (axes x, y, z)")
    _CACHE[key] = a
    _CACHE_BYTES[0] += a.nbytes
    while _CACHE_BYTES[0] > MESH_CACHE_MB * 2 ** 20 and len(_CACHE) > 1:
        _, old = _CACHE.popitem(last=False)
        _CACHE_BYTES[0] -= old.nbytes
    return a


def clear_cache():
    _CACHE.clear()
    _CACHE_BYTES[0] = 0


# ----------------------------------------------------------------------------- sampling
@nb.njit(parallel=True, cache=True)
def _sample_kernel(P, idx, data, lo, origin, dx, shift, linear, cellc, per, N, out, done):
    """for the points P[idx[q]] not done yet: the values of `data` (nv, nx, ny, nz; NaN = not stored; index (0,0,0) is level index lo) at the point,
    trilinear or nearest; done[p] = True where this level covers the point."""
    nv, nx, ny, nz = data.shape
    dims = (nx, ny, nz)
    for q in nb.prange(idx.shape[0]):
        p = idx[q]
        if done[p]:
            continue
        f = np.empty(3)
        for a in range(3):
            f[a] = (P[p, a] - origin[a]) / dx - shift - lo[a]
        ok = False
        if linear:
            i0 = np.empty(3, np.int64)
            t = np.empty(3)
            good = True
            for a in range(3):
                fl = np.floor(f[a])
                i0[a] = np.int64(fl)
                t[a] = f[a] - fl
            # the 8 corners; a corner of weight 0 is not needed (a point on a face, edge or vertex)
            for k in range(nv):
                out[k, p] = 0.0
            for c in range(8):
                w = 1.0
                ii = np.empty(3, np.int64)
                for a in range(3):
                    bit = (c >> a) & 1
                    w *= t[a] if bit else 1.0 - t[a]
                    ii[a] = i0[a] + bit
                    if per[a]:
                        ii[a] = ii[a] % N[a]
                if w == 0.0:
                    continue
                if ii[0] < 0 or ii[0] >= nx or ii[1] < 0 or ii[1] >= ny or ii[2] < 0 or ii[2] >= nz:
                    good = False
                    break
                v0 = data[0, ii[0], ii[1], ii[2]]
                if v0 != v0:
                    good = False
                    break
                for k in range(nv):
                    out[k, p] += w * data[k, ii[0], ii[1], ii[2]]
            ok = good
        if not ok and (not linear or cellc):
            ii = np.empty(3, np.int64)
            inside = True
            for a in range(3):
                ii[a] = np.int64(np.floor(f[a] + 0.5))
                if per[a]:
                    ii[a] = ii[a] % N[a]
                if ii[a] < 0 or ii[a] >= dims[a]:
                    inside = False
            if inside and data[0, ii[0], ii[1], ii[2]] == data[0, ii[0], ii[1], ii[2]]:
                for k in range(nv):
                    out[k, p] = data[k, ii[0], ii[1], ii[2]]
                ok = True
        if ok:
            done[p] = True
        else:
            for k in range(nv):
                out[k, p] = np.nan


def _wrap(P, level):
    """positions wrapped into the periodic level's box (its index box, N points of spacing dx per axis)."""
    lo, hi = _level_bbox(level)
    s = _shift(level)
    start = np.asarray(level.origin) + (lo + s) * level.dx
    L = (hi - lo + 1) * level.dx
    return start + np.mod(P - start, L)


def sample(obj, path, P, variables, interp="linear", plane=None, stats=None):
    """values of `variables` at the points P (n, 3) world coordinates: (len(variables), n) float64, NaN where no level covers a point. The finest level
    covering a point gives its value. plane = (point, unit normal): only patches within reach of that plane are read (a slice); stats: dict filled with
    the number of points per level."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    sig = _file_sig(path) if path is not None else ("obj", id(obj), None)
    n = len(P)
    nv = len(variables)
    out = np.full((nv, n), np.nan)
    done = np.zeros(n, np.bool_)
    order = sorted(range(len(obj.levels)), key=lambda l: obj.levels[l].dx)
    linear = interp == "linear"
    for li in order:
        lev = obj.levels[li]
        if not lev.patches:
            continue
        idx = np.nonzero(~done)[0]
        if not len(idx):
            break
        s = _shift(lev)
        Q = _wrap(P[idx], lev) if lev.periodic else P[idx]
        F = (Q - np.asarray(lev.origin)) / lev.dx - s
        blo, bhi = _level_bbox(lev)
        nlo = np.maximum(np.floor(F.min(0)).astype(np.int64) - 1, blo)
        nhi = np.minimum(np.ceil(F.max(0)).astype(np.int64) + 1, bhi)
        if lev.periodic:
            nlo, nhi = blo.copy(), bhi.copy()
        if np.any(nhi < nlo):
            continue
        reads = []
        for pi, (plo, phi) in enumerate(lev.patches):
            plo, phi = np.asarray(plo, np.int64), np.asarray(phi, np.int64)
            ilo, ihi = np.maximum(plo, nlo), np.minimum(phi, nhi)
            if np.any(ihi < ilo):
                continue
            if plane is not None and not lev.periodic:
                c0, c1 = box_world(lev, ilo, ihi)
                corners = np.array([[(c0, c1)[(k >> a) & 1][a] for a in range(3)] for k in range(8)])
                d = (corners - plane[0]) @ plane[1]
                reach = 2.0 * lev.dx
                if d.min() > reach or d.max() < -reach:
                    continue
            rlo = np.where(nhi - nlo + 1 <= THIN, ilo, plo)
            rhi = np.where(nhi - nlo + 1 <= THIN, ihi, phi)
            reads.append((pi, rlo, rhi, ilo, ihi))
        if not reads:
            continue
        dlo = np.min([r[3] for r in reads], axis=0)
        dhi = np.max([r[4] for r in reads], axis=0)
        data = np.full((nv,) + tuple(int(x) for x in dhi - dlo + 1), np.nan)
        for pi, rlo, rhi, ilo, ihi in reads:
            for k, var in enumerate(variables):
                a = _read(obj, sig, var, li, pi, rlo, rhi)
                src = tuple(slice(int(ilo[d] - rlo[d]), int(ihi[d] - rlo[d] + 1)) for d in range(3))
                dst = (k,) + tuple(slice(int(ilo[d] - dlo[d]), int(ihi[d] - dlo[d] + 1)) for d in range(3))
                data[dst] = a[src]
        per = np.array([bool(lev.periodic)] * 3)
        N = (bhi - blo + 1).astype(np.int64)
        sub_out = np.full((nv, len(idx)), np.nan)
        sub_done = np.zeros(len(idx), np.bool_)
        _sample_kernel(np.ascontiguousarray(Q), np.arange(len(idx), dtype=np.int64), data, dlo.astype(np.int64),
                       np.asarray(lev.origin, np.float64), lev.dx, s, linear, lev.centring == "cell", per, N, sub_out, sub_done)
        got = idx[sub_done]
        out[:, got] = sub_out[:, sub_done]
        done[got] = True
        if stats is not None:
            stats[li] = int(sub_done.sum())
    return out


# ----------------------------------------------------------------------------- maps
def plane_points(frame, view):
    """(points (npix * npix, 3) world coordinates of the pixel centres, row-major with j (vertical) slow, plane point, unit normal) of the view's plane."""
    from .maps import AXES
    ia, ib, il = AXES[view.axes]
    R = frame.rot
    half = 0.5 * view.width
    pix = view.width / view.npix
    a = view.cx - half + (np.arange(view.npix) + 0.5) * pix
    b = view.cy - half + (np.arange(view.npix) + 0.5) * pix
    A, B = np.meshgrid(a, b)                                  # [j, i]
    base = np.asarray(frame.center, np.float64) + view.z0 * R[il]
    P = base + A.reshape(-1, 1) * R[ia] + B.reshape(-1, 1) * R[ib]
    return P, base, R[il].copy()


def to_window(frame, view, pos):
    """world positions (N, 3) -> (x, y, depth) along the view's axes, relative to the frame (window coordinates and the distance from the plane's frame origin along the line of sight)."""
    from .maps import AXES
    ia, ib, il = AXES[view.axes]
    q = frame.pos(np.asarray(pos, np.float64).reshape(-1, 3))
    return q[:, ia], q[:, ib], q[:, il]


def _plain(v):
    """a header value as JSON-able Python (numpy arrays and scalars to lists and numbers, bytes to str, anything else to its repr)."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    elif isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v if isinstance(v, (int, float, str, bool)) or v is None else repr(v)


def snap_dict(path, kind=None):
    """the snapshot dict of a kind's file (what io.load_snapshot returns for it): time, header, path, kind, mesh (the opened object) and, when the
    object has points(), the point masses under the black-hole keys the frame rules read."""
    kind = kind or kinds.kind_of_file(path)
    obj = kinds.open_mesh(path, kind)
    hdr = {str(k): _plain(v) for k, v in dict(getattr(obj, "header", None) or {}).items()}
    names = kinds.units(kind)
    hdr["_unit_names"] = {"L": names.get("length", "L"), "T": names.get("time", "")}
    out = {"time": float(obj.time), "header": hdr, "path": os.path.abspath(str(path)), "kind": kind, "mesh": obj}
    pts = obj.points() if callable(getattr(obj, "points", None)) else None
    if pts is not None and len(np.asarray(pts.get("pos", []))):
        pos = np.asarray(pts["pos"], np.float64).reshape(-1, 3)
        n = len(pos)
        mass = np.asarray(pts.get("mass", np.ones(n)), np.float64)
        out.update(bh_pos=pos, bh_mass=mass, bh_mdyn=mass, bh_vel=np.asarray(pts.get("vel", np.zeros((n, 3))), np.float64).reshape(-1, 3),
                   bh_id=np.asarray(pts.get("id", np.arange(1, n + 1))))
    return out


def resolve_field(snap, name):
    specs = kinds.field_specs(snap["kind"], snap["mesh"])
    if name not in specs:
        raise KeyError(f"unknown field {name!r} for a {snap['kind']} run; available: {sorted(specs)}")
    return specs[name]


def render_mesh(snap, view, frame=None):
    """(image (npix, npix) float32, meta) of a slice of a mesh snapshot: the output contract of maps.render_map."""
    if view.mode != "slice":
        raise ValueError(f"mesh data ({snap['kind']} run): mode 'slice' only (projections and volume renderings of mesh data are not implemented)")
    obj = snap["mesh"]
    spec = resolve_field(snap, view.field)
    frame = frame or make_frame(snap, view.center, view.align, view.tilt)
    P, p0, nrm = plane_points(frame, view)
    stats = {}
    vals = sample(obj, snap["path"], P, list(spec["needs"]), interp=view.interp, plane=(p0, nrm), stats=stats)
    n = view.npix
    if spec["fn"] is None:
        img = vals[0]
    else:
        v = {name: vals[k].reshape(n, n) for k, name in enumerate(spec["needs"])}
        v.update(x=P[:, 0].reshape(n, n), y=P[:, 1].reshape(n, n), z=P[:, 2].reshape(n, n))
        with np.errstate(invalid="ignore", divide="ignore"):
            img = np.asarray(spec["fn"](v), dtype=np.float64)
    img = np.asarray(img, dtype=np.float32).reshape(n, n)
    from .maps import AXES
    ia, ib, il = AXES[view.axes]
    half = 0.5 * view.width
    names = kinds.units(snap["kind"])
    lu, tu = names.get("length") or "code length", names.get("time") or "code time"
    unit = spec["unit"]
    meta = {
        "quantity_unit": {"quantity": view.field, "unit": unit or "", "factor": 1.0, "system": view.units, "code_unit": unit or "", "cgs_unit": "", "cgs_factor": 1.0,
                          "known": unit is not None},
        "unit_system": view.units, "unit_scales": {}, "unit_override": unit or "",
        "length_unit": {"unit": lu, "factor": 1.0}, "time_unit": {"unit": tu, "factor": 1.0}, "time_physical": float(snap["time"]),
        "extent_physical": [view.cx - half, view.cx + half, view.cy - half, view.cy + half],
        "view": view.as_dict(), "time": float(snap["time"]), "frame": frame.describe(), "weight": "none", "label": spec["label"],
        "extent": [view.cx - half, view.cx + half, view.cy - half, view.cy + half], "n_particles": 0,
        "kernel": f"mesh slice: finest level covering each pixel, {view.interp} interpolation", "origin": "lower", "units": snap.get("header", {}),
        "xlabel": "xyz"[ia], "ylabel": "xyz"[ib], "kind": snap["kind"],
        "mesh": {"pixels_per_level": {str(k): v for k, v in sorted(stats.items())}, "levels": len(obj.levels)},
    }
    return transforms.finish(img, meta, view.post)


# ----------------------------------------------------------------------------- overlays: refinement boxes, markers
def plane_box_polygon(c0, c1, p0, nrm, ea, eb):
    """the polygon where the plane (point p0, unit normal nrm) cuts the box [c0, c1], as [(a, b)] along the in-plane unit axes ea, eb (relative to p0),
    in order around its centre; [] when the plane misses the box."""
    corners = np.array([[(c0, c1)[(k >> a) & 1][a] for a in range(3)] for k in range(8)], dtype=np.float64)
    d = (corners - p0) @ nrm
    pts = []
    for i in range(8):
        for a in range(3):
            j = i | (1 << a)
            if j == i:
                continue
            di, dj = d[i], d[j]
            if di == 0.0 and dj == 0.0:
                pts += [corners[i], corners[j]]
            elif (di <= 0.0 <= dj) or (dj <= 0.0 <= di):
                t = di / (di - dj)
                pts.append(corners[i] + t * (corners[j] - corners[i]))
    if len(pts) < 3:
        return []
    Q = np.array(pts) - p0
    ab = np.stack([Q @ ea, Q @ eb], axis=1)
    ab = np.unique(np.round(ab, 12), axis=0)
    if len(ab) < 3:
        return []
    c = ab.mean(0)
    ang = np.arctan2(ab[:, 1] - c[1], ab[:, 0] - c[0])
    return ab[np.argsort(ang)].tolist()


def _flag(v):
    return v.lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)


LEVEL_COLORS = ("white", "cyan", "yellow", "magenta", "lime", "orange", "deepskyblue", "pink", "red", "violet")


def box_outlines(snap, frame, view, levels="all", color="", lw=1.0, alpha=0.8, labels=False):
    """overlay `mesh_boxes`: the outlines of the refinement boxes of each level where the view's plane cuts them (levels: 'all' or 'a;b;c' level indices;
    color: one colour for all, default one per level; labels: the level number at each box)."""
    from .maps import AXES
    from .overlays import color_rgb
    if "mesh" not in snap:
        return [{"kind": "note", "text": "mesh_boxes: not a mesh snapshot"}]
    labels, lw, alpha = _flag(labels), float(lw), float(alpha)
    obj = snap["mesh"]
    ia, ib, il = AXES[view.axes]
    R = frame.rot
    p0 = np.asarray(frame.center, np.float64) + view.z0 * R[il]
    want = None if str(levels) in ("all", "") else {int(x) for x in str(levels).replace(";", " ").split()}
    order = sorted(range(len(obj.levels)), key=lambda l: -obj.levels[l].dx)
    items = []
    for rank, li in enumerate(order):
        if want is not None and li not in want:
            continue
        lev = obj.levels[li]
        col = color_rgb(color) if color else color_rgb(LEVEL_COLORS[rank % len(LEVEL_COLORS)])
        for lo, hi in lev.boxes:
            c0, c1 = box_world(lev, lo, hi)
            poly = plane_box_polygon(c0, c1, p0, R[il], R[ia], R[ib])
            if not poly:
                continue
            pts = [[float(a + 0.0), float(b + 0.0)] for a, b in poly]
            pts.append(pts[0])
            items.append({"kind": "polyline", "points": pts, "magnitude": [0.0] * len(pts), "head": None, "color": col, "linewidth": float(lw),
                          "alpha": float(alpha), "halo": 0.0, "level": li, "overlay": "mesh_boxes"})
            if labels:
                items.append({"kind": "text", "x": pts[0][0], "y": pts[0][1], "text": f"L{li}", "color": col, "size_px": 10, "dx_px": 3, "dy_px": 3})
    return items


def markers_overlay(snap, frame, view, labels=True, size=6.0):
    """overlay `mesh_markers`: the points the kind's object returns from markers() (e.g. positions it reads from its file), in the view's window."""
    from .overlays import color_rgb
    if "mesh" not in snap:
        return [{"kind": "note", "text": "mesh_markers: not a mesh snapshot"}]
    labels, size = _flag(labels), float(size)
    obj = snap["mesh"]
    fn = getattr(obj, "markers", None)
    groups = fn() if callable(fn) else []
    items = []
    for g in groups or ():
        pos = np.asarray(g.get("pos", []), np.float64).reshape(-1, 3)
        if not len(pos):
            continue
        x, y, d = to_window(frame, view, pos)
        col = color_rgb(g.get("color") or "white")
        items.append({"kind": "points", "type": str(g.get("name", "markers")), "x": [float(v) for v in x], "y": [float(v) for v in y], "depth": [float(v) for v in d],
                      "size_px": float(size), "style": "ring", "color": col, "alpha": 1.0, "n_total": int(len(x)), "n_drawn": int(len(x)), "overlay": "mesh_markers"})
        if labels and g.get("labels"):
            for k, t in enumerate(g["labels"][:len(x)]):
                items.append({"kind": "text", "x": float(x[k]), "y": float(y[k]), "text": str(t), "color": col, "size_px": 10, "dx_px": 5, "dy_px": 5})
    return items


plugins.register_overlay("mesh_boxes", box_outlines, options=[("labels", "level numbers", False)])
plugins.register_overlay("mesh_markers", markers_overlay, options=[("labels", "labels", True)])
