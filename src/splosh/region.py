"""Partial loading: read only the gas a zoomed view needs.

GIZMO writes the gas in domain order along a space-filling curve, so consecutive particles are close in space (a 0.9M-particle snapshot: consecutive particles
are on average 1.8% as far apart as random pairs), and a small window touches few runs of consecutive rows. An index per snapshot, made once and kept next to
the map store (`<store>/index/<snapshot signature>.npz`), holds for every chunk of CHUNK consecutive particles its bounding box and its largest support H.

A region load (`load_region`) computes the frame from the black holes only (centre rules bh, ids of black holes, xyz; no tilt L), takes the chunks whose box,
seen in the frame, overlaps the window enlarged by `margin` window widths on each side plus the chunk's H, and reads only those rows of every dataset. Every
particle whose kernel reaches the enlarged window is inside (the chunk test is conservative), so maps and tiles of any window inside it are exact.
Rules that need all the gas (com, mass-weighted, peak, gas ids, tilt L) and windows that would read more than `max_fraction` of the rows load everything.

    get_index(path, root) -> {"chunk", "n", "lo" (nc, 3), "hi" (nc, 3), "hmax" (nc,)}
    plan(index, frame, view, margin) -> (rows [(start, stop), ...] merged, bounds {a: [lo, hi], b: [lo, hi]} of the enlarged window in frame coordinates)
    load_region(path, view, gas_fields, root, margin, max_fraction, min_particles) -> snapshot dict with snap["region"], or None (load everything instead)"""
import os

import h5py
import numpy as np

from .frames import make_frame
from .maps import AXES

CHUNK = 4096
INDEX_VERSION = 1
ALL_GAS_CENTRES = ("com", "mass-weighted", "peak")


def build_index(path, chunk=None):
    chunk = chunk or CHUNK
    with h5py.File(path, "r") as f:
        g = f["PartType0"]
        x = g["Coordinates"][...]
        h = g["SmoothingLength"][...]
    n = len(x)
    nc = -(-n // chunk)
    lo, hi, hm = np.empty((nc, 3)), np.empty((nc, 3)), np.empty(nc)
    for k in range(nc):
        s = slice(k * chunk, min(n, (k + 1) * chunk))
        lo[k], hi[k], hm[k] = x[s].min(0), x[s].max(0), h[s].max()
    return {"chunk": chunk, "n": n, "lo": lo, "hi": hi, "hmax": hm, "version": INDEX_VERSION}


def index_file(path, root, chunk=None):
    from .mapstore import snapshot_hash
    return os.path.join(root, "index", snapshot_hash(path) + f".c{chunk or CHUNK}.npz")


def get_index(path, root=None, chunk=None):
    """the chunk index of a snapshot: from the cache file when there is one, else built (reads Coordinates and SmoothingLength once) and saved."""
    chunk = chunk or CHUNK
    fn = index_file(path, root, chunk) if root else None
    if fn and os.path.exists(fn):
        try:
            with np.load(fn) as d:
                if int(d["version"]) == INDEX_VERSION and int(d["chunk"]) == chunk:
                    return {k: (d[k] if d[k].ndim else d[k].item()) for k in d.files}
        except (OSError, ValueError, KeyError):
            pass
    ix = build_index(path, chunk)
    if fn:
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        tmp = fn + f".{os.getpid()}.tmp.npz"
        np.savez(tmp, **ix)
        os.replace(tmp, fn)
    return ix


def needs_all_gas(view):
    """True when the frame of the view cannot be found from the black holes alone."""
    key = view.center.split(":")[0].strip().lower()
    return key in ALL_GAS_CENTRES or any(t.strip().split(":")[0] == "L" for t in (view.tilt or "none").split("+"))


def plan(index, frame, view, margin=0.5):
    """the row ranges whose chunk boxes, in the frame, overlap the window enlarged by `margin` widths on each side (plus each chunk's H)."""
    ia, ib, il = AXES[view.axes]
    lo, hi = index["lo"], index["hi"]
    corners = np.stack([np.stack([np.where(m & 1, hi[:, 0], lo[:, 0]), np.where(m & 2, hi[:, 1], lo[:, 1]), np.where(m & 4, hi[:, 2], lo[:, 2])], 1)
                        for m in range(8)], 1)                                  # (nc, 8, 3)
    q = frame.pos(corners.reshape(-1, 3)).reshape(-1, 8, 3)
    qlo, qhi = q.min(1), q.max(1)
    h = index["hmax"] * view.hscale
    half = 0.5 * view.width * (1 + 2 * margin)
    a0, a1, b0, b1 = view.cx - half, view.cx + half, view.cy - half, view.cy + half
    hit = (qhi[:, ia] + h > a0) & (qlo[:, ia] - h < a1) & (qhi[:, ib] + h > b0) & (qlo[:, ib] - h < b1)
    if view.mode == "slice":
        hit &= (qhi[:, il] + h > view.z0) & (qlo[:, il] - h < view.z0)
    rows, c, n, nc = [], index["chunk"], index["n"], len(hit)
    k = 0
    while k < nc:
        if not hit[k]:
            k += 1
            continue
        j = k
        while j + 1 < nc and hit[j + 1]:
            j += 1
        rows.append((k * c, min(n, (j + 1) * c)))
        k = j + 1
    return rows, {"a": [a0, a1], "b": [b0, b1]}


def load_region(path, view, gas_fields, root=None, margin=0.5, max_fraction=0.5, min_particles=0):
    """the snapshot dict with only the gas rows the enlarged window needs (snap["region"] says which), or None when a full load is the right thing."""
    from .io import load_snapshot
    if needs_all_gas(view):
        return None
    with h5py.File(path, "r") as f:
        n = int(f["Header"].attrs["NumPart_ThisFile"][0])
    if n < min_particles:
        return None
    base = load_snapshot(path, gas_fields=())                  # header and black holes only: enough for the frame of these rules
    try:
        frame = make_frame(base, view.center, view.align, view.tilt)
    except (KeyError, ValueError):
        return None                                             # e.g. ids of gas particles: they need the gas, load it all
    ix = get_index(path, root)
    rows, bounds = plan(ix, frame, view, margin)
    nrows = sum(b - a for a, b in rows)
    if nrows > max_fraction * ix["n"]:
        return None
    snap = load_snapshot(path, gas_fields=gas_fields, rows=rows)
    snap["region"] = {"rows": nrows, "n": ix["n"], "fraction": nrows / ix["n"] if ix["n"] else 0.0, "runs": len(rows), "bounds": bounds,
                      "frame_key": [view.center, view.align, view.tilt, view.axes, view.mode, float(view.z0), float(view.hscale)]}
    return snap


def covers(region, view):
    """True when a loaded region also serves `view` (same frame and plane; its window inside the region's enlarged window)."""
    if region is None:
        return False
    if region["frame_key"] != [view.center, view.align, view.tilt, view.axes, view.mode, float(view.z0), float(view.hscale)]:
        return False
    half = 0.5 * view.width
    (a0, a1), (b0, b1) = region["bounds"]["a"], region["bounds"]["b"]
    return a0 <= view.cx - half and view.cx + half <= a1 and b0 <= view.cy - half and view.cy + half <= b1
