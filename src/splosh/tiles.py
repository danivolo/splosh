"""Tiled rendering of a View: deep zoom and images larger than View.npix allows, in bounded memory, with cacheable pieces.

A tile grid is defined by the tile size in pixels (`tile`, default 256), the world width `tw` of one tile and an origin (`ox`, `oy`) in frame coordinates (after
centring and alignment, like View.cx / cy). Tile (ix, iy) covers [ox + ix tw, ox + (ix + 1) tw) x [oy + iy tw, oy + (iy + 1) tw) and holds `tile` x `tile`
pixels; indices can be negative. Row 0 of a tile is its LOWEST y (like maps; flip for display). Two kinds of grid:

    global grid   (`tile_grid`, `render_tile`): origin (0, 0), zoom levels L (any integer) with tw = base_width / 2^L, so the pixel size is base_width / (tile 2^L):
                  widths are powers of two of the base width, tile edges of neighbouring levels coincide and any window at the same level reuses the same tiles
                  (panning and small zooms hit cached tiles). `base_width` defaults to the view width rounded up to a power of two.
    window grid   (`render_tiled`): origin at the lower-left corner of the view window, pixel size = width / npix, so that the tiles reproduce render_map(view)
                  pixel for pixel (the last tiles are cropped when npix is not a multiple of tile).

Each tile is rendered independently by `render_map`, with a margin of HALO pixels that is cropped (see HALO), on the particles whose support (radius H, times hscale) intersects the tile: a spatial binning of the particles
by tile cell is built once (in size classes of powers of two of the tile width, each looked up in the 3 x 3 cells around the tile), so a particle that
straddles a tile boundary or a corner is deposited in every tile it touches. The frame (centre, rotation) is computed once for the whole snapshot, not per tile.
Tiles are cached in the mapstore (`mapstore.tile_key`: snapshot signature, view without its window, tile size, grid geometry, indices, level).

Accuracy: a tile uses the same engine and pixel size as the monolithic render. With the exact engine (View.exact) the assembled window grid equals render_map up
to float32 summation order (a few 1e-7 relative to the local value). With the default multi-resolution engine the large particles are merged on coarser grids of
the tile instead of the whole image, so the two differ by the engine's own error (documented 1e-3 level of the map maximum for column density, tests/test_tiles.py
checks 5e-3): neither is the exact answer, both are within that of the exact kernel. A slice with `weight=mass` of a non-density field masks pixels under 1e-6 of
the tile's own weight maximum (the monolithic render uses the maximum of the whole image): pixels that faint can differ in NaN-or-value.

Parallelism: tiles run on `workers` Python threads; each sets its numba thread count to ncpu // workers (numba's thread mask is per Python thread), so the total
never exceeds the CPUs. Everything inside a tile is the existing band-parallel engine, nothing is nested."""
import json
import math
import os
import threading
import time
import zlib
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numba as nb
import numpy as np

from . import imaging
from .frames import make_frame
from .maps import AXES, render_map, required_gas_fields
from .mapstore import MapStore, snapshot_hash, tile_key

DEFAULT_TILE = 256
TILE_SIZES = (64, 128, 256, 512, 1024)


# ------------------------------------------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class TileGrid:
    """Tile grid: tile size in pixels, world width of a tile, origin; `level` and `base_width` are set for a global grid. Tile ranges (ix0..ix1, iy0..iy1, inclusive)
    are those covering the requested window."""
    tile: int
    tw: float
    ox: float = 0.0
    oy: float = 0.0
    level: int = None
    base_width: float = None
    ix0: int = 0
    ix1: int = -1
    iy0: int = 0
    iy1: int = -1

    @property
    def pixel(self):
        return self.tw / self.tile

    @property
    def nx(self):
        return self.ix1 - self.ix0 + 1

    @property
    def ny(self):
        return self.iy1 - self.iy0 + 1

    def tiles(self):
        """(ix, iy) of the covering tiles, rows of constant iy (lowest first), ix ascending."""
        return [(ix, iy) for iy in range(self.iy0, self.iy1 + 1) for ix in range(self.ix0, self.ix1 + 1)]

    def extent(self, ix, iy):
        """[x0, x1, y0, y1] of a tile in frame coordinates."""
        return [self.ox + ix * self.tw, self.ox + (ix + 1) * self.tw, self.oy + iy * self.tw, self.oy + (iy + 1) * self.tw]

    def centre(self, ix, iy):
        return self.ox + (ix + 0.5) * self.tw, self.oy + (iy + 0.5) * self.tw

    def at(self, ix0, ix1, iy0, iy1):
        return TileGrid(self.tile, self.tw, self.ox, self.oy, self.level, self.base_width, ix0, ix1, iy0, iy1)

    def describe(self):
        return {"tile": self.tile, "tile_width": self.tw, "pixel": self.pixel, "origin": [self.ox, self.oy], "level": self.level, "base_width": self.base_width,
                "ix": [self.ix0, self.ix1], "iy": [self.iy0, self.iy1], "n_tiles": max(self.nx, 0) * max(self.ny, 0)}


def default_base_width(width):
    """`width` rounded up to a power of two (in code length units)."""
    return 2.0 ** math.ceil(math.log2(width))


def _check_tile(tile):
    if tile not in TILE_SIZES:
        raise ValueError(f"tile must be one of {TILE_SIZES}")
    return int(tile)


def level_for(view, tile=DEFAULT_TILE, base_width=None):
    """Smallest zoom level whose pixel size base_width / (tile 2^L) is at most the view's pixel size width / npix."""
    bw = base_width or default_base_width(view.width)
    return math.ceil(math.log2(bw * view.npix / (tile * view.width)) - 1e-9)


def _cover(lo, hi, ox, tw):
    ix0 = math.floor((lo - ox) / tw + 1e-9)
    ix1 = math.ceil((hi - ox) / tw - 1e-9) - 1
    return ix0, max(ix1, ix0)


def tile_grid(view, tile=DEFAULT_TILE, base_width=None, level=None):
    """The global tile grid covering the window of `view`: level = `level` if given, else the coarsest level at least as fine as the view's pixel size.
    `base_width` defaults to the view width rounded up to a power of two (a server uses one fixed value so that every request shares tiles)."""
    tile = _check_tile(tile)
    bw = float(base_width or default_base_width(view.width))
    L = level_for(view, tile, bw) if level is None else int(level)
    tw = bw / 2.0 ** L
    h = 0.5 * view.width
    ix0, ix1 = _cover(view.cx - h, view.cx + h, 0.0, tw)
    iy0, iy1 = _cover(view.cy - h, view.cy + h, 0.0, tw)
    return TileGrid(tile, tw, 0.0, 0.0, L, bw, ix0, ix1, iy0, iy1)


def window_grid(view, tile=DEFAULT_TILE):
    """The grid anchored at the window's lower-left corner with the view's own pixel size: its tiles cover exactly the pixels of render_map(view)."""
    tile = _check_tile(tile)
    n = -(-view.npix // tile)
    h = 0.5 * view.width
    return TileGrid(tile, tile * view.width / view.npix, view.cx - h, view.cy - h, None, None, 0, n - 1, 0, n - 1)


HALO = 16       # pixels of margin rendered around a tile and cropped: the engine renormalises the weight of particles with a footprint under 6 pixels over the part of
                # the footprint inside the image, so a particle at a tile edge must have its whole footprint (at most 2 x 6 + 2 pixels beyond the edge) in the rendered image;
                # 16 also keeps tile + 2 HALO divisible by 2^levels of the multi-resolution engine (levels while npix / 2^k >= 64: 8 for 512, 16 for 1024), so that its coarse
                # grids have exactly 2^k times the pixel size (with 14 a 512 tile had 540 pixels, 135 / 67 coarse pixels and 7e-3 errors)


LEVELS = 3      # multi-resolution levels inside a tile: particles above 12 x 2^3 pixels go to a grid 8 times coarser (the engine's default stops at 64 pixels, i.e. 2 levels for a tile);
                # 8 x HALO-reach of the bicubic upsampling is 16 pixels = the margin, so the clamped border of the coarse grids stays in the margin


def _tile_view(view, grid, ix, iy):
    """The View that renders tile (ix, iy) plus its HALO margin."""
    cx, cy = grid.centre(ix, iy)
    return view.with_(cx=cx, cy=cy, width=grid.tw * (grid.tile + 2 * HALO) / grid.tile, npix=grid.tile + 2 * HALO, post="none")      # a transform needs the whole window, never one tile


# ------------------------------------------------------------------------------------------------------ per-tile culling
def _subset(snap, idx, n):
    """snap restricted to the gas particles `idx` (every array with n rows, except the black-hole arrays)."""
    out = {}
    for k, v in snap.items():
        out[k] = v[idx] if (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n and not k.startswith("bh_")) else v
    return out


class TileRenderer:
    """Per snapshot, view fields and tile grid: the frame and the particle positions once, a cell binning for the culls; `render(ix, iy)` is thread safe."""

    def __init__(self, snap, view, grid):
        view.with_(npix=grid.tile).validate()              # the window's own npix may exceed View's limit: only the tiles are rendered
        if view.mode == "volume":
            raise ValueError("tiles are additive maps: a volume rendering (front-to-back compositing over the whole line of sight) is rendered as one map, without tiles")
        if view.post != "none":
            raise ValueError("tiles are raw maps: View.post (a transform of the whole window) cannot be tiled; render the map without tiles")
        self.snap, self.view, self.grid = snap, view, grid
        self.frame = make_frame(snap, view.center, view.align, view.tilt)
        ia, ib, il = AXES[view.axes]
        pos = self.frame.pos(snap["Coordinates"])
        self.n = len(pos)
        h = np.asarray(snap["SmoothingLength"], dtype=np.float64) * view.hscale
        a, b = pos[:, ia], pos[:, ib]
        ids = np.arange(self.n)
        if view.mode == "slice":
            k = np.abs(pos[:, il] - view.z0) < h                  # only particles that reach the plane can contribute to any tile
            ids, a, b, h = ids[k], a[k], b[k], h[k]
        self.ids, self.a, self.b, self.h = ids, a, b, h
        tw = grid.tw
        # size classes: class c holds the particles with 2^(c-1) tw < H <= 2^c tw (class 0: H <= tw), binned by cell of width 2^c tw. Tile (ix, iy) lies inside the
        # cell (ix >> c, iy >> c) of every class, and a particle of class c that touches the tile has its centre in that cell or one of its 8 neighbours.
        cls = np.maximum(np.ceil(np.log2(np.maximum(h, 1e-300) / tw)), 0).astype(np.int64)
        self.classes = {}
        for c in np.unique(cls):
            members = np.nonzero(cls == c)[0]
            size = tw * 2.0 ** int(c)
            cx = np.floor((a[members] - grid.ox) / size).astype(np.int64)
            cy = np.floor((b[members] - grid.oy) / size).astype(np.int64)
            cxmin, cxmax, cymin, cymax = int(cx.min()), int(cx.max()), int(cy.min()), int(cy.max())
            M = cymax - cymin + 1
            key = (cx - cxmin) * M + (cy - cymin)
            order = np.argsort(key, kind="stable")
            self.classes[int(c)] = (cxmin, cxmax, cymin, cymax, M, key[order], members[order])

    def candidates(self, ix, iy):
        """positions (into self.ids) of the particles whose support box intersects tile (ix, iy) (a superset of the particles that change its pixels; render_map
        does the exact cull): the 3 x 3 cells around the tile in each size class, then a vectorised box test."""
        parts = []
        for c, (cxmin, cxmax, cymin, cymax, M, skey, sidx) in self.classes.items():
            tx, ty = ix >> c, iy >> c
            y0, y1 = max(ty - 1, cymin), min(ty + 1, cymax)
            if y0 > y1:
                continue
            for cx in (tx - 1, tx, tx + 1):
                if cxmin <= cx <= cxmax:
                    base = (cx - cxmin) * M - cymin
                    lo = np.searchsorted(skey, base + y0, side="left")
                    hi = np.searchsorted(skey, base + y1, side="right")
                    if hi > lo:
                        parts.append(sidx[lo:hi])
        if not parts:
            return np.zeros(0, np.int64)
        cand = np.concatenate(parts)
        pad = HALO * self.grid.pixel                  # the cloud-in-cell deposit of a particle under a pixel reaches a pixel past its support box
        e = self.grid.extent(ix, iy)
        e = [e[0] - pad, e[1] + pad, e[2] - pad, e[3] + pad]
        hc, ac, bc = self.h[cand], self.a[cand], self.b[cand]
        return cand[(ac + hc > e[0]) & (ac - hc < e[1]) & (bc + hc > e[2]) & (bc - hc < e[3])]

    def render(self, ix, iy):
        """(image (tile, tile) float32, meta) of one tile, by render_map on the culled particles."""
        cand = self.candidates(ix, iy)
        sub = _subset(self.snap, self.ids[cand], self.n)
        tv = _tile_view(self.view, self.grid, ix, iy)
        kw = None if self.view.exact else {"min_npix": tv.npix // 2 ** LEVELS}
        img, meta = render_map(sub, tv, frame=self.frame, engine_kw=kw)
        meta["n_candidates"] = int(len(cand))
        return np.ascontiguousarray(img[HALO:-HALO, HALO:-HALO]), meta


# ------------------------------------------------------------------------------------------------------ sources, cache, scheduling
class _Source:
    """The snapshot (a loaded dict, or a path loaded on the first cache miss) plus its signature and the renderer (built once, lazily, thread safe)."""

    def __init__(self, snap, view, grid):
        self.view, self.grid = view, grid
        self._path = snap if isinstance(snap, (str, os.PathLike)) else snap.get("path")
        self._snap = None if isinstance(snap, (str, os.PathLike)) else snap
        self._r = None
        self._lock = threading.Lock()
        self._sig = None
        self.load_seconds = 0.0

    @property
    def sig(self):
        if self._sig is None and self._path is not None and os.path.exists(self._path):
            self._sig = snapshot_hash(str(self._path))
        return self._sig

    def renderer(self):
        with self._lock:
            if self._r is None:
                if self._snap is None:
                    from .io import load_snapshot
                    t0 = time.perf_counter()
                    self._snap = load_snapshot(str(self._path), gas_fields=tuple(sorted(required_gas_fields(self.view))))
                    self.load_seconds = time.perf_counter() - t0
                self._r = TileRenderer(self._snap, self.view, self.grid)
            return self._r


def _ncpu():
    return nb.config.NUMBA_NUM_THREADS


def _auto_workers(ntiles, tile):
    return max(1, min(_ncpu(), ntiles))


def _one(src, store, ix, iy):
    key = None
    if store is not None and src.sig is not None:
        g = src.grid
        key = tile_key(src.sig, src.view, g.tile, g.tw, g.ox, g.oy, ix, iy, g.level)
        got = store.get(key)
        if got is not None:
            store.hits += 1
            return got[0], got[1], True
    t0 = time.perf_counter()
    img, meta = src.renderer().render(ix, iy)
    meta["render_seconds"] = round(time.perf_counter() - t0, 4)
    meta["tile"] = {"tile": src.grid.tile, "ix": ix, "iy": iy, "level": src.grid.level, "extent": src.grid.extent(ix, iy)}
    meta.pop("units", None)
    if key is not None:
        meta["key"] = key
        store.misses += 1
        store.put(key, img, meta)
    return img, meta, False


def iter_tiles(snap, view, grid, tiles=None, store=None, workers=None):
    """Yield (ix, iy, image, meta, cache_hit) for the tiles of `grid` (default: all its covering tiles), in order, holding at most about 2 x workers images.
    `snap` is a loaded snapshot dict or a path (read only if a tile is not in the cache)."""
    tiles = list(grid.tiles() if tiles is None else tiles)
    src = _Source(snap, view, grid)
    w = workers or _auto_workers(len(tiles), grid.tile)
    w = max(1, min(int(w), max(1, len(tiles))))
    nthreads = max(1, _ncpu() // w)

    def work(t):
        nb.set_num_threads(nthreads)
        return _one(src, store, *t)
    if w == 1:
        for t in tiles:
            yield (t[0], t[1], *work(t))
        return
    with ThreadPoolExecutor(w) as ex:
        pend = deque()
        it = iter(tiles)
        for t in it:
            pend.append((t, ex.submit(work, t)))
            if len(pend) >= 2 * w:
                break
        while pend:
            t, f = pend.popleft()
            r = f.result()
            nxt = next(it, None)
            if nxt is not None:
                pend.append((nxt, ex.submit(work, nxt)))
            yield (t[0], t[1], *r)


def render_tile(snap, view, level, ix, iy, tile=DEFAULT_TILE, base_width=None, store=None, renderer=None):
    """One tile of the global grid: (image (tile, tile) float32, meta). Cached in `store` when given and the snapshot has a path (or `snap` is a path).
    `renderer`: a TileRenderer for the same snapshot, view and grid, to reuse the frame and the particle binning over many tiles (render_tiles)."""
    tile = _check_tile(tile)
    bw = float(base_width or default_base_width(view.width))
    grid = TileGrid(tile, bw / 2.0 ** int(level), 0.0, 0.0, int(level), bw)
    if renderer is not None:
        src = _Source(renderer.snap, view, grid)
        src._r = renderer
    else:
        src = _Source(snap, view, grid)
    img, meta, hit = _one(src, store, int(ix), int(iy))
    return img, dict(meta, cached=hit)


def render_tiles(snap, view, grid, store=None, workers=None):
    """List of (ix, iy, image, meta, hit) for all tiles of a grid (parallel; for many tiles prefer iter_tiles, which does not keep them)."""
    return list(iter_tiles(snap, view, grid, store=store, workers=workers))


# ------------------------------------------------------------------------------------------------------ assembled image
def render_tiled(snap, view, tile=DEFAULT_TILE, out=None, store=None, workers=None, progress=None, limit_npix=65536):
    """The image of `view` assembled from tiles of the window grid: (image (npix, npix) float32, meta). Equal to render_map(snap, view) within the engine tolerance
    (module docstring), for any npix up to `limit_npix` (View.npix itself is limited to 8192: the view passed here is validated with the npix capped, the tiles are
    rendered at the requested pixel size). `out`: None (an array in memory), a path ending in .npy (a memory-mapped array written progressively, so a 16384^2 image
    never needs to fit in RAM), or an existing array / memmap of shape (npix, npix) to fill. `progress(done, total)` is called after each tile.
    Rows are y, row 0 the lowest y, as render_map."""
    if view.npix > limit_npix:
        raise ValueError(f"npix {view.npix} above the tiled limit {limit_npix}")
    tile = _check_tile(tile)
    view.with_(npix=tile).validate()
    grid = window_grid(view, tile)
    n = view.npix
    if out is None:
        img = np.empty((n, n), np.float32)
    elif isinstance(out, (str, os.PathLike)):
        img = np.lib.format.open_memmap(str(out), mode="w+", dtype=np.float32, shape=(n, n))
    else:
        img = out
        if img.shape != (n, n) or img.dtype != np.float32:
            raise ValueError(f"out must be float32 with shape {(n, n)}")
    first = None
    nt = grid.nx * grid.ny
    hits = 0
    for k, (ix, iy, t_img, meta, hit) in enumerate(iter_tiles(snap, view, grid, store=store, workers=workers)):
        j0, i0 = iy * tile, ix * tile
        h, w = min(tile, n - j0), min(tile, n - i0)
        img[j0:j0 + h, i0:i0 + w] = t_img[:h, :w]
        hits += int(hit)
        if first is None:
            first = meta
        if progress:
            progress(k + 1, nt)
    if hasattr(img, "flush"):
        img.flush()
    half = 0.5 * view.width
    meta = dict(first or {})
    f = meta.get("length_unit", {}).get("factor", 1.0)
    ext = [view.cx - half, view.cx + half, view.cy - half, view.cy + half]
    meta.update(view=view.as_dict(), extent=ext, extent_physical=[e * f for e in ext], tiles=dict(grid.describe(), cache_hits=hits, kind="window"))
    meta.pop("tile", None); meta.pop("key", None); meta.pop("n_candidates", None)
    return img, meta


# ------------------------------------------------------------------------------------------------------ streaming PNG for big images
def write_png_streaming(path, img, cmap="viridis", scale="log", vmin=None, vmax=None, band=512, sample=4_000_000):
    """Colourised PNG of a big (memory-mapped) map without holding the RGB image: limits from a regular subsample of at most `sample` pixels (or the given
    ones), then bands of `band` rows (top of the picture first, as imaging.colorize) are coloured, compressed and written. Returns (vmin, vmax)."""
    ny, nx = img.shape
    if vmin is None or vmax is None:
        s = max(1, int(math.ceil(math.sqrt(ny * nx / sample))))
        lo, hi = imaging.limits(np.asarray(img[::s, ::s]), scale=scale)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    comp = zlib.compressobj(6)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", nx, ny, 8, 2, 0, 0, 0)))
        for top in range(0, ny, band):                   # picture rows top..bottom = map rows ny-1..0
            r1 = ny - top
            r0 = max(0, r1 - band)
            rgb, _ = imaging.colorize(np.asarray(img[r0:r1]), cmap, scale, vmin, vmax)
            raw = b"".join(b"\x00" + rgb[j].tobytes() for j in range(rgb.shape[0]))
            data = comp.compress(raw)
            if data:
                f.write(chunk(b"IDAT", data))
        f.write(chunk(b"IDAT", comp.flush()) + chunk(b"IEND", b""))
    return float(vmin), float(vmax)


def tile_index(grid, store, view, snapshot_path):
    """Listing of the grid's tiles: dicts with ix, iy, extent, key and whether the store has it."""
    sig = snapshot_hash(snapshot_path)
    out = []
    for ix, iy in grid.tiles():
        key = tile_key(sig, view, grid.tile, grid.tw, grid.ox, grid.oy, ix, iy, grid.level)
        out.append({"ix": ix, "iy": iy, "extent": grid.extent(ix, iy), "key": key, "cached": store.get(key) is not None if store is not None else False})
    return out
