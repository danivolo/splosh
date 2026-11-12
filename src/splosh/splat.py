"""Numba splatting of SPH/GIZMO particles onto a regular image.

Given particles (x, y[, z], H, weight, nq quantities) it returns an image of shape (ny, nx, nq+1):
    out[..., 0]   = sum_i  w_i * K_i(pixel)                   (the weight image)
    out[..., k+1] = sum_i  w_i * q_k,i * K_i(pixel)
with K the column-integrated kernel (mode 'proj') or the kernel at the plane z = z0 (mode 'slice').
With w = mass:     out0 = column density, out_k / out0 = mass-weighted mean of q_k along the line of sight.
With w = m / rho:  slice: out_k = interpolated field q_k at z0, out0 ~ 1 (normalised interpolation: out_k / out0).
Variants (same result up to float summation order), kept for the benchmark:
    splat_serial      one thread, reference for speed
    splat_tl          thread-local images + reduction
    splat_bands       image rows split into bands owned by threads (no reduction, no atomics), cost-balanced
    splat_bands_pyr   as bands, but particles much larger than a pixel go to coarser grids that are upsampled (large-H speed-up)
Particles with a support radius under one pixel are deposited cloud-in-cell on the 4 nearest pixel centres; footprints of up to ~6 pixels are
renormalised to the exact particle weight (the kernel sampled at a few pixel centres does not integrate to one).
"""
import numpy as np
import numba as nb
from numba import njit, prange

from .kernel import tables

_T2, _T3 = tables()
NT = len(_T2) - 1
import os as _os
_NEAREST = _os.environ.get("SPLOSH_LOOKUP", "nearest") == "nearest"      # nearest-bin lookup: same accuracy at 4096 bins, 1.4x faster (measured)
_FM = _os.environ.get("SPLOSH_FASTMATH", "0") == "1"


# ------------------------------------------------------------------------------------------------------ geometry helpers
@njit(cache=True, inline="always")
def _lookup(tab, q2):
    if _NEAREST:
        return tab[int(q2 * NT + 0.5)]
    u = q2 * NT
    k = int(u)
    if k >= NT:
        return tab[NT]
    f = u - k
    return tab[k] * (1.0 - f) + tab[k + 1] * f


@njit(cache=True)
def _footprint(px, py, hp, nx, ny):
    i0 = max(0, int(np.floor(px - hp)))
    i1 = min(nx - 1, int(np.ceil(px + hp)))
    j0 = max(0, int(np.floor(py - hp)))
    j1 = min(ny - 1, int(np.ceil(py + hp)))
    return i0, i1, j0, j1


@njit(cache=True)
def _norm_scale(px, py, hp, i0, i1, j0, j1, tab, pref_pix):
    """1 / (sum of the sampled kernel over the footprint * pixel area) for small footprints; weight conservation."""
    s = 0.0
    inv = 1.0 / (hp * hp)
    for j in range(j0, j1 + 1):
        dy = j + 0.5 - py
        for i in range(i0, i1 + 1):
            dx = i + 0.5 - px
            q2 = (dx * dx + dy * dy) * inv
            if q2 < 1.0:
                s += _lookup(tab, q2)
    s *= pref_pix
    return 1.0 / s if s > 0 else 0.0


@njit(cache=True)
def _cic(img, nq1, vals, p, px, py, k, nx, ny, rlo, rhi):
    """cloud-in-cell: k (weight per unit area) shared between the 4 pixel centres around (px, py); only rows rlo <= j < rhi are written."""
    fx = px - 0.5; fy = py - 0.5
    i0 = int(np.floor(fx)); j0 = int(np.floor(fy))
    tx = fx - i0; ty = fy - j0
    for dj in range(2):
        j = j0 + dj
        if j < max(rlo, 0) or j >= min(rhi, ny):
            continue
        wy = ty if dj else 1.0 - ty
        for di in range(2):
            i = i0 + di
            if i < 0 or i >= nx:
                continue
            kk = k * wy * (tx if di else 1.0 - tx)
            img[j, i, 0] += kk
            for c in range(1, nq1):
                img[j, i, c] += kk * vals[c - 1, p]


@njit(cache=True, inline="always", fastmath=_FM)
def _deposit(img, nq1, vals, p, px, py, hp, i0, i1, j0, j1, wgt, tab, scale, nx):
    """add particle p over the pixel box [i0..i1] x [j0..j1] of img (shape (rows, nx, nq1)); wgt = w / (H^2 in pixel^2 -> per-pixel units)."""
    inv = 1.0 / (hp * hp)
    for j in range(j0, j1 + 1):
        dy = j + 0.5 - py
        dy2 = dy * dy
        for i in range(i0, i1 + 1):
            dx = i + 0.5 - px
            q2 = (dx * dx + dy2) * inv
            if q2 < 1.0:
                k = wgt * scale * _lookup(tab, q2)
                img[j, i, 0] += k
                for c in range(1, nq1):
                    img[j, i, c] += k * vals[c - 1, p]


# ------------------------------------------------------------------------------------------------------ serial (projection only for now)
@njit(cache=True)
def _splat_range(x, y, h, w, vals, cx, cy, dx, nx, ny, img, lo, hi, small_cut, t2):
    inv_dx = 1.0 / dx
    nq1 = img.shape[2]
    for p in range(lo, hi):
        px = (x[p] - cx) * inv_dx + 0.5 * nx
        py = (y[p] - cy) * inv_dx + 0.5 * ny
        hp = h[p] * inv_dx
        if hp <= 0.0:
            continue
        i0, i1, j0, j1 = _footprint(px, py, hp, nx, ny)
        if px + hp < 0 or py + hp < 0 or px - hp > nx or py - hp > ny:
            continue
        wgt = w[p] / (h[p] * h[p])          # integral of the projected kernel over one pixel area dx^2 -> image per unit area
        if hp < 1.0:                         # support smaller than a pixel: cloud-in-cell onto the 4 nearest pixel centres (conserves the weight)
            _cic(img, nq1, vals, p, px, py, w[p] / (dx * dx), nx, ny, 0, ny)
            continue
        scale = 1.0
        if hp < small_cut:
            scale = _norm_scale(px, py, hp, i0, i1, j0, j1, t2, 1.0 / (hp * hp))   # sampled integral = sum f / hp^2: make it exactly 1
        _deposit(img, nq1, vals, p, px, py, hp, i0, i1, j0, j1, wgt, t2, scale, nx)


def _check_inputs(x, y, h, w, vals):
    n = len(x)
    assert len(y) == n and len(h) == n and len(w) == n and vals.shape[1] == n and vals.flags["C_CONTIGUOUS"]


def splat_serial(x, y, h, w, vals, center, width, npix, small_cut=6.0):
    """Single-threaded reference. vals: (nq, N) float32 C-contiguous. width: image width in the units of x. Returns (npix, npix, nq+1) float32."""
    _check_inputs(x, y, h, w, vals)
    img = np.zeros((npix, npix, vals.shape[0] + 1), np.float32)
    _splat_range(x, y, h, w, vals, center[0], center[1], width / npix, npix, npix, img, 0, len(x), small_cut, _T2)
    return img


@njit(cache=True, parallel=True)
def _tl_kernel(x, y, h, w, vals, cx, cy, dx, nx, ny, nthreads, small_cut, t2):
    nq1 = vals.shape[0] + 1
    imgs = np.zeros((nthreads, ny, nx, nq1), np.float32)
    n = len(x)
    chunk = (n + nthreads - 1) // nthreads
    for t in prange(nthreads):
        _splat_range(x, y, h, w, vals, cx, cy, dx, nx, ny, imgs[t], t * chunk, min(n, (t + 1) * chunk), small_cut, t2)
    out = np.zeros((ny, nx, nq1), np.float32)
    for t in range(nthreads):
        out += imgs[t]
    return out


def splat_tl(x, y, h, w, vals, center, width, npix, nthreads=None, small_cut=6.0):
    """Thread-local images + reduction. Particles are processed in contiguous chunks of the input order."""
    _check_inputs(x, y, h, w, vals)
    nthreads = nthreads or nb.get_num_threads()
    return _tl_kernel(x, y, h, w, vals, center[0], center[1], width / npix, npix, npix, nthreads, small_cut, _T2)


# ------------------------------------------------------------------------------------------------------ row bands (no reduction)
@njit(cache=True, parallel=True)
def _row_ranges(x, y, h, cx, cy, dx, nx, ny):
    """per particle: first and last image row touched (j1 < j0 = none) and the cost (pixels in the footprint)."""
    n = len(x)
    j0a = np.empty(n, np.int32); j1a = np.empty(n, np.int32); cost = np.empty(n, np.float32)
    inv = 1.0 / dx
    for p in prange(n):
        px = (x[p] - cx) * inv + 0.5 * nx
        py = (y[p] - cy) * inv + 0.5 * ny
        hp = h[p] * inv
        if px + hp < 0 or px - hp > nx or py + hp < 0 or py - hp > ny or hp <= 0:
            j0a[p] = 1; j1a[p] = 0; cost[p] = 0
            continue
        i0, i1, j0, j1 = _footprint(px, py, hp, nx, ny)
        if hp < 1.0:
            j0 = max(int(np.floor(py - 0.5)), 0); j1 = min(int(np.floor(py - 0.5)) + 1, ny - 1)
            cost[p] = 4.0
        else:
            cost[p] = (i1 - i0 + 1) * (j1 - j0 + 1)
        j0a[p] = j0; j1a[p] = j1
    return j0a, j1a, cost


@njit(cache=True)
def _nchunks(n, nthreads):
    """chunks of consecutive particles for the parallel set-up passes: a few per thread, at least ~20k particles each (the thread count is an
    argument: a function that asks numba for it cannot be cached)."""
    return max(1, min(4 * nthreads, (n + 19999) // 20000))


@njit(cache=True, parallel=True)
def _band_edges(j0a, j1a, cost, ny, nbands, nthreads):
    """row edges (nbands+1) with equal total cost (the row costs summed per chunk of particles in parallel)."""
    n = len(j0a)
    nch = _nchunks(n, nthreads)
    size = (n + nch - 1) // nch
    rc = np.zeros((nch, ny + 1))
    for ch in prange(nch):
        for p in range(ch * size, min(n, (ch + 1) * size)):
            if j1a[p] >= j0a[p]:
                c = cost[p] / (j1a[p] - j0a[p] + 1)
                for j in range(j0a[p], j1a[p] + 1):
                    rc[ch, j] += c
    rowcost = np.zeros(ny + 1)
    for ch in range(nch):
        rowcost += rc[ch]
    tot = rowcost.sum()
    edges = np.zeros(nbands + 1, np.int64)
    acc = 0.0
    b = 1
    for j in range(ny):
        acc += rowcost[j]
        while b < nbands and acc >= tot * b / nbands:
            edges[b] = j + 1
            b += 1
    for k in range(b, nbands + 1):
        edges[k] = ny
    edges[nbands] = ny
    return edges


@njit(cache=True, parallel=True)
def _band_lists(j0a, j1a, edges, ny, nthreads):
    """the particle list of every band (start, idx), particles in ascending order inside a band: counted per (chunk, band) in parallel, then filled
    in parallel at the offsets of each chunk. The order (hence every pixel's sum) is the same as a serial pass."""
    nb_ = len(edges) - 1
    band_of_row = np.empty(ny, np.int32)
    for b in range(nb_):
        for j in range(edges[b], edges[b + 1]):
            band_of_row[j] = b
    n = len(j0a)
    nch = _nchunks(n, nthreads)
    size = (n + nch - 1) // nch
    counts = np.zeros((nch, nb_), np.int64)
    for ch in prange(nch):
        for p in range(ch * size, min(n, (ch + 1) * size)):
            if j1a[p] >= j0a[p]:
                for b in range(band_of_row[j0a[p]], band_of_row[j1a[p]] + 1):
                    counts[ch, b] += 1
    start = np.zeros(nb_ + 1, np.int64)
    off = np.empty((nch, nb_), np.int64)
    for b in range(nb_):
        acc = start[b]
        for ch in range(nch):
            off[ch, b] = acc
            acc += counts[ch, b]
        start[b + 1] = acc
    idx = np.empty(start[nb_], np.int64)
    for ch in prange(nch):
        fill = off[ch].copy()
        for p in range(ch * size, min(n, (ch + 1) * size)):
            if j1a[p] >= j0a[p]:
                for b in range(band_of_row[j0a[p]], band_of_row[j1a[p]] + 1):
                    idx[fill[b]] = p
                    fill[b] += 1
    return start, idx


@njit(cache=True, parallel=True)
def _norm_scales(x, y, h, cx, cy, dx, nx, ny, small_cut, t2):
    """the weight-conservation scale of every small particle (1 <= hp < small_cut; 1.0 otherwise), once per particle: a particle spanning several
    row bands would otherwise compute it in every band (a sum over its whole footprint), and the number of bands grows with the threads."""
    inv_dx = 1.0 / dx
    out = np.ones(len(x))
    for p in prange(len(x)):
        hp = h[p] * inv_dx
        if 1.0 <= hp < small_cut:
            px = (x[p] - cx) * inv_dx + 0.5 * nx
            py = (y[p] - cy) * inv_dx + 0.5 * ny
            i0, i1, j0, j1 = _footprint(px, py, hp, nx, ny)
            out[p] = _norm_scale(px, py, hp, i0, i1, j0, j1, t2, 1.0 / (hp * hp))
    return out


@njit(cache=True)
def _band_work(x, y, h, w, vals, cx, cy, dx, nx, ny, img, rlo, rhi, plist, small_cut, t2, scales):
    """splat the listed particles into rows [rlo, rhi) of img (full-size image; only those rows are touched). scales: _norm_scales."""
    inv_dx = 1.0 / dx
    nq1 = img.shape[2]
    for a in range(len(plist)):
        p = plist[a]
        px = (x[p] - cx) * inv_dx + 0.5 * nx
        py = (y[p] - cy) * inv_dx + 0.5 * ny
        hp = h[p] * inv_dx
        if hp < 1.0:
            _cic(img, nq1, vals, p, px, py, w[p] / (dx * dx), nx, ny, rlo, rhi)
            continue
        i0, i1, j0, j1 = _footprint(px, py, hp, nx, ny)
        j0c, j1c = max(j0, rlo), min(j1, rhi - 1)
        if j1c < j0c:
            continue
        wgt = w[p] / (h[p] * h[p])
        _deposit(img, nq1, vals, p, px, py, hp, i0, i1, j0c, j1c, wgt, t2, scales[p], nx)


@njit(cache=True, parallel=True)
def _bands_kernel(x, y, h, w, vals, cx, cy, dx, nx, ny, nthreads, bands_per_thread, small_cut, t2):
    nq1 = vals.shape[0] + 1
    j0a, j1a, cost = _row_ranges(x, y, h, cx, cy, dx, nx, ny)
    nbands = nthreads * bands_per_thread
    edges = _band_edges(j0a, j1a, cost, ny, nbands, nthreads)
    start, idx = _band_lists(j0a, j1a, edges, ny, nthreads)
    scales = _norm_scales(x, y, h, cx, cy, dx, nx, ny, small_cut, t2)
    img = np.zeros((ny, nx, nq1), np.float32)
    for t in prange(nthreads):
        for b in range(t, nbands, nthreads):  # interleave the bands over the threads
            if edges[b + 1] > edges[b]:
                _band_work(x, y, h, w, vals, cx, cy, dx, nx, ny, img, edges[b], edges[b + 1], idx[start[b]:start[b + 1]], small_cut, t2, scales)
    return img


def splat_bands(x, y, h, w, vals, center, width, npix, nthreads=None, bands_per_thread=4, small_cut=6.0):
    """Rows of the image are owned by bands (cost-balanced); each band's particle list is processed by one thread: no reduction, no atomics."""
    _check_inputs(x, y, h, w, vals)
    nthreads = nthreads or nb.get_num_threads()
    return _bands_kernel(x, y, h, w, vals, center[0], center[1], width / npix, npix, npix, nthreads, bands_per_thread, small_cut, _T2)


# ------------------------------------------------------------------------------------------------------ pyramid for large particles
@njit(cache=True, parallel=True)
def _levels_order(hp, big, nlev, nthreads):
    """(order, start): the particles sorted by level, stable (= argsort(level, kind="stable")), by a parallel counting sort."""
    n = len(hp)
    lev = np.empty(n, np.int64)
    for p in prange(n):
        k = np.ceil(np.log2(hp[p] / big))
        lev[p] = 0 if k < 0 else (nlev if k > nlev else int(k))
    nch = _nchunks(n, nthreads)
    size = (n + nch - 1) // nch
    counts = np.zeros((nch, nlev + 1), np.int64)
    for ch in prange(nch):
        for p in range(ch * size, min(n, (ch + 1) * size)):
            counts[ch, lev[p]] += 1
    start = np.zeros(nlev + 2, np.int64)
    off = np.empty((nch, nlev + 1), np.int64)
    for k in range(nlev + 1):
        acc = start[k]
        for ch in range(nch):
            off[ch, k] = acc
            acc += counts[ch, k]
        start[k + 1] = acc
    order = np.empty(n, np.int64)
    for ch in prange(nch):
        fill = off[ch].copy()
        for p in range(ch * size, min(n, (ch + 1) * size)):
            order[fill[lev[p]]] = p
            fill[lev[p]] += 1
    return order, start


@njit(cache=True, parallel=True)
def _gather(a, idx):
    out = np.empty(len(idx), a.dtype)
    for k in prange(len(idx)):
        out[k] = a[idx[k]]
    return out


@njit(cache=True, parallel=True)
def _gather_rows(v, idx):
    out = np.empty((v.shape[0], len(idx)), v.dtype)
    for k in prange(len(idx)):
        for r in range(v.shape[0]):
            out[r, k] = v[r, idx[k]]
    return out


@njit(cache=True, parallel=True)
def _footprints(h, inv_dx):
    out = np.empty(len(h), np.float64)
    for p in prange(len(h)):
        out[p] = max(h[p] * inv_dx, 1e-9)
    return out


def _pyr_levels(hp, npix, big, min_npix):
    """Level of each particle (footprint radius hp pixels): the smallest k with hp / 2^k <= big, at most the coarsest level (npix / 2^k >= min_npix)."""
    nlev = 0
    while npix // 2 ** (nlev + 1) >= min_npix:
        nlev += 1
    order, start = _levels_order(np.asarray(hp, np.float64), float(big), nlev, nb.get_num_threads())
    return nlev, order, start


def _pyr_render(hp, npix, nq1, big, min_npix, render):
    """Render the levels with render(idx, n_level) -> image or None (idx = particle indices of that level) and merge them coarse to fine."""
    nlev, order, start = _pyr_levels(hp, npix, big, min_npix)
    levels = []                                       # (npix_level, image or None), finest first
    for k in range(nlev + 1):
        n_l = npix // 2 ** k
        lo, hi = start[k], start[k + 1]
        levels.append((n_l, render(order[lo:hi], n_l) if hi > lo else None))
    out = None
    for n_l, im in reversed(levels):                  # coarsest first
        if out is not None:
            out = _upsample(out, n_l, im)             # fused: upsample + add this level's own image (None = just upsample)
        else:
            out = im
    if out is None:
        return np.zeros((npix, npix, nq1), np.float32)
    return out if out.shape[0] == npix else _upsample(out, npix, None)


def splat_bands_pyr(x, y, h, w, vals, center, width, npix, nthreads=None, big=12.0, min_npix=64, bands_per_thread=4, small_cut=6.0):
    """Particles whose footprint radius exceeds `big` pixels are rendered on a grid coarsened by 2^k (so their radius is at most `big` coarse pixels)
    and the levels are merged from the coarsest to the finest by 2x bicubic upsampling and adding. The kernel is smooth, so the error is small
    (median ~1e-4 relative at big = 12) while the cost of the biggest particles drops by 4^k."""
    _check_inputs(x, y, h, w, vals)
    nthreads = nthreads or nb.get_num_threads()
    hp = np.maximum(h / (width / npix), 1e-9)

    def render(idx, n_l):                             # the level's particles gathered in parallel (numpy fancy indexing is serial)
        return _bands_kernel(_gather(x, idx), _gather(y, idx), _gather(h, idx), _gather(w, idx), _gather_rows(vals, idx), center[0], center[1], width / n_l, n_l, n_l,
                             nthreads, bands_per_thread, small_cut, _T2)
    return _pyr_render(hp, npix, vals.shape[0] + 1, big, min_npix, render)


# ------------------------------------------------------------------------------------------------------ slices (cross-section)
# A slice is the 3D kernel evaluated on the plane z = z0:  out0 = sum_i w_i W(r_i) with W = w3(q)/H^3, q^2 = (dx^2 + dy^2 + dz^2) / H^2.
# Only particles with |dz| < H contribute, with the in-plane footprint radius H sqrt(1 - dz^2/H^2). Pixel centres are sampled (no cloud-in-cell, no
# renormalisation): the kernel of a particle smaller than a pixel is point-sampled, so a slice should be rendered at a resolution finer than the
# typical H (or use the normalised form out_k / out0 with w = m / rho, where the sampling cancels to first order).
@njit(cache=True)
def _slice_work(x, y, h, q2z, w, vals, cx, cy, dx, nx, ny, img, rlo, rhi, plist, t3):
    inv_dx = 1.0 / dx
    nq1 = img.shape[2]
    for a in range(len(plist)):
        p = plist[a]
        px = (x[p] - cx) * inv_dx + 0.5 * nx
        py = (y[p] - cy) * inv_dx + 0.5 * ny
        hp = h[p] * inv_dx
        he = hp * np.sqrt(1.0 - q2z[p])
        i0, i1, j0, j1 = _footprint(px, py, he, nx, ny)
        j0 = max(j0, rlo)
        j1 = min(j1, rhi - 1)
        wgt = w[p] / (h[p] * h[p] * h[p])
        inv = 1.0 / (hp * hp)
        qz = q2z[p]
        for j in range(j0, j1 + 1):
            dy = j + 0.5 - py
            dy2 = dy * dy
            for i in range(i0, i1 + 1):
                ddx = i + 0.5 - px
                q2 = (ddx * ddx + dy2) * inv + qz
                if q2 < 1.0:
                    k = wgt * _lookup(t3, q2)
                    img[j, i, 0] += k
                    for c in range(1, nq1):
                        img[j, i, c] += k * vals[c - 1, p]


@njit(cache=True, parallel=True)
def _bands_kernel_slice(x, y, h, q2z, w, vals, cx, cy, dx, nx, ny, nthreads, bands_per_thread, t3):
    nq1 = vals.shape[0] + 1
    heff = h * np.sqrt(1.0 - q2z)
    j0a, j1a, cost = _row_ranges(x, y, heff, cx, cy, dx, nx, ny)
    nbands = nthreads * bands_per_thread
    edges = _band_edges(j0a, j1a, cost, ny, nbands, nthreads)
    start, idx = _band_lists(j0a, j1a, edges, ny, nthreads)
    img = np.zeros((ny, nx, nq1), np.float32)
    for t in prange(nthreads):
        for r in range(bands_per_thread):
            b = r * nthreads + t
            if edges[b + 1] > edges[b]:
                _slice_work(x, y, h, q2z, w, vals, cx, cy, dx, nx, ny, img, edges[b], edges[b + 1], idx[start[b]:start[b + 1]], t3)
    return img


def splat_slice(x, y, z, h, w, vals, center, width, npix, nthreads=None, big=12.0, min_npix=64, bands_per_thread=4, pyramid=True):
    """Slice at z = center[2] through the particles: (npix, npix, nq+1) float32, out0 = sum w W, out_k = sum w q_k W (W = 3D kernel, units 1/length^3).
    With w = m: out0 is the SPH density and out_k / out0 the mass-weighted mean; with w = m/rho: out_k / out0 the normalised interpolation of q_k.
    pyramid=True renders particles larger than `big` pixels on coarser grids (see splat_bands_pyr); False is the exact (slower) variant."""
    _check_inputs(x, y, h, w, vals)
    nthreads = nthreads or nb.get_num_threads()
    cz = center[2] if len(center) > 2 else 0.0
    q2z_all = ((z - cz) / h).astype(np.float64) ** 2
    keep = np.nonzero(q2z_all < 1.0)[0]
    nq1 = vals.shape[0] + 1
    if len(keep) == 0:
        return np.zeros((npix, npix, nq1), np.float32)
    xs, ys, hs, ws, qs = x[keep], y[keep], h[keep], w[keep], q2z_all[keep]
    vs = np.ascontiguousarray(vals[:, keep])
    if not pyramid:
        return _bands_kernel_slice(xs, ys, hs, qs, ws, vs, center[0], center[1], width / npix, npix, npix, nthreads, bands_per_thread, _T3)
    hp = np.maximum(hs / (width / npix), 1e-9)

    def render(idx, n_l):
        return _bands_kernel_slice(xs[idx], ys[idx], hs[idx], qs[idx], ws[idx], np.ascontiguousarray(vs[:, idx]), center[0], center[1], width / n_l, n_l, n_l,
                                   nthreads, bands_per_thread, _T3)
    return _pyr_render(hp, npix, nq1, big, min_npix, render)


@njit(cache=True)
def _cr_weights(n_in, n_out):
    """Catmull-Rom taps for resampling n_in point samples (at pixel centres) to n_out: indices (n_out,4) and weights (n_out,4)."""
    f = n_in / n_out
    idx = np.empty((n_out, 4), np.int64)
    wt = np.empty((n_out, 4), np.float32)
    for o in range(n_out):
        s = (o + 0.5) * f - 0.5
        k = int(np.floor(s))
        t = s - k
        t2 = t * t
        t3 = t2 * t
        wt[o, 0] = 0.5 * (-t3 + 2.0 * t2 - t)
        wt[o, 1] = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
        wt[o, 2] = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
        wt[o, 3] = 0.5 * (t3 - t2)
        for a in range(4):
            idx[o, a] = min(max(k - 1 + a, 0), n_in - 1)
    return idx, wt


@njit(cache=True, parallel=True)
def _upsample(im, npix, base=None):
    """bicubic (Catmull-Rom) upsampling of a coarse image (n,n,c) to (npix,npix,c), plus `base` (same size) if given; fused, separable weights."""
    n = im.shape[0]
    nc = im.shape[2]
    ix, wx = _cr_weights(n, npix)
    out = np.empty((npix, npix, nc), np.float32)
    for j in prange(npix):
        r0, r1, r2, r3 = ix[j, 0], ix[j, 1], ix[j, 2], ix[j, 3]
        w0, w1, w2, w3 = wx[j, 0], wx[j, 1], wx[j, 2], wx[j, 3]
        for c in range(nc):
            for i in range(npix):
                c0, c1, c2, c3 = ix[i, 0], ix[i, 1], ix[i, 2], ix[i, 3]
                v0 = wx[i, 0] * im[r0, c0, c] + wx[i, 1] * im[r0, c1, c] + wx[i, 2] * im[r0, c2, c] + wx[i, 3] * im[r0, c3, c]
                v1 = wx[i, 0] * im[r1, c0, c] + wx[i, 1] * im[r1, c1, c] + wx[i, 2] * im[r1, c2, c] + wx[i, 3] * im[r1, c3, c]
                v2 = wx[i, 0] * im[r2, c0, c] + wx[i, 1] * im[r2, c1, c] + wx[i, 2] * im[r2, c2, c] + wx[i, 3] * im[r2, c3, c]
                v3 = wx[i, 0] * im[r3, c0, c] + wx[i, 1] * im[r3, c1, c] + wx[i, 2] * im[r3, c2, c] + wx[i, 3] * im[r3, c3, c]
                v = w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3
                if base is not None:
                    v += base[j, i, c]
                out[j, i, c] = v
    return out


# ------------------------------------------------------------------------------------------------------ volume rendering (emission + absorption)
# Particles sorted FRONT TO BACK. Each particle contributes, at every pixel it covers, the optical depth tau = kappa * Sigma_i (its column density there, the
# same projected kernel as the projections) and emits its source value S: the pixel accumulates  E += T (1 - exp(-tau)) S  and  T *= exp(-tau)  (T = the light
# that still gets through from behind, 1 at the start). Rows are owned by bands as in splat_bands, and the bands keep the particle order, so the compositing
# order is exact without locks. Output (ny, nx, 2): [..., 0] = E, [..., 1] = 1 - T (the opacity).

@njit(cache=True, inline="always")
def _composite(img, trans, j, i, tau, s):
    t = trans[j, i]
    if t < 1e-6:
        return
    a = 1.0 - np.exp(-tau)
    img[j, i, 0] += t * a * s
    img[j, i, 1] += t * a
    trans[j, i] = t * (1.0 - a)


@njit(cache=True)
def _volume_band(x, y, h, w, src, kappa, cx, cy, dx, nx, ny, img, trans, rlo, rhi, plist, small_cut, t2):
    inv_dx = 1.0 / dx
    for a in range(len(plist)):
        p = plist[a]
        px = (x[p] - cx) * inv_dx + 0.5 * nx
        py = (y[p] - cy) * inv_dx + 0.5 * ny
        hp = h[p] * inv_dx
        s = src[p]
        if hp < 1.0:                                    # smaller than a pixel: its column density shared cloud-in-cell between the 4 pixel centres
            sig = kappa * w[p] / (dx * dx)
            fx = px - 0.5; fy = py - 0.5
            i0 = int(np.floor(fx)); j0 = int(np.floor(fy))
            tx = fx - i0; ty = fy - j0
            for dj in range(2):
                j = j0 + dj
                if j < max(rlo, 0) or j >= min(rhi, ny):
                    continue
                wy = ty if dj else 1.0 - ty
                for di in range(2):
                    i = i0 + di
                    if 0 <= i < nx:
                        _composite(img, trans, j, i, sig * wy * (tx if di else 1.0 - tx), s)
            continue
        i0, i1, j0, j1 = _footprint(px, py, hp, nx, ny)
        scale = 1.0
        if hp < small_cut:
            scale = _norm_scale(px, py, hp, i0, i1, j0, j1, t2, 1.0 / (hp * hp))
        wgt = kappa * w[p] / (h[p] * h[p]) * scale
        inv = 1.0 / (hp * hp)
        for j in range(max(j0, rlo), min(j1, rhi - 1) + 1):
            dy = j + 0.5 - py
            for i in range(i0, i1 + 1):
                ddx = i + 0.5 - px
                q2 = (ddx * ddx + dy * dy) * inv
                if q2 < 1.0:
                    _composite(img, trans, j, i, wgt * _lookup(t2, q2), s)


@njit(cache=True, parallel=True)
def _volume_kernel(x, y, h, w, src, kappa, cx, cy, dx, nx, ny, nthreads, bands_per_thread, small_cut, t2):
    j0a, j1a, cost = _row_ranges(x, y, h, cx, cy, dx, nx, ny)
    nbands = nthreads * bands_per_thread
    edges = _band_edges(j0a, j1a, cost, ny, nbands, nthreads)
    start, idx = _band_lists(j0a, j1a, edges, ny, nthreads)          # each band's list keeps the input (front-to-back) order
    img = np.zeros((ny, nx, 2), np.float32)
    trans = np.ones((ny, nx), np.float64)
    for t in prange(nthreads):
        for r in range(bands_per_thread):
            b = r * nthreads + t
            if edges[b + 1] > edges[b]:
                _volume_band(x, y, h, w, src, kappa, cx, cy, dx, nx, ny, img, trans, edges[b], edges[b + 1], idx[start[b]:start[b + 1]], small_cut, t2)
    return img


def splat_volume(x, y, depth, h, w, src, kappa, center, width, npix, nthreads=None, bands_per_thread=4, small_cut=6.0):
    """Volume rendering seen from +depth (the largest depth is in front): (npix, npix, 2) float32, [..., 0] the emission sum E = sum T_i (1 - e^-tau_i) S_i,
    [..., 1] the opacity 1 - T. tau_i = kappa * w_i * F2D(r, H_i) (w = mass: kappa is an opacity per unit mass, code units length^2 / mass)."""
    o = np.argsort(-np.asarray(depth), kind="stable")
    x, y, h, w, src = (np.ascontiguousarray(np.asarray(a)[o]) for a in (x, y, h, w, src))
    _check_inputs(x, y, h, w, np.zeros((0, len(x)), np.float32))
    nthreads = nthreads or nb.get_num_threads()
    return _volume_kernel(x.astype(np.float64), y.astype(np.float64), h.astype(np.float64), w.astype(np.float64), src.astype(np.float64), float(kappa),
                          center[0], center[1], width / npix, npix, npix, nthreads, bands_per_thread, small_cut, _T2)
