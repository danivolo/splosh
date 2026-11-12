"""Post-processing of a finished map: what is left after the dominant component is removed, and where the map changes fastest.

A transform is a spec string, stored in View.post (so it is part of the cache key, the sidecar and every layout):

    none                 the map as rendered
    resid                contrast against the azimuthal average about the frame origin (the black hole, or whatever the centre rule gives): alias of resid:azim
    resid:azim           f / f_bar(R) - 1, with f_bar(R) the mean of the finite pixels in rings of one pixel width about the frame origin of the view's plane
    resid:mean           f / mean(f) - 1 with the mean over the window
    resid:smooth:S       f / G_S(f) - 1, G_S a Gaussian of standard deviation S (code length units, > 0): an unsharp mask
    grad                 |grad log10 f| (the change of the logarithm per length; the logarithm is of the smoothed map)
    grad:S               the same with the Gaussian S in code length units (default: two pixels)

For quantities that are positive by nature (rho, u, H, speed, T, cs, pressure, entropy, mach, beta) the residual is a contrast (dimensionless), and the gradient is of log10 f.
Any other field (velocities, jz, vR, vphi, plugin fields) has no safe ratio: the residual is the difference f - f_bar in the field's own unit, and the gradient is |grad f| per length.
Pixels that are NaN stay NaN. The azimuthal average is taken over the pixels of the window, so in a zoomed window that shows only part of each ring it is the average of the visible
part (the residual is then relative to that, not to the whole ring): say so in a caption when it matters."""
import math

import numpy as np
from numba import njit

POSITIVE_FIELDS = frozenset({"rho", "u", "H", "speed", "T", "cs", "pressure", "entropy", "mach", "beta"})
KINDS = ("none", "resid", "grad")


def parse(spec):
    """spec -> (kind, how, sigma): kind 'none' | 'resid' | 'grad'; how 'azim' | 'mean' | 'smooth' (resid); sigma in code length units or None (default)."""
    s = (spec or "none").strip()
    if s in ("", "none"):
        return "none", None, None
    parts = s.split(":")
    kind = parts[0]
    if kind == "resid":
        how = parts[1] if len(parts) > 1 else "azim"
        if how not in ("azim", "mean", "smooth"):
            raise ValueError(f"post {spec!r}: resid takes azim, mean or smooth:S")
        if how == "smooth":
            if len(parts) != 3:
                raise ValueError(f"post {spec!r}: resid:smooth needs a width, e.g. resid:smooth:0.5")
            sigma = _positive(parts[2], spec)
        else:
            if len(parts) > 2:
                raise ValueError(f"post {spec!r}: unexpected text after {how}")
            sigma = None
        return "resid", how, sigma
    if kind == "grad":
        if len(parts) > 2:
            raise ValueError(f"post {spec!r}: grad or grad:S")
        return "grad", None, (_positive(parts[1], spec) if len(parts) == 2 else None)
    raise ValueError(f"post {spec!r}: use none, resid[:azim|mean|smooth:S] or grad[:S]")


def _positive(text, spec):
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f"post {spec!r}: {text!r} is not a number") from None
    if not (v > 0 and math.isfinite(v)):
        raise ValueError(f"post {spec!r}: the width must be > 0")
    return v


def canonical(spec):
    kind, how, sigma = parse(spec)
    if kind == "none":
        return "none"
    if kind == "resid":
        return "resid:azim" if how == "azim" else ("resid:mean" if how == "mean" else f"resid:smooth:{sigma:g}")
    return "grad" if sigma is None else f"grad:{sigma:g}"


def style(spec):
    """What the display should do by default for this transform: dict(cmap, scale, symmetric) or None for none."""
    kind, _, _ = parse(spec)
    if kind == "resid":
        return {"cmap": "RdBu_r", "scale": "linear", "symmetric": True}
    if kind == "grad":
        return {"cmap": "magma", "scale": "asinh", "symmetric": False}
    return None


@njit(cache=True)
def _conv_axis(src, ker, axis):
    ny, nx = src.shape
    r = len(ker) // 2
    out = np.zeros_like(src)
    if axis == 1:
        for j in range(ny):
            for i in range(nx):
                acc = 0.0
                for k in range(-r, r + 1):
                    ii = i + k
                    if 0 <= ii < nx:
                        acc += ker[k + r] * src[j, ii]
                out[j, i] = acc
    else:
        for j in range(ny):
            for i in range(nx):
                acc = 0.0
                for k in range(-r, r + 1):
                    jj = j + k
                    if 0 <= jj < ny:
                        acc += ker[k + r] * src[jj, i]
                out[j, i] = acc
    return out


def gaussian_nan(a, sigma_px):
    """Gaussian smoothing of a 2D array that ignores NaN (normalised convolution); the result is NaN where `a` is NaN. sigma in pixels."""
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    if sigma_px < 0.3:
        return a.copy()
    r = int(min(math.ceil(4 * sigma_px), max(a.shape)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2)
    k /= k.sum()
    v = np.where(ok, a, 0.0)
    num = _conv_axis(_conv_axis(v, k, 1), k, 0)
    den = _conv_axis(_conv_axis(ok.astype(np.float64), k, 1), k, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(ok & (den > 1e-12), num / den, np.nan)
    return out


def ring_radius(extent, shape):
    """Distance of every pixel centre from the frame origin, in the length unit of `extent` (x0, x1, y0, y1)."""
    ny, nx = shape
    x = extent[0] + (np.arange(nx) + 0.5) * (extent[1] - extent[0]) / nx
    y = extent[2] + (np.arange(ny) + 0.5) * (extent[3] - extent[2]) / ny
    return np.hypot(x[None, :], y[:, None])


def azimuthal_mean(img, extent, max_bins=4096):
    """The image of the azimuthal mean about the frame origin. The pixels are binned in rings one pixel wide; each ring gives the mean value at the mean radius of its pixels,
    and every pixel's reference is the profile interpolated at its own radius (so a steep profile is not biased by the radius spread inside a ring). Rings with no finite pixel
    are bridged by the interpolation. Returns (reference image float64, ring radii, ring means)."""
    ny, nx = img.shape
    pix = (extent[1] - extent[0]) / nx
    r = ring_radius(extent, img.shape)
    idx = np.minimum((r / pix).astype(np.int64), max_bins - 1)
    ok = np.isfinite(img)
    nb = int(idx.max()) + 1
    cnt = np.bincount(idx[ok], minlength=nb)
    tot = np.bincount(idx[ok], weights=img[ok].astype(np.float64), minlength=nb)
    rsum = np.bincount(idx[ok], weights=r[ok], minlength=nb)
    good = cnt > 0
    if not good.any():
        return np.full(img.shape, np.nan), np.zeros(0), np.zeros(0)
    rc, prof = rsum[good] / cnt[good], tot[good] / cnt[good]
    return np.interp(r, rc, prof), rc, prof


def apply(img, meta, spec):
    """Transform the map. Returns (new image float32, info dict) with info = dict(post, kind, mode, label, unit, symmetric, ...); for 'none' the same image and None.
    `meta` is the render_map metadata (extent, field, length_unit, quantity_unit, view)."""
    kind, how, sigma = parse(spec)
    if kind == "none":
        return img, None
    field = meta["view"]["field"]
    positive = field in POSITIVE_FIELDS
    extent = meta["extent"]
    ny, nx = img.shape
    pix = (extent[1] - extent[0]) / nx
    a = np.asarray(img, dtype=np.float64)
    if positive:
        a = np.where(a > 0, a, np.nan)
    qlabel = meta.get("label", field)
    qunit = (meta.get("quantity_unit") or {}).get("unit", "")
    if kind == "resid":
        if how == "azim":
            ref, _, _ = azimuthal_mean(a, extent)
            what = "azimuthal mean"
        elif how == "mean":
            ref = np.full(a.shape, np.nanmean(a) if np.isfinite(a).any() else np.nan)
            what = "window mean"
        else:
            ref = gaussian_nan(a, sigma / pix)
            what = f"Gaussian {sigma:g}"
        with np.errstate(invalid="ignore", divide="ignore"):
            if positive:
                out = np.where(ref > 0, a / ref - 1.0, np.nan)
                mode, unit, label = "contrast", "", f"contrast of {qlabel} against its {what}"
            else:
                out = a - ref
                mode, unit, label = "difference", qunit, f"{qlabel} minus its {what}"
        info = {"post": canonical(spec), "kind": "resid", "mode": mode, "label": label, "unit": unit, "symmetric": True, "reference": what}
    else:
        sp = (sigma / pix) if sigma is not None else 2.0
        with np.errstate(invalid="ignore", divide="ignore"):
            base = np.log10(a) if positive else a
        sm = gaussian_nan(base, sp)
        gy, gx = np.gradient(sm, pix)
        out = np.hypot(gx, gy)
        lf = (meta.get("length_unit") or {}).get("factor", 1.0) or 1.0
        lu = (meta.get("length_unit") or {}).get("unit", "")
        out = out / lf
        if positive:
            mode, unit, label = "log-gradient", f"dex / {lu}".strip(), f"|grad log10 {qlabel}|"
        else:
            mode, unit, label = "gradient", f"{qunit} / {lu}".strip(" /"), f"|grad {qlabel}|"
        info = {"post": canonical(spec), "kind": "grad", "mode": mode, "label": label, "unit": unit, "symmetric": False, "smoothing_px": sp}
    return out.astype(np.float32), info


def finish(img, meta, spec):
    """Apply `spec` to a rendered map and fold the result into its metadata (what render_map does for View.post): returns (image, meta) - for 'none' the arguments themselves,
    otherwise a new image and a copy of meta with meta['post'] (the transform's info), the new label, the unit (meta['unit_override']) and view['post'] = spec."""
    out, info = apply(img, meta, spec)
    if info is None:
        return img, meta
    meta = dict(meta)
    meta["post"] = info
    meta["label"] = info["label"]
    meta["quantity_unit"] = {**meta.get("quantity_unit", {}), "unit": info["unit"], "known": True, "transformed": info["post"]}
    meta["unit_override"] = info["unit"]
    meta["view"] = {**meta["view"], "post": spec}
    return out, meta
