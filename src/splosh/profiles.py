"""Radial profiles of particle data (no kernel: particles are binned by their position).

radial_profile(pos, weights, values, rmin, rmax, nbins, ...) -> dict
    ndim = 2: cylindrical radius R = hypot(x, y) about `center` (a disc seen face-on); ndim = 3: spherical radius.
    spacing 'log' | 'lin' sets the bin EDGES; `rbins` are the arithmetic centres of the bins (what pynbody's Profile reports), `rgeo` the geometric ones.
    For every entry of `values` the weighted mean in each bin (sum w q / sum w; NaN for empty bins); `weights` is typically the particle mass (mass-weighted mean),
    `mass` the summed weight, `n` the number of particles, `sigma` the weight per unit area (ndim 2) or volume (ndim 3), `density` is that same quantity (alias).
    `<name>_std` is the weighted standard deviation of the values in the bin (the scatter of the profile: sqrt(<q^2> - <q>^2)).
    modes = (m, ...): azimuthal Fourier modes about the z axis through `center` (the angle phi = atan2(y, x) of each particle): `A<m>` = |sum w exp(i m phi)| / sum w in the bin (0: axisymmetric, 1: one-armed
    or off-centre, 2: bar or two arms) and `phi<m>` = atan2(sum w sin m phi, sum w cos m phi) / m, the angle of the maximum of the m-fold pattern (radians, in (-pi/m, pi/m])."""
import numpy as np


def radial_profile(pos, weights, values=None, rmin=0.1, rmax=20.0, nbins=50, spacing="log", ndim=2, center=(0.0, 0.0, 0.0), modes=()):
    pos = np.asarray(pos, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    r = np.hypot(pos[:, 0], pos[:, 1]) if ndim == 2 else np.sqrt((pos ** 2).sum(1))
    edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1) if spacing == "log" else np.linspace(rmin, rmax, nbins + 1)
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    i, wi = idx[ok], w[ok]
    den = np.bincount(i, weights=wi, minlength=nbins)
    n = np.bincount(i, minlength=nbins)
    out = {"edges": edges, "rbins": 0.5 * (edges[1:] + edges[:-1]), "rgeo": np.sqrt(edges[1:] * edges[:-1]), "n": n, "mass": den}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, v in (values or {}).items():
            num = np.bincount(i, weights=wi * np.asarray(v, dtype=np.float64)[ok], minlength=nbins)
            out[k] = np.where(den > 0, num / den, np.nan)
            sq = np.bincount(i, weights=wi * np.asarray(v, dtype=np.float64)[ok] ** 2, minlength=nbins)
            out[k + "_std"] = np.sqrt(np.maximum(np.where(den > 0, sq / den, np.nan) - out[k] ** 2, 0.0))
        phi = np.arctan2(pos[ok, 1], pos[ok, 0])
        for m in modes:
            cm = np.bincount(i, weights=wi * np.cos(m * phi), minlength=nbins)
            sm = np.bincount(i, weights=wi * np.sin(m * phi), minlength=nbins)
            out[f"A{m}"] = np.where(den > 0, np.hypot(cm, sm) / den, np.nan)
            out[f"phi{m}"] = np.where(den > 0, np.arctan2(sm, cm) / m, np.nan)
    measure = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) if ndim == 2 else 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    out["sigma"] = out["density"] = den / measure
    return out


def value_histogram(x, weights, values=None, lo=1e-3, hi=1e3, nbins=50, spacing="log"):
    """Distribution of the particle quantity `x` (a probability density function): bins in x between lo and hi (`spacing` 'log' | 'lin' sets the EDGES), weighted by `weights`
    (mass: the mass-weighted PDF; volume: the volume-weighted one). Same keys as radial_profile so that the writers and the profile panels take it unchanged, with
    `rbins` = the arithmetic bin centres IN x, `rgeo` the geometric ones, plus
        frac   weight in the bin / total weight of all particles (so the bins inside lo..hi sum to the fraction of the weight in the range)
        pdf    frac per unit log10(x) (log spacing) or per unit x (lin): its integral over the range is the fraction of the weight in the range
    `sigma` is an alias of pdf. For each entry of `values` the weighted mean in each bin (and `<name>_std`), e.g. the temperature at each density."""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1) if spacing == "log" else np.linspace(lo, hi, nbins + 1)
    fin = np.isfinite(x) & (x > 0 if spacing == "log" else True)
    idx = np.digitize(x, edges) - 1
    ok = fin & (idx >= 0) & (idx < nbins)
    i, wi = idx[ok], w[ok]
    den = np.bincount(i, weights=wi, minlength=nbins)
    n = np.bincount(i, minlength=nbins)
    total = float(w[fin].sum()) if fin.any() else 1.0
    width = np.diff(np.log10(edges)) if spacing == "log" else np.diff(edges)
    pdf = den / total / width
    out = {"edges": edges, "rbins": 0.5 * (edges[1:] + edges[:-1]), "rgeo": np.sqrt(edges[1:] * edges[:-1]), "n": n, "mass": den, "frac": den / total, "pdf": pdf, "sigma": pdf}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, v in (values or {}).items():
            vv = np.asarray(v, dtype=np.float64)[ok]
            num = np.bincount(i, weights=wi * vv, minlength=nbins)
            sq = np.bincount(i, weights=wi * vv ** 2, minlength=nbins)
            mean = np.where(den > 0, num / den, np.nan)
            out[k] = mean
            out[k + "_std"] = np.sqrt(np.maximum(np.where(den > 0, sq / den, np.nan) - mean ** 2, 0.0))
    return out
