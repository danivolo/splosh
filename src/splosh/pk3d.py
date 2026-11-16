"""3D power spectra of the gas in a cube of the disc (MRI turbulence): the snapshot product `pk3d` (opt-in, like `pk`). Description: docs/PLOTS.md
("3D spectra in a cube of the disc").

Frame (modes.disc_frame, the rules of disc.py): two black holes: their centre of mass, z along the orbit normal, lengths in units of the osculating
semi-major axis a; one black hole: about it, z along the gas angular momentum within `r_max`, code length; none: the gas centre of mass. The cube of
side L = `box` (units of a, or code length) is centred in the midplane at radius `r0` along the frame's in-plane x axis (the simulation x projected on the
plane; rotated by `phi0` degrees about z). Box axes: x = radial, y = azimuthal at the box centre, z = the disc normal.

Deposit (SPH, the GIZMO cubic spline with support H = SmoothingLength): every particle whose kernel overlaps the cube is spread over the n^3 cells with the
support max(H, hmin dx) (dx = L / n, hmin = 2 cells: a particle smaller than a cell still reaches its neighbours; this is the smoothing scale of the grid:
max(H, 2 dx)), its sampled kernel renormalised to integrate to one (footprints under 6 cells, as splat.py). Per cell, with W_j the kernel of particle j:
    rho = sum m_j W_j                                      (the SPH density of the widened kernel)
    dv  = sum m_j dv_j W_j / sum m_j W_j                   (mass-weighted)
    B   = sum V_j B_j W_j / sum V_j W_j, V_j = m_j / rho_j  (volume-weighted, normalised: a uniform field is exact wherever the gas reaches)
Cells no kernel reaches hold no gas (rho = dv = B = 0). B in code units with mu_0 = 1 (P_mag = B^2 / 2, as fields.f_pmag).
dv, the turbulent velocity, is the velocity relative to the frame minus the mean flow (`rotation`): shell (default) the mass-weighted means of v_R, v_phi,
v_z of the cube's own gas in cylindrical shells of R (n / 4 shells across the cube's range of R, linearly interpolated in R), kepler: only
Omega_K(R) R e_phi (Omega_K^2 = G M / R^3 [1 + 3/4 q / (1 + q)^2 (a / R)^2], disc.py's), none: nothing.

Spectra (spectrum3d): each field f (minus its window-weighted mean, `subtract_mean`) times the 3D Hann (or Tukey, none) window w, F = rfftn(f w),
k = 2 pi / L * (integer mode numbers); kn = |k| / k_f, k_f = 2 pi / L. Shell sums over bins of kn with half-integer edges (0.5, 1.5, ... : integer shells
at low k, log-spaced groups of them at high k, up to the Nyquist n / 2 + 0.5), divided by the bin width in kn: a spectral density per unit kn, so that
    sum_bins E(kn) dkn + E_above_Nyquist = 1/2 <f'^2 w^2> / <w^2>         (Parseval; f' = f minus its weighted mean, the k = 0 term then vanishes)
i.e. the spectrum integrates to the window-weighted mean energy density of the fluctuation (code pressure units for B and v). The bin's kn is the mean |kn|
of its Fourier modes (fixed by n and nk: the same family rows in every snapshot of the same n, nk). E(k) ~ k^-5/3 for a 3D power P(k) ~ k^-11/3.
    E_B  = 1/2 |B_k|^2 summed over the three components;   E_BR, E_Bphi, E_Bz: each component alone (box frame)
    E_v  = 1/2 |(sqrt(rho) dv)_k|^2 (the kinetic energy spectrum)
    P_rho = |delta_k|^2 with delta = rho / <rho>_w - 1 (no 1/2): sum P dkn = the window-weighted variance of delta
"""
import numpy as np
from numba import njit, prange
import numba as nb

from . import plugins
from .kernel import tables
from .pk import slope, window1d

_T2, _T3 = tables()
_NT = len(_T3) - 1
SMALL_CUT = 6.0          # footprints up to this radius (cells) are renormalised to the exact particle weight
ROTATION = ("shell", "kepler", "none")


@njit(cache=True, inline="always")
def _w(tab, q2):
    return tab[int(q2 * _NT + 0.5)]


@njit(cache=True, parallel=True)
def _scales(g, hp, small_cut, tab):
    """1 / (the sampled kernel summed over the particle's whole footprint, in cell units) for footprints up to small_cut cells, else 1."""
    N = g.shape[0]
    out = np.ones(N)
    for p in prange(N):
        h = hp[p]
        if h > small_cut:
            continue
        inv = 1.0 / (h * h)
        s = 0.0
        for i in range(int(np.floor(g[p, 0] - h)), int(np.ceil(g[p, 0] + h)) + 1):
            dx2 = (i + 0.5 - g[p, 0]) ** 2
            for j in range(int(np.floor(g[p, 1] - h)), int(np.ceil(g[p, 1] + h)) + 1):
                dxy2 = dx2 + (j + 0.5 - g[p, 1]) ** 2
                for k in range(int(np.floor(g[p, 2] - h)), int(np.ceil(g[p, 2] + h)) + 1):
                    q2 = (dxy2 + (k + 0.5 - g[p, 2]) ** 2) * inv
                    if q2 < 1.0:
                        s += _w(tab, q2)
        s /= h * h * h
        if s > 0:
            out[p] = 1.0 / s
    return out


@njit(cache=True, parallel=True)
def _deposit(g, hp, scale, contrib, n, nbands, tab):
    """grid[c, i, j, k] = sum_p contrib[c, p] W_p(cell): the x planes split into bands, each band owned by one thread (no races)."""
    nq, N = contrib.shape
    grid = np.zeros((nq, n, n, n), np.float32)
    for b in prange(nbands):
        lo = b * n // nbands
        hi = (b + 1) * n // nbands
        for p in range(N):
            h = hp[p]
            i0 = max(lo, int(np.floor(g[p, 0] - h)))
            i1 = min(hi - 1, int(np.ceil(g[p, 0] + h)))
            if i0 > i1:
                continue
            j0 = max(0, int(np.floor(g[p, 1] - h)))
            j1 = min(n - 1, int(np.ceil(g[p, 1] + h)))
            k0 = max(0, int(np.floor(g[p, 2] - h)))
            k1 = min(n - 1, int(np.ceil(g[p, 2] + h)))
            inv = 1.0 / (h * h)
            wn = scale[p] / (h * h * h)
            for i in range(i0, i1 + 1):
                dx2 = (i + 0.5 - g[p, 0]) ** 2
                for j in range(j0, j1 + 1):
                    dxy2 = dx2 + (j + 0.5 - g[p, 1]) ** 2
                    for k in range(k0, k1 + 1):
                        q2 = (dxy2 + (k + 0.5 - g[p, 2]) ** 2) * inv
                        if q2 < 1.0:
                            W = _w(tab, q2) * wn
                            for c in range(nq):
                                grid[c, i, j, k] += contrib[c, p] * W
    return grid


def select(xl, H, L):
    """the particles (local coordinates xl about the cube's centre) whose kernel overlaps the cube of side L."""
    return np.all(np.abs(xl) < 0.5 * L + np.asarray(H)[:, None], axis=1)


def deposit(xl, H, contrib, L, n, hmin=2.0, nthreads=None):
    """SPH deposit of per-particle quantities contrib (nq, N) onto the n^3 cells of the cube [-L/2, L/2]^3 (xl: local coordinates): (nq, n, n, n)
    float32, each particle's weight spread with the support max(H, hmin dx) and summing to its contrib (in cell units: divide by dx^3 for densities)."""
    dx = float(L) / n
    g = np.ascontiguousarray((np.asarray(xl, np.float64) + 0.5 * L) / dx)
    hp = np.maximum(np.asarray(H, np.float64) / dx, float(hmin))
    contrib = np.ascontiguousarray(np.asarray(contrib, np.float64).reshape(-1, len(hp)))
    if len(hp) == 0:
        return np.zeros((contrib.shape[0], n, n, n), np.float32)
    sc = _scales(g, hp, SMALL_CUT, _T3)
    nthreads = nthreads or nb.get_num_threads()
    return _deposit(g, hp, sc, contrib, int(n), int(min(n, 4 * nthreads)), _T3)


def grid_fields(xl, dv, m, rho, H, B, L, n, hmin=2.0):
    """the cube's fields from particles in local coordinates: {"rho" (n,n,n), "dv" (3,n,n,n), "B" (3,n,n,n) or None, "vfill" (sum V W: ~1 where the
    gas fills the cell)}; dv and B in the box frame (x radial, y azimuthal, z normal) as given."""
    m = np.asarray(m, np.float64)
    V = m / np.asarray(rho, np.float64)
    rows = [V, m] + [m * dv[:, c] for c in range(3)]
    if B is not None:
        rows += [V * B[:, c] for c in range(3)]
    grid = deposit(xl, H, np.array(rows), L, n, hmin)
    dx3 = (float(L) / n) ** 3
    vg, mg = grid[0], grid[1]
    out = {"rho": mg / np.float32(dx3), "vfill": vg / np.float32(dx3)}           # float32 grids: 4 n^3 bytes each
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dv"] = np.where(mg > 0, grid[2:5] / mg, np.float32(0))
        out["B"] = np.where(vg > 0, grid[5:8] / vg, np.float32(0)) if B is not None else None
    return out


# ----------------------------------------------------------------------------- spectra

def kedges(n, nk=32):
    """bin edges in kn: nk log-spaced edges from 0.5 to n/2 + 0.5 snapped to half-integers (each bin holds whole integer shells; fewer bins at low k)."""
    e = np.geomspace(0.5, n / 2 + 0.5, int(nk) + 1)
    return np.unique(np.floor(e) + 0.5)


_KCACHE = {}


def _kgrid(n, nk):
    key = (n, nk)
    if key not in _KCACHE:
        fr = np.fft.fftfreq(n) * n
        fz = np.fft.rfftfreq(n) * n
        kn = np.sqrt(fr[:, None, None] ** 2 + fr[None, :, None] ** 2 + fz[None, None, :] ** 2).ravel()
        mult = np.full(len(fz), 2.0)                      # the half spectrum: kz > 0 modes stand for their conjugates too
        mult[0] = 1.0
        if n % 2 == 0:
            mult[-1] = 1.0
        mult = np.broadcast_to(mult, (n, n, len(fz))).ravel().copy()
        edges = kedges(n, nk)
        idx = np.digitize(kn, edges) - 1
        nb_ = len(edges) - 1
        idx[(kn >= edges[-1])] = nb_                     # above the Nyquist shell
        idx[kn < edges[0]] = nb_ + 1                     # k = 0
        cnt = np.bincount(idx, weights=mult, minlength=nb_ + 2)
        kc = np.bincount(idx, weights=mult * kn, minlength=nb_ + 2)[:nb_] / cnt[:nb_]
        _KCACHE[key] = (idx, mult, edges, kc, cnt[:nb_])
    return _KCACHE[key]


def window3d(n, kind="hann"):
    w1 = window1d(n, kind)
    return w1[:, None, None] * w1[None, :, None] * w1[None, None, :]


def spectrum3d(f, L, window="hann", nk=32, subtract_mean=True, factor=0.5, w=None):
    """the shell-summed spectrum of a field f (n,n,n) or of the sum over components (c,n,n,n) on the cube of side L, as in the module docstring:
    {"kn", "k" (1/length), "E" (per unit kn), "dkn", "n" (Fourier modes per bin), "edges", "hi" (above the Nyquist shell), "k0" (the k = 0 term),
    "total" (factor <f'^2 w^2> / <w^2>: = sum E dkn + hi + k0), "kf", "mean" (the window-weighted means)}."""
    f = np.asarray(f)
    if f.ndim == 3:
        f = f[None]
    n = f.shape[-1]
    if f.shape[1:] != (n, n, n):
        raise ValueError("spectrum3d: a cubic grid")
    if w is None:
        w = window3d(n, window)
    w2 = float(np.mean(w * w))
    idx, mult, edges, kc, cnt = _kgrid(n, int(nk))
    nb_ = len(edges) - 1
    tot = np.zeros(nb_ + 2)
    total, means = 0.0, []
    norm = factor / (float(n) ** 6 * w2)
    for comp in f:
        comp = comp.astype(np.float64)                   # one component at a time in double precision
        mu = float((comp * w).sum() / w.sum()) if subtract_mean else 0.0
        means.append(mu)
        g = (comp - mu) * w
        total += factor * float(np.mean(g * g)) / w2
        F = np.fft.rfftn(g)
        P = (F.real ** 2 + F.imag ** 2).ravel()
        del F
        tot += np.bincount(idx, weights=P * mult, minlength=nb_ + 2) * norm
    dkn = np.diff(edges)
    kf = 2 * np.pi / float(L)
    return {"kn": kc, "k": kc * kf, "E": tot[:nb_] / dkn, "dkn": dkn, "n": cnt, "edges": edges, "hi": float(tot[nb_]), "k0": float(tot[nb_ + 1]),
            "total": total, "kf": kf, "mean": np.array(means)}


# ----------------------------------------------------------------------------- the product

def _B_code(s):
    """MagneticField in code units with mu_0 = 1 (B^2 / 2 = magnetic pressure in code units; the conversion of fields.f_pmag / disc.py)."""
    from .fields import magnetic_unit_gauss
    from .maps import View
    from .units import Units
    un = Units.from_header(s.get("header", {}))
    bunit, _ = magnetic_unit_gauss(s, View())
    return np.asarray(s["MagneticField"], np.float64) * bunit / np.sqrt(4 * np.pi * un.density_cgs * un.velocity_cm_s ** 2)


def mean_flow(R, vcyl, m, nbins):
    """the mass-weighted shell means of the cylindrical velocity components vcyl (N, 3) in nbins shells of R, linearly interpolated at each R."""
    lo, hi = float(R.min()), float(R.max())
    if hi <= lo:
        return np.broadcast_to((vcyl * m[:, None]).sum(0) / m.sum(), vcyl.shape)
    edges = np.linspace(lo, hi, int(nbins) + 1)
    idx = np.clip(np.digitize(R, edges) - 1, 0, nbins - 1)
    w = np.bincount(idx, weights=m, minlength=nbins)
    ok = w > 0
    xc = (np.bincount(idx, weights=m * R, minlength=nbins)[ok] / w[ok])
    out = np.empty_like(vcyl)
    for c in range(3):
        mc = np.bincount(idx, weights=m * vcyl[:, c], minlength=nbins)[ok] / w[ok]
        out[:, c] = np.interp(R, xc, mc)
    return out


def box_particles(pos, vel, m, rho, H, B, frame, params):
    """the particles of the cube in its own frame: (xl, dv, m, rho, H, B or None, info) for the snapshot's gas arrays and modes.disc_frame `frame`."""
    a = float(frame["scale"])
    L = float(params.get("box") or 2.0) * a
    r0 = float(params.get("r0") if params.get("r0") is not None else 3.0) * a
    ph = np.radians(float(params.get("phi0") or 0.0))
    zhat = frame["zhat"]
    e1 = np.cos(ph) * frame["e1"] + np.sin(ph) * frame["e2"]
    e2 = np.cross(zhat, e1)
    E = np.array([e1, e2, zhat])                         # rows: the box axes
    center = frame["com"] + r0 * e1
    xl = (np.asarray(pos, np.float64) - center) @ E.T
    sel = select(xl, H, L)
    xl = xl[sel]
    rel = np.asarray(pos, np.float64)[sel] - frame["com"]
    vrel = np.asarray(vel, np.float64)[sel] - frame["vcom"]
    m, rho, H = (np.asarray(x, np.float64)[sel] for x in (m, rho, H))
    x1, x2, z = rel @ frame["e1"], rel @ frame["e2"], rel @ zhat
    R = np.hypot(x1, x2)
    phi = np.arctan2(x2, x1)
    eR = np.cos(phi)[:, None] * frame["e1"] + np.sin(phi)[:, None] * frame["e2"]
    ephi = -np.sin(phi)[:, None] * frame["e1"] + np.cos(phi)[:, None] * frame["e2"]
    rot = params.get("rotation") or "shell"
    if rot not in ROTATION:
        raise ValueError(f"pk3d: rotation must be one of {ROTATION}")
    if rot == "kepler" and frame.get("M") is None:
        rot = "shell"                                    # no black hole: no Keplerian rotation to subtract
    if len(m) == 0:
        dv = vrel
    elif rot == "shell":
        vcyl = np.column_stack([(vrel * eR).sum(1), (vrel * ephi).sum(1), vrel @ zhat])
        mf = mean_flow(R, vcyl, m, max(4, int(params.get("n") or 128) // 4))
        dv = vrel - mf[:, 0:1] * eR - mf[:, 1:2] * ephi - mf[:, 2:3] * zhat
    elif rot == "kepler":
        GM = frame["G"] * frame["M"]
        q = frame.get("q") or 0.0
        quad = 0.75 * q / (1 + q) ** 2 if frame.get("binary") else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            OmK = np.sqrt(GM / R ** 3 * (1.0 + quad * (a / R) ** 2))
        dv = vrel - np.where(R > 0, OmK * R, 0.0)[:, None] * ephi
    else:
        dv = vrel
    Bl = np.asarray(B, np.float64)[sel] @ E.T if B is not None else None
    return xl, dv @ E.T, m, rho, H, Bl, {"L": L, "r0": r0, "center": center, "axes": E, "rotation": rot, "a": a}


def spectra(fields, L, params, out=None):
    """the product's scalars from grid_fields(...) of the cube of side L (see the module docstring and docs/PLOTS.md)."""
    out = {} if out is None else out
    n = fields["rho"].shape[0]
    window = params.get("window") or "hann"
    nk = int(params.get("nk") or 32)
    sm = params.get("subtract_mean", True)
    sm = True if sm is None else bool(sm)
    lo, hi = float(params.get("kn_lo") or 4), float(params.get("kn_hi") or 16)
    w = window3d(n, window)
    dx3 = (float(L) / n) ** 3
    rho = fields["rho"]
    key = lambda k: f"{k:.4g}"

    def put(name, r):
        for kc, e in zip(r["kn"], r["E"]):
            out[f"{name}_{key(kc)}"] = float(e)
        sfx = name[name.index("_"):]                     # E_B -> _B: tot_B, hi_B, slope_B (names outside the family E_B_<kn>)
        out["tot" + sfx] = r["total"]                    # = sum E dkn + hi: the window-weighted mean energy density of the fluctuation
        out["hi" + sfx] = r["hi"]                        # the part above the Nyquist shell
        out["slope" + sfx] = slope(r["kn"], r["E"], lo, hi)
        return r

    rhom = float((rho * w).sum(dtype=np.float64) / w.sum())
    delta = rho / rhom - 1.0 if rhom > 0 else np.zeros_like(rho)
    put("P_rho", spectrum3d(delta, L, window, nk, sm, factor=1.0, w=w))
    g = np.sqrt(rho)[None] * fields["dv"]
    put("E_v", spectrum3d(g, L, window, nk, sm, w=w))
    out["Ekin"] = float(0.5 * np.sum(g.astype(np.float64) ** 2) * dx3)
    out["ekin"] = out["Ekin"] / float(L) ** 3
    del g
    B = fields.get("B")
    if B is not None:
        put("E_B", spectrum3d(B, L, window, nk, sm, w=w))
        for c, nm in enumerate(("BR", "Bphi", "Bz")):
            put("E_" + nm, spectrum3d(B[c], L, window, nk, sm, w=w))
        out["Emag"] = float(0.5 * np.sum(B.astype(np.float64) ** 2) * dx3)
        out["emag"] = out["Emag"] / float(L) ** 3
        Bm = B.reshape(3, -1).mean(1, dtype=np.float64)
        out["emag_mean"] = float(0.5 * Bm @ Bm)
        out["BR_mean"], out["Bphi_mean"], out["Bz_mean"] = (float(x) for x in Bm)
    out["Mbox"] = float(rho.sum(dtype=np.float64) * dx3)
    out["rho_mean"] = float(rho.mean(dtype=np.float64))
    out["fill"] = float(np.mean(fields["vfill"] > 0.5))
    out["kf"] = 2 * np.pi / float(L)
    return out


def pk3d_arrays(s, params):
    """the product for a loaded snapshot dict (gas datasets Coordinates, Velocities, Masses, Density, SmoothingLength, MagneticField optional)."""
    from .modes import disc_frame
    pos = np.asarray(s["Coordinates"], np.float64)
    vel = np.asarray(s["Velocities"], np.float64)
    m = np.asarray(s["Masses"], np.float64)
    frame = disc_frame(s, pos, vel, m, r_max=float(params.get("r_max") or 12.0))
    B = _B_code(s) if "MagneticField" in s else None
    xl, dv, m, rho, H, Bl, info = box_particles(pos, vel, m, s["Density"], s["SmoothingLength"], B, frame, params)
    del pos, vel, B
    n = int(params.get("n") or 128)
    out = {"L": info["L"], "box": info["L"] / info["a"], "r0": info["r0"] / info["a"], "a": info["a"], "n": n, "dx": info["L"] / n, "npart": len(m)}
    if frame.get("binary"):
        out.update(e=frame["e"], q=frame["q"])
    if len(m) == 0:
        return out
    f = grid_fields(xl, dv, m, rho, H, Bl, info["L"], n, float(params.get("hmin") or 2.0))
    del xl, dv, Bl
    spectra(f, info["L"], params, out)
    return out


def pk3d(path, params):
    from .io import load_snapshot
    gas = ("Coordinates", "Velocities", "Masses", "Density", "SmoothingLength", "MagneticField")
    s = load_snapshot(path, gas_fields=gas[:5] + ("?MagneticField", "?ParticleIDs"))
    ex = params.get("exclude_ids") or []           # cells left out by ID (e.g. spawned wind cells); not a default parameter, so cached results stay valid
    if ex and "ParticleIDs" in s:
        keep = ~np.isin(np.asarray(s["ParticleIDs"]), np.asarray(ex, dtype=np.int64))
        s = {**s, **{k: np.asarray(s[k])[keep] for k in gas if k in s}}
    return {"scalars": pk3d_arrays(s, params)}


_KU = "k in units of 2 pi / L (kn)"
plugins.register_product(
    "pk3d", pk3d, needs=("Coordinates", "Velocities", "Masses", "Density", "SmoothingLength", "?MagneticField", "PartType5"), version=1,
    label="pk3d: 3D spectra of B, kinetic energy, density in a cube of the disc (opt-in)", applies=None,
    params={"box": 2.0, "r0": 3.0, "phi0": 0.0, "n": 128, "hmin": 2.0, "rotation": "shell", "window": "hann", "nk": 32, "subtract_mean": True,
            "kn_lo": 4, "kn_hi": 16, "r_max": 12.0},
    defaults=["slope_B", "slope_v"],
    meta={"E_B_": {"label": f"E_B(k) = 1/2 |B_k|^2 per unit kn, {_KU}", "group": "spectrum"},
          "E_BR_": {"label": f"E_BR(k), radial component, {_KU}", "group": "spectrum"},
          "E_Bphi_": {"label": f"E_Bphi(k), azimuthal component, {_KU}", "group": "spectrum"},
          "E_Bz_": {"label": f"E_Bz(k), vertical component, {_KU}", "group": "spectrum"},
          "E_v_": {"label": f"E_v(k) = 1/2 |(sqrt(rho) dv)_k|^2 per unit kn, {_KU}", "group": "spectrum"},
          "P_rho_": {"label": f"P_rho(k) = |delta_k|^2 per unit kn, {_KU}", "group": "spectrum"},
          "Emag": {"label": "magnetic energy in the cube, sum B^2/2 dV [code]", "group": "pk3d"},
          "Ekin": {"label": "turbulent kinetic energy in the cube, sum rho dv^2/2 dV [code]", "group": "pk3d"},
          "emag": {"label": "mean magnetic energy density B^2/2 [code pressure]", "group": "pk3d"},
          "ekin": {"label": "mean turbulent kinetic energy density [code pressure]", "group": "pk3d"},
          "emag_mean": {"label": "energy density of the cube's mean field |<B>|^2/2", "group": "pk3d"},
          "slope_B": {"label": "log-slope of E_B(k)", "group": "pk3d"}, "slope_v": {"label": "log-slope of E_v(k)", "group": "pk3d"},
          "slope_rho": {"label": "log-slope of P_rho(k)", "group": "pk3d"},
          "kf": {"label": "2 pi / L [1/code length]", "group": "pk3d"}, "L": {"label": "cube side [code length]", "group": "pk3d"},
          "fill": {"label": "fraction of cells filled by gas (sum V W > 0.5)", "group": "pk3d"}})
plugins.PRODUCTS["pk3d"]["auto"] = False
