"""Azimuthal modes of the gas about the disc frame's centre, per snapshot: the snapshot product `modes` (products.py), read as the series source prod_modes.

Frame (`disc_frame`, the same rules as disc.py, which is not changed): two or more black holes: the two lowest-ID ones (dynamical masses), their centre
of mass, z along the binary's orbit normal, radii in units of the osculating semi-major axis a (the separation when unbound); one black hole: about it, z
along the gas angular momentum within `r_max`, radii in code length; no black hole: about the gas centre of mass, z along the gas angular momentum.
The in-plane basis is fixed by the simulation x axis projected on the plane (as in disc.py), so phases are comparable in time (an inertial frame).

Per cylindrical shell (columns <name>_<R>, R the shell's geometric centre in units of a, or code length), m = 1 .. `mmax`:
    A<m>      |sum m_i exp(i m phi_i)| / sum m_i: the m-th azimuthal Fourier amplitude of the surface density relative to its mean (Sigma ~ 1 + 2 A_m
              cos m (phi - phi_m): a perturbation eps cos m(phi - phi0) has A_m = eps / 2)
    phi<m>    the pattern's orientation in degrees: arg(sum m_i exp(i m phi_i)) / m, in (-180/m, 180/m]: the azimuth of one of its m maxima
    W, Wphi   the m = 1 vertical (bending) mode: W = 2 |sum m_i z_i exp(i phi_i)| / sum m_i R_i (tan of the tilt of a tilted ring), Wphi the azimuth of
              its highest point [deg]
Scalars: A<m>, phi<m>, W, Wphi over the band g_min < R < g_max (all the gas of the band at once); with a black hole, the tidal tensor of the gas at the
centre (the l = 2 multipole of the gas potential about the centre): T_ij = G sum m (3 x_i x_j - r^2 delta_ij) / r^5 over the gas at r > q_rmin (units
of the frame), in units of G M / a^3 (Omega_b^2 for a binary): Qzz (the vertical component; -G M_ring / R^3 for a ring), Q22 = |(T_xx - T_yy) / 2 + i T_xy|
(the in-plane m = 2 part) and phiQ22 (its orientation, deg); Mband (the gas mass of the band).
A heatmap of the family A2_ against time is A_2(R, t); the radial panel draws A2_ against R; the columns A1 .. A8 also form a family `A` (A against m).
Parameters: r_min / r_max / nbins (0.5, 12, 32, units of the frame), mmax (8), g_min / g_max (1.5, 5), q_rmin (1.0), z_cut (|z| < z_cut R only; None:
all gas), exclude_ids, center (auto | com: force the gas centre of mass)."""
import numpy as np

from . import plugins

MMAX = 8


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0, 1.0])


def _basis(zhat):
    from .disc import _basis
    return _basis(zhat)


def disc_frame(s, pos, vel, m, r_max=12.0, center="auto"):
    """the frame of disc.py for a loaded snapshot dict `s` and the gas arrays used: {com, vcom, zhat, e1, e2, scale (a, or 1), binary, a, e, q, M (the
    black holes' mass, or None), G, bh_pos, bh_vel, bh_mass (sorted by ID)}."""
    from .fields import gravitational_constant
    G = gravitational_constant(s)
    out = {"G": G, "binary": False, "a": None, "e": None, "q": None, "M": None}
    nbh = len(s["bh_pos"]) if ("bh_pos" in s and center != "com") else 0
    if nbh:
        o = np.argsort(np.asarray(s["bh_id"]).astype(np.int64), kind="stable") if "bh_id" in s else np.arange(nbh)
        bm = np.asarray(s["bh_mdyn"] if "bh_mdyn" in s else s["bh_mass"], dtype=np.float64)[o]
        bp, bv = np.asarray(s["bh_pos"], np.float64)[o], np.asarray(s["bh_vel"], np.float64)[o]
        out.update(bh_pos=bp, bh_vel=bv, bh_mass=bm)
    if nbh >= 2:
        m0, m1 = bm[0], bm[1]
        M = m0 + m1
        com, vcom = (m0 * bp[0] + m1 * bp[1]) / M, (m0 * bv[0] + m1 * bv[1]) / M
        r_rel, v_rel = bp[1] - bp[0], bv[1] - bv[0]
        sep = float(np.linalg.norm(r_rel))
        E = 0.5 * v_rel @ v_rel - G * M / sep
        hvec = np.cross(r_rel, v_rel)
        a = -G * M / (2 * E) if E < 0 else sep
        out.update(binary=True, a=float(a), e=float(np.sqrt(max(0.0, 1 + 2 * E * (hvec @ hvec) / (G * M) ** 2))), q=float(min(m0, m1) / max(m0, m1)),
                   M=float(M), sep=sep)
        zhat, scale = _unit(hvec), float(a)
    else:
        if nbh == 1:
            com, vcom = bp[0], bv[0]
            out["M"] = float(bm[0])
        else:
            com, vcom = (pos * m[:, None]).sum(0) / m.sum(), (vel * m[:, None]).sum(0) / m.sum()
        rr = pos - com
        near = np.linalg.norm(rr, axis=1) < float(r_max)
        if not near.any():
            near = np.ones(len(m), bool)
        zhat, scale = _unit(np.sum(m[near, None] * np.cross(rr[near], vel[near] - vcom), axis=0)), 1.0
    e1, e2 = _basis(zhat)
    out.update(com=com, vcom=vcom, zhat=zhat, e1=e1, e2=e2, scale=scale)
    return out


def coefficients(R, phi, z, w, edges, mmax=MMAX):
    """per shell of `edges` (same units as R): the weights W (n_shells), the complex sums C[m - 1] = sum w exp(i m phi) (mmax, n_shells), the vertical
    sum V = sum w z exp(i phi), sum w R and the particle counts."""
    nb = len(edges) - 1
    idx = np.digitize(R, edges) - 1
    ok = (idx >= 0) & (idx < nb)
    i, ph, ww = idx[ok], phi[ok], w[ok]
    W = np.bincount(i, weights=ww, minlength=nb)
    C = np.empty((mmax, nb), complex)
    for k in range(1, mmax + 1):
        C[k - 1] = np.bincount(i, weights=ww * np.cos(k * ph), minlength=nb) + 1j * np.bincount(i, weights=ww * np.sin(k * ph), minlength=nb)
    V = np.bincount(i, weights=ww * z[ok] * np.cos(ph), minlength=nb) + 1j * np.bincount(i, weights=ww * z[ok] * np.sin(ph), minlength=nb)
    WR = np.bincount(i, weights=ww * R[ok], minlength=nb)
    return W, C, V, WR, np.bincount(i, minlength=nb)


def amplitudes(W, C, V, WR):
    """(A [mmax, n], phi [mmax, n] in degrees (orientation: arg / m), W_bend [n], Wphi [n] in degrees) from `coefficients`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.abs(C) / W
        mm = np.arange(1, C.shape[0] + 1)[:, None]
        ph = np.degrees(np.angle(C)) / mm
        Wb = 2.0 * np.abs(V) / WR
        Wphi = np.degrees(np.angle(V))
    return A, ph, Wb, Wphi


def tidal_tensor(rel, m, G, r_min):
    """T_ij = G sum m (3 x_i x_j - r^2 delta_ij) / r^5 over the particles at r > r_min (rel: positions in the frame, (N, 3))."""
    r2 = (rel * rel).sum(1)
    sel = r2 > r_min * r_min
    x, mm, r2 = rel[sel], m[sel], r2[sel]
    f = G * mm / r2 ** 2.5
    T = 3.0 * np.einsum("n,ni,nj->ij", f, x, x) - np.eye(3) * np.sum(f * r2)
    return T


def _y(v):
    return f"{v:.4g}"


def modes(path, params):
    from .io import load_snapshot
    s = load_snapshot(path, gas_fields=("Coordinates", "Velocities", "Masses", "?ParticleIDs"))
    keep = np.ones(len(s["Masses"]), bool)
    ex = params.get("exclude_ids") or []
    if ex and "ParticleIDs" in s:
        keep &= ~np.isin(s["ParticleIDs"], np.asarray(ex, dtype=np.int64))
    pos = np.asarray(s["Coordinates"], np.float64)[keep]
    vel = np.asarray(s["Velocities"], np.float64)[keep]
    m = np.asarray(s["Masses"], np.float64)[keep]
    if not len(m):
        return {}
    fr = disc_frame(s, pos, vel, m, r_max=float(params.get("r_max") or 12.0), center=params.get("center") or "auto")
    sc = fr["scale"]
    rel = pos - fr["com"]
    x1, x2, z = rel @ fr["e1"], rel @ fr["e2"], rel @ fr["zhat"]
    R = np.hypot(x1, x2)
    phi = np.arctan2(x2, x1)
    if params.get("z_cut") is not None:
        body = np.abs(z) < float(params["z_cut"]) * R
        R, phi, z, mw = R[body], phi[body], z[body], m[body]
    else:
        mw = m
    mmax = int(params.get("mmax") or MMAX)
    edges = np.logspace(np.log10(float(params.get("r_min") or 0.5)), np.log10(float(params.get("r_max") or 12.0)), int(params.get("nbins") or 32) + 1)
    W, C, V, WR, n = coefficients(R / sc, phi, z, mw, edges, mmax)
    A, ph, Wb, Wphi = amplitudes(W, C, V, WR)
    few = n < 20
    Rm = np.sqrt(edges[1:] * edges[:-1])
    out = {}
    for k in range(mmax):
        for y, a, p, f in zip(Rm, A[k], ph[k], few):
            out[f"A{k + 1}_{_y(y)}"] = np.nan if f else a
            out[f"phi{k + 1}_{_y(y)}"] = np.nan if f else p
    for y, a, p, f in zip(Rm, Wb, Wphi, few):
        out[f"W_{_y(y)}"] = np.nan if f else a
        out[f"Wphi_{_y(y)}"] = np.nan if f else p
    g = [float(params.get("g_min") or 1.5), float(params.get("g_max") or 5.0)]
    Wg, Cg, Vg, WRg, ng = coefficients(R / sc, phi, z, mw, np.array(g), mmax)
    Ag, phg, Wbg, Wphig = amplitudes(Wg, Cg, Vg, WRg)
    for k in range(mmax):
        out[f"A{k + 1}"], out[f"phi{k + 1}"] = (float(Ag[k, 0]), float(phg[k, 0])) if ng[0] >= 20 else (np.nan, np.nan)
    out["W"], out["Wphi"] = (float(Wbg[0]), float(Wphig[0])) if ng[0] >= 20 else (np.nan, np.nan)
    out["Mband"] = float(Wg[0])
    if fr["M"]:
        Rot = np.vstack([fr["e1"], fr["e2"], fr["zhat"]])
        T = tidal_tensor(rel @ Rot.T, m, fr["G"], float(params.get("q_rmin") or 1.0) * sc)
        unit = fr["G"] * fr["M"] / sc ** 3
        c = 0.5 * (T[0, 0] - T[1, 1]) + 1j * T[0, 1]
        out["Qzz"], out["Q22"], out["phiQ22"] = T[2, 2] / unit, abs(c) / unit, float(np.degrees(np.angle(c)) / 2)
    if fr["binary"]:
        out.update(a=fr["a"], e=fr["e"], q=fr["q"])
    return {"scalars": out}


META = {**{f"A{k}": {"label": f"A_{k} over the band g_min < R < g_max", "group": "modes"} for k in range(1, MMAX + 1)},
        **{f"phi{k}": {"label": f"m = {k} orientation [deg]", "group": "modes", "wrap": 360.0 / k} for k in range(1, MMAX + 1)},
        "W": {"label": "bending (m = 1 vertical) amplitude: tan of the tilt", "group": "modes"},
        "Wphi": {"label": "bending: azimuth of the highest point [deg]", "group": "modes", "wrap": 360.0},
        "Mband": {"label": "gas mass in the band", "group": "modes"},
        "Qzz": {"label": "gas tidal tensor T_zz at the centre [G M / a^3]", "group": "quadrupole"},
        "Q22": {"label": "gas tidal tensor, in-plane m = 2 part [G M / a^3]", "group": "quadrupole"},
        "phiQ22": {"label": "orientation of the gas m = 2 tide [deg]", "group": "quadrupole", "wrap": 180.0},
        "a": {"label": "binary a", "group": "orbit"}, "e": {"label": "binary e", "group": "orbit"}, "q": {"label": "mass ratio", "group": "orbit"},
        **{f"A{k}_": {"label": f"A_{k}", "group": "profiles"} for k in range(1, MMAX + 1)},
        **{f"phi{k}_": {"label": f"m = {k} orientation [deg]", "group": "profiles"} for k in range(1, MMAX + 1)},
        "W_": {"label": "bending amplitude", "group": "profiles"}, "Wphi_": {"label": "bending azimuth [deg]", "group": "profiles"}}

def _applies(path):
    import h5py
    with h5py.File(path, "r") as f:
        return "PartType0" in f and len(f["PartType0"]["Masses"]) > 0


plugins.register_product("modes", modes, needs=("Coordinates", "Velocities", "Masses", "?ParticleIDs", "PartType5"), version=1,
                         label="modes: azimuthal Fourier amplitudes A_m(R), phases, bending mode, gas quadrupole at the centre", applies=_applies,
                         params={"r_min": 0.5, "r_max": 12.0, "nbins": 32, "mmax": MMAX, "g_min": 1.5, "g_max": 5.0, "q_rmin": 1.0, "z_cut": None,
                                 "exclude_ids": [], "center": "auto"},
                         defaults=["A1", "A2"], meta=META)
