"""Disc diagnostics about a binary (two black holes) or a single black hole, per snapshot: the snapshot product `disc` (products.py).

Frames. Binary: the two lowest-ID black holes (dynamical masses); the circumbinary disc (CBD) about their centre of mass, z along the orbit normal, radii in
units of the osculating semi-major axis a (the separation when unbound); each minidisc about its black hole, z along the angular momentum of the gas in its
Eggleton Roche lobe, radii in units of the Roche radius R_L. Single black hole: the disc about it, z along the gas angular momentum within `r_max`, radii in
code length. The in-plane basis is fixed by the simulation x axis (projected on the plane), so angles (eccentricity, m = 1 phase) are comparable in time.

Per cylindrical shell (columns <name>_<R>, a family per quantity, R the shell's geometric centre in units of a, R_L or code length):
    sigma       surface density sum m / area                                 hr       H/R, H = sqrt(<(z - <z>)^2>) (mass-weighted)
    rhomid      midplane density sum m / sum V over |z - <z>| < H/4           cs2      mass-weighted sound speed^2 gamma (gamma - 1) u
    vphi, vR    mass-weighted azimuthal / radial velocity                     vrat     v_phi / v_K with Omega_K^2 = G M / R^3 [1 + 3/4 q/(1+q)^2 (a/R)^2]
    mdot        -sum m v_R / dR  (> 0: inflow)                                (the binary quadrupole term; single BH or minidisc: Keplerian)
    vsup        H Omega_K / c_s (vertical support by thermal pressure ~ 1)   vsupA    H Omega_K / sqrt(c_s^2 + v_A^2)
    gauss       sigma / (sqrt(2 pi) rhomid H)  (1 for a Gaussian vertical profile)
    aR, aM      Reynolds sum m dv_R dv_phi / sum V P and Maxwell -sum V B_R B_phi / sum V P (dv: about the shell's mass-weighted means)
    toomre      c_s Omega_K / (pi G Sigma)                                    hH, nH   resolution: <h> / H and cells per H (H / <(m/rho)^(1/3)>)
    ecc, eccphi magnitude and angle of the mass-weighted eccentricity vector of the gas orbits (Keplerian about M)
    A1, A1phi   m = 1 azimuthal Fourier amplitude |sum m e^{i phi}| / sum m and its phase
  CBD only, the angular-momentum budget (outward fluxes through the cylinder R, in units of a^2 Omega_b, Omega_b = sqrt(G M / a^3); divide by mdot
  for the specific angular momentum ell = flux / Mdot in a^2 Omega_b):
    Fadv        advective sum m v_R j / dR (j = R v_phi)                      Frey     Reynolds R sum m dv_R dv_phi / dR
    Fmax        Maxwell -R sum V B_R B_phi / dR                               dTdR     torque of the binary on the gas per unit R/a (sum -m tau_bin / d(R/a))
    Tout        torque of the binary on all gas outside R                     jdot     -(Fadv + Frey + Fmax) - Tout: in steady state Mdot ell_0, the same at
                                                                                        every R (Mdot ell_0 / Mdot = ell_0: divide jdot by mdot)
    nu, alphaeff, tvisc   the effective viscosity of all the stresses (F = 3 pi nu Sigma R^2 Omega), nu Omega / c_s^2, and R^2 / nu in binary orbits
  With exclude_ids, the left-out gas (e.g. injected wind cells): mdotx, Fx (its advective J flux), jdotx = jdot - Fx, mdott = mdot + mdotx.
  Magnetic (snapshots with MagneticField; B in code units, mu_0 = 1, via UnitMagneticField_in_gauss):
    beta        sum V P / sum V B^2/2           Brms  sqrt(sum V B^2 / sum V)        vA2   sum V B^2 / sum m
    Psi         height-integrated toroidal field sum V B_phi / area (signed); PsiAbs  |Psi above <z>| + |Psi below|
    vRPsi       |v_R| PsiAbs (its log-slope is q of the flux budget)
    Qz, Qphi    MRI quality lambda/h with lambda = 2 pi v_A,z / Omega_K (or v_A,phi), h = (m / rho)^(1/3): mass-weighted geometric mean
    epsP        vertical loss of toroidal flux per Omega_K: -([v_phi B_z - v_z B_phi] at +2H minus at -2H) / (Omega_K Psi)  (layers 1.5-2.5 H; experimental)
Scalars: r_cav (R/a where sigma first reaches half its maximum), e_disc / varpi_disc (eccentricity vector of the gas between 1.5 and 5 a), A1_lump /
phi_lump (m = 1 between 2 and 4 a), r_steady (the largest R with t_visc < the time so far), mdot_3a, Mcbd, Mmini0/1 (gas in the Roche lobes), r50_mini0/1, r90_mini0/1 (radii holding 50 / 90% of the lobe's
gas, units of R_L: the minidisc truncation), a, e (osculating), q.
Extra budget terms (an explicit viscosity of the code, ...): register_flux(name, fn); the parameter `fluxes` lists those a run computes (columns F<name>_<R>).
Parameters: gamma (5/3), r_min / r_max / nbins of the CBD (0.5, 12, 44, units of a), exclude_ids (particle IDs left out, e.g. spawned wind cells), z_cut (the
|z| < z_cut R disc body for the stresses, Q and beta; None = all gas)."""
import numpy as np

from . import plugins

GAMMA = 5.0 / 3.0
FLUXES = {}         # extra angular-momentum fluxes of a code's own physics (an explicit viscosity, ...): name -> (fn, label); see register_flux


def register_flux(name, fn, label=None):
    """add a term F<name>_<R> to the CBD budget (and to jdot): fn(ctx) -> per-particle weights w such that the outward angular-momentum flux through
    the cylinder R is sum over the shell of w / dR (code units; e.g. an explicit viscosity: w = -m nu R^2 dOmega/dR). ctx: the gas arrays after
    exclude_ids (pos, vel, m, rho, u, ids), the frame (com, vcom, zhat, bh_pos, bh_vel, bh_mass, a, sep, G), the cylindrical R and per-particle shell
    dOmega/dR about the centre (R, dOmega_dR), gamma and the product's params. A run computes it when its name is in the product parameter `fluxes`."""
    FLUXES[name] = (fn, label or name)


MINI_EDGES = np.logspace(np.log10(0.05), 0.0, 17)


def _roche(q, a):
    q23 = q ** (2.0 / 3.0)
    return a * 0.49 * q23 / (0.6 * q23 + np.log1p(q ** (1.0 / 3.0)))


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0, 1.0])


def _basis(zhat):
    e1 = np.array([1.0, 0.0, 0.0]) - zhat[0] * zhat
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0.0, 1.0, 0.0]) - zhat[1] * zhat
    e1 = _unit(e1)
    return e1, np.cross(zhat, e1)


def _y(v):
    return f"{v:.4g}"


def _bin(idx, nb, w):
    ok = (idx >= 0) & (idx < nb)
    return np.bincount(idx[ok], weights=w[ok], minlength=nb)


def shells(rel, vel, m, rho, u, edges, zhat, GM, gamma, B=None, quad=0.0, a=1.0, z_cut=None, scale=1.0, hsml=None, G=None):
    """per-shell statistics of the gas at positions `rel` (relative to the centre, code length) and velocities `vel` (relative): a dict of arrays over the
    shells `edges` (in units of `scale`), as listed in the module docstring (without the CBD budget). quad: 3/4 q/(1+q)^2 for the binary's Omega_K."""
    e1, e2 = _basis(zhat)
    z = rel @ zhat
    x1, x2 = rel @ e1, rel @ e2
    R = np.hypot(x1, x2)
    phi = np.arctan2(x2, x1)
    cph, sph = np.cos(phi), np.sin(phi)
    v1, v2, vz = vel @ e1, vel @ e2, vel @ zhat
    vR, vphi = v1 * cph + v2 * sph, -v1 * sph + v2 * cph
    nb = len(edges) - 1
    Rs = R / scale
    idx = np.digitize(Rs, edges) - 1
    idx[(idx < 0) | (idx >= nb)] = -1
    ok = idx >= 0
    ic = np.where(ok, idx, 0)
    Rm = np.sqrt(edges[1:] * edges[:-1]) * scale
    dR = np.diff(edges) * scale
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * scale ** 2
    w = _bin(idx, nb, m)
    n = np.bincount(idx[ok], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = lambda x: _bin(idx, nb, m * x) / w
        zm = mean(z)
        dz = z - zm[ic]
        H = np.sqrt(mean(dz * dz))
        V = m / rho
        mid = ok & (np.abs(dz) < 0.25 * H[ic])
        rhomid = _bin(np.where(mid, idx, -1), nb, m) / _bin(np.where(mid, idx, -1), nb, V)
        cs2 = mean(gamma * (gamma - 1.0) * np.maximum(u, 0.0))
        mvphi, mvR = mean(vphi), mean(vR)
        OmK = np.sqrt(GM / Rm ** 3 * (1.0 + quad * (a / Rm) ** 2))
        dvR, dvphi = vR - mvR[ic], vphi - mvphi[ic]
        body = ok if z_cut is None else ok & (np.abs(dz) < z_cut * R)
        ib = np.where(body, idx, -1)
        P = (gamma - 1.0) * rho * np.maximum(u, 0.0)
        PV = _bin(ib, nb, P * V)
        out = {"sigma": w / area, "hr": H / Rm, "rhomid": rhomid, "cs2": cs2, "vphi": mvphi, "vR": mvR, "vrat": mvphi / (Rm * OmK),
               "mdot": -_bin(idx, nb, m * vR) / dR, "vsup": H * OmK / np.sqrt(cs2), "gauss": (w / area) / (np.sqrt(2 * np.pi) * rhomid * H),
               "aR": _bin(ib, nb, m * dvR * dvphi) / PV,
               "toomre": np.sqrt(cs2) * OmK / (np.pi * (G or 1.0) * w / area)}
        if hsml is not None:                         # resolution: the kernel size against the disc's scale height
            out["hH"] = mean(hsml) / H                                         # <h> / H (mass-weighted h)
            out["nH"] = H / mean(np.cbrt(m / rho))                              # cells per scale height (mean spacing (m / rho)^(1/3))
        # eccentricity vectors (Keplerian about GM) and the m = 1 mode, in the plane basis
        h = x1 * v2 - x2 * v1
        r3 = np.sqrt(x1 * x1 + x2 * x2 + z * z)
        ex = v2 * h / GM - x1 / r3
        ey = -v1 * h / GM - x2 / r3
        Ex, Ey = mean(ex), mean(ey)
        out["ecc"], out["eccphi"] = np.hypot(Ex, Ey), np.degrees(np.arctan2(Ey, Ex))
        Cx, Cy = mean(cph), mean(sph)
        out["A1"], out["A1phi"] = np.hypot(Cx, Cy), np.degrees(np.arctan2(Cy, Cx))
        if B is not None:
            b1, b2, Bz = B @ e1, B @ e2, B @ zhat
            BR, Bphi = b1 * cph + b2 * sph, -b1 * sph + b2 * cph
            B2 = (B * B).sum(1)
            out["aM"] = -_bin(ib, nb, V * BR * Bphi) / PV
            out["beta"] = PV / _bin(ib, nb, 0.5 * V * B2)
            out["Brms"] = np.sqrt(_bin(idx, nb, V * B2) / _bin(idx, nb, V))
            vA2 = _bin(idx, nb, V * B2) / w
            out["vA2"] = vA2
            out["vsupA"] = H * OmK / np.sqrt(cs2 + vA2)
            up = ok & (dz > 0)
            pu = _bin(np.where(up, idx, -1), nb, V * Bphi) / area
            pl = _bin(np.where(ok & ~up, idx, -1), nb, V * Bphi) / area
            out["Psi"], out["PsiAbs"] = pu + pl, np.abs(pu) + np.abs(pl)
            out["vRPsi"] = np.abs(mvR) * out["PsiAbs"]
            hh = np.cbrt(V)
            om = OmK[ic]
            for key, comp in (("Qz", Bz), ("Qphi", Bphi)):
                lam = 2 * np.pi * np.abs(comp) / np.sqrt(rho) / om
                lq = np.log10(np.maximum(lam / hh, 1e-30))
                out[key] = 10.0 ** (_bin(ib, nb, m * lq) / _bin(ib, nb, m))
            emf = vphi * Bz - vz * Bphi
            lay = lambda s: ok & (s * dz > 1.5 * H[ic]) & (s * dz < 2.5 * H[ic])
            top = _bin(np.where(lay(1), idx, -1), nb, V * emf) / _bin(np.where(lay(1), idx, -1), nb, V)
            bot = _bin(np.where(lay(-1), idx, -1), nb, V * emf) / _bin(np.where(lay(-1), idx, -1), nb, V)
            out["epsP"] = -(top - bot) / (OmK * out["Psi"])
    few = n < 20
    for k in out:
        out[k] = np.where(few, np.nan, out[k])
    out["_n"], out["_w"], out["_dR"], out["_Rm"], out["_idx"], out["_R"], out["_vR"], out["_vphi"], out["_dvR"], out["_dvphi"] = n, w, dR, Rm, idx, R, vR, vphi, dvR, dvphi
    if B is not None:
        out["_BR"], out["_Bphi"] = BR, Bphi
    return out


def _load(path, params):
    from .io import load_snapshot
    s = load_snapshot(path, gas_fields=("Coordinates", "Velocities", "Masses", "Density", "InternalEnergy", "SmoothingLength", "?MagneticField", "?ParticleIDs"))
    keep = np.ones(len(s["Masses"]), bool)
    ex = params.get("exclude_ids") or []
    if ex and "ParticleIDs" in s:
        keep &= ~np.isin(s["ParticleIDs"], np.asarray(ex, dtype=np.int64))
    return s, keep


def disc(path, params):
    from .fields import binary_torque, gravitational_constant, magnetic_unit_gauss
    from .maps import View
    from .units import Units
    s, keep = _load(path, params)
    gamma = float(params.get("gamma") or GAMMA)
    G = gravitational_constant(s)
    if "bh_pos" not in s or not len(s["bh_pos"]):
        return {}
    o = np.argsort(np.asarray(s["bh_id"]).astype(np.int64), kind="stable") if "bh_id" in s else np.arange(len(s["bh_pos"]))
    bm = np.asarray(s["bh_mdyn"] if "bh_mdyn" in s else s["bh_mass"], dtype=np.float64)[o]
    bp, bv = np.asarray(s["bh_pos"], np.float64)[o], np.asarray(s["bh_vel"], np.float64)[o]
    pos = np.asarray(s["Coordinates"], np.float64)[keep]
    vel = np.asarray(s["Velocities"], np.float64)[keep]
    m = np.asarray(s["Masses"], np.float64)[keep]
    rho = np.asarray(s["Density"], np.float64)[keep]
    u = np.asarray(s["InternalEnergy"], np.float64)[keep]
    B = None
    if "MagneticField" in s:
        un = Units.from_header(s.get("header", {}))
        bunit, _ = magnetic_unit_gauss(s, View())
        B = np.asarray(s["MagneticField"], np.float64)[keep] * bunit / np.sqrt(4 * np.pi * un.density_cgs * un.velocity_cm_s ** 2)   # code units, mu_0 = 1
    zc = params.get("z_cut")
    sc, out_arr = {}, {}
    binary = len(bp) >= 2
    if binary:
        m0, m1 = bm[0], bm[1]
        M = m0 + m1
        com, vcom = (m0 * bp[0] + m1 * bp[1]) / M, (m0 * bv[0] + m1 * bv[1]) / M
        r_rel, v_rel = bp[1] - bp[0], bv[1] - bv[0]
        sep = float(np.linalg.norm(r_rel))
        E = 0.5 * v_rel @ v_rel - G * M / sep
        hvec = np.cross(r_rel, v_rel)
        a = -G * M / (2 * E) if E < 0 else sep
        e = float(np.sqrt(max(0.0, 1 + 2 * E * (hvec @ hvec) / (G * M) ** 2)))
        zhat = _unit(hvec)
        q = min(m0, m1) / max(m0, m1)
        quad = 0.75 * q / (1 + q) ** 2
        sc.update({"a": a, "e": e, "q": q})
    else:
        M = bm[0]
        com, vcom = bp[0], bv[0]
        rr = pos - com
        near = np.linalg.norm(rr, axis=1) < float(params.get("r_max") or 12.0)
        zhat = _unit(np.sum(m[near, None] * np.cross(rr[near], vel[near] - vcom), axis=0))
        a, quad = 1.0, 0.0
    GM = G * M
    edges = np.logspace(np.log10(float(params.get("r_min") or 0.5)), np.log10(float(params.get("r_max") or 12.0)), int(params.get("nbins") or 44) + 1)
    rel, vrel = pos - com, vel - vcom
    cbd_sel = np.ones(len(m), bool)
    if binary:
        rl = [_roche(m0 / m1, sep), _roche(m1 / m0, sep)]
        d = [np.linalg.norm(pos - bp[k], axis=1) for k in (0, 1)]
        lobes = (d[0] < rl[0]) | (d[1] < rl[1])
        sc["Mcbd"] = float(m[~lobes].sum())
    hs = np.asarray(s["SmoothingLength"], np.float64)[keep]
    st = shells(rel, vrel, m, rho, u, edges, zhat, GM, gamma, B=B, quad=quad, a=a, z_cut=zc, scale=a, hsml=hs, G=G)
    Rmid = np.sqrt(edges[1:] * edges[:-1])
    Om_b = np.sqrt(GM / a ** 3)
    lunit = a * a * Om_b
    idx, R = st["_idx"], st["_R"]
    okb = idx >= 0
    j = R * st["_vphi"]
    fl = lambda wts: _bin(idx, len(Rmid), wts) / st["_dR"] / lunit
    fam = {k: v for k, v in st.items() if not k.startswith("_")}
    fam["Fadv"] = fl(m * st["_vR"] * j)
    fam["Frey"] = fl(R * m * st["_dvR"] * st["_dvphi"])
    jin = -(fam["Fadv"] + fam["Frey"])
    if B is not None:
        V = m / rho
        fam["Fmax"] = fl(-R * V * st["_BR"] * st["_Bphi"])
        jin = jin - fam["Fmax"]
    names = [n for n in (params.get("fluxes") or []) if n in FLUXES]
    if names:
        Om = st["vphi"] / st["_Rm"]
        dOm = np.gradient(np.nan_to_num(Om), st["_Rm"])
        ctx = {"pos": pos, "vel": vel, "m": m, "rho": rho, "u": u, "ids": np.asarray(s["ParticleIDs"])[keep] if "ParticleIDs" in s else None, "com": com,
               "vcom": vcom, "zhat": zhat, "bh_pos": bp, "bh_vel": bv, "bh_mass": bm, "a": a, "sep": sep if binary else None, "G": G, "R": R,
               "dOmega_dR": np.where(okb, dOm[np.clip(idx, 0, len(dOm) - 1)], 0.0), "gamma": gamma, "params": params, "path": path}
        for n in names:
            fam["F" + n] = fl(np.asarray(FLUXES[n][0](ctx), dtype=np.float64))
            jin = jin - fam["F" + n]
    if binary:
        tau = binary_torque({**s, "Coordinates": pos, "SmoothingLength": np.asarray(s["SmoothingLength"])[keep]})
        tg = -_bin(idx, len(Rmid), m * tau) / lunit               # torque of the binary on the gas, per shell
        fam["dTdR"] = tg / np.diff(edges)
        fam["Tout"] = np.cumsum(tg[::-1])[::-1]
        fam["jdot"] = jin - fam["Tout"]
    else:
        fam["jdot"] = jin
    # the gas left out by exclude_ids (e.g. injected wind cells): its mass and advective angular-momentum fluxes through the same cylinders
    if not keep.all():
        xp = np.asarray(s["Coordinates"], np.float64)[~keep] - com
        xv = np.asarray(s["Velocities"], np.float64)[~keep] - vcom
        xm = np.asarray(s["Masses"], np.float64)[~keep]
        e1, e2 = _basis(zhat)
        X1, X2 = xp @ e1, xp @ e2
        XR = np.hypot(X1, X2)
        V1, V2 = xv @ e1, xv @ e2
        XvR, Xj = (X1 * V1 + X2 * V2) / np.maximum(XR, 1e-30), X1 * V2 - X2 * V1
        xi = np.digitize(XR / a, edges) - 1
        xi[(xi < 0) | (xi >= len(Rmid))] = -1
        fam["mdotx"] = -_bin(xi, len(Rmid), xm * XvR) / st["_dR"]
        fam["Fx"] = _bin(xi, len(Rmid), xm * XvR * Xj) / st["_dR"] / lunit
        fam["jdotx"] = fam["jdot"] - fam["Fx"]                     # J inflow including the excluded gas
        fam["mdott"] = fam["mdot"] + fam["mdotx"]                  # Mdot of all the gas
    # effective viscosity of all the stresses (Reynolds, Maxwell, registered terms) and the viscous time: F = 3 pi nu Sigma R^2 Omega (Keplerian)
    fstress = fam["Frey"] + (fam["Fmax"] if "Fmax" in fam else 0.0) + sum(fam["F" + n] for n in names)
    with np.errstate(invalid="ignore", divide="ignore"):
        OmR = np.sqrt(GM / st["_Rm"] ** 3)
        nu = fstress * lunit / (3 * np.pi * fam["sigma"] * st["_Rm"] ** 2 * OmR)
        fam["nu"] = nu
        fam["tvisc"] = np.where(nu > 0, st["_Rm"] ** 2 / nu, np.nan) / (2 * np.pi / Om_b)     # in binary orbits (2 pi / Omega_b)
        fam["alphaeff"] = nu * OmR / fam["cs2"]
    t_orb = float(s["time"]) / (2 * np.pi / Om_b)
    reach = np.where(np.nan_to_num(fam["tvisc"], nan=np.inf) < t_orb)[0]
    sc["r_steady"] = float(Rmid[reach.max()]) if len(reach) else np.nan          # largest R whose viscous time is shorter than the time so far
    few = st["_n"] < 20
    for k in ["Fadv", "Frey", "Fmax", "jdot", "nu", "tvisc", "alphaeff"] + ["F" + n for n in names]:
        if k in fam:
            fam[k] = np.where(few, np.nan, fam[k])
    for k, arr in fam.items():
        for y, v in zip(Rmid, arr):
            sc[f"{k}_{_y(y)}"] = v
    # global scalars of the CBD
    sig = fam["sigma"]
    if np.isfinite(sig).any():
        k = int(np.nanargmax(sig))
        half = np.where(np.nan_to_num(sig[:k + 1]) >= 0.5 * sig[k])[0]
        sc["r_cav"] = float(Rmid[half[0]]) if len(half) else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        zr = rel @ zhat
        e1, e2 = _basis(zhat)
        x1, x2, v1, v2 = rel @ e1, rel @ e2, vrel @ e1, vrel @ e2
        Rcyl = np.hypot(x1, x2) / a
        h = x1 * v2 - x2 * v1
        r3 = np.sqrt(x1 * x1 + x2 * x2 + zr * zr)
        band = (Rcyl > 1.5) & (Rcyl < 5.0)
        if band.any():
            ex = np.sum(m[band] * (v2 * h / GM - x1 / r3)[band]) / m[band].sum()
            ey = np.sum(m[band] * (-v1 * h / GM - x2 / r3)[band]) / m[band].sum()
            sc["e_disc"], sc["varpi_disc"] = float(np.hypot(ex, ey)), float(np.degrees(np.arctan2(ey, ex)))
        lump = (Rcyl > 2.0) & (Rcyl < 4.0)
        if lump.any():
            ph = np.arctan2(x2, x1)[lump]
            cx, cy = np.sum(m[lump] * np.cos(ph)) / m[lump].sum(), np.sum(m[lump] * np.sin(ph)) / m[lump].sum()
            sc["A1_lump"], sc["phi_lump"] = float(np.hypot(cx, cy)), float(np.degrees(np.arctan2(cy, cx)))
    sc["mdot_3a"] = float(np.interp(np.log(3.0), np.log(Rmid), np.nan_to_num(fam["mdot"]))) if binary else np.nan
    # minidiscs
    if binary:
        for k in (0, 1):
            sel = d[k] < rl[k]
            sc[f"Mmini{k}"] = float(m[sel].sum())
            if sel.sum() < 100:
                continue
            rr, vv = pos[sel] - bp[k], vel[sel] - bv[k]
            zk = _unit(np.sum(m[sel, None] * np.cross(rr, vv), axis=0))
            mk = bm[k]
            stk = shells(rr, vv, m[sel], rho[sel], u[sel], MINI_EDGES, zk, G * mk, gamma, B=None if B is None else B[sel], z_cut=zc, scale=rl[k], hsml=hs[sel], G=G)
            Rk = np.sqrt(MINI_EDGES[1:] * MINI_EDGES[:-1])
            for name, arr in stk.items():
                if not name.startswith("_") and name not in ("Psi", "PsiAbs", "vRPsi", "epsP"):
                    for y, v in zip(Rk, arr):
                        sc[f"{name}_mini{k}_{_y(y)}"] = v
            Rc = np.linalg.norm(rr - (rr @ zk)[:, None] * zk, axis=1) / rl[k]
            o2 = np.argsort(Rc)
            cm = np.cumsum(m[sel][o2]) / m[sel].sum()
            sc[f"r50_mini{k}"], sc[f"r90_mini{k}"] = float(np.interp(0.5, cm, Rc[o2])), float(np.interp(0.9, cm, Rc[o2]))
    return {"scalars": sc}


def _applies(path):
    import h5py
    with h5py.File(path, "r") as f:
        return "PartType5" in f and len(f["PartType5"]["Masses"]) >= 1 and "PartType0" in f


META = {"r_cav": ("cavity radius (sigma at half its maximum) [a]", "disc"), "e_disc": ("disc eccentricity (1.5-5 a)", "disc"),
        "varpi_disc": ("disc periapsis angle [deg]", "disc"), "A1_lump": ("m = 1 amplitude (2-4 a)", "disc"), "phi_lump": ("m = 1 phase [deg]", "disc"),
        "mdot_3a": ("Mdot at 3 a (> 0 inflow)", "disc"), "r_steady": ("steady-state radius (t_visc < t) [a]", "disc"), "Mcbd": ("gas outside the Roche lobes", "disc"), "a": ("binary a", "orbit"), "e": ("binary e", "orbit"),
        "q": ("mass ratio", "orbit"), **{f"{x}_mini{k}": (f"minidisc {k}: radius holding {x[1:]}% of the lobe gas [R_L]", "minidiscs") for x in ("r50", "r90") for k in (0, 1)},
        **{f"Mmini{k}": (f"gas in Roche lobe {k}", "minidiscs") for k in (0, 1)}}
# log-slopes d ln X / d ln R of standard steady-disc closures, for reference lines on slope panels (radial panel `refs = true`): Shakura & Sunyaev (1973)
# zones B (gas pressure, electron scattering) and C (gas pressure, Kramers); a magnetically supported disc with B ~ rho^(2/3) (Hopkins et al. 2024);
# the flux-frozen magnetically supported solution with pure steady advection of toroidal flux (B H |v_R| = const). rho_mid = Sigma / H.
CLOSURE_SLOPES = {
    "sigma": {"SS73 B": -3 / 5, "SS73 C": -3 / 4, "B~rho^2/3": -5 / 6, "flux advection": -11 / 10},
    "hr": {"SS73 B": 1 / 20, "SS73 C": 1 / 8, "B~rho^2/3": 1 / 6, "flux advection": 3 / 10},
    "cs2": {"SS73 B": -9 / 10, "SS73 C": -3 / 4, "B~rho^2/3": -3 / 4},
    "rhomid": {"SS73 B": -3 / 5 - 21 / 20, "SS73 C": -3 / 4 - 9 / 8, "B~rho^2/3": -2.0, "flux advection": -11 / 10 - 13 / 10},
    "Brms": {"B~rho^2/3": -4 / 3, "flux advection": -7 / 5},
}

FAMILIES = {"sigma_": "Sigma", "hr_": "H/R", "rhomid_": "rho_mid", "cs2_": "c_s^2", "vphi_": "v_phi", "vR_": "v_R", "vrat_": "v_phi / v_K", "mdot_": "Mdot (> 0 in)",
            "vsup_": "H Omega_K / c_s", "vsupA_": "H Omega_K / sqrt(c_s^2 + v_A^2)", "gauss_": "Sigma / (sqrt(2 pi) rho_mid H)", "aR_": "alpha Reynolds",
            "aM_": "alpha Maxwell", "ecc_": "e_d", "eccphi_": "periapsis [deg]", "A1_": "m = 1 amplitude", "A1phi_": "m = 1 phase [deg]", "Fadv_": "advective",
            "Frey_": "Reynolds", "Fmax_": "Maxwell", "dTdR_": "binary torque density", "Tout_": "binary torque outside R", "jdot_": "J inflow - torque outside",
            "beta_": "beta", "Brms_": "B_rms", "vA2_": "v_A^2", "Psi_": "Psi", "PsiAbs_": "|Psi|", "vRPsi_": "|v_R| |Psi|", "Qz_": "Q_z", "Qphi_": "Q_phi",
            "epsP_": "epsilon_P", "toomre_": "Toomre Q", "hH_": "<h> / H", "nH_": "cells per H", "mdotx_": "Mdot of the excluded gas",
            "Fx_": "excluded gas (advective)", "jdotx_": "J inflow incl. excluded gas", "mdott_": "Mdot of all gas", "nu_": "nu_eff", "tvisc_": "t_visc [orbits]",
            "alphaeff_": "alpha_eff (all stresses)"}

plugins.register_product("disc", disc, needs=("Coordinates", "Velocities", "Masses", "Density", "InternalEnergy", "SmoothingLength", "?MagneticField", "?ParticleIDs",
                                              "PartType5"),
                         version=2, label="disc: profiles, angular-momentum budget, eccentricity, minidiscs", applies=_applies,
                         params={"gamma": None, "r_min": 0.5, "r_max": 12.0, "nbins": 44, "exclude_ids": [], "z_cut": None, "fluxes": []}, defaults=["r_cav"],
                         meta={**{k: {"label": a, "group": g} for k, (a, g) in META.items()}, "varpi_disc": {"label": META["varpi_disc"][0], "group": "disc", "wrap": 360.0},
                               "phi_lump": {"label": META["phi_lump"][0], "group": "disc", "wrap": 360.0},
                               **{k: {"label": v, "group": "profiles"} for k, v in FAMILIES.items()}})
