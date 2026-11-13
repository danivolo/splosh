"""The core product `disc` (disc.py) on synthetic discs with known answers: a Keplerian disc about one black hole with Sigma ~ 1/R, H/R = 0.1 and a Gaussian
vertical profile; an eccentric ring; a circumbinary ring (quadrupole Omega_K, the budget columns)."""
import h5py
import numpy as np
import pytest

from splosh import plugins, products as P

HDR = {"UnitLength_In_CGS": 3.0857e18, "UnitMass_In_CGS": 1.989e33, "UnitVelocity_In_CGS": 1e5, "Gravitational_Constant_In_Code_Inits": 1.0}


def write(path, pos, vel, m, rho, u, bh_pos, bh_vel, bh_m, B=None):
    n = len(pos)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["Time"] = 0.0
        for k, v in HDR.items():
            h.attrs[k] = v
        g = f.create_group("PartType0")
        g["Coordinates"], g["Velocities"], g["Masses"], g["Density"], g["InternalEnergy"] = pos, vel, m, rho, u
        g["SmoothingLength"] = np.full(n, 0.01)
        g["ParticleIDs"] = np.arange(1, n + 1)
        if B is not None:
            g["MagneticField"] = B
        b = f.create_group("PartType5")
        b["Coordinates"], b["Velocities"], b["Masses"] = np.atleast_2d(bh_pos), np.atleast_2d(bh_vel), np.atleast_1d(bh_m)
        b["ParticleIDs"] = np.arange(10_000_000, 10_000_000 + len(np.atleast_1d(bh_m)))
    return str(path)


def kepler_disc(n=200_000, M=1.0, hr=0.1, r0=1.0, r1=10.0, ecc=0.0, seed=1):
    rng = np.random.default_rng(seed)
    R = r0 * (r1 / r0) ** rng.random(n) if ecc == 0 else np.full(n, 3.0)      # Sigma ~ 1/R: uniform in R (dN ~ Sigma R dR = dR)
    if ecc == 0:
        R = r0 + (r1 - r0) * rng.random(n)
    phi = 2 * np.pi * rng.random(n)
    H = hr * R
    z = rng.normal(0, 1, n) * H
    vk = np.sqrt(M / R)
    if ecc:                                                   # orbits of semi-major axis 3, eccentricity ecc, periapsis along x, at random mean anomaly
        a, Mn = 3.0, 2 * np.pi * rng.random(n)
        E = Mn.copy()
        for _ in range(50):
            E = Mn + ecc * np.sin(E)
        x, y = a * (np.cos(E) - ecc), a * np.sqrt(1 - ecc ** 2) * np.sin(E)
        r = np.hypot(x, y)
        fac = np.sqrt(M * a) / r
        vx, vy = -fac * np.sin(E), fac * np.sqrt(1 - ecc ** 2) * np.cos(E)
        pos = np.column_stack([x, y, 0.001 * rng.normal(size=n)])
        vel = np.column_stack([vx, vy, np.zeros(n)])
        cs2 = np.full(n, 1e-4)
    else:
        pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), z])
        vel = np.column_stack([-vk * np.sin(phi), vk * np.cos(phi), np.zeros(n)])
        cs2 = (hr * vk) ** 2                                   # H = c_s / Omega
    gamma = 5 / 3
    u = cs2 / (gamma * (gamma - 1))
    m = np.full(n, 1e-6)
    rho = m[0] * 100 / (np.sqrt(2 * np.pi) * H) / (2 * np.pi * R) * np.exp(-0.5 * (pos[:, 2] / H) ** 2) * n / (r1 - r0) / 100 if not ecc else np.full(n, 1.0)
    return pos, vel, m, rho, u


@pytest.fixture
def run(tmp_path):
    (tmp_path / "output").mkdir()
    return tmp_path


def test_kepler_disc_single_bh(run):
    pos, vel, m, rho, u = kepler_disc()
    p = write(run / "output" / "snapshot_000.hdf5", pos, vel, m, rho, u, [0, 0, 0], [0, 0, 0], 1.0)
    plugins.load_plugins()
    s = P.PRODUCTS["disc"]["fn"](p, P.params_of("disc", {"r_min": 1.5, "r_max": 9.0, "nbins": 12}))["scalars"]
    def fam(k):
        out = []
        for c, v in s.items():
            try:
                float(c[len(k) + 1:]) if c.startswith(k + "_") else None
            except ValueError:
                continue
            if c.startswith(k + "_"):
                out.append(v)
        return np.array(out, dtype=float)
    R = np.array([float(c[len("sigma_"):]) for c in s if c.startswith("sigma_")])
    sig = fam("sigma")
    slope = np.polyfit(np.log(R), np.log(sig), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.05)                         # Sigma ~ 1/R
    assert np.allclose(fam("hr"), 0.1, rtol=0.03)                          # H/R
    assert np.allclose(fam("vrat"), 1.0, rtol=0.01)                        # Keplerian
    assert np.allclose(fam("vsup"), 1.0, rtol=0.04)                        # H Omega / c_s
    assert np.allclose(fam("gauss"), 1.0, rtol=0.15)                       # Gaussian vertical profile
    assert np.abs(fam("mdot")).max() < 1e-3 * sig.max() * R.max() * 1.0    # no radial flow
    assert fam("ecc").max() < 0.02 and fam("A1").max() < 0.03


def test_eccentric_ring(run):
    pos, vel, m, rho, u = kepler_disc(n=60_000, ecc=0.2)
    p = write(run / "output" / "snapshot_000.hdf5", pos, vel, m, rho, u, [0, 0, 0], [0, 0, 0], 1.0)
    plugins.load_plugins()
    s = P.PRODUCTS["disc"]["fn"](p, P.params_of("disc", {"r_min": 2.0, "r_max": 4.0, "nbins": 4}))["scalars"]
    e = np.array([v for c, v in s.items() if c.startswith("ecc_") and v is not None and np.isfinite(v)])
    phi = np.array([v for c, v in s.items() if c.startswith("eccphi_") and v is not None and np.isfinite(v)])
    assert np.allclose(e, 0.2, atol=0.01) and np.allclose(phi, 0.0, atol=2.0)    # e and the periapsis along +x
    assert s["e_disc"] == pytest.approx(0.2, abs=0.01)


def test_circumbinary_budget_columns(run):
    pos, vel, m, rho, u = kepler_disc(n=100_000, r0=1.5, r1=8.0)
    a, M = 1.0, 1.0
    v = 0.5 * np.sqrt(M / a)
    p = write(run / "output" / "snapshot_000.hdf5", pos, vel, m, rho, u, [[0.5, 0, 0], [-0.5, 0, 0]], [[0, v, 0], [0, -v, 0]], [0.5, 0.5])
    plugins.load_plugins()
    s = P.PRODUCTS["disc"]["fn"](p, P.params_of("disc", {"r_min": 2.0, "r_max": 7.0, "nbins": 8}))["scalars"]
    assert s["a"] == pytest.approx(1.0, rel=1e-6) and s["e"] < 1e-6 and s["q"] == 1.0
    vr = np.array([v for c, v in s.items() if c.startswith("vrat_")])
    assert np.all(vr < 1.0) and np.all(vr > 0.97)                          # the quadrupole raises Omega_K a little above the Keplerian v_phi of the gas
    for k in ("Fadv", "Frey", "dTdR", "Tout", "jdot", "mdot"):
        assert any(c.startswith(k + "_") for c in s), k
    assert "Fmax_2.17" not in s                                            # no field: no Maxwell column
    assert s["Mcbd"] == pytest.approx(m.sum(), rel=1e-6)


def test_radial_and_spectrum_panels(tmp_path):
    """layout panels `radial` (a family averaged over time windows, ratios, slopes) and `spectrum` (Lomb-Scargle) on a toy product and a toy source."""
    from splosh import layout as L, series as SR
    from test_server_fields import _snap

    def fn(path, params):
        k = int(path[-6])                                   # snapshot index
        return {"scalars": {**{f"f_{R:g}": float(R ** -1 * (1 + k)) for R in (1.0, 2.0, 4.0, 8.0)}, **{f"g_{R:g}": 2.0 for R in (1.0, 2.0, 4.0, 8.0)}}}
    plugins.register_product("toyr", fn, version=1, params={})
    try:
        run = tmp_path / "run"; (run / "output").mkdir(parents=True)
        paths = [_snap(run / "output" / f"snapshot_{k:03d}.hdf5") for k in range(4)]
        P.register_sources()
        P.compute(str(run), ["toyr"], paths, workers=1, log=lambda m: None)
        ctx = SR.context(str(run), paths)
        lay = L.validate_layout({"schema": "splosh-layout-1", "name": "t", "scope": "run", "panel": [
            {"type": "radial", "id": "a", "source": "prod_toyr", "prefix": "f_", "windows": 1, "xaxis": "time"},
            {"type": "radial", "id": "b", "source": "prod_toyr", "prefix": "f_", "divide": "g_", "slope": True, "xaxis": "time"},
            {"type": "spectrum", "id": "c", "source": "prod_toyr", "columns": ["f_1"]}]})
        pa, pb, pc = lay["panel"]
        t = SR.radial_tile(pa, ctx)
        (lab, x, y), = t["curves"]
        assert list(x) == [1.0, 2.0, 4.0, 8.0] and np.allclose(y, 2.5 / x)                     # mean of (1 + k) / R over k = 0..3
        assert "band" in t["styles"][0]
        (_, x, y), = SR.radial_tile(pb, ctx)["curves"]
        assert np.allclose(y, -1.0)                                                            # d ln(f / g) / d ln R
        with pytest.raises(L.LayoutError):
            L.validate_layout({"schema": "splosh-layout-1", "name": "t", "panel": [{"type": "radial", "id": "x", "source": "s"}]})
        assert SR.spectrum_tile(pc, ctx)["xlog"]
    finally:
        plugins.PRODUCTS.pop("toyr", None); plugins.SERIES.pop("prod_toyr", None)


def test_extra_flux_hook(run):
    """register_flux: a code's own angular-momentum flux becomes F<name>_<R> and is taken out of jdot."""
    from splosh import disc as D
    pos, vel, m, rho, u = kepler_disc(n=50_000, r0=1.5, r1=8.0)
    v = 0.5
    p = write(run / "output" / "snapshot_000.hdf5", pos, vel, m, rho, u, [[0.5, 0, 0], [-0.5, 0, 0]], [[0, v, 0], [0, -v, 0]], [0.5, 0.5])
    plugins.load_plugins()
    D.register_flux("toy", lambda ctx: ctx["m"] * 0 + 1e-3)
    try:
        base = P.params_of("disc", {"r_min": 2.0, "r_max": 7.0, "nbins": 5, "exclude_ids": [], "fluxes": []})
        s0 = P.PRODUCTS["disc"]["fn"](p, base)["scalars"]
        s1 = P.PRODUCTS["disc"]["fn"](p, {**base, "fluxes": ["toy"]})["scalars"]
        k = next(c for c in s1 if c.startswith("Ftoy_"))
        R = k[len("Ftoy_"):]
        assert s1[k] > 0 and s1["jdot_" + R] == pytest.approx(s0["jdot_" + R] - s1[k], rel=1e-9)
        assert not any(c.startswith("Ftoy_") for c in s0)
    finally:
        D.FLUXES.pop("toy", None)


def test_fit_scatter_parray(tmp_path):
    """fit_profile recovers a torqued-disc Sigma and a power law; scatter and parray tiles on a toy product."""
    from splosh import series as SR
    from test_server_fields import _snap
    x = np.geomspace(1.5, 10, 30)
    y = 3.0 * x ** -1.2 * (1 - 0.7 / np.sqrt(x))
    r = SR.fit_profile(x, y, "torqued")
    assert r["params"]["n"] == pytest.approx(1.2, abs=1e-3) and r["params"]["ell_0"] == pytest.approx(0.7, abs=1e-3)
    assert SR.fit_profile(x, 2 * x ** -0.5, "power")["params"]["slope"] == pytest.approx(-0.5)

    def fn(path, params):
        k = int(path[-6])
        R = (1.0, 2.0, 4.0)
        return {"scalars": {**{f"b_{q:g}": 10.0 ** (q + k) for q in R}, **{f"m_{q:g}": 0.1 / 10.0 ** (q + k) for q in R}},
                "arrays": {"h": np.full((3, 4), float(k + 1)), "xe": np.array([1.0, 2, 4, 8]), "ye": np.linspace(0, 1, 5)}}
    plugins.register_product("toys", fn, version=1, params={})
    try:
        run = tmp_path / "run"; (run / "output").mkdir(parents=True)
        paths = [_snap(run / "output" / f"snapshot_{k:03d}.hdf5") for k in range(3)]
        P.register_sources()
        P.compute(str(run), ["toys"], paths, workers=1, log=lambda m: None)
        ctx = SR.context(str(run), paths)
        t = SR.scatter_tile({"source": "prod_toys", "x": "b_", "y": "m_", "windows": 1, "xaxis": "time", "color": "R", "xlog": True, "ylog": True}, ctx)
        assert len(t["scatter"]["x"]) == 9 and np.allclose(t["scatter"]["x"] * t["scatter"]["y"], 0.1)
        h = SR.parray_tile({"product": "toys", "array": "h", "xedges": "xe", "yedges": "ye", "windows": 1, "xaxis": "time", "norm": "column"}, ctx)
        assert h["heat"]["z"].shape == (4, 3) and np.allclose(h["heat"]["z"].sum(axis=0), 1.0)
    finally:
        plugins.PRODUCTS.pop("toys", None); plugins.SERIES.pop("prod_toys", None)


def test_resolution_fields(run):
    """hres = (c_s / Omega_K) / h on the Keplerian disc with c_s = 0.1 v_K and h = 0.01."""
    from splosh.frames import make_frame
    from splosh.io import load_snapshot
    from splosh.maps import FieldContext, View, _field_values
    pos, vel, m, rho, u = kepler_disc(n=20_000)
    p = write(run / "output" / "snapshot_000.hdf5", pos, vel, m, rho, u, [0, 0, 0], [0, 0, 0], 1.0)
    s = load_snapshot(p, gas_fields=("Coordinates", "SmoothingLength", "Masses", "Density", "InternalEnergy", "Velocities"))
    v = View(center="bh", field="hres")
    hres = _field_values("hres", FieldContext(s, make_frame(s, "bh", "none"), v))
    r = np.linalg.norm(pos, axis=1)
    R = np.hypot(pos[:, 0], pos[:, 1])
    assert np.allclose(hres, 0.1 * R ** -0.5 * r ** 1.5 / 0.01, rtol=1e-4)              # c_s = 0.1 v_K(R), Omega_K = r^-3/2 (G = M = 1)
