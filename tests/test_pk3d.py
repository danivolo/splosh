"""The snapshot product pk3d (3D spectra in a cube of the disc): known answers."""
import h5py
import numpy as np
import pytest

from splosh import pk3d as K, plugins, products as P

L = 2.0
HDR = {"UnitLength_In_CGS": 3.0857e18, "UnitMass_In_CGS": 1.989e33, "UnitVelocity_In_CGS": 1e5, "Gravitational_Constant_In_Code_Inits": 1.0}


def lattice(m1, extent):
    """particles on a cubic lattice of m1^3 points over [-extent/2, extent/2]^3 (unit density): positions, spacing."""
    sp = extent / m1
    g = (np.arange(m1) + 0.5) * sp - 0.5 * extent
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3), sp


def kernel_ft(k, H):
    """the Fourier transform of the GIZMO cubic spline of support H at wavenumber k (W normalised to 1): the amplitude a smoothed sine keeps."""
    from splosh.kernel import w3
    r = np.linspace(0, H, 20001)
    f = w3(r / H) / H ** 3 * 4 * np.pi * r ** 2 * np.sinc(k * r / np.pi)
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(r)))


def test_parseval_grid():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(3, 32, 32, 32)) + np.linspace(0, 3, 32)[None, :, None, None]       # a non-periodic field with a gradient
    for window in ("hann", "tukey", "none"):
        r = K.spectrum3d(f, L, window, nk=20)
        assert (r["E"] * r["dkn"]).sum() + r["hi"] + r["k0"] == pytest.approx(r["total"], rel=1e-12)
        assert abs(r["k0"]) < 1e-12 * r["total"]                                            # the weighted mean removed: no k = 0 term
        w = K.window3d(32, window)
        g = f - (f * w).sum((1, 2, 3), keepdims=True) / w.sum()
        assert r["total"] == pytest.approx(0.5 * np.mean((g * w) ** 2) * 3 / np.mean(w * w), rel=1e-12)
    r = K.spectrum3d(f, L, subtract_mean=False)
    assert (r["E"] * r["dkn"]).sum() + r["hi"] + r["k0"] == pytest.approx(r["total"], rel=1e-12) and r["k0"] > 0
    e = K.kedges(64, 32)
    assert e[0] == 0.5 and e[-1] == 32.5 and np.all(np.diff(e) >= 1) and np.all(np.mod(e, 1) == 0.5)


def test_single_mode_in_B_from_particles():
    """B_z = B0 sin(2 pi 4 x / L) on a particle lattice: the peak at kn = 4 with the energy B0^2 / 4 times the kernel's smoothing |W(k)|^2."""
    X, sp = lattice(44, 1.3 * L)
    N = len(X)
    H = 2.0 * sp
    B = np.zeros((N, 3))
    B[:, 2] = 1.0 * np.sin(2 * np.pi * 4 * X[:, 0] / L)
    f = K.grid_fields(X, np.zeros((N, 3)), np.full(N, sp ** 3), np.ones(N), np.full(N, H), B, L, 64)
    assert f["rho"].mean() == pytest.approx(1.0, rel=0.01) and np.ptp(f["rho"]) < 0.02          # uniform: no empty cells, density 1
    s = K.spectra(f, L, {"nk": 32})
    fam = sorted((float(c[4:]), v) for c, v in s.items() if c.startswith("E_B_"))
    kn, E = np.array(fam).T
    assert kn[np.argmax(E)] == pytest.approx(4.0, abs=0.15)
    ok = np.abs(kn - 4) < 1.6                                                                   # the Hann window spreads the line over kn = 3 .. 5
    expect = 0.25 * kernel_ft(2 * np.pi * 4 / L, H) ** 2
    assert np.sum(E[ok]) == pytest.approx(expect, rel=0.03)                                     # dkn = 1 at these kn
    assert np.sum(E[~ok]) < 1e-3 * np.sum(E[ok])
    assert s["tot_Bz"] == pytest.approx(s["tot_B"], rel=1e-9) and s["tot_BR"] == 0 and s["tot_Bphi"] == 0
    assert s["emag"] == pytest.approx(expect, rel=0.03) and s["tot_v"] == 0
    assert s["tot_B"] == pytest.approx(sum(E * np.diff(K.kedges(64, 32))) + s["hi_B"], rel=1e-9)   # Parseval through the product's columns


def test_power_law_recovered():
    """random-phase fields with the 3D power P(k) ~ k^-11/3: the shell spectrum E(k) ~ k^2 P ~ k^-5/3, averaged over three realisations. The window's
    own Fourier width mixes the lowest kn: Hann up to kn ~ 3, Tukey (taper 1/8 of the side) up to kn ~ 8, so the fits start there."""
    n = 64
    fr = np.fft.fftfreq(n) * n
    fz = np.fft.rfftfreq(n) * n
    k = np.sqrt(fr[:, None, None] ** 2 + fr[None, :, None] ** 2 + fz[None, None, :] ** 2)
    amp = np.where(k > 0, np.maximum(k, 1e-9) ** (-11 / 6), 0.0)
    acc = {}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        f = np.fft.irfftn(amp * np.exp(2j * np.pi * rng.random(k.shape)), s=(n, n, n), axes=(0, 1, 2))
        for window in ("none", "hann", "tukey"):
            r = K.spectrum3d(f, L, window, nk=32)
            acc[window] = acc.get(window, 0) + r["E"]
    kn = r["kn"]
    assert K.slope(kn, acc["none"], 4, 24) == pytest.approx(-5 / 3, abs=0.05)
    assert K.slope(kn, acc["hann"], 4, 24) == pytest.approx(-5 / 3, abs=0.08)
    assert K.slope(kn, acc["tukey"], 8, 31) == pytest.approx(-5 / 3, abs=0.08)


def _write_disc(path, n=40_000, seed=2, bfield=False, kepler=True):
    """a thin disc (H/R = 0.05, Sigma ~ 1/R between 1.5 and 4.5 a) about a circular equal-mass binary (a = 1, G M = 1), on exact circular orbits with
    Omega^2 = G M / R^3 (1 + 3/16 (a/R)^2) (the binary quadrupole, q = 1)."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(1.5, 4.5, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    z = rng.normal(0, 0.05 * R)
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), z])
    Om = np.sqrt(R ** -3 * (1 + 0.1875 / R ** 2)) if kepler else 0.0
    vel = np.column_stack([-R * Om * np.sin(phi), R * Om * np.cos(phi), np.zeros(n)])
    m = np.full(n, 1.0 / n)
    rho = m / (2 * np.pi * R * 3.0 / n) / (np.sqrt(2 * np.pi) * 0.05 * R) * np.exp(-0.5 * (z / (0.05 * R)) ** 2)
    hs = np.minimum(2.0 * (m / rho) ** (1 / 3), 1.0)
    with h5py.File(path, "w") as f:
        hd = f.create_group("Header")
        hd.attrs["Time"] = 0.0
        for k_, v in HDR.items():
            hd.attrs[k_] = v
        g = f.create_group("PartType0")
        g["Coordinates"], g["Velocities"], g["Masses"], g["Density"] = pos, vel, m, rho
        g["SmoothingLength"], g["InternalEnergy"], g["ParticleIDs"] = hs, np.ones(n), np.arange(1, n + 1)
        if bfield:
            g["MagneticField"] = np.tile([0.0, 0.0, 1e-3], (n, 1))
        b = f.create_group("PartType5")
        b["Coordinates"] = np.array([[0.5, 0, 0], [-0.5, 0, 0]])
        b["Velocities"] = np.array([[0, 0.5, 0], [0, -0.5, 0]])
        b["Masses"] = np.array([0.5, 0.5])
        b["ParticleIDs"] = np.array([10_000_001, 10_000_002])
    return str(path)


def test_keplerian_disc_has_no_turbulence(tmp_path):
    p = _write_disc(tmp_path / "snapshot_000.hdf5", bfield=True)
    plugins.load_plugins()
    assert "pk3d" in P.PRODUCTS and not P.PRODUCTS["pk3d"].get("auto", True)
    run = tmp_path
    assert "pk3d" not in P.default_names(str(run))                                             # opt-in
    base = {"n": 32, "nk": 12}
    none = P.PRODUCTS["pk3d"]["fn"](p, P.params_of("pk3d", {**base, "rotation": "none"}))["scalars"]
    assert none["box"] == pytest.approx(2.0) and none["L"] == pytest.approx(2.0) and none["r0"] == pytest.approx(3.0)
    assert none["npart"] > 1000 and none["Ekin"] > 0 and none["fill"] > 0
    for rot in ("kepler", "shell"):
        s = P.PRODUCTS["pk3d"]["fn"](p, P.params_of("pk3d", {**base, "rotation": rot}))["scalars"]
        assert s["Ekin"] < 1e-4 * none["Ekin"], rot
        assert s["tot_v"] < 1e-4 * none["tot_v"], rot
    # a uniform vertical field: all of it in the mean (Bz_mean ~ B0 where the gas is), no fluctuation of B_R, B_phi
    from splosh.fields import magnetic_unit_gauss
    from splosh.io import load_snapshot
    from splosh.maps import View
    from splosh.units import Units
    sn = load_snapshot(p, gas_fields=("Coordinates",))
    un = Units.from_header(sn["header"])
    B0 = 1e-3 * magnetic_unit_gauss(sn, View())[0] / np.sqrt(4 * np.pi * un.density_cgs * un.velocity_cm_s ** 2)
    assert s["tot_BR"] == 0 and s["tot_Bphi"] == 0 and s["BR_mean"] == 0
    assert 0 < s["Bz_mean"] <= B0 * (1 + 1e-6) and s["emag"] <= 0.5 * B0 ** 2 * (1 + 1e-6)
    fam = [c for c in s if c.startswith("E_Bz_")]
    assert len(fam) == len([c for c in s if c.startswith("P_rho_")]) == len([c for c in none if c.startswith("E_v_")]) >= 8


def test_turbulent_velocity_in_disc(tmp_path):
    """a velocity perturbation dv_z = A sin(2 pi 4 x' / L) in the box frame on top of the Keplerian flow: E_v peaks at kn ~ 4 after the subtraction."""
    p = _write_disc(tmp_path / "snapshot_000.hdf5", n=60_000)
    with h5py.File(p, "r+") as f:
        x = f["PartType0/Coordinates"][:, 0] - 3.0
        v = f["PartType0/Velocities"][...]
        v[:, 2] += 0.01 * np.sin(2 * np.pi * 4 * x / L)
        f["PartType0/Velocities"][...] = v
    s = K.pk3d(p, P.params_of("pk3d", {"n": 32, "nk": 16}))["scalars"]
    fam = sorted((float(c[4:]), v) for c, v in s.items() if c.startswith("E_v_"))
    kn, E = np.array(fam).T
    assert kn[np.argmax(E)] == pytest.approx(4.0, abs=0.3)
    assert np.isfinite(s["slope_v"]) and np.isfinite(s["slope_rho"]) and "slope_B" not in s           # no MagneticField: no B columns


def test_product_series_and_layout(tmp_path):
    """two snapshots through products.compute: the families as a heatmap against time and as radial panels; the example layout validates."""
    import os
    from splosh import layout as LY, series as SR
    (tmp_path / "output").mkdir()
    paths = []
    for k in range(2):
        p = _write_disc(tmp_path / "output" / f"snapshot_{k:03d}.hdf5", n=20_000, seed=k, bfield=True)
        with h5py.File(p, "r+") as f:
            f["Header"].attrs["Time"] = float(k)
            b = f["PartType0/MagneticField"][...]
            b[:, 0] = 1e-3 * np.sin(2 * np.pi * 3 * f["PartType0/Coordinates"][:, 1] / L)
            f["PartType0/MagneticField"][...] = b
        paths.append(p)
    plugins.load_plugins()
    P.register_sources()
    assert "pk3d" not in P.default_names(str(tmp_path))
    d = tmp_path / "analysis_viewer" / "products"
    d.mkdir(parents=True)
    (d / "params.toml").write_text("[pk3d]\nn = 32\nnk = 12\n")                              # the run's own parameters switch the opt-in product on
    assert "pk3d" in P.default_names(str(tmp_path))
    P.compute(str(tmp_path), ["pk3d"], paths, workers=1, log=lambda m: None)
    ctx = SR.context(str(tmp_path), paths)
    nfam = len([c for c in SR.table("prod_pk3d", ctx).columns if c.startswith("E_B_")])
    t = SR.heat_tile({"source": "prod_pk3d", "prefix": "E_B_", "nx": 10, "yscale": "log"}, ctx)
    assert nfam >= 8 and t["heat"]["z"].shape[0] == nfam
    r = SR.radial_tile({"source": "prod_pk3d", "prefixes": ["E_BR_", "E_Bz_"], "windows": 1, "xaxis": "time"}, ctx)
    kn, eR = r["curves"][0][1], r["curves"][0][2]
    assert kn[np.nanargmax(eR)] == pytest.approx(3.0, abs=0.5)                                  # the B_R mode of 3 wavelengths along y
    lay = LY.validate_layout(LY.load_layout(os.path.join(os.path.dirname(__file__), "..", "docs", "layouts", "mri_spectra.toml")))
    assert {p["source"] for p in lay["panel"]} == {"prod_pk3d"}
