"""Histograms / CDFs (hist.py), azimuthal modes (modes.py), map power spectra (pk.py), DMD (dmd.py) and their layout panels (plotpanels.py), on synthetic
data with known answers."""
import os

import h5py
import numpy as np
import pytest

from splosh import plugins, products as P

HDR = {"UnitLength_In_CGS": 3.0857e18, "UnitMass_In_CGS": 1.989e33, "UnitVelocity_In_CGS": 1e5, "Gravitational_Constant_In_Code_Inits": 1.0}


def write(path, pos, m=None, rho=None, u=None, h=0.05, time=0.0, bh_pos=None, bh_vel=None, bh_m=None, vel=None):
    n = len(pos)
    with h5py.File(path, "w") as f:
        hd = f.create_group("Header")
        hd.attrs["Time"] = time
        for k, v in HDR.items():
            hd.attrs[k] = v
        g = f.create_group("PartType0")
        g["Coordinates"] = np.asarray(pos, float)
        g["Velocities"] = np.zeros((n, 3)) if vel is None else vel
        g["Masses"] = np.full(n, 1.0 / n) if m is None else m
        g["Density"] = np.ones(n) if rho is None else rho
        g["InternalEnergy"] = np.ones(n) if u is None else u
        g["SmoothingLength"] = np.full(n, h)
        g["ParticleIDs"] = np.arange(1, n + 1)
        if bh_m is not None:
            b = f.create_group("PartType5")
            b["Coordinates"], b["Masses"] = np.atleast_2d(bh_pos).astype(float), np.atleast_1d(bh_m).astype(float)
            b["Velocities"] = np.zeros_like(np.atleast_2d(bh_pos), dtype=float) if bh_vel is None else np.atleast_2d(bh_vel).astype(float)
            b["ParticleIDs"] = np.arange(10_000_000, 10_000_000 + len(np.atleast_1d(bh_m)))
    return str(path)


BINARY = {"bh_pos": [[0.5, 0, 0], [-0.5, 0, 0]], "bh_vel": [[0, 0.5, 0], [0, -0.5, 0]], "bh_m": [0.5, 0.5]}      # a = 1, circular, G M = 1


def ring(n, R0=3.0, dR=0.2, eps=0.0, m_mode=2, phi0=0.0, tilt=0.0, node=0.0, seed=1):
    """particles in an annulus, Sigma ~ 1 + eps cos m (phi - phi0) (rejection sampling); optionally tilted by `tilt` about the line at azimuth `node`."""
    rng = np.random.default_rng(seed)
    phi = np.empty(0)
    while len(phi) < n:
        p = 2 * np.pi * rng.random(2 * n)
        keep = rng.random(2 * n) * (1 + abs(eps)) < 1 + eps * np.cos(m_mode * (p - phi0))
        phi = np.concatenate([phi, p[keep]])
    phi = phi[:n]
    R = R0 + dR * (rng.random(n) - 0.5)
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), np.zeros(n)])
    if tilt:                                                   # rotate about the node line by `tilt`
        k = np.array([np.cos(node), np.sin(node), 0.0])
        c, s = np.cos(tilt), np.sin(tilt)
        pos = pos * c + np.cross(k, pos) * s + np.outer(pos @ k, k) * (1 - c)
    return pos


@pytest.fixture
def run(tmp_path):
    (tmp_path / "output").mkdir()
    return tmp_path


# ----------------------------------------------------------------------------- hist

def test_distribution_known_answers():
    from splosh import hist as H
    v = np.linspace(0.0, 1.0, 100_001)[1:]
    d = H.distribution(v, np.ones_like(v), bins=10, xlog=False, xr=[0, 1])
    assert np.allclose(d["pdf"], 1.0, rtol=1e-3) and np.allclose(d["cdf"], np.linspace(0.1, 1, 10), atol=1e-4)
    assert d["median"] == pytest.approx(0.5, abs=1e-4) and d["p16"] == pytest.approx(0.16, abs=1e-4)
    d = H.distribution(v, np.ones_like(v), bins=5, xlog=False, xr=[0.5, 1.0])        # the weight below the range enters the CDF
    assert d["cdf"][0] == pytest.approx(0.6, abs=1e-4) and d["in_range"] / d["total"] == pytest.approx(0.5, abs=1e-4)
    x = 10.0 ** np.linspace(-2, 2, 400_001)                   # log-uniform over 4 dex: 1/4 per dex, median 1
    d = H.distribution(x, np.ones_like(x), bins=8, xlog=True, xr=[-2, 2])
    assert np.allclose(d["pdf"], 0.25, rtol=1e-3) and d["median"] == pytest.approx(1.0, rel=1e-3)
    w = np.where(x > 1, 3.0, 1.0)                             # weights: 3/4 of the weight above 1
    d = H.distribution(x, w, bins=4, xlog=True, xr=[-2, 2])
    assert d["cdf"][1] == pytest.approx(0.25, abs=1e-3)


def _box_snapshot(path, n=40_000, seed=3, time=0.0, dens_range=(-2, 2)):
    rng = np.random.default_rng(seed)
    pos = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(-0.1, 0.1, n)])
    rho = 10.0 ** rng.uniform(*dens_range, n)
    return write(path, pos, rho=rho, time=time, h=0.05)


def test_hist_panel_and_renderers(run):
    from splosh import layout as L, layout_mpl, layout_preview as lp
    from splosh.io import load_snapshot
    from splosh.mapstore import MapStore
    paths = [_box_snapshot(run / "output" / f"snapshot_{k:03d}.hdf5", seed=k, time=float(k)) for k in range(3)]
    lay = {"schema": "splosh-layout-1", "name": "h", "scope": "snapshot", "panel": [
        {"type": "hist", "id": "x", "quantities": "x", "xlog": False, "bins": 20, "xr": [-1, 1], "center": "xyz:0,0,0", "width": 1.0, "show": "both"},
        {"type": "hist", "id": "d", "quantities": ["density", "u"], "bins": 16, "xr": [-2, 2], "center": "xyz:0,0,0", "width": 4.0}]}
    path = paths[1]
    lo, tiles, notes, _ = lp.tiles_for(lay, path, lambda f: load_snapshot(path, gas_fields=f), MapStore(str(run / "store")))
    assert not notes, notes
    (lab, x, pdf), (_, xc, cdf) = tiles["x"]["curves"]
    assert np.allclose(pdf[5:15], 1.0, rtol=0.1) and not pdf[:5].any()       # the window [-0.5, 0.5]: uniform in x, 1 per unit inside it
    assert cdf[-1] == pytest.approx(1.0) and tiles["x"]["styles"][1]["twin"]
    d = tiles["d"]["curves"]
    assert len(d) == 2 and np.allclose(d[0][2][2:-2], 0.25, rtol=0.15)      # density log-uniform over 4 dex
    assert layout_mpl.render(lo, tiles) is not None and L.render_layout(lo, tiles).ndim == 3
    run_lay = dict(lay, scope="run", panel=[dict(lay["panel"][1], quantities=["density"], snaps=[0, -1], show="cdf")])
    lo, tiles, notes, _ = lp.tiles_for(run_lay, path, lambda f: load_snapshot(path, gas_fields=f), MapStore(str(run / "store")))
    assert [c[0] for c in tiles["d"]["curves"]] == ["t = 0", "t = 2"]
    for bad in ({"type": "hist", "id": "q"}, {"type": "hist", "id": "q", "quantities": "x", "show": "pmf"}, {"type": "hist", "id": "q", "quantities": "x", "snaps": "all"}):
        with pytest.raises(L.LayoutError):
            L.validate_layout({"name": "b", "panel": [bad]})


def test_hist_product_families_and_heatmap(run):
    from splosh import series as SR
    paths = [_box_snapshot(run / "output" / f"snapshot_{k:03d}.hdf5", seed=k, time=float(k), dens_range=(-2, 2 + k)) for k in range(2)]
    plugins.load_plugins()
    s = P.PRODUCTS["hist"]["fn"](paths[0], P.params_of("hist", {"quantities": ["density"], "center": "xyz:0,0,0"}))["scalars"]
    pdf = {float(c[len("density_"):]): v for c, v in s.items() if c.startswith("density_") and c[8:9] in "-0123456789"}
    assert len(pdf) == 40 and np.allclose(sorted(pdf), np.arange(-2, 2, 0.1) + 0.05, atol=1e-6)     # a fixed grid of 0.1 dex
    assert np.allclose(list(pdf.values()), 0.25, rtol=0.1)
    assert s["density_med"] == pytest.approx(1.0, rel=0.05) and s["densitycdf_2"] == pytest.approx(1.0) and s["density_frac"] == pytest.approx(1.0)
    P.register_sources()
    P.compute(str(run), ["hist"], paths, workers=1, log=lambda m: None)
    ctx = SR.context(str(run), paths)
    t = SR.heat_tile({"source": "prod_hist", "prefix": "density_", "nx": 10}, ctx)
    assert t["heat"]["z"].shape == (50, 2)                                  # the union of the rows: -2 .. 3 in 0.1 dex
    assert np.nansum(t["heat"]["z"][:, 1]) * 0.1 == pytest.approx(1.0, rel=1e-6)


def test_run_params_file_and_opt_in(run):
    paths = [_box_snapshot(run / "output" / "snapshot_000.hdf5")]
    assert "pk" not in P.default_names(str(run)) and "hist" in P.default_names(str(run))
    d = P.product_dir(str(run), "")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "params.toml"), "w") as f:
        f.write('[hist]\nquantities = ["u"]\nper_dex = 4\n\n[pk]\nnpix = 64\n')
    assert "pk" in P.default_names(str(run))
    assert P.params_of("hist", run_dir=str(run))["quantities"] == ["u"] and P.params_of("hist")["quantities"] == ["density", "T"]
    r = P.compute_one(str(run), "hist", paths[0])
    assert any(c.startswith("u_") for c in r["scalars"]) and not any(c.startswith("density_") for c in r["scalars"])
    assert P.load_result(str(run), "hist", paths[0]) is not None                      # read back with the run's parameters


# ----------------------------------------------------------------------------- modes

def test_modes_m2_ring_binary_frame(run):
    eps, phi0 = 0.4, np.radians(30.0)
    pos = ring(200_000, eps=eps, phi0=phi0)
    p = write(run / "output" / "snapshot_000.hdf5", pos, **BINARY)
    plugins.load_plugins()
    s = P.PRODUCTS["modes"]["fn"](p, P.params_of("modes", {"r_min": 2.0, "r_max": 4.5, "nbins": 5, "g_min": 2.5, "g_max": 3.5}))["scalars"]
    assert s["a"] == pytest.approx(1.0) and s["A2"] == pytest.approx(eps / 2, abs=0.01) and s["phi2"] == pytest.approx(30.0, abs=1.0)
    assert s["A1"] < 0.01 and s["A3"] < 0.01 and s["A4"] < 0.01
    A2 = [v for c, v in s.items() if c.startswith("A2_") and v is not None and np.isfinite(v)]
    assert len(A2) == 1 and A2[0] == pytest.approx(eps / 2, abs=0.01)       # the one populated shell
    # the gas tide at the centre: ring mass 1 at R = 3: T_zz = -1/27, the m = 2 part 0.75 eps / 27 at phi0 (units G M / a^3 = 1)
    assert s["Qzz"] == pytest.approx(-1 / 27, rel=0.01) and s["Q22"] == pytest.approx(0.75 * eps / 27, rel=0.05)
    assert s["phiQ22"] == pytest.approx(30.0, abs=2.0) and s["W"] < 1e-6


def test_modes_bending_and_single_bh(run):
    tilt, node = np.radians(10.0), np.radians(40.0)
    pos = ring(100_000, tilt=tilt, node=node)
    # one black hole: z from the gas angular momentum, which the tilt changes; set velocities of a circular orbit in the ring's own plane
    p = write(run / "output" / "snapshot_000.hdf5", pos, bh_pos=[0, 0, 0], bh_m=1.0, vel=np.zeros_like(pos))
    plugins.load_plugins()
    s = P.PRODUCTS["modes"]["fn"](p, P.params_of("modes", {"r_min": 2.0, "r_max": 4.5, "nbins": 6, "g_min": 2.5, "g_max": 3.5}))["scalars"]
    # no gas angular momentum: z falls back to +z of the simulation (a zero vector -> z), the ring is tilted by 10 deg in that frame
    assert s["W"] == pytest.approx(np.tan(tilt), rel=0.02)
    assert s["Wphi"] == pytest.approx(np.degrees(node) + 90.0, abs=1.5)    # the highest point: 90 deg ahead of the ascending node
    assert "a" not in s and np.isfinite(s["Qzz"])


def test_modes_heatmap_and_radial(run):
    from splosh import series as SR
    paths = [write(run / "output" / f"snapshot_{k:03d}.hdf5", ring(40_000, eps=0.5, phi0=0.2 * k, seed=k), time=float(k), **BINARY) for k in range(3)]
    P.register_sources()
    P.compute(str(run), ["modes"], paths, workers=1, log=lambda m: None)
    ctx = SR.context(str(run), paths)
    tab = SR.table("prod_modes", ctx)
    assert np.allclose(np.diff(tab.columns["phi2"]), np.degrees(0.2), atol=1.0)            # the pattern turns by 0.2 rad per snapshot
    t = SR.heat_tile({"source": "prod_modes", "prefix": "A2_", "nx": 10, "yscale": "log"}, ctx)
    assert t["heat"]["z"].shape[0] == 32
    r = SR.radial_tile({"source": "prod_modes", "prefix": "A2_", "windows": 1, "xaxis": "time"}, ctx)
    assert np.nanmax(r["curves"][0][2]) == pytest.approx(0.25, abs=0.02)


# ----------------------------------------------------------------------------- pk

def _gaussian_field(n, slope, seed=0):
    rng = np.random.default_rng(seed)
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing="ij"))
    amp = np.where(k > 0, k ** (slope / 2.0), 0.0)
    f = np.fft.ifft2(amp * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))).real
    return f / f.std()


def test_power_spectrum_known_answers():
    from splosh import pk as PK
    n, L = 128, 2.0
    x = (np.arange(n) + 0.5) * L / n
    img = 1.0 + 0.3 * np.sin(2 * np.pi * 8 * x / L)[None, :] * np.ones((n, 1))
    r = PK.power_spectrum(img, L, window="none", subtract="delta", nk=40)
    kpk = r["k"][np.nanargmax(r["P"])]
    assert kpk / r["kf"] == pytest.approx(8.0, rel=0.06) and r["kf"] == pytest.approx(np.pi)
    f = PK.prepare(img, "delta")
    P2 = np.abs(np.fft.fft2(f) * (L / n) ** 2) ** 2 / L ** 2                # Parseval with the module's normalisation
    assert P2.sum() / L ** 2 == pytest.approx(f.var(), rel=1e-9) and r["var"] == pytest.approx(f.var(), rel=1e-9)
    g = _gaussian_field(256, -3.0)
    for win, tol in (("none", 0.15), ("hann", 0.2), ("tukey", 0.2)):
        r = PK.power_spectrum(g, 1.0, window=win, subtract="mean")
        assert PK.slope(r["kn"], r["P"], 4, 64) == pytest.approx(-3.0, abs=tol), win
    lg = PK.power_spectrum(10 ** g, 1.0, window="none", subtract="log")
    assert PK.slope(lg["kn"], lg["P"], 4, 64) == pytest.approx(-3.0, abs=0.15)
    assert PK.slope_label(-5 / 3) == "k^-5/3" and PK.slope_label(-11 / 3) == "k^-11/3" and PK.slope_label(-2) == "k^-2"
    (lab, kk, pp), = PK.reference_lines(r["k"], r["P"], [-2.0])
    assert lab == "k^-2" and np.allclose(np.diff(np.log(pp)) / np.diff(np.log(kk)), -2.0)
    (lab, kk, pp), = PK.reference_lines(r["k"], r["P"], [-2.0], compensate=2.0)                 # on a k^2 P(k) curve: flat, still named k^-2
    assert lab == "k^-2" and np.allclose(pp, pp[0]) and PK.power_text(11 / 3) == "11/3"


def test_pk_panel_and_product(run):
    from splosh import layout as L, layout_mpl, layout_preview as lp
    from splosh.io import load_snapshot
    from splosh.mapstore import MapStore
    rng = np.random.default_rng(5)
    n = 30_000
    pos = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.normal(0, 0.05, n)])
    paths = [write(run / "output" / f"snapshot_{k:03d}.hdf5", pos, h=0.08, time=float(k), **BINARY) for k in range(2)]
    lay = {"schema": "splosh-layout-1", "name": "p", "scope": "run", "panel": [
        {"type": "pk", "id": "k", "center": "xyz:0,0,0", "width": 3.0, "npix": 64, "slopes": [-1.6667, -3.6667], "snaps": 2}]}
    lo, tiles, notes, _ = lp.tiles_for(lay, paths[0], lambda f: load_snapshot(paths[0], gas_fields=f), MapStore(str(run / "store")))
    assert not notes, notes
    labs = [c[0] for c in tiles["k"]["curves"]]
    assert labs == ["t = 0", "t = 1", "k^-5/3", "k^-11/3"] and tiles["k"]["xlog"] and tiles["k"]["ylog"]
    k = tiles["k"]["curves"][0][1]
    from splosh.pk import kbins
    e = kbins(64)
    assert e[-1] == 32 and np.all(np.isfinite(tiles["k"]["curves"][0][2]))
    assert k[0] / (2 * np.pi / 3.0) <= 1.5 and k[-1] == pytest.approx(2 * np.pi / 3.0 * np.sqrt(e[-1] * e[-2]), rel=1e-6)   # 1/code length, fundamental to Nyquist
    assert layout_mpl.render(lo, tiles) is not None and L.render_layout(lo, tiles).ndim == 3
    with pytest.raises(L.LayoutError):
        L.validate_layout({"name": "b", "panel": [{"type": "pk", "id": "q", "window": "kaiser"}]})
    plugins.load_plugins()
    s = P.PRODUCTS["pk"]["fn"](paths[0], P.params_of("pk", {"npix": 64, "nk": 12}))["scalars"]
    assert s["width"] == pytest.approx(8.0) and s["kf"] == pytest.approx(2 * np.pi / 8.0)             # 8 a about the binary
    assert 8 <= len([c for c in s if c.startswith("P_")]) <= 12 and np.isfinite(s["slope"])


# ----------------------------------------------------------------------------- DMD

def test_exact_dmd_recovers_frequencies_and_growth():
    from splosh import dmd as D
    n, dt, nt = 32, 0.05, 60
    g = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(g, g)
    R, PH = np.hypot(X, Y), np.arctan2(Y, X)
    f1, f2, gam = 1.5, 0.4, -0.8                 # an m = 2 pattern turning at Omega_p = pi f1 (temporal frequency f1); a decaying m = 1 pattern turning at 2 pi f2
    t = np.arange(nt) * dt
    cols = [2.0 - R + 0.3 * np.exp(-((R - 0.5) / 0.2) ** 2) * np.cos(2 * PH - 2 * np.pi * f1 * tk)
            + 0.2 * np.exp(gam * tk) * np.exp(-((R - 0.3) / 0.15) ** 2) * np.cos(PH - 2 * np.pi * f2 * tk) for tk in t]
    M = np.array([c.ravel() for c in cols]).T
    res = D.analyse(t, M, rank=20)
    pos = [i for i in range(len(res["freq"])) if res["freq"][i] > 0.05]
    got = sorted((round(float(res["freq"][i]), 6), float(res["growth"][i])) for i in pos[:2])
    assert got[0][0] == pytest.approx(f2, abs=1e-6) and got[0][1] == pytest.approx(gam, abs=1e-6)
    assert got[1][0] == pytest.approx(f1, abs=1e-6) and got[1][1] == pytest.approx(0.0, abs=1e-6)
    assert res["rank"] == 5 and abs(res["freq"][0]) < 1e-9 and abs(res["growth"][0]) < 1e-9                  # the steady part: f = 0
    sm = D.analyse(t, M, rank=20, subtract_mean=True)                                                         # the DFT of the record: biased
    assert sm["rank"] == 4 and abs(sm["freq"][0]) == pytest.approx(f1, abs=0.01)
    # in units of a period P = 2: frequencies x 2, rates x 2
    r2 = D.analyse(t, M, rank=20, period=2.0)
    assert np.max(np.abs(r2["freq"])) == pytest.approx(2 * np.max(np.abs(res["freq"])))
    # uneven sampling is resampled onto a uniform grid
    tt = t + np.where(np.arange(nt) % 2, 0.01, 0.0)
    tt[-1] = t[-1]
    assert D.resample_uniform(tt, M)[0].size == nt


def test_dmd_cli_rotating_pattern_and_panel(run):
    from splosh import cli, dmd as D, layout as L, layout_mpl, layout_preview as lp
    from splosh.io import load_snapshot
    from splosh.mapstore import MapStore
    base = ring(30_000, R0=1.0, dR=0.8, eps=0.8, seed=2)
    Om, dt = 1.0, 0.2
    paths = []
    for k in range(16):
        a = Om * dt * k
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        paths.append(write(run / "output" / f"snapshot_{k:03d}.hdf5", base @ rot.T, h=0.15, time=dt * k))
    cli.main(["dmd", str(run), "--width", "3", "--npix", "40", "--rank", "12", "--center", "xyz:0,0,0", "--quiet"])
    res = D.load(str(run / "analysis_viewer" / "dmd" / "dmd.npz"))
    top = D.leading(res, 1, fmin=0.05)[0]
    assert res["freq"][top] == pytest.approx(2 * Om / (2 * np.pi), rel=0.02) and abs(res["growth"][top]) < 0.02    # m = 2 at 2 Omega_p / 2 pi
    out = run / "analysis_viewer" / "dmd"
    assert (out / "dmd.npz").is_file() and (out / "dmd.png").is_file()
    r = D.load(str(out / "dmd.npz"))
    assert r["modes"].shape[:2] == (40, 40) and r["unit"] == "code time" and r["params"]["width"] == 3.0
    lay = {"schema": "splosh-layout-1", "name": "d", "scope": "run", "ncols": 2, "panel": [
        {"type": "dmd", "id": "s"}, {"type": "dmd", "id": "e", "show": "eigs"}, {"type": "dmd", "id": "v", "show": "sv"},
        {"type": "dmd", "id": "m", "show": "mode", "mode": 0}, {"type": "dmd", "id": "x", "name": "other"}]}
    lo, tiles, notes, _ = lp.tiles_for(lay, paths[0], lambda f: load_snapshot(paths[0], gas_fields=f), MapStore(str(run / "store")))
    assert set(notes) == {"x"} and "splosh dmd" in notes["x"]
    assert tiles["m"]["heat"]["z"].shape == (40, 40) and tiles["m"]["heat"]["zlim"][0] == -tiles["m"]["heat"]["zlim"][1]
    assert len(tiles["e"]["curves"]) == 2 and tiles["s"]["ylog"]
    assert layout_mpl.render(lo, tiles) is not None and L.render_layout(lo, tiles).ndim == 3
    with pytest.raises(L.LayoutError):
        L.validate_layout({"name": "b", "panel": [{"type": "dmd", "id": "q", "show": "modes"}]})
