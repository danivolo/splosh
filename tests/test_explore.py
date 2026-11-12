"""Exploration tools: residual and gradient maps (transforms.py), the asinh softening and histogram-equalisation stretches, symmetric limits, and their plumbing through View, shot, sheet, tiles and the CLI."""
import json
import os

import numpy as np
import pytest

from splosh import cli, figures, imaging, limits, transforms
from splosh.mapstore import MapStore, view_key
from splosh.maps import View, render_map
from conftest import lattice_snapshot

N = 160


def meta_for(extent=(-4.0, 4.0, -4.0, 4.0), field="rho", length=("kpc", 1.0)):
    return {"extent": list(extent), "view": {"field": field}, "label": "column density", "quantity_unit": {"unit": "Msun pc^-2"}, "length_unit": {"unit": length[0], "factor": length[1]}}


def disc_map(amp=0.1, m=2, extent=(-4.0, 4.0, -4.0, 4.0), n=N):
    """rho(R, phi) = R^-1.5 (1 + amp cos(m phi)) on the pixel centres."""
    x = extent[0] + (np.arange(n) + 0.5) * (extent[1] - extent[0]) / n
    X, Y = np.meshgrid(x, x)
    R, phi = np.hypot(X, Y), np.arctan2(Y, X)
    return (R ** -1.5 * (1 + amp * np.cos(m * phi))).astype(np.float32), R, phi


# ----------------------------------------------------------------------------- spec strings
@pytest.mark.parametrize("spec,expect", [("none", "none"), ("", "none"), ("resid", "resid:azim"), ("resid:azim", "resid:azim"), ("resid:mean", "resid:mean"),
                                         ("resid:smooth:0.50", "resid:smooth:0.5"), ("grad", "grad"), ("grad:0.25", "grad:0.25")])
def test_canonical(spec, expect):
    assert transforms.canonical(spec) == expect


@pytest.mark.parametrize("bad", ["resid:smooth", "resid:smooth:-1", "resid:smooth:x", "resid:azim:3", "grad:0", "grad:1:2", "log", "resid:foo"])
def test_bad_specs(bad):
    with pytest.raises(ValueError):
        transforms.parse(bad)


def test_view_validates_post_and_keys_differ():
    with pytest.raises(ValueError):
        View(post="bogus").validate()
    assert view_key("s", View(post="none")) != view_key("s", View(post="resid"))
    assert View().post == "none"


# ----------------------------------------------------------------------------- the transforms
def test_azimuthal_residual_recovers_the_perturbation():
    img, R, phi = disc_map(0.1, 2)
    out, info = transforms.apply(img, meta_for(), "resid")
    ok = (R > 0.6) & (R < 3.6)
    want = 0.1 * np.cos(2 * phi)
    assert np.abs(out[ok] - want[ok]).max() < 0.02          # ring means of the cos(2 phi) part vanish up to the discretisation
    assert info["mode"] == "contrast" and info["unit"] == "" and info["symmetric"] and info["post"] == "resid:azim"
    assert "azimuthal" in info["label"]


def test_axisymmetric_map_has_zero_residual():
    img, R, _ = disc_map(0.0)
    out, _ = transforms.apply(img, meta_for(), "resid")
    assert np.nanmax(np.abs(out[(R > 0.6) & (R < 3.6)])) < 0.03       # rings one pixel wide of a steep profile: the pixel's own R spread, not noise


def test_mean_and_smooth_residuals():
    img, _, _ = disc_map(0.0)
    out, _ = transforms.apply(np.full((N, N), 3.0, np.float32), meta_for(), "resid:mean")
    assert np.allclose(out, 0.0)
    flat, info = transforms.apply(np.full((N, N), 3.0, np.float32), meta_for(), "resid:smooth:0.4")
    assert np.allclose(flat, 0.0, atol=1e-6) and "Gaussian" in info["label"]
    spike = np.ones((N, N), np.float32); spike[80, 80] = 5.0
    out, _ = transforms.apply(spike, meta_for(), "resid:smooth:0.4")      # 0.4 length = 8 px
    assert out[80, 80] > 3.0 and abs(out[20, 20]) < 1e-3                 # the spike stands out against its smoothed self, the far field does not


def test_signed_fields_use_a_difference_in_their_unit():
    img, _, _ = disc_map(0.1)
    sig = img - np.mean(img)
    out, info = transforms.apply(sig, meta_for(field="vx"), "resid:mean")
    assert info["mode"] == "difference" and info["unit"] == "Msun pc^-2"
    assert np.allclose(out, sig - np.mean(sig), atol=1e-6)


def test_nan_stays_nan_and_unreferenced_rings_interpolate():
    img, R, _ = disc_map(0.0)
    img[R > 3.0] = np.nan
    img[(R > 1.0) & (R < 1.1)] = np.nan
    out, _ = transforms.apply(img, meta_for(), "resid")
    assert np.isnan(out[R > 3.0]).all() and np.isnan(out[(R > 1.0) & (R < 1.1)]).all()


def test_gaussian_nan_conserves_and_ignores_nan():
    a = np.zeros((101, 101)); a[50, 50] = 1.0
    g = transforms.gaussian_nan(a, 3.0)
    assert abs(g.sum() - 1.0) < 1e-3 and g[50, 50] == g.max()
    peak = 1.0 / (2 * np.pi * 9.0)
    assert abs(g[50, 50] - peak) / peak < 0.02
    b = np.full((40, 40), 2.0); b[10:20, 10:20] = np.nan
    gb = transforms.gaussian_nan(b, 2.0)
    assert np.isnan(gb[10:20, 10:20]).all() and np.allclose(gb[np.isfinite(gb)], 2.0)


def test_gradient_of_an_exponential():
    n, a = 200, 0.5
    x = -4.0 + (np.arange(n) + 0.5) * 8.0 / n
    img = np.exp(a * x)[None, :].repeat(n, 0).astype(np.float32)
    out, info = transforms.apply(img, meta_for(), "grad")
    want = a / np.log(10.0)                                  # dex per length
    assert np.allclose(out[40:160, 40:160], want, rtol=2e-3)
    assert info["mode"] == "log-gradient" and info["unit"] == "dex / kpc"
    out2, _ = transforms.apply(img, meta_for(length=("pc", 1000.0)), "grad")      # physical length factor: per pc
    assert np.allclose(out2[40:160, 40:160], want / 1000.0, rtol=2e-3)


def test_gradient_of_a_signed_field_is_linear():
    n = 120
    x = -4.0 + (np.arange(n) + 0.5) * 8.0 / n
    img = (2.0 * x)[None, :].repeat(n, 0).astype(np.float32)
    out, info = transforms.apply(img, meta_for(field="vx"), "grad")
    assert np.allclose(out[30:90, 30:90], 2.0, rtol=1e-3) and info["mode"] == "gradient"


def test_style_defaults():
    assert transforms.style("none") is None
    assert transforms.style("resid") == {"cmap": "RdBu_r", "scale": "linear", "symmetric": True}
    assert transforms.style("grad")["cmap"] == "magma"


# ----------------------------------------------------------------------------- stretches and limits
def test_asinh_default_is_the_legacy_stretch_and_softening_changes_it():
    a = np.random.default_rng(0).lognormal(size=(64, 64))
    lo, hi = imaging.limits(a, scale="asinh")
    legacy = imaging.stretch(a, lo, hi, "asinh")
    s = 0.5 * (abs(hi) + abs(lo)) / 10.0
    want = np.clip((np.arcsinh(a / s) - np.arcsinh(lo / s)) / (np.arcsinh(hi / s) - np.arcsinh(lo / s)), 0, 1)
    assert np.allclose(legacy, want)
    assert np.allclose(imaging.stretch(a, lo, hi, "asinh:0.1"), legacy)
    soft, hard = imaging.stretch(a, lo, hi, "asinh:0.001"), imaging.stretch(a, lo, hi, "asinh:50")
    lin = imaging.stretch(a, lo, hi, "linear")
    assert np.nanmean(soft) > np.nanmean(legacy) > np.nanmean(lin)            # smaller F: more logarithmic, the faint end is lifted
    assert np.abs(hard - lin).max() < 0.02                                    # a huge F is linear


def test_scale_parsing_errors():
    for bad in ("asinh:0", "asinh:x", "asinh:101", "log:3", "histeq:2", "sqrt"):
        with pytest.raises(ValueError):
            imaging.parse_scale(bad)
    assert imaging.parse_scale("asinh") == "asinh" and str(imaging.parse_scale("asinh:0.05")) == "asinh:0.05"
    assert imaging.parse_scale("asinh:0.05") == "asinh"           # compares with the bare name


def test_histeq_is_flat_and_monotone():
    a = np.random.default_rng(1).lognormal(sigma=2.0, size=(200, 200))
    lo, hi = imaging.limits(a, scale="linear", pclip=(0.0, 100.0))
    sc = imaging.fit_scale(a, lo, hi, "histeq")
    t = imaging.stretch(a, lo, hi, sc)
    h, _ = np.histogram(t, bins=10, range=(0, 1))
    assert h.min() > 0.8 * h.mean() and h.max() < 1.2 * h.mean()
    order = np.argsort(a.ravel())
    assert np.all(np.diff(t.ravel()[order]) >= -1e-12)
    ticks = figures.make_ticks(lo, hi, sc)
    assert ticks == sorted(ticks) and len(ticks) >= 4


def test_histeq_plateau_and_nan():
    a = np.ones((50, 50)); a[:, 25:] = 2.0; a[0, 0] = np.nan
    t = imaging.stretch(a, 1.0, 2.0, imaging.fit_scale(a, 1.0, 2.0, "histeq"))
    assert np.isnan(t[0, 0]) and 0.2 < t[10, 5] < 0.3 and 0.7 < t[10, 40] < 0.8       # each plateau sits at the middle of its ranks


def test_symmetric_limits():
    a = np.concatenate([np.linspace(-1, 3, 1000), [np.nan]])
    lo, hi = imaging.limits(a, pclip=(1.0, 99.0), scale="linear", symmetric=True)
    assert lo == -hi and abs(hi - np.percentile(np.abs(a[:-1]), 99)) < 1e-9
    acc = limits.Accumulator("linear", symmetric=True)
    acc.add(np.linspace(-5.0, 1.0, 1001)); acc.add(np.linspace(-2.0, 3.0, 1001))
    lo, hi = acc.result()
    assert lo == -hi and abs(hi - np.percentile(np.abs(np.concatenate([np.linspace(-5.0, 1.0, 1001), np.linspace(-2.0, 3.0, 1001)])), 99)) < 0.02
    assert acc.result((1.0, 100.0)) == (-5.0, 5.0)
    assert limits.resolve("frame", "linear", None, None, None, img=a, symmetric=True)[0] < 0


def test_colorize_symmetric_puts_zero_in_the_middle():
    img = np.array([[-2.0, 0.0, 2.0]], np.float32)
    rgb, lim = imaging.colorize(img, "RdBu_r", "linear", symmetric=True)
    assert lim == (-2.0, 2.0)
    assert rgb[0, 0, 2] > rgb[0, 0, 0] and rgb[0, 2, 0] > rgb[0, 2, 2]       # blue, then red
    assert np.abs(rgb[0, 1].astype(int) - np.array([247, 247, 247])).max() < 12     # near white at zero


# ----------------------------------------------------------------------------- plumbing
def ring_snapshot():
    return lattice_snapshot(lambda x, y: (-y, x))


def test_render_map_with_post_carries_the_metadata():
    snap = ring_snapshot()
    base = View(center="com", width=5.0, npix=64)
    img0, m0 = render_map(snap, base)
    img1, m1 = render_map(snap, base.with_(post="resid"))
    assert img1.shape == img0.shape and "post" not in m0 and m1["post"]["kind"] == "resid"
    assert m1["view"]["post"] == "resid" and "contrast" in m1["label"] and m1["unit_override"] == ""
    assert abs(float(np.nanmedian(img1))) < 0.05                            # a uniform lattice: the contrast is ~0 in the bulk
    img2, m2 = render_map(snap, base.with_(post="grad:0.3"))
    assert m2["post"]["unit"].startswith("dex /") and np.nanmax(img2) >= 0


def test_tiles_refuse_a_transform():
    from splosh import tiles
    snap = ring_snapshot()
    grid = tiles.tile_grid(View(width=5.0, npix=64), tile=64)
    with pytest.raises(ValueError, match="tiles are raw"):
        tiles.TileRenderer(snap, View(width=5.0, npix=64, post="resid"), grid)


def test_shot_defaults_follow_the_transform_and_replay(tmp_path, snapfile):
    store = MapStore(str(tmp_path / "store"))
    v = View(center="bh", width=6.0, npix=96, post="resid")
    pic = figures.shot(snapfile, v, store=store)
    sc = pic.sidecar
    assert sc["colormap"] == "RdBu_r" and sc["stretch"] == "linear" and sc["limits"]["vmin"] == -sc["limits"]["vmax"]
    assert sc["view"]["post"] == "resid"
    out = str(tmp_path / "r.png")
    pic.save(out)
    cmd, rep = figures.replay(out + ".json" if os.path.exists(out + ".json") else os.path.splitext(out)[0] + ".json", store=store)
    assert cmd == "shot" and os.path.exists(rep)
    pic2 = figures.shot(snapfile, v.with_(post="none"), store=store)             # an ordinary map keeps viridis / log
    assert pic2.sidecar["colormap"] == "viridis" and pic2.sidecar["stretch"] == "log"
    pic3 = figures.shot(snapfile, v.with_(post="none"), scale="asinh:0.02", store=store)
    assert pic3.sidecar["stretch"] == "asinh:0.02"
    pic4 = figures.shot(snapfile, v.with_(post="none"), scale="histeq", store=store)
    assert pic4.sidecar["stretch"] == "histeq" and pic4.rgba.shape[2] == 4


def test_sheet_with_a_transform_uses_one_symmetric_range(tmp_path, snapdir):
    store = MapStore(str(tmp_path / "store"))
    pic = figures.contact_sheet(snapdir, View(center="bh", width=6.0, npix=64, post="resid"), store=store)
    lo, hi = pic.limits
    assert lo == -hi and pic.sidecar["colormap"] == "RdBu_r"
    with pytest.raises(ValueError, match="histeq"):
        figures.contact_sheet(snapdir, View(center="bh", width=6.0, npix=64), scale="histeq", store=store)


def test_cli_render_post_and_scale(tmp_path, snapfile):
    out = str(tmp_path / "o.png")
    store = str(tmp_path / "st")
    assert cli.main(["render", snapfile, "--npix", "64", "--width", "6", "--post", "resid", "--out", out, "--store", store]) in (0, None)
    assert os.path.getsize(out) > 100
    assert cli.main(["render", snapfile, "--npix", "64", "--width", "6", "--scale", "asinh:0.02", "--out", out, "--store", store]) in (0, None)
    with pytest.raises(SystemExit):
        cli.main(["render", snapfile, "--npix", "64", "--width", "6", "--post", "resid", "--tiles", "64", "--out", out, "--store", store])


# ----------------------------------------------------------------------------- contours
from splosh import overlays


def test_marching_squares_circle_and_open_lines():
    n = 120
    x = (np.arange(n) + 0.5) * 0.1 - 6.0
    X, Y = np.meshgrid(x, x)
    Z = np.hypot(X, Y)
    paths = overlays.contour_paths(Z, 3.0, x[0], 0.1, x[0], 0.1)
    assert len(paths) == 1
    P = np.array(paths[0])
    assert np.allclose(P[0], P[-1]) and np.abs(np.hypot(P[:, 0], P[:, 1]) - 3.0).max() < 2e-3        # one closed line on the circle
    paths = overlays.contour_paths(Z, 7.0, x[0], 0.1, x[0], 0.1)                                        # leaves the window: arcs, no closed loop
    assert len(paths) == 4 and all(not np.allclose(p[0], p[-1]) for p in paths)
    ramp = X
    lines = overlays.contour_paths(ramp, 1.05, x[0], 0.1, x[0], 0.1)
    assert len(lines) == 1 and np.allclose([q[0] for q in lines[0]], 1.05) and len(lines[0]) == n
    assert overlays.contour_paths(Z, 100.0, x[0], 0.1, x[0], 0.1) == []


def test_marching_squares_saddle_and_nan():
    Z = np.array([[1.0, 0.0], [0.0, 1.0]])
    def ends(level):
        return sorted(tuple(sorted((round(p[0][0], 3), round(p[0][1], 3), round(p[-1][0], 3), round(p[-1][1], 3)) for p in [q])[0]) for q in overlays.contour_paths(Z, level, 0, 1, 0, 1))
    low = overlays.contour_paths(Z, 0.4, 0, 1, 0, 1)          # centre (0.5) above 0.4: the high corners are joined, each low corner is cut off by its own arc
    high = overlays.contour_paths(Z, 0.6, 0, 1, 0, 1)         # centre below 0.6: each high corner is cut off
    assert len(low) == 2 and len(high) == 2
    pts = lambda paths: {(round(float(x), 3), round(float(y), 3)) for p in paths for x, y in p}
    assert pts(low) == {(0.6, 0.0), (1.0, 0.4), (0.0, 0.6), (0.4, 1.0)}
    assert pts(high) == {(0.4, 0.0), (0.0, 0.4), (1.0, 0.6), (0.6, 1.0)}
    W = np.array([[np.nan, 0.0], [0.0, 1.0]])
    assert overlays.contour_paths(W, 0.5, 0, 1, 0, 1) == []                     # a cell with a NaN corner is skipped


def test_contours_overlay_items_and_validation(snapfile):
    from splosh import load_snapshot
    snap = load_snapshot(snapfile)
    v = View(center="bh", width=6.0, npix=96)
    items = overlays.compute(snap, v, ["contours:rho,levels=lin,n=3,smooth=1"])["items"]
    lines = [i for i in items if i["kind"] == "polyline"]
    assert lines and all(i["head"] is None and len(set(i["magnitude"])) == 1 for i in lines)
    assert items[-1]["kind"] == "legend" and items[-1]["text"].startswith("contours rho")
    levels = sorted({i["magnitude"][0] for i in lines})
    assert len(levels) <= 3
    assert overlays.compute(snap, v, ["contours:rho,levels=log,n=4,step=0.3,legend=false,smooth=0"])["items"][-1]["kind"] == "polyline"
    for bad in ("contours:rho,levels=wat", "contours:rho,n=x", "contours:nofield", "contours:rho,bogus=1"):
        with pytest.raises((ValueError, KeyError)):
            overlays.compute(snap, v, [bad])
    r = overlays.compute(snap, v, ["contours:rho,levels=1e9"])
    assert r["items"] == [] and r["notes"]                                       # no level crosses: a note, not a crash
    assert overlays.spec_string("contours:rho,n=3") .startswith("contours")


def test_contours_of_the_residual_and_of_another_field_in_a_shot(tmp_path, snapfile):
    store = MapStore(str(tmp_path / "store"))
    v = View(center="bh", width=6.0, npix=96, post="resid")
    pic = figures.shot(snapfile, v, overlays=["contours:rho,n=4,smooth=2"], store=store)           # contours of the raw density on a contrast map
    assert pic.sidecar["overlays"][0].startswith("contours")
    pic2 = figures.shot(snapfile, v.with_(post="none"), overlays=["contours:u,levels=lin,n=3"], store=store)
    assert pic2.rgba.shape[2] == 4


# ----------------------------------------------------------------------------- analyze products and layouts
def test_analyze_map_product_with_a_transform(tmp_path):
    from test_analyze import make_run, go
    run = make_run(tmp_path)
    cfg = {"map": [{"name": "contrast", "center": "com", "width": 4.0, "npix": 48, "post": "resid", "limits": "global"},
                   {"name": "slope", "center": "com", "width": 4.0, "npix": 48, "post": "grad:0.2", "limits": "frame"},
                   {"name": "soft", "center": "com", "width": 4.0, "npix": 48, "scale": "asinh:0.02"}],
           "dashboard": [{"name": "dash", "panels": ["contrast", "slope"]}]}
    r = go(run, cfg)
    assert not r.failures
    out = run / "analysis"
    c = json.load(open(out / "contrast" / "snapshot_001.json"))
    assert c["colormap"] == "RdBu_r" and c["stretch"] == "linear" and c["limits"]["vmin"] == -c["limits"]["vmax"] and c["view"]["post"] == "resid"
    g = json.load(open(out / "slope" / "snapshot_001.json"))
    assert g["colormap"] == "magma" and g["stretch"] == "asinh" and g["limits"]["vmin"] > -1
    assert json.load(open(out / "soft" / "snapshot_001.json"))["stretch"] == "asinh:0.02"
    assert (out / "dash" / "snapshot_001.png").exists()
    r2 = go(run, cfg)
    assert stages_total(r2) == 0, r2.counts                                         # incremental: nothing to redo


def stages_total(r):
    return sum(r.counts.values())


def test_analyze_rejects_histeq(tmp_path):
    from test_analyze import make_run, go
    run = make_run(tmp_path, n=1)
    with pytest.raises(ValueError, match="histeq"):
        go(run, {"map": [{"name": "m", "center": "com", "width": 4.0, "npix": 32, "scale": "histeq"}]})


# ----------------------------------------------------------------------------- Fourier modes, profile scatter
def test_profile_modes_and_scatter():
    from splosh.profiles import radial_profile
    rng = np.random.default_rng(5)
    n = 400_000
    R = rng.uniform(1.0, 3.0, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    phi0, a = 0.4, 0.3
    w = (1 + a * np.cos(2 * (phi - phi0))) * 1.0
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), np.zeros(n)])
    q = rng.normal(5.0, 2.0, n)
    prof = radial_profile(pos, w, {"q": q}, rmin=1.0, rmax=3.0, nbins=4, spacing="lin", ndim=2, modes=[1, 2, 3])
    assert np.allclose(prof["A2"], a / 2, atol=0.01) and np.allclose(prof["phi2"], phi0, atol=0.05)
    assert np.all(prof["A1"] < 0.01) and np.all(prof["A3"] < 0.01)
    assert np.allclose(prof["q"], 5.0, atol=0.1) and np.allclose(prof["q_std"], 2.0, atol=0.1)
    flat = radial_profile(pos, np.ones(n), None, rmin=1.0, rmax=3.0, nbins=4, spacing="lin", ndim=2, modes=[2])
    assert np.all(flat["A2"] < 0.01) and "q_std" not in flat


def test_one_armed_offset_is_mode_one():
    from splosh.profiles import radial_profile
    rng = np.random.default_rng(6)
    n = 300_000
    R = rng.uniform(1.0, 2.0, n); phi = rng.uniform(-np.pi, np.pi, n)
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), np.zeros(n)])
    w = 1 + 0.5 * np.cos(phi - 1.0)                                              # one-armed, maximum at phi = 1
    prof = radial_profile(pos, w, None, rmin=1.0, rmax=2.0, nbins=2, spacing="lin", ndim=2, modes=[1, 2])
    assert np.allclose(prof["A1"], 0.25, atol=0.01) and np.allclose(prof["phi1"], 1.0, atol=0.05) and np.all(prof["A2"] < 0.01)


def test_analyze_profile_modes_and_modes_scalar(tmp_path):
    from test_analyze import make_run, go
    run = make_run(tmp_path, n=2)
    cfg = {"profile": [{"name": "m", "center": "com", "quantities": ["u", "A2", "phi2"], "rmin": 0.2, "rmax": 3.0, "bins": 6, "geometry": "cylindrical"}],
           "scalar": [{"name": "modes", "names": ["modes"], "params": {"center": "com", "orders": [1, 2]}}, {"name": "b", "names": ["counts"]}]}
    r = go(run, cfg)
    assert not r.failures
    head = open(run / "analysis" / "m" / "snapshot_000.csv").readline().strip().split(",")
    assert {"A2", "phi2", "u", "u_std"} <= set(head)
    sc = open(run / "analysis" / "scalars.csv").readline().strip().split(",")
    assert {"A1", "A2", "phi1", "modes_rmin", "modes_rmax", "n_gas"} <= set(sc) and "A3" not in sc
    default = go(make_run(tmp_path / "second", n=1))                             # `builtin` does not drag the opt-in scalar in
    assert "A2" not in open(tmp_path / "second" / "run" / "analysis" / "scalars.csv").readline()


# ----------------------------------------------------------------------------- PDFs
def test_value_histogram_lognormal():
    from splosh.profiles import value_histogram
    rng = np.random.default_rng(8)
    x = 10 ** rng.normal(0.0, 0.5, 300_000)
    w = np.ones_like(x)
    h = value_histogram(x, w, {"y": 2.0 * x}, lo=1e-2, hi=1e2, nbins=40, spacing="log")
    dlog = np.diff(np.log10(h["edges"]))
    assert abs((h["pdf"] * dlog).sum() - 1.0) < 1e-3 and abs(h["frac"].sum() - 1.0) < 1e-3
    centre = h["rgeo"][np.argmax(h["pdf"])]
    assert 0.8 < centre < 1.25
    peak = 1.0 / (0.5 * np.sqrt(2 * np.pi))                                     # per dex
    assert abs(h["pdf"].max() - peak) / peak < 0.1
    ok = h["n"] > 100
    assert np.allclose(h["y"][ok] / h["rgeo"][ok], 2.0, rtol=0.1) and np.all(h["y_std"][ok] >= 0)
    lin = value_histogram(x, w, None, lo=0.0, hi=10.0, nbins=20, spacing="lin")
    assert abs((lin["pdf"] * np.diff(lin["edges"])).sum() - lin["frac"].sum()) < 1e-12 and lin["frac"].sum() < 1.0 + 1e-9


def test_analyze_pdf_product(tmp_path):
    from test_analyze import make_run, go
    run = make_run(tmp_path, n=1)
    cfg = {"profile": [{"name": "rhopdf", "center": "com", "geometry": "pdf", "quantities": ["density", "u"], "weighting": "mass", "rmin": 1e-2, "rmax": 1e3, "bins": 12, "spacing": "log"}]}
    r = go(run, cfg)
    assert not r.failures
    rows = open(run / "analysis" / "rhopdf" / "snapshot_000.csv").read().strip().splitlines()
    cols = rows[0].split(",")
    assert {"rbins", "pdf", "frac", "u", "u_std"} <= set(cols) and len(rows) == 13
    for bad in ({"rmin": "auto", "rmax": 10}, {"rmin": 0.0, "rmax": 10.0}, {"rmin": 5.0, "rmax": 1.0}):
        with pytest.raises(ValueError, match="pdf"):
            go(run, {"profile": [{"name": "bad", "geometry": "pdf", "quantities": ["density"], **bad}]})


def test_histeq_ignores_exact_zeros():
    a = np.zeros((100, 100)); a[:50] = np.random.default_rng(2).lognormal(size=(50, 100))
    lo, hi = imaging.limits(a, scale="linear")
    t = imaging.stretch(a, lo, hi, imaging.fit_scale(a, lo, hi, "histeq"))
    nz = t[:50]
    h, _ = np.histogram(nz, bins=10, range=(0, 1))
    assert h.min() > 0.8 * h.mean() and h.max() < 1.2 * h.mean()           # the non-zero half is spread evenly over the colours, the background does not eat a third of them
    assert np.all(t[50:] == 0.0)


# ----------------------------------------------------------------------------- the discovery sheet
def test_explore_sheet_and_replay(tmp_path, snapfile):
    store = MapStore(str(tmp_path / "store"))
    pic = figures.explore(snapfile, View(center="bh", width=6.0, npix=96), store=store)
    sc = pic.sidecar
    assert sc["command"] == "explore" and [p["post"] for p in sc["panels"]][:3] == ["none", "none", "none"] and len(sc["panels"]) == 6
    assert sc["panels"][3]["cmap"] == "RdBu_r" and sc["panels"][3]["limits"][0] == -sc["panels"][3]["limits"][1] and sc["panels"][5]["post"].startswith("grad:")
    assert sc["panels"][2]["scale"] == "histeq" and sc["panels"][1]["scale"] == "asinh:0.02"
    assert pic.rgba.shape[1] > 3 * 96 and pic.rgba.shape[2] == 4
    out = str(tmp_path / "e.png")
    pic.save(out)
    cmd, rep = figures.replay(os.path.splitext(out)[0] + ".json", store=store)
    assert cmd == "explore" and os.path.exists(rep)
    a = np.asarray(figures.explore(snapfile, View(center="bh", width=6.0, npix=96), store=store).rgba)


def test_explore_signed_field_and_custom_panels(tmp_path, snapfile):
    store = MapStore(str(tmp_path / "store"))
    pic = figures.explore(snapfile, View(center="bh", width=6.0, npix=64, field="vx"), store=store, ncols=2)
    assert pic.sidecar["panels"][0]["scale"] == "linear"                              # a signed field has no log panel
    pic2 = figures.explore(snapfile, View(center="bh", width=6.0, npix=64), panels=[{"title": "x", "post": "resid:smooth:{w25}"}, {"post": "grad", "cmap": "inferno"}], store=store)
    assert [p["post"] for p in pic2.sidecar["panels"]] == [f"resid:smooth:{6.0 / 25:.3g}", "grad"] and pic2.sidecar["panels"][1]["cmap"] == "inferno"
    with pytest.raises(ValueError):
        figures.explore(snapfile, View(), panels=[], store=store)


def test_cli_explore(tmp_path, snapfile):
    out = str(tmp_path / "x.png")
    assert cli.main(["explore", snapfile, "--npix", "64", "--width", "6", "--out", out, "--store", str(tmp_path / "st")]) in (0, None)
    assert os.path.exists(out) and os.path.exists(os.path.splitext(out)[0] + ".json")
    assert cli.main(["explore", "--from-sidecar", os.path.splitext(out)[0] + ".json", "--store", str(tmp_path / "st")]) in (0, None)
    assert os.path.exists(str(tmp_path / "x_repro.png"))
    with pytest.raises(SystemExit):
        cli.main(["explore", snapfile])




# ----------------------------------------------------------------------------- server
@pytest.fixture
def served(snapdir, tmp_path):
    import threading
    from splosh.server import make_server
    srv, state = make_server([snapdir], token="t", store=MapStore(str(tmp_path / "store")), defaults=View(center="bh", align="bh", width=6.0, npix=64))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def test_server_post_param_png_style_and_tiles(served):
    import urllib.error, urllib.request
    def get(path):
        return urllib.request.urlopen(urllib.request.Request(served + path, headers={"X-Splosh-Token": "t"}), timeout=60)
    r = get("/api/map?snap=0&post=resid&token=t")
    meta = json.loads(r.headers["X-Splosh-Meta"])
    assert meta["post"]["kind"] == "resid" and meta["view"]["post"] == "resid" and meta["unit_override"] == ""
    raw = np.frombuffer(r.read(), "<f4")
    plain = np.frombuffer(get("/api/map?snap=0&token=t").read(), "<f4")
    assert raw.shape == plain.shape and not np.allclose(np.nan_to_num(raw), np.nan_to_num(plain))
    png = get("/api/png?snap=0&post=resid&token=t").read()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(urllib.error.HTTPError) as e:
        get("/api/map?snap=0&post=wat&token=t")
    assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        get("/api/tile?snap=0&level=0&ix=0&iy=0&post=resid&token=t")
    assert e.value.code == 400


def test_mpl_backend_with_a_transform_and_stretches(tmp_path, snapfile):
    pytest.importorskip("matplotlib")
    store = MapStore(str(tmp_path / "store"))
    for kw in ({"post": "resid"}, {"post": "none", "scale": "asinh:0.02"}, {"post": "none", "scale": "histeq"}):
        v = View(center="bh", width=6.0, npix=64, post=kw["post"])
        pic = figures.shot(snapfile, v, scale=kw.get("scale"), backend="mpl", store=store)
        out = str(tmp_path / "m.png")
        pic.save(out)
        assert os.path.getsize(out) > 2000
        assert pic.sidecar["backend"] == "mpl"
