import numpy as np
import pytest

from splosh import View, render_map
from splosh.profiles import radial_profile
from splosh import tables
from test_maps import synthetic_snap


def write(path, rows, comment=None):
    with open(path, "w") as f:
        if comment:
            f.write(comment + "\n")
        for r in rows:
            f.write(" ".join(str(x) for x in r) + "\n")


def test_read_rows_fast_and_mixed_layouts(tmp_path):
    a = tmp_path / "a.dat"
    write(a, [[1.0, 10, 5, 6], [2.0, 10, 7, 8], [3.0, 10, 9, "-nan"]])
    r = tables.read_rows(str(a), 4)
    assert r.shape == (3, 4) and np.isnan(r[2, 3])
    b = tmp_path / "b.dat"                                          # an old narrower layout and a truncated last line
    with open(b, "w") as f:
        f.write("1.0 10 5\n2.0 10 7 8\n3.0 10 9 1\n4.0 1")
    r = tables.read_rows(str(b), 4, widths={3, 4})
    assert r.shape == (3, 4) and np.isnan(r[0, 3]) and r[1, 3] == 8                       # width-3 line padded with NaN, the truncated line dropped
    assert tables.n_columns(str(b), 3) == 3 and tables.last_line_columns(str(b), 3) == 4


def test_restart_overlap_dropped_and_sorted(tmp_path):
    # task files for two particles; a restart resumed at t = 3 and rewrote 3 < t <= 5 (the earlier copies are stale)
    f0 = [[1, 1, 0.1], [2, 1, 0.2], [3, 1, 0.3], [4, 1, 0.4], [5, 1, 0.5], [3.5, 1, 9.9], [4.5, 1, 9.9], [6, 1, 0.6]]       # time, id, value
    f1 = [[1, 2, 1.1], [2, 2, 1.2], [3, 2, 1.3], [4, 2, 1.4], [5, 2, 1.5], [6, 2, 1.6]]
    write(tmp_path / "t0.dat", f0)
    write(tmp_path / "t1.dat", f1)
    arr, info = tables.load_table([str(tmp_path / "t0.dat"), str(tmp_path / "t1.dat")], id_col=1, time_col=0, min_cols=3, return_info=True)
    assert info["n_stale"] >= 2
    one = arr[arr[:, 1] == 1]
    assert np.all(np.diff(one[:, 0]) > 0)                           # time strictly increasing: the stale rows (4, 5 before the restart) are gone
    assert set(one[:, 0]) == {1, 2, 3, 3.5, 4.5, 6} and 4.0 not in one[:, 0] and 5.0 not in one[:, 0]
    assert np.all(np.diff(arr[:, 0]) >= 0)                          # sorted by time overall
    assert tables.load_table([], 1, 0, 3) is None


def test_schema():
    s = tables.Schema({"t": 0, "id": 1, "m": 2, "extra": 5}, widths=[3, 6])
    assert s["m"] == 2 and s.has("m", 3) and not s.has("extra", 3) and s.has("extra", 6) and s.widths == (3, 6)


def test_radial_profile_uniform_disc_and_weights():
    rng = np.random.default_rng(0)
    n = 200000
    R = 10.0 * np.sqrt(rng.random(n)); phi = 2 * np.pi * rng.random(n)
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi), rng.normal(size=n) * 0.1])
    m = np.full(n, 0.01)
    q = 3.0 + R                                                       # a known function of R
    p = radial_profile(pos, m, {"q": q}, rmin=1.0, rmax=8.0, nbins=7, spacing="lin")
    assert np.allclose(p["sigma"], n * 0.01 / (np.pi * 100.0), rtol=0.06)                      # uniform surface density (Poisson noise ~1-2% per bin)
    assert np.allclose(p["q"], 3.0 + p["rbins"], atol=0.2)                     # bin mean of 3 + R, with the density weighting inside each bin
    assert np.all(p["n"] > 0) and p["rbins"][0] == pytest.approx(1.5)
    pl = radial_profile(pos, m, {"q": q}, rmin=1.0, rmax=8.0, nbins=7, spacing="log")
    assert pl["rbins"][0] == pytest.approx(0.5 * (pl["edges"][0] + pl["edges"][1])) and pl["rgeo"][0] == pytest.approx(np.sqrt(pl["edges"][0] * pl["edges"][1]))
    w2 = radial_profile(pos, m * (1 + (R > 5)), {"q": q}, rmin=1.0, rmax=8.0, nbins=7, spacing="lin")        # weights change the mean, not the counts
    assert np.array_equal(w2["n"], p["n"]) and w2["q"][-1] > p["q"][-1] - 1e-9
    empty = radial_profile(pos, m, {"q": q}, rmin=50.0, rmax=60.0, nbins=3)
    assert np.all(np.isnan(empty["q"])) and np.all(empty["n"] == 0)


def test_hscale_equals_doubling_h():
    s = synthetic_snap(hmin=0.4)
    v = View(center="com", width=8.0, npix=48, exact=True)
    legacy, meta = render_map(s, v.with_(hscale=2.0))
    s2 = dict(s, SmoothingLength=(2.0 * s["SmoothingLength"]).astype(np.float32))
    ref, _ = render_map(s2, v)
    assert np.allclose(legacy, ref, rtol=1e-5, atol=1e-6 * ref.max()) and "legacy" in meta["kernel"]
    plain, m1 = render_map(s, v)
    assert not np.allclose(plain, legacy) and "legacy" not in m1["kernel"]
    assert legacy.sum() == pytest.approx(plain.sum(), rel=0.02)                                        # the mass is conserved (inside the window)
