"""Several runs in the server: grouping of the arguments, snapshot times, time alignment (nearest snapshot, tolerance, offsets, ties) and the `run=` parameter of the endpoints."""
import json
import math
import os
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from splosh import View
from splosh import runs as R
from splosh.mapstore import MapStore
from splosh.server import make_server

from conftest import write_snapshot

UNIT_MYR = 3.0857e18 / 1e5 / 3.156e13           # code time (length / velocity of the test header) in Myr


def make_run(d, times, seed=1, **kw):
    d.mkdir(parents=True, exist_ok=True)
    for k, t in enumerate(times):
        write_snapshot(str(d / f"snap_{k:03d}.hdf5"), n_gas=300, seed=seed + k, time=t, **kw)
    return str(d)


@pytest.fixture
def two_runs(tmp_path):
    a = make_run(tmp_path / "fast", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], seed=1)           # spacing 1
    b = make_run(tmp_path / "slow", [0.0, 2.5, 5.0], seed=20)                          # spacing 2.5
    return a, b


@pytest.fixture
def server(two_runs, tmp_path):
    srv, state = make_server(list(two_runs), token="tok", store=MapStore(str(tmp_path / "store")), defaults=View(center="bh", width=6.0, npix=32))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown(); srv.server_close()


def getj(base, path, **q):
    from urllib.parse import urlencode
    return json.loads(urllib.request.urlopen(f"{base}{path}?token=tok&" + urlencode(q, doseq=True), timeout=120).read())


# ---------------------------------------------------------------- grouping
def test_each_directory_is_a_run_and_label_path_names_it(two_runs):
    g = R.group_runs([two_runs[0], "other=" + two_runs[1]])
    assert [(l, len(p)) for l, p in g] == [("fast", 6), ("other", 3)]
    assert [os.path.basename(p) for p in g[0][1]] == [f"snap_{k:03d}.hdf5" for k in range(6)]


def test_loose_files_form_one_run_and_one_directory_is_one_run(two_runs):
    fast = sorted(os.path.join(two_runs[0], f) for f in os.listdir(two_runs[0]))
    assert [(l, len(p)) for l, p in R.group_runs(fast[:3])] == [("fast", 3)]
    assert [l for l, _ in R.group_runs([two_runs[0]])] == ["fast"]


def test_duplicate_labels_are_suffixed_and_natural_order(tmp_path):
    for k in (2, 10, 1):
        (tmp_path / "a").mkdir(exist_ok=True); (tmp_path / "a" / f"snap_{k}.hdf5").write_bytes(b"")
    (tmp_path / "b" / "a").mkdir(parents=True); (tmp_path / "b" / "a" / "snap_0.hdf5").write_bytes(b"")
    g = R.group_runs([str(tmp_path / "a"), str(tmp_path / "b" / "a")])
    assert [l for l, _ in g] == ["a", "a-2"]
    assert [os.path.basename(p) for p in g[0][1]] == ["snap_1.hdf5", "snap_2.hdf5", "snap_10.hdf5"]


def test_run_directory_with_an_output_subdirectory(tmp_path):
    make_run(tmp_path / "r1" / "output", [0.0, 1.0])
    assert [(l, len(p)) for l, p in R.group_runs([str(tmp_path / "r1")])] == [("r1", 2)]


# ---------------------------------------------------------------- times
def test_times_and_physical_time(tmp_path):
    run = R.Run("x", R.group_runs([make_run(tmp_path / "x", [0.0, 2.0])])[0][1])
    assert run.times == [0.0, 2.0] and run.has_myr
    assert run.times_myr[1] == pytest.approx(2.0 * UNIT_MYR, rel=1e-9)


def test_no_myr_for_comoving_or_unit_less_headers(tmp_path):
    for name, kw in (("co", {"comoving": 1}), ("nu", {"units": False})):
        run = R.Run(name, R.group_runs([make_run(tmp_path / name, [0.5, 1.0], **kw)])[0][1])
        assert run.times == [0.5, 1.0] and run.times_myr == [None, None] and not run.has_myr
        assert np.isnan(run.coords("time_myr")).all()


# ---------------------------------------------------------------- nearest and align
def test_nearest_picks_the_closest_within_tolerance():
    c = np.array([0.0, 1.0, 2.0, 3.0])
    assert R.nearest(c, 1.2) == (1, "")
    assert R.nearest(c, 2.6)[0] == 3
    assert R.nearest(c, 3.4)[0] == 3                                  # 0.4 <= half the spacing
    j, why = R.nearest(c, 4.0)                                        # 1.0 away > 0.5
    assert j is None and "no snapshot within 0.5 of 4" in why
    assert R.nearest(c, 4.0, tol=1.0)[0] == 3                         # explicit tolerance
    assert R.nearest(c, 1.5)[0] == 1                                  # exact tie: the first (no preference)
    assert R.nearest(c, 1.5, prefer=2)[0] == 2                        # ... or the one nearest to the preferred index


def test_equal_times_go_to_the_preferred_index_and_tolerance_zero_works():
    c = np.array([0.25, 0.25, 0.25])
    assert R.nearest(c, 0.25, prefer=2)[0] == 2 and R.nearest(c, 0.25, prefer=0)[0] == 0 and R.nearest(c, 0.25)[0] == 0
    assert R.spacing(c) == 0.0 and R.nearest(c, 0.25)[0] is not None   # auto tolerance 0 still accepts the exact time


def test_nan_coordinates_and_single_snapshots():
    assert R.nearest(np.array([np.nan, np.nan]), 1.0)[0] is None
    j, _ = R.nearest(np.array([5.0]), 100.0)                           # one snapshot: spacing inf, always accepted
    assert j == 0


def test_align_modes_offsets_and_tolerance(tmp_path, two_runs):
    runs = [R.Run(l, p) for l, p in R.group_runs(list(two_runs))]
    a = R.align(runs, "time", 2.5)
    assert [x["snap"] for x in a] == [2, 1] and a[1]["coord"] == 2.5 and a[1]["distance"] == 0.0 and a[0]["tol"] == 0.5 and a[1]["tol"] == 1.25
    a = R.align(runs, "time", 3.9)                                     # fast: 4.0 (d 0.1); slow: 5.0 is 1.1 away, tol 1.25: ok; 2.5 is 1.4
    assert [x["snap"] for x in a] == [4, 2]
    a = R.align(runs, "time", 10.0)
    assert a[0]["snap"] is None and a[1]["snap"] is None and "no snapshot within" in a[0]["reason"]
    a = R.align(runs, "time", 3.0, offsets=[0.0, -10.0])               # slow shifted: aligned = t + 10 (10, 12.5, 15)
    assert a[0]["snap"] == 3 and a[1]["snap"] is None
    a = R.align(runs, "time", 13.0, offsets=[0.0, -10.0])
    assert a[0]["snap"] is None and a[1]["snap"] == 1
    a = R.align(runs, "index", 1.0)
    assert [x["snap"] for x in a] == [1, 1]
    a = R.align(runs, "index", 2.0, offsets=[0, 1])                    # slow index minus 1: -1, 0, 1 -> index 2 aligned 1... nearest to 2.0 is 1 (d 1) > tol 0.5
    assert a[0]["snap"] == 2 and a[1]["snap"] is None
    a = R.align(runs, "time_myr", 2.5 * UNIT_MYR)
    assert [x["snap"] for x in a] == [2, 1]
    a = R.align(runs, "time", 2.5, tol=0.1)
    assert [x["snap"] for x in a] == [None, 1]
    a = R.align(runs, "time", 1.0, prefer=[None, 0])
    assert a[1]["snap"] == 0                                           # 1.0 is nearer 0.0 (1.0) than 2.5 (1.5)
    with pytest.raises(ValueError):
        R.align(runs, "offset", 1.0)                                   # offsets are required


# ---------------------------------------------------------------- server
def test_runs_endpoint_lists_labels_snapshots_and_times(server):
    base, state = server
    r = getj(base, "/api/runs")
    assert [x["label"] for x in r["runs"]] == ["fast", "slow"] and r["aligns"] == ["time", "time_myr", "index", "offset", "orbits"]
    fast, slow = r["runs"]
    assert fast["n"] == 6 and [s["time"] for s in fast["snapshots"]] == [0, 1, 2, 3, 4, 5] and [s["time"] for s in slow["snapshots"]] == [0, 2.5, 5]
    assert slow["snapshots"][1]["time_myr"] == pytest.approx(2.5 * UNIT_MYR) and fast["time_myr"] and slow["snapshots"][2]["name"] == "snap_002.hdf5"
    st = getj(base, "/api/state")
    assert [s["time"] for s in st["snapshots"]] == [0, 1, 2, 3, 4, 5] and [x["label"] for x in st["runs"]] == ["fast", "slow"]      # the first run, as before


def test_run_parameter_selects_the_run_and_the_default_is_the_first(server):
    base, state = server
    from urllib.parse import urlencode
    def meta(**q):
        r = urllib.request.urlopen(f"{base}/api/map?token=tok&" + urlencode(dict(snap=1, field="rho", npix=32, width=5, **q)), timeout=120)
        return json.loads(r.headers["X-Splosh-Meta"]), np.frombuffer(r.read(), "<f4")
    m0, d0 = meta()
    mf, df = meta(run="fast")
    ms, ds = meta(run="slow")
    mn, dn = meta(run=1)
    assert np.array_equal(d0, df) and m0["time"] == mf["time"] == 1.0                     # unchanged URLs: the first run
    assert ms["time"] == 2.5 and not np.array_equal(df, ds) and np.array_equal(ds, dn)       # snap 1 of the slow run is its own file
    assert ms["snapshot"] == "snap_001.hdf5"
    for path in ("/api/bh", "/overlay"):
        extra = {"spec": "particles:bh"} if path == "/overlay" else {}
        a = getj(base, path, snap=2, width=5, npix=32, run="slow", **extra)
        assert a["time"] == 5.0
    assert getj(base, "/inspect", snap=2, x=0.1, y=0.1, width=5, npix=32, run="slow")["n_total"] >= 0


def test_unknown_run_and_snapshot_out_of_a_shorter_run(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        getj(base, "/api/map", snap=0, run="nope")
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        getj(base, "/api/map", snap=4, run="slow")                                           # the slow run has 3 snapshots
    assert e.value.code == 404


def test_tile_endpoint_takes_a_run(server):
    base, _ = server
    r = urllib.request.urlopen(f"{base}/api/tile?token=tok&snap=2&level=1&ix=0&iy=0&base=8&run=slow", timeout=120)
    meta = json.loads(r.headers["X-Splosh-Meta"])
    assert meta["time"] == 5.0 and meta["snapshot"] == "snap_002.hdf5"


def test_align_endpoint_matches_the_library(server):
    base, state = server
    r = getj(base, "/api/align", mode="time", at=3.9)
    assert [x["snap"] for x in r["runs"]] == [4, 2] and r["runs"][0]["label"] == "fast"
    r = getj(base, "/api/align", mode="time", at=3.0, offset="0,-10")
    assert [x["snap"] for x in r["runs"]] == [3, None] and "no snapshot within" in r["runs"][1]["reason"]
    r = getj(base, "/api/align", mode="time", at=2.5, tol=0.1)
    assert [x["snap"] for x in r["runs"]] == [None, 1]
    r = getj(base, "/api/align", mode="time", at=1.0, prefer=",0")
    assert r["runs"][1]["snap"] == 0
    with pytest.raises(urllib.error.HTTPError) as e:
        getj(base, "/api/align", mode="offset", at=1.0)
    assert e.value.code == 400


def test_binary_period_from_a_circular_orbit_and_the_orbits_alignment(tmp_path):
    import math
    import h5py
    from conftest import write_snapshot
    from splosh import figures
    from splosh.runs import Run, binary_period, G_CGS, _PERIODS
    from splosh.units import Units
    d = tmp_path / "run"
    d.mkdir()
    for k in range(3):
        write_snapshot(str(d / f"snap_{k:03d}.hdf5"), n_gas=200, seed=k, bh_pos=((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), time=0.25 * k)
    f0 = str(d / "snap_000.hdf5")
    with h5py.File(f0, "r+") as f:
        h = {k: np.asarray(v).tolist() for k, v in f["Header"].attrs.items()}
        u = Units.from_header(h)
        G = G_CGS * u.mass_g * u.time_s ** 2 / u.length_cm ** 3
        M = float(np.asarray(f["PartType5/Masses"]).sum())
        r = 2.0
        vrel = math.sqrt(G * M / r)                                      # circular: relative speed sqrt(G M / r)
        m1, m2 = np.asarray(f["PartType5/Masses"])[:2]
        v = np.zeros((2, 3)); v[0, 1] = vrel * m2 / M; v[1, 1] = -vrel * m1 / M
        f["PartType5/Velocities"][...] = v
    _PERIODS.clear()
    P = binary_period(f0)
    assert P == pytest.approx(2 * math.pi * math.sqrt(r ** 3 / (G * M)), rel=1e-6)
    run = Run("run", sorted(str(p) for p in d.iterdir()))
    assert run.period == P and run.describe(2)["orbits"] == pytest.approx(0.5 / P)
    assert np.allclose(run.coords("orbits"), np.array([0.0, 0.25, 0.5]) / P)
    assert figures.orbit_period(str(d / "snap_002.hdf5")) == P                 # labels use the folder's first snapshot
    assert f"{0.5 / P:.3g} P0" in figures.time_text(0.5, {"tcgs": None}, period=P)
    with h5py.File(str(d / "snap_001.hdf5"), "r+") as f:                      # far apart and fast: unbound, no period
        f["PartType5/Velocities"][...] = np.array([[0, 100.0, 0], [0, -100.0, 0]])
    assert binary_period(str(d / "snap_001.hdf5")) is None
