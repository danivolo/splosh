"""Overlays: gridded vectors and streamlines (rigid rotation, uniform flow, B), particles by type, click-to-inspect, the numpy rasteriser, plugin registry, shot / sheet / movie / CLI / server."""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

import splosh
from splosh import cli, figures, overlays, plugins
from splosh.io import load_snapshot
from splosh.mapstore import MapStore
from splosh.maps import View, render_map
from splosh.frames import make_frame

from conftest import lattice_snapshot, write_snapshot

OMEGA = 2.0
VIEW = View(center="xyz:0,0,0", width=4.0, npix=64)


def rigid(x, y):
    return -OMEGA * y, OMEGA * x


@pytest.fixture(scope="module")
def rot_snap():
    return lattice_snapshot(rigid)


@pytest.fixture
def store(tmp_path):
    return MapStore(str(tmp_path / "store"))


def arrows(items):
    return [i for i in items if i["kind"] == "arrow"]


# ----------------------------------------------------------------------------- specs

def test_spec_parsing_and_errors():
    assert overlays.parse_spec("vectors:B,grid=32,color=#00ff00") == {"name": "vectors", "args": ["B"], "opts": {"grid": "32", "color": "#00ff00"}}
    assert overlays.parse_spec("particles:bh,star,star.color=gold")["args"] == ["bh", "star"]
    assert overlays.spec_string({"name": "vectors", "args": ["v"], "opts": {"grid": 12}}) == "vectors:v,grid=12"
    assert overlays.color_rgb("#ff8000") == [255, 128, 0] and overlays.color_rgb("white") == [255, 255, 255]
    with pytest.raises(ValueError, match="unknown colour"):
        overlays.color_rgb("chartreuse-ish")
    with pytest.raises(ValueError, match="unknown option"):
        overlays.vectors({}, None, VIEW, "v", gridd=3)
    with pytest.raises(ValueError, match="unknown overlay"):
        overlays.compute(lattice_snapshot(rigid, n=8), VIEW, ["nope"])
    with pytest.raises(ValueError, match="unknown particle type"):
        overlays.required_data(["particles:wombat"], VIEW)


# ----------------------------------------------------------------------------- vectors and streamlines

def test_grid_equals_weighted_projection(rot_snap):
    frame = make_frame(rot_snap, VIEW.center)
    U, W, ext, names = overlays.vector_grid(rot_snap, VIEW, "v", 16, frame)
    assert names == ["vx", "vy"] and ext == [-2.0, 2.0, -2.0, 2.0]
    ref, _ = render_map(rot_snap, VIEW.with_(field="vx", npix=16), frame)
    assert np.array_equal(U, ref.astype(np.float64))
    # the mass-weighted projection of a linear field is the field itself at the cell centre
    c = (np.arange(16) + 0.5) * 0.25 - 2.0
    # (the outermost rows and columns see a truncated kernel at the window edge and are left out)
    assert np.allclose(U[1:-1, 1:-1], (-OMEGA * c[:, None] + 0 * c[None, :])[1:-1, 1:-1], atol=0.02) and np.allclose(W[1:-1, 1:-1], (OMEGA * c[None, :] + 0 * c[:, None])[1:-1, 1:-1], atol=0.02)
    # axes xz: the vertical component is vz (zero here)
    U2, W2, _, n2 = overlays.vector_grid(rot_snap, VIEW.with_(axes="xz"), "v", 16, frame)
    assert n2 == ["vx", "vz"] and np.nanmax(np.abs(W2)) < 1e-6


def test_rigid_rotation_arrows_are_tangential_with_the_right_magnitude(rot_snap):
    items = overlays.compute(rot_snap, VIEW, ["vectors:v,grid=16"])["items"]
    ar = arrows(items)
    assert len(ar) > 200
    n = 0
    for a in ar:
        cx, cy = a["x"] + 0.5 * a["dx"], a["y"] + 0.5 * a["dy"]          # arrows are centred on the cell centre
        r = np.hypot(cx, cy)
        if r < 0.4 or r > 1.8:
            continue
        n += 1
        L = np.hypot(a["dx"], a["dy"])
        assert abs((a["dx"] * cx + a["dy"] * cy) / (L * r)) < 0.05                      # tangential
        assert (cx * a["dy"] - cy * a["dx"]) > 0                                         # counter-clockwise for Omega > 0
        assert a["magnitude"] == pytest.approx(OMEGA * r, rel=0.04)
    assert n > 80
    # lengths proportional to |v| with the stated reference: ref (p99 of the magnitudes) is drawn as `len` cells
    leg = [i for i in items if i["kind"] == "legend"][0]
    cell = 4.0 / 16
    assert leg["length"] == pytest.approx(0.9 * cell) and leg["value"] > 0
    for a in ar:
        assert np.hypot(a["dx"], a["dy"]) == pytest.approx(0.9 * cell * a["magnitude"] / leg["value"], rel=1e-3)
    assert "km/s" in leg["text"]


def test_vector_options_unit_ref_min_max(rot_snap):
    unit = arrows(overlays.compute(rot_snap, VIEW, ["vectors:v,grid=16,arrows=unit,len=0.5"])["items"])
    assert {round(np.hypot(a["dx"], a["dy"]), 6) for a in unit} == {round(0.5 * 0.25, 6)}
    ref = arrows(overlays.compute(rot_snap, VIEW, ["vectors:v,grid=16,ref=4,len=1,legend=false"])["items"])
    for a in ref:
        assert np.hypot(a["dx"], a["dy"]) == pytest.approx(0.25 * a["magnitude"] / 4.0, rel=1e-3)
    lo = arrows(overlays.compute(rot_snap, VIEW, ["vectors:v,grid=16,min=1.5"])["items"])
    assert min(a["magnitude"] for a in lo) >= 1.5 and len(lo) < len(ref)
    hi = arrows(overlays.compute(rot_snap, VIEW, ["vectors:v,grid=16,max=2,ref=2,len=1"])["items"])
    assert max(np.hypot(a["dx"], a["dy"]) for a in hi) == pytest.approx(0.25, rel=1e-3) and max(a["magnitude"] for a in hi) > 2.5      # lengths saturate, magnitudes stay


def test_uniform_flow_gives_parallel_arrows_and_straight_streamlines():
    snap = lattice_snapshot(lambda x, y: (1.5 + 0 * x, 0.0 * x))
    res = overlays.compute(snap, VIEW, ["vectors:v,grid=12", "streamlines:v"])
    ar = arrows(res["items"])
    assert len(ar) == 144
    assert np.allclose([a["dy"] for a in ar], 0, atol=1e-3 * 0.33) and np.allclose([a["magnitude"] for a in ar], 1.5, rtol=1e-3)
    assert len({round(a["dx"], 5) for a in ar}) == 1 and ar[0]["dx"] > 0
    lines = [i for i in res["items"] if i["kind"] == "polyline"]
    assert len(lines) >= 8
    for ln in lines:
        p = np.array(ln["points"])
        assert np.ptp(p[:, 1]) < 1e-3 and np.all(np.diff(p[:, 0]) > 0)                  # horizontal, in flow order
    # window edge: nothing leaves the window, and the lines reach it
    allp = np.concatenate([np.array(l["points"]) for l in lines])
    assert allp[:, 0].min() >= -2.0 and allp[:, 0].max() <= 2.0 and allp[:, 0].min() < -1.7 and allp[:, 0].max() > 1.7


def test_rigid_rotation_streamlines_are_circles(rot_snap):
    items = overlays.compute(rot_snap, VIEW, ["streamlines:v"])["items"]
    lines = [i for i in items if i["kind"] == "polyline"]
    assert len(lines) >= 4
    checked = 0
    for ln in lines:
        p = np.array(ln["points"])
        r = np.hypot(p[:, 0], p[:, 1])
        if len(p) < 12 or r.mean() < 0.4:
            continue
        checked += 1
        assert np.ptp(r) < 0.06 * r.mean()                                              # circular
        cross = p[:-1, 0] * np.diff(p[:, 1]) - p[:-1, 1] * np.diff(p[:, 0])
        assert np.all(cross > 0)                                                        # points in flow order, counter-clockwise
        assert np.allclose(ln["magnitude"], OMEGA * r, rtol=0.06)
        assert 0 < ln["head"] < len(p)
    assert checked >= 3


def test_streamline_limits_density_length_and_stagnation(rot_snap):
    U, W, ext, _ = overlays.vector_grid(rot_snap, VIEW, "v", 32)
    sparse = overlays.streamline_paths(U, W, ext, seeds=14, density=6)
    dense = overlays.streamline_paths(U, W, ext, seeds=14, density=40)
    assert 0 < len(sparse) < len(dense)
    short = overlays.streamline_paths(U, W, ext, seeds=14, density=28, maxlen=0.1)
    step = 0.25 * 4.0 / 32
    for pts, _ in short:
        assert len(pts) * step <= 2 * 0.1 * 4.0 + 3 * step                              # both directions, each at most maxlen
    # stagnation: nothing is traced where |v| is below the threshold (the centre of the rotation)
    for pts, m in overlays.streamline_paths(U, W, ext, stag=0.3):
        assert min(m) > 0.1 * max(OMEGA * 2.0, 1e-9)
    # a single seed at rest: no line
    z = np.zeros((16, 16))
    assert overlays.streamline_paths(z, z, [-1, 1, -1, 1]) == []
    # an undefined (NaN) field stops the line
    Un = U.copy()
    Un[:, : U.shape[1] // 2] = np.nan
    for pts, _ in overlays.streamline_paths(Un, W, ext):
        assert min(p[0] for p in pts) > -0.1


def test_magnetic_overlay_and_error_without_field():
    snap = lattice_snapshot(lambda x, y: (0 * x, 0 * x), B=(1.0, 0.5, 0.2))
    res = overlays.compute(snap, VIEW, ["vectors:B,grid=8"])
    ar = arrows(res["items"])
    assert np.allclose([a["magnitude"] for a in ar], np.hypot(1.0, 0.5), rtol=1e-3)
    assert np.allclose([a["dy"] / a["dx"] for a in ar], 0.5, rtol=1e-3)
    assert "G" in [i for i in res["items"] if i["kind"] == "legend"][0]["text"]
    # B in the aligned frame is rotated with the frame (vec:0,1,0 -> the old +y axis becomes +x): (Bx, By) -> (By, -Bx)
    snap2 = dict(snap, BH=0)
    r2 = overlays.compute(snap2, VIEW.with_(align="vec:0,1,0"), ["vectors:B,grid=8"])
    a = arrows(r2["items"])[0]
    assert a["dx"] > 0 > a["dy"] and a["dy"] / a["dx"] == pytest.approx(-1.0 / 0.5, rel=1e-3)
    # the xz projection shows (Bx, Bz)
    r3 = arrows(overlays.compute(snap, VIEW.with_(axes="xz"), ["vectors:B,grid=8"])["items"])
    assert r3[0]["dy"] / r3[0]["dx"] == pytest.approx(0.2, rel=1e-2)
    with pytest.raises(ValueError, match="MagneticField"):
        overlays.compute(lattice_snapshot(rigid, n=8), VIEW, ["vectors:B"])
    with pytest.raises(ValueError, match="velocity"):
        overlays.vector_grid(lattice_snapshot(rigid, n=8), VIEW, "x")
    # B is also a map field now
    img, _ = render_map(snap, VIEW.with_(field="Bx", npix=16), make_frame(snap, VIEW.center))
    assert np.nanmax(np.abs(img - 1.0)) < 1e-3


def test_slice_mode_uses_volume_weights(rot_snap):
    res = overlays.compute(rot_snap, VIEW.with_(mode="slice", z0=0.0), ["vectors:v,grid=16"])
    ar = arrows(res["items"])
    assert len(ar) > 150
    mid = [a for a in ar if 0.5 < np.hypot(a["x"] + 0.5 * a["dx"], a["y"] + 0.5 * a["dy"]) < 1.5]
    assert all(a["magnitude"] == pytest.approx(OMEGA * np.hypot(a["x"] + 0.5 * a["dx"], a["y"] + 0.5 * a["dy"]), rel=0.05) for a in mid)


# ----------------------------------------------------------------------------- particles

def test_loader_types_are_optional_and_readonly(tmp_path):
    p = str(tmp_path / "s.hdf5")
    write_snapshot(p, n_gas=50, n_dm=300, n_star=120)
    base = load_snapshot(p)
    assert "pt" not in base                                                               # existing loads unchanged
    s = load_snapshot(p, types=(1, 4, 2, 5))
    assert set(s["pt"]) == {1, 4} and s["pt"][1]["pos"].shape == (300, 3) and s["pt"][4]["id"][0] == 200000 and "bh_pos" in s
    assert s["pt"][4]["mass"].shape == (120,)


def test_particle_overlay_by_type_cap_and_determinism(tmp_path):
    p = str(tmp_path / "s.hdf5")
    write_snapshot(p, n_gas=200, n_dm=900, n_star=400)
    v = View(center="bh", width=6.0, npix=64)
    snap = overlays.load_for(p, v, ["particles:bh,star,dm"])
    res = overlays.compute(snap, v, ["particles:bh,star,dm,max=100"])
    by = {i["type"]: i for i in res["items"]}
    assert set(by) == {"bh", "star", "dm"}
    for t in ("star", "dm"):
        it = by[t]
        assert it["n_drawn"] == 100 <= it["n_total"] and len(it["x"]) == len(it["y"]) == len(it["id"]) == len(it["depth"]) == 100
        assert np.all(np.abs(np.array(it["x"])) <= 3.0 + 1e-9)
    assert by["bh"]["style"] == "ring" and sorted(by["bh"]["id"]) == [3, 7] and by["bh"]["n_drawn"] == 2
    again = overlays.compute(snap, v, ["particles:star,max=100"])["items"][0]
    assert again["id"] == by["star"]["id"]                                                  # deterministic subsample
    other = overlays.compute(snap, v, ["particles:star,max=100,seed=5"])["items"][0]
    assert other["id"] != again["id"]
    custom = overlays.compute(snap, v, ["particles:star,dm,star.color=#102030,dm.size=7,color=red"])["items"]
    d = {i["type"]: i for i in custom}
    assert d["star"]["color"] == [16, 32, 48] and d["dm"]["color"] == [235, 60, 60] and d["dm"]["size_px"] == 7.0
    json.dumps(res)                                                                         # plain data
    # a type that is not in the file gives a note, not an error; depth cut keeps only the slab
    nores = overlays.compute(snap, v, ["particles:dm2"])
    assert nores["items"] == [] and "no particles" in nores["notes"][0]
    slab = overlays.compute(snap, v, ["particles:dm,depth=0.1,max=5000"])["items"][0]
    assert 0 < slab["n_total"] < by["dm"]["n_total"] and np.all(np.abs(slab["depth"]) <= 0.1 + 1e-6)


# ----------------------------------------------------------------------------- inspect

def one_particle(pos=(0.3, -0.2, 0.0), H=1.0, **kw):
    s = lattice_snapshot(rigid, n=8)
    s = {k: (v[:1] if isinstance(v, np.ndarray) and len(v) == len(s["Coordinates"]) else v) for k, v in s.items()}
    s["Coordinates"] = np.array([pos])
    s["SmoothingLength"] = np.array([H], np.float32)
    s["Velocities"] = np.array([[1.0, 2.0, 3.0]], np.float32)
    s["Density"] = np.array([0.5], np.float32)
    s.update(kw)
    return s


def test_inspect_single_particle_projection_and_slice():
    snap = one_particle(ElectronAbundance=np.array([1.0], np.float32))
    v = View(center="xyz:0,0,0", width=4.0, npix=64)
    r = overlays.inspect(snap, v, 0.5, -0.2)
    assert r["n_total"] == 1 and r["mode"] == "proj" and r["pixel"] == [int((0.5 + 2) / 4 * 64), int((-0.2 + 2) / 4 * 64)]
    e = r["particles"][0]
    assert e["id"] == 5000 and e["index"] == 0 and e["H"] == pytest.approx(1.0) and e["rho"] == pytest.approx(0.5) and e["u"] == pytest.approx(3.0)
    assert e["pos"] == pytest.approx([0.3, -0.2, 0.0]) and e["vel"] == pytest.approx([1.0, 2.0, 3.0])
    assert e["circle"] == {"x": pytest.approx(0.3), "y": pytest.approx(-0.2), "r": pytest.approx(1.0)}
    assert e["contribution"] == pytest.approx(1.0) and e["T"] is not None and e["T"] > 0
    assert overlays.inspect(snap, v, 1.5, -0.2)["n_total"] == 0                          # beyond H
    assert overlays.inspect(snap, v, 1.5, -0.2)["particles"] == []
    # without ElectronAbundance and without View.mu the temperature is not derivable
    assert overlays.inspect(one_particle(), v, 0.5, -0.2)["particles"][0]["T"] is None
    assert overlays.inspect(one_particle(), v.with_(mu=0.6), 0.5, -0.2)["particles"][0]["T"] > 0
    # slice: the particle counts when the point (x, y, z0) is within H, the circle is the cross-section sqrt(H^2 - dz^2)
    sl = v.with_(mode="slice", z0=0.6)
    rs = overlays.inspect(snap, sl, 0.3, -0.2)
    assert rs["n_total"] == 1 and rs["particles"][0]["circle"]["r"] == pytest.approx(np.sqrt(1 - 0.36)) and rs["particles"][0]["H"] == pytest.approx(1.0)
    assert overlays.inspect(snap, sl.with_(z0=1.2), 0.3, -0.2)["n_total"] == 0           # plane farther than H
    assert overlays.inspect(snap, sl, 1.2, -0.2)["n_total"] == 0                          # 3D distance sqrt(0.81 + 0.36) > 1
    assert overlays.inspect(snap, sl, 1.0, -0.2)["n_total"] == 1                          # sqrt(0.49 + 0.36) < 1
    json.dumps(r)


def test_inspect_ranking_proj_and_slice(rot_snap):
    v = View(center="xyz:0,0,0", width=4.0, npix=64)
    r = overlays.inspect(rot_snap, v, 0.0, 0.0, limit=500)
    c = [e["contribution"] for e in r["particles"]]
    assert r["n_total"] > 40 and c == sorted(c, reverse=True) and r["contribution_listed"] == pytest.approx(1.0) and r["particles"][0]["rank"] == 1
    top = r["particles"][0]
    d = np.hypot(top["circle"]["x"], top["circle"]["y"])
    assert d < 0.35                                                                      # the most central kernels weigh most
    assert len(overlays.inspect(rot_snap, v, 0.0, 0.0, limit=5)["particles"]) == 5
    sl = overlays.inspect(rot_snap, v.with_(mode="slice", z0=0.05, field="u"), 0.2, 0.1, limit=50)
    dist = [np.sqrt((e["pos"][0] - 0.2) ** 2 + (e["pos"][1] - 0.1) ** 2 + (e["pos"][2] - 0.05) ** 2) for e in sl["particles"]]
    assert dist == sorted(dist) and all(x < e["H"] for x, e in zip(dist, sl["particles"]))      # nearest first, all within H
    assert sl["weight"] == "volume"
    # brute-force check of the number of contributing particles
    pos = rot_snap["Coordinates"]
    h = rot_snap["SmoothingLength"]
    assert sl["n_total"] == int((((pos - [0.2, 0.1, 0.05]) ** 2).sum(1) < h.astype(np.float64) ** 2).sum())
    # frame: an aligned frame reports positions in the frame
    ra = overlays.inspect(one_particle(pos=(1.0, 0.0, 0.0)), v.with_(align="vec:0,1,0"), 0.0, -1.0)
    assert ra["particles"][0]["pos"] == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)
    assert ra["particles"][0]["vel"] == pytest.approx([2.0, -1.0, 3.0], abs=1e-6)


def test_inspect_items_are_circles():
    snap = one_particle()
    r = overlays.inspect(snap, VIEW, 0.3, -0.2)
    it = overlays.inspect_items(r)
    assert it[0]["kind"] == "circle" and it[0]["r"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- rasteriser

def blank(n=64, v=0):
    return np.full((n, n, 3), v, np.uint8)


EXT = [0.0, 64.0, 0.0, 64.0]          # one window unit = one pixel


def test_polyline_anti_aliasing_and_orientation():
    img = blank()
    ln = {"kind": "polyline", "points": [[10, 32.0], [50, 32.0]], "head": None, "color": [255, 255, 255], "linewidth": 1.0, "alpha": 1.0, "halo": 0}
    overlays.draw_items(img, [ln], EXT)
    rows = img[..., 0].astype(int)
    row = 64 - 32                                                          # y = 32 is the boundary between rows 31 and 32 (row 0 at the top)
    assert rows[row - 1, 30] == pytest.approx(128, abs=4) and rows[row, 30] == pytest.approx(128, abs=4)     # half a pixel each side: coverage 1/2 + 1/2
    assert rows[row + 3, 30] == 0 and rows[row, 5] == 0 and rows[row, 55] == 0
    img2 = blank()
    ln2 = dict(ln, points=[[10, 32.5], [50, 32.5]])
    overlays.draw_items(img2, [ln2], EXT)
    r2 = img2[..., 0].astype(int)
    assert r2[64 - 33, 30] == 255 and r2[64 - 33 - 1, 30] == 0 and r2[64 - 33 + 1, 30] == 0
    # +y is up: a line rising in y rises in the picture
    img3 = blank()
    overlays.draw_items(img3, [dict(ln, points=[[10, 10], [50, 50]])], EXT)
    j = np.nonzero(img3[..., 0][:, 12])[0].mean(), np.nonzero(img3[..., 0][:, 48])[0].mean()
    assert j[1] < j[0]
    # the halo darkens a bright background next to the line
    bg = blank(64, 200)
    overlays.draw_items(bg, [dict(ln, halo=1.5, color=[255, 0, 0])], EXT)
    assert bg[row - 2, 30, 1] < 200 and bg[row, 30, 0] > 150 and bg[row, 30, 1] < 120


def test_arrow_head_and_direction():
    img = blank(128)
    ar = {"kind": "arrow", "x": 20.0, "y": 64.0, "dx": 80.0, "dy": 0.0, "head": 0.25, "color": [255, 255, 255], "linewidth": 1.0, "alpha": 1.0, "halo": 0}
    overlays.draw_items(img, [ar], [0, 128, 0, 128])
    ink = img[..., 0] > 100
    cols = np.nonzero(ink.any(0))[0]
    assert cols.min() <= 21 and cols.max() >= 98
    # the head: ink above and below the shaft near the tip, none near the tail
    tip_rows = np.nonzero(ink[:, 90])[0]
    tail_rows = np.nonzero(ink[:, 30])[0]
    assert len(tip_rows) >= 3 and np.ptp(tip_rows) >= 4 and np.ptp(tail_rows) <= 1
    # direction: reversing dx moves the head to the other end
    img2 = blank(128)
    overlays.draw_items(img2, [dict(ar, x=100.0, dx=-80.0)], [0, 128, 0, 128])
    assert np.ptp(np.nonzero((img2[..., 0] > 100)[:, 30])[0]) >= 4
    # a tiny arrow is skipped
    img3 = blank(128)
    overlays.draw_items(img3, [dict(ar, dx=0.5)], [0, 128, 0, 128])
    assert not img3.any()


def test_discs_rings_points_and_clipping():
    img = blank(100)
    overlays.draw_items(img, [{"kind": "circle", "x": 50.0, "y": 50.0, "r": 20.0, "fill": True, "color": [255, 255, 255], "alpha": 1.0}], [0, 100, 0, 100])
    assert img[..., 0].astype(float).sum() / 255 == pytest.approx(np.pi * 400, rel=0.02)
    img = blank(100)
    overlays.draw_items(img, [{"kind": "circle", "x": 50.0, "y": 50.0, "r": 20.0, "fill": False, "color": [0, 255, 0], "linewidth": 1.0}], [0, 100, 0, 100])
    assert img[50, 69:71, 1].max() > 100 and img[50, 50, 1] == 0 and img[50, 60, 1] == 0
    assert img[..., 1].astype(float).sum() / 255 == pytest.approx(2 * np.pi * 20, rel=0.03)         # a ring of width 1 has area 2 pi r
    pts = {"kind": "points", "x": [10, 90, 500, -5], "y": [10, 90, 10, 10], "size_px": 6.0, "style": "dot", "color": [255, 0, 0]}
    img = blank(100)
    overlays.draw_items(img, [pts], [0, 100, 0, 100])
    assert img[90, 10, 0] == 255 and img[10, 90, 0] == 255                     # y = 10 is near the bottom: row 90
    # half transparent
    img = blank(100, 100)
    overlays.draw_items(img, [{"kind": "circle", "x": 50, "y": 50, "r": 10, "fill": True, "color": [200, 200, 200], "alpha": 0.5}], [0, 100, 0, 100])
    assert img[50, 50, 0] == 150
    # items far outside, empty polylines and old-style markers do not fail
    overlays.draw_items(img, [{"kind": "polyline", "points": [[1e6, 1e6], [2e6, 2e6]], "color": [1, 2, 3]}, {"kind": "polyline", "points": []}, {"id": 1, "x": 50.0, "y": 20.0}], [0, 100, 0, 100])
    assert img[80, 54, 0] != 100 or img[80, 46, 0] != 100


def test_legend_is_drawn_top_right():
    img = blank(200)
    leg = {"kind": "legend", "text": "5 km/s", "length": 30.0, "color": [255, 255, 255], "head": 0.3, "linewidth": 1.0}
    overlays.draw_items(img, [leg], [0, 200, 0, 200], font_scale=1)
    ink = img[..., 0] > 0
    ys, xs = np.nonzero(ink)
    assert ys.max() < 20 and xs.min() > 200 - 30 - 6 * 8 - 40 and xs.max() > 190
    img2 = blank(200)
    overlays.draw_items(img2, [leg], [0, 200, 0, 200], legend=False)
    assert not img2.any()


# ----------------------------------------------------------------------------- registry

def test_plugin_overlay_registry_compat(monkeypatch):
    monkeypatch.setattr(plugins, "OVERLAYS", dict(plugins.OVERLAYS))
    plugins.register_overlay("toy", lambda snap, frame, view: [{"x": 0.5, "y": 0.25, "note": "m"}, {"kind": "circle", "x": 0.0, "y": 0.0, "r": 0.5}])
    plugins.register_overlay("toyopt", lambda snap, frame, view, n=1: [{"kind": "circle", "x": 0.0, "y": 0.0, "r": float(n)}])
    snap = lattice_snapshot(rigid, n=8)
    res = overlays.compute(snap, VIEW, ["toy", "toyopt:n=3"])
    assert [i["overlay"] for i in res["items"]] == ["toy", "toy", "toyopt"] and res["items"][2]["r"] == 3.0 and res["extent"] == [-2.0, 2.0, -2.0, 2.0]
    with pytest.raises(ValueError, match="takes"):
        overlays.compute(snap, VIEW, ["toy:k=1"])
    img = blank(64)
    overlays.draw_items(img, res["items"], res["extent"])
    assert img.any()
    # the built-ins are registered under the documented signature function(snap, frame, view)
    assert {"vectors", "streamlines", "particles"} <= set(plugins.OVERLAYS)
    out = plugins.OVERLAYS["vectors"](snap, make_frame(snap, VIEW.center), VIEW)
    assert isinstance(out, list) and out[0]["kind"] == "arrow"
    assert "vectors" in overlays.describe()


# ----------------------------------------------------------------------------- pictures

OV = ["vectors:v,grid=12", "streamlines:v,seeds=8", "particles:bh"]
SV = View(center="bh", width=6.0, npix=128)


def test_shot_with_overlays_sidecar_and_replay(snapfile, store, tmp_path):
    plain = splosh.shot(snapfile, SV, store=store)
    pic = splosh.shot(snapfile, SV, store=store, overlays=OV)
    assert pic.shape == plain.shape and not np.array_equal(pic.rgba, plain.rgba)
    diff = np.any(pic.rgba != plain.rgba, axis=2)
    assert diff.sum() > 400 and diff[:20, :].sum() > 0                                   # something drawn on the map; the legend sits at the top
    assert pic.sidecar["overlays"] == OV and plain.sidecar["overlays"] == []
    out = str(tmp_path / "ov.png")
    files = pic.save(out)
    assert json.load(open(files[1]))["overlays"] == OV
    cli.main(["shot", "--from-sidecar", files[1], "--store", str(tmp_path / "store")])
    assert open(str(tmp_path / "ov_repro.png"), "rb").read() == open(out, "rb").read()          # the overlays are reproduced, byte for byte
    with pytest.raises(ValueError, match="built-in renderer"):
        splosh.shot(snapfile, SV, store=store, overlays=OV, backend="mpl")
    with pytest.raises(ValueError, match="unknown overlay"):
        splosh.shot(snapfile, SV, store=store, overlays=["bogus"])
    with pytest.raises(ValueError, match="spec strings"):
        splosh.shot(snapfile, SV, store=store, overlays=[lambda s, f, v: []])


def test_overlay_errors_are_clear_on_a_snapshot_without_B(snapfile, store):
    with pytest.raises(ValueError, match="MagneticField"):
        splosh.shot(snapfile, SV, store=store, overlays=["vectors:B"])


def test_shot_with_B_from_file(tmp_path, store):
    p = str(tmp_path / "b.hdf5")
    write_snapshot(p, bfield=(1.0, 0.0, 0.0))
    pic = splosh.shot(p, SV, store=store, overlays=["vectors:B,color=cyan,lw=3,halo=0", "streamlines:B,color=#ff00ff,lw=3,halo=0"])
    cy = np.array([60, 220, 235])
    assert (np.abs(pic.rgba[..., :3].astype(int) - cy).sum(2) < 60).sum() > 30
    assert (np.abs(pic.rgba[..., :3].astype(int) - np.array([255, 0, 255])).sum(2) < 60).sum() > 30


def test_cli_overlay_option_for_shot_sheet_movie(snapfile, snapdir, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))                                              # no ffmpeg: numbered frames
    st = str(tmp_path / "st")
    out = str(tmp_path / "s.png")
    cli.main(["shot", snapfile, "--center", "bh", "--width", "6", "--npix", "96", "--overlay", "vectors:v,grid=10", "--overlay", "particles:bh", "--out", out, "--store", st])
    sc = json.load(open(str(tmp_path / "s.json")))
    assert sc["overlays"] == ["vectors:v,grid=10", "particles:bh"]
    plain = str(tmp_path / "p.png")
    cli.main(["shot", snapfile, "--center", "bh", "--width", "6", "--npix", "96", "--out", plain, "--store", st])
    assert open(out, "rb").read() != open(plain, "rb").read()
    sheet = str(tmp_path / "sheet.png")
    cli.main(["sheet", snapdir, "--center", "bh", "--width", "6", "--npix", "64", "--overlay", "vectors:v,grid=8", "--out", sheet, "--store", st])
    assert json.load(open(str(tmp_path / "sheet.json")))["overlays"] == ["vectors:v,grid=8"]
    cli.main(["sheet", "--from-sidecar", str(tmp_path / "sheet.json"), "--store", st])
    assert open(str(tmp_path / "sheet_repro.png"), "rb").read() == open(sheet, "rb").read()
    mv = str(tmp_path / "m.mp4")
    cli.main(["movie", snapdir, "--center", "bh", "--width", "6", "--npix", "64", "--overlay", "streamlines:v,seeds=6", "--out", mv, "--store", st])
    msc = json.load(open(str(tmp_path / "m.json")))
    assert msc["overlays"] == ["streamlines:v,seeds=6"] and msc["frames_dir"]
    cli.main(["overlays"])
    assert "streamlines" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["shot", snapfile, "--overlay", "vectors:v,gridd=3", "--out", str(tmp_path / "x.png"), "--store", st])


def test_cli_inspect(snapfile, capsys):
    cli.main(["inspect", snapfile, "0.0", "0.0", "--center", "bh", "--width", "6", "--limit", "3"])
    r = json.loads(capsys.readouterr().out)
    assert len(r["particles"]) == 3 and r["n_total"] >= 3 and r["particles"][0]["circle"]["r"] > 0


# ----------------------------------------------------------------------------- server

@pytest.fixture
def server(snapdir, tmp_path):
    from splosh.server import make_server
    srv, state = make_server([snapdir], token="s3cret", store=MapStore(str(tmp_path / "store")), defaults=View(center="bh", align="bh", width=6.0, npix=64))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=60))


def test_server_overlay_and_inspect(server):
    r = get(server + "/overlay?token=s3cret&snap=1&spec=vectors:v,grid=8&spec=particles:bh&spec=streamlines:v,seeds=6")
    kinds = {i["kind"] for i in r["items"]}
    assert {"arrow", "points", "polyline", "legend"} <= kinds and r["extent"] == pytest.approx([-3, 3, -3, 3]) and r["npix"] == 64 and r["time"] == pytest.approx(0.25)
    ins = get(server + "/inspect?token=s3cret&snap=0&x=0.1&y=0.1&limit=4")
    assert len(ins["particles"]) == 4 and ins["particles"][0]["circle"]["r"] > 0 and ins["mode"] == "proj"
    sl = get(server + "/inspect?token=s3cret&snap=0&x=0.1&y=0.1&mode=slice&z0=0.0&limit=2")
    assert sl["mode"] == "slice"
    for bad, code in (("/overlay?token=s3cret&snap=0&spec=nope", 400), ("/overlay?token=s3cret&snap=0&spec=vectors:B", 400), ("/inspect?token=s3cret&snap=7&x=0&y=0", 404), ("/overlay?snap=0", 403)):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(server + bad)
        assert e.value.code == code, bad
