"""Kinds of runs and block-structured mesh data (kinds.py, mesh.py, docs/KINDS.md), on the tests' reference format (meshref.py): the picker claims the
folder and offers only its mesh snapshots, slices have exact values at pixels and across level and patch boundaries, the finest level wins, arbitrary
planes, box outlines, markers, the map store, the server."""
import json
import os
import threading
import urllib.request

import numpy as np
import pytest

import meshref
from splosh import View, kinds, mesh, plugins
from splosh.io import load_snapshot
from splosh.maps import render_map, required_gas_fields
from splosh.mapstore import MapStore
from conftest import write_snapshot


@pytest.fixture
def meshrun(tmp_path):
    """a run folder: output/ with two mesh snapshots and two particle dummies next to them."""
    out = tmp_path / "meshrun" / "output"
    out.mkdir(parents=True)
    for k in range(2):
        meshref.write_mesh(str(out / f"snapshot_{k:03d}{meshref.EXT}"), time=1.5 * k)
        write_snapshot(str(out / f"snapshot_{k:03d}.hdf5"), n_gas=200, seed=k, time=1.5 * k)
    mesh.clear_cache()
    return tmp_path / "meshrun"


def snap0(run):
    return str(run / "output" / f"snapshot_000{meshref.EXT}")


def pixel_centres(view):
    pix = view.width / view.npix
    a = view.cx - view.width / 2 + (np.arange(view.npix) + 0.5) * pix
    b = view.cy - view.width / 2 + (np.arange(view.npix) + 0.5) * pix
    return np.meshgrid(a, b)                    # [j, i]


# ----------------------------------------------------------------------------- picker, runs
def test_kind_claims_the_folder_and_lists_only_its_snapshots(meshrun):
    from splosh import runs
    out = meshrun / "output"
    assert kinds.kind_of_dir(str(out)) == "meshref" and kinds.kind_of_dir(str(meshrun)) is None
    snaps = runs.run_snapshots(str(meshrun))
    assert [os.path.basename(p) for p in snaps] == [f"snapshot_000{meshref.EXT}", f"snapshot_001{meshref.EXT}"]       # not the particle dummies
    assert runs.snapshot_dir(str(meshrun)) == (str(out), 2) and runs.folder_kind(str(meshrun)) == "meshref"
    assert kinds.kind_of_file(snaps[0]) == "meshref" and kinds.kind_of_file(str(out / "snapshot_000.hdf5")) is None
    r = runs.Run("r", snaps)
    assert r.kind == "meshref" and r.times == [0.0, 1.5] and r.period is None
    assert "2 mesh test snapshots in output/, t = 0-1.5" in runs.folder_description(str(meshrun))


def test_browse_counts_the_kind_snapshots(meshrun, tmp_path):
    from splosh.server import ViewerState
    st = ViewerState([], root=str(tmp_path), store=MapStore(str(tmp_path / "store")))
    e = {x["name"]: x for x in st.browse("")["entries"]}["meshrun"]
    assert e["n"] == 2 and e["output"] and e["kind"] == "meshref"
    deep = {x["name"]: x for x in st.browse("", deep=True)["entries"]}
    assert deep["meshrun"]["n"] == 2 and deep["meshrun"]["kind"] == "meshref"


def test_particle_folders_are_unchanged(snapdir):
    from splosh import runs
    assert kinds.kind_of_dir(snapdir) is None
    assert all(p.endswith(".hdf5") for p in runs.run_snapshots(snapdir)) and runs.Run("p", runs.run_snapshots(snapdir)).kind is None


# ----------------------------------------------------------------------------- slices
def test_linear_field_is_exact_everywhere(meshrun):
    s = load_snapshot(snap0(meshrun))
    assert s["kind"] == "meshref" and s["time"] == 0.0 and "mesh" in s
    v = View(center="origin", mode="slice", field="lin", width=12.0, npix=96, cx=0.3, cy=-0.2, z0=0.37)
    img, meta = render_map(s, v)
    X, Y = pixel_centres(v)
    assert img.dtype == np.float32 and img.shape == (96, 96)
    np.testing.assert_allclose(img, meshref.lin(X, Y, 0.37), rtol=0, atol=2e-5)                  # trilinear reproduces a linear function (also across seams)
    assert meta["extent"] == [0.3 - 6, 0.3 + 6, -0.2 - 6, -0.2 + 6] and meta["xlabel"] == "x" and meta["ylabel"] == "y"
    assert meta["length_unit"]["unit"] == "M" and meta["time_unit"]["unit"] == "M" and meta["label"] == "linear test field"
    assert set(meta["mesh"]["pixels_per_level"]) == {"0", "1", "2"}


def test_values_at_vertices_are_the_stored_ones_and_nearest_mode(meshrun):
    s = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="pot", width=8.0, npix=32, cx=0.125, cy=0.125, z0=0.0)     # pixel centres on the 0.25 grid
    X, Y = pixel_centres(v)
    fine = (np.abs(X) <= 1) & (np.abs(Y) <= 1)
    for interp in ("linear", "nearest"):
        img, _ = render_map(s, v.with_(interp=interp))
        np.testing.assert_allclose(img[fine], meshref.pot(X, Y, 0.0)[fine], rtol=1e-6)        # level 2 has a point at every pixel
        on1 = ~fine & (np.abs(X) <= 2) & (np.abs(Y) <= 2) & (np.mod(X, 0.5) == 0) & (np.mod(Y, 0.5) == 0)
        np.testing.assert_allclose(img[on1], meshref.pot(X, Y, 0.0)[on1], rtol=1e-6)          # level-1 points
    img_n, _ = render_map(s, v.with_(interp="nearest", field="lin"))
    between = (~fine) & (np.abs(X) < 2) & (np.abs(Y) < 2) & (np.mod(X, 0.5) == 0.25) & (np.mod(Y, 0.5) == 0)
    assert between.any()                                                                        # nearest: a stored neighbour's value, not the analytic one
    assert not np.allclose(img_n[between], meshref.lin(X, Y, 0.0)[between])


def test_pot_interpolation_matches_an_independent_trilinear(meshrun):
    s = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="pot", width=3.0, npix=40, cx=0.1, cy=0.05, z0=0.3)
    img, _ = render_map(s, v)
    X, Y = pixel_centres(v)

    def tri(x, y, z, dx):                                              # trilinear in the uniform grid of spacing dx through -8
        out = 0.0
        f = [(c + 8.0) / dx for c in (x, y, z)]
        i0 = [np.floor(c) for c in f]
        t = [c - i for c, i in zip(f, i0)]
        for k in range(8):
            w, p = 1.0, []
            for a in range(3):
                bit = (k >> a) & 1
                w = w * (t[a] if bit else 1 - t[a])
                p.append(-8.0 + (i0[a] + bit) * dx)
            out = out + w * meshref.pot(*p)
        return out
    level = np.where((np.abs(X) <= 1) & (np.abs(Y) <= 1), 0.25, np.where((np.abs(X) <= 2) & (np.abs(Y) <= 2), 0.5, 1.0))
    want = np.where(level == 0.25, tri(X, Y, 0.3, 0.25), np.where(level == 0.5, tri(X, Y, 0.3, 0.5), tri(X, Y, 0.3, 1.0)))
    np.testing.assert_allclose(img, want, rtol=1e-5, atol=1e-6)


def test_finest_level_wins_and_outside_is_nan(meshrun):
    s = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="level", width=20.0, npix=80, z0=0.0)
    img, meta = render_map(s, v)
    X, Y = pixel_centres(v)
    want = np.where((np.abs(X) <= 1) & (np.abs(Y) <= 1), 2.0, np.where(((np.abs(X) <= 2) & (np.abs(Y) <= 2)) | ((X >= 3) & (X <= 5) & (np.abs(Y) <= 1)), 1.0,
                                                                         np.where((np.abs(X) <= 8) & (np.abs(Y) <= 8), 0.0, np.nan)))
    np.testing.assert_array_equal(img, want.astype(np.float32))
    assert int(meta["mesh"]["pixels_per_level"]["2"]) == int((want == 2).sum())
    img3, _ = render_map(s, v.with_(z0=1.5))                                                   # above level 2 (|z| <= 1), inside level 1's first box
    assert np.nanmax(img3) == 1.0


def test_slice_reads_only_the_layers_it_needs(meshrun):
    s = load_snapshot(snap0(meshrun))
    obj = s["mesh"]
    obj.reads.clear()
    render_map(s, View(center="origin", mode="slice", field="lin", width=20.0, npix=40, z0=5.0))    # only level 0's second patch (k 8..16) reaches z = 5
    assert obj.reads and all(r[1] == 0 and r[2] == 1 for r in obj.reads)
    lo, hi = obj.reads[0][3], obj.reads[0][4]
    assert hi[2] - lo[2] + 1 <= meshref.LEVELS[0][3][0][1][2] and lo[2] >= 12 and hi[2] <= 14         # a slab around k = 13, the whole patch in x and y
    n = len(obj.reads)
    render_map(s, View(center="origin", mode="slice", field="lin", width=10.0, npix=40, z0=5.0, cx=1.0))    # panned: served from the patch cache
    assert len(obj.reads) == n


def test_arbitrary_planes(meshrun):
    s = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="lin", axes="xz", width=6.0, npix=48, z0=0.4)
    img, meta = render_map(s, v)
    X, Z = pixel_centres(v)
    np.testing.assert_allclose(img, meshref.lin(X, 0.4, Z), atol=2e-5)                           # xz: y = z0
    assert meta["xlabel"] == "x" and meta["ylabel"] == "z"
    from splosh.frames import make_frame
    vt = View(center="xyz:0.3,-0.2,0.1", mode="slice", field="lin", tilt="euler:30,50,10", width=7.0, npix=48, z0=0.25, cx=0.2)
    img, _ = render_map(s, vt)
    R = make_frame(s, vt.center, vt.align, vt.tilt).rot
    A, B = pixel_centres(vt)
    P = np.array([0.3, -0.2, 0.1]) + A[..., None] * R[0] + B[..., None] * R[1] + 0.25 * R[2]
    assert abs(np.linalg.det(R) - 1) < 1e-12
    ok = np.isfinite(img)
    assert ok.mean() > 0.95
    np.testing.assert_allclose(img[ok], meshref.lin(P[..., 0], P[..., 1], P[..., 2])[ok], atol=3e-5)


def test_derived_fields_and_frame_points(meshrun):
    s = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="lin2", width=4.0, npix=16, z0=0.1)
    img, meta = render_map(s, v)
    X, Y = pixel_centres(v)
    np.testing.assert_allclose(img, 2 * meshref.lin(X, Y, 0.1) + 1, atol=5e-5)
    assert meta["quantity_unit"]["unit"] == "M" and meta["unit_override"] == "M"
    img, _ = render_map(s, v.with_(field="radius"))
    np.testing.assert_allclose(img, np.sqrt(X ** 2 + Y ** 2 + 0.01), atol=1e-5)
    # points() feed the frame rules like black holes: centre bh puts (0.5, 0.25, 0) at the window centre
    imgb, metab = render_map(s, v.with_(center="bh", field="lin"))
    np.testing.assert_allclose(imgb, meshref.lin(X + 0.5, Y + 0.25, 0.1), atol=5e-5)
    from splosh.maps import bh_markers
    assert bh_markers(s, v.with_(center="bh"))[0]["x"] == pytest.approx(0.0)
    with pytest.raises(KeyError):
        render_map(s, v.with_(field="nosuch"))
    with pytest.raises(ValueError):
        render_map(s, v.with_(mode="proj"))
    assert required_gas_fields(v.with_(field="lin2")) == {"Coordinates", "SmoothingLength", "Masses"}


def test_cell_centred_and_periodic_levels():
    class Uniform:                                             # one periodic cell-centred level built in memory: values at origin + (i + 1/2) dx
        time = 0.0
        variables = ["f"]

        def __init__(self):
            self.levels = [mesh.Level(dx=1.0, origin=(0.0, 0.0, 0.0), patches=[((0, 0, 0), (7, 7, 7))], centring="cell", periodic=True)]
            c = np.arange(8) + 0.5
            X = c[:, None, None] + 0 * c[None, :, None] + 0 * c[None, None, :]
            self.data = np.sin(2 * np.pi * X / 8.0)

        def read(self, var, level, patch, lo, hi):
            return self.data[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    obj = Uniform()
    P = np.array([[0.5, 1.5, 2.5], [8.5, 1.0, 1.0], [-0.5, 3, 3], [0.1, 3.0, 3.0], [4.0, 4.0, 4.0]])
    out = mesh.sample(obj, None, P, ["f"])[0]
    f = lambda x: np.sin(2 * np.pi * x / 8.0)
    assert out[0] == pytest.approx(f(0.5)) and out[1] == pytest.approx(f(0.5))               # a cell centre; the periodic image of x = 0.5
    assert out[2] == pytest.approx(f(7.5))                                                       # -0.5 wraps to 7.5
    assert out[3] == pytest.approx(0.6 * f(0.5) + 0.4 * f(7.5))                                 # across the periodic boundary (cells 7 and 0)
    assert out[4] == pytest.approx(0.5 * (f(3.5) + f(4.5)))
    near = mesh.sample(obj, None, P, ["f"], interp="nearest")[0]
    assert near[3] == pytest.approx(f(0.5))


# ----------------------------------------------------------------------------- overlays
def test_box_outlines_on_the_plane(meshrun):
    from splosh import overlays
    s = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="lin", width=20.0, npix=64, z0=0.0)
    res = overlays.compute(s, v, ["mesh_boxes"])
    lines = [it for it in res["items"] if it["kind"] == "polyline"]
    assert sorted(it["level"] for it in lines) == [0, 1, 1, 2]                                  # level 1: two boxes (the first merged from two patches)
    box1 = [it for it in lines if it["level"] == 1]
    xs = sorted((min(p[0] for p in it["points"]), max(p[0] for p in it["points"])) for it in box1)
    assert xs[0] == pytest.approx((-2.0, 2.0)) and xs[1] == pytest.approx((3.0, 5.0))
    assert all(it["points"][0] == it["points"][-1] for it in lines)                              # closed
    assert len([it for it in overlays.compute(s, v.with_(z0=1.5), ["mesh_boxes"])["items"] if it["kind"] == "polyline"]) == 2      # level 2 and the second level-1 box (|z| <= 1) missed
    assert len(overlays.compute(s, v.with_(z0=1.5), ["mesh_boxes:levels=0"])["items"]) == 1
    tl = overlays.compute(s, v.with_(tilt="euler:0,45,0"), ["mesh_boxes:labels=1"])
    poly = [it for it in tl["items"] if it["kind"] == "polyline" and it["level"] == 2][0]
    ys = [p[1] for p in poly["points"]]
    assert max(ys) - min(ys) == pytest.approx(2 * np.sqrt(2), rel=1e-6)                         # a cube of side 2 cut by a plane at 45 degrees
    assert any(it["kind"] == "text" for it in tl["items"])


def test_markers_overlay_and_drawing(meshrun):
    from splosh import overlays
    s = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="lin", width=6.0, npix=64)
    res = overlays.compute(s, v, ["mesh_markers"])
    pts = [it for it in res["items"] if it["kind"] == "points"][0]
    assert pts["x"] == [1.0, -1.0] and pts["y"] == [1.0, 0.5] and pts["color"] == [235, 60, 60]
    assert [it["text"] for it in res["items"] if it["kind"] == "text"] == ["A", "B"]
    rgb = np.zeros((64, 64, 3), np.uint8)
    overlays.draw_items(rgb, res["items"] + overlays.compute(s, v, ["mesh_boxes"])["items"], res["extent"])
    assert rgb.any()


# ----------------------------------------------------------------------------- downstream: store, shot, tiles, server
def test_map_store_and_shot(meshrun, tmp_path):
    from splosh import figures
    store = MapStore(str(tmp_path / "store"))
    v = View(center="origin", mode="slice", field="pot", width=8.0, npix=48)
    a, meta, hit = store.get_or_render(snap0(meshrun), v)
    b, meta2, hit2 = store.get_or_render(snap0(meshrun), v)
    assert not hit and hit2 and np.array_equal(a, b, equal_nan=True) and meta2["snapshot"] == f"snapshot_000{meshref.EXT}"
    c, _, hit3 = store.get_or_render(snap0(meshrun), v.with_(interp="nearest"))
    assert not hit3 and not np.array_equal(a, c)
    pic = figures.shot(snap0(meshrun), v, store=store, overlays=["mesh_boxes", "mesh_markers"], scale="linear")
    assert pic.rgba.shape[2] == 4 and pic.sidecar["snapshots"][0]["name"] == f"snapshot_000{meshref.EXT}"
    assert "width 8 M" in figures.view_summary(meta)
    assert figures.time_text(0.0, figures.unit_names(meta["units"])) == "t = 0 M"


def test_cli_uses_the_kind_default_view(meshrun):
    import argparse
    from splosh.cli import add_view_options, view_from_args
    p = argparse.ArgumentParser()
    p.add_argument("snapshot")
    add_view_options(p)
    v = view_from_args(p.parse_args([snap0(meshrun), "--width", "5"]))
    assert (v.center, v.mode, v.field, v.width, v.npix) == ("origin", "slice", "pot", 5.0, 64)


def test_tiles_of_a_mesh_slice(meshrun, tmp_path):
    from splosh import tiles
    s = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="lin", width=8.0, npix=128)
    img, _ = tiles.render_tiled(s, v, tile=64, store=None)
    ref, _ = render_map(s, v)
    np.testing.assert_allclose(img, ref, atol=1e-5)


def test_post_transform(meshrun):
    s0 = load_snapshot(snap0(meshrun))
    v = View(center="origin", mode="slice", field="pot", width=8.0, npix=32, post="grad")
    img, meta = render_map(s0, v)
    assert np.isfinite(img).mean() > 0.9 and meta["view"]["post"] == "grad"


@pytest.fixture
def server(meshrun, tmp_path):
    from splosh.server import make_server
    srv, state = make_server([str(meshrun)], token="t", store=MapStore(str(tmp_path / "store")))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown(); srv.server_close()


def test_server_opens_a_mesh_run(server):
    base, state = server
    st = json.load(urllib.request.urlopen(base + "/api/state?token=t"))
    assert [s["name"] for s in st["snapshots"]] == [f"snapshot_000{meshref.EXT}", f"snapshot_001{meshref.EXT}"]
    assert st["runs"][0]["kind"] == "meshref" and st["defaults"]["center"] == "origin" and st["defaults"]["field"] == "pot" and st["defaults"]["mode"] == "slice"
    assert {"lin", "pot", "level", "lin2", "radius"} <= set(st["fields"]) and "rho" not in st["fields"]
    assert {"mesh_boxes", "mesh_markers"} <= {o["name"] for o in st["plugin_overlays"]}
    r = urllib.request.urlopen(base + "/api/map?token=t&snap=1&field=lin&mode=slice&center=origin&width=4&npix=32")
    meta = json.loads(r.headers["X-Splosh-Meta"])
    img = np.frombuffer(r.read(), "<f4").reshape(32, 32)
    X, Y = pixel_centres(View(width=4.0, npix=32))
    np.testing.assert_allclose(img, meshref.lin(X, Y, 0.0), atol=2e-5)
    assert meta["snapshot"] == f"snapshot_001{meshref.EXT}" and meta["time"] == 1.5
    ov = json.load(urllib.request.urlopen(base + "/overlay?token=t&snap=0&field=lin&mode=slice&center=origin&width=20&npix=32&spec=mesh_boxes"))
    assert len([i for i in ov["items"] if i["kind"] == "polyline"]) == 4
    bh = json.load(urllib.request.urlopen(base + "/api/bh?token=t&snap=0&field=lin&mode=slice&center=origin&width=4&npix=32"))
    assert len(bh["markers"]) == 1 and bh["markers"][0]["x"] == pytest.approx(0.5)
    runs = json.load(urllib.request.urlopen(base + "/api/runs?token=t"))["runs"]
    assert runs[0]["kind"] == "meshref" and [s["time"] for s in runs[0]["snapshots"]] == [0.0, 1.5]


def test_particle_server_hides_mesh_overlays(snapdir, tmp_path):
    from splosh.server import make_server
    srv, state = make_server([snapdir], token="t", store=MapStore(str(tmp_path / "store")))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/state?token=t"))
        assert not {"mesh_boxes", "mesh_markers"} & {o["name"] for o in st["plugin_overlays"]} and st["runs"][0]["kind"] is None and "rho" in st["fields"]
    finally:
        srv.shutdown(); srv.server_close()


def test_register_kind_api():
    assert "meshref" in plugins.KINDS and plugins.PRESETS["meshref"]["field"] == "pot"
    k = plugins.KINDS["meshref"]
    assert set(k) == {"detect", "snapshots", "open", "fields", "units", "default_view", "label", "time", "doc"}
    assert mesh.merge_boxes([((0, 0, 0), (3, 3, 3)), ((4, 0, 0), (7, 3, 3)), ((0, 4, 0), (7, 5, 3))]) == [((0, 0, 0), (7, 5, 3))]


def test_particle_products_do_not_apply_to_a_mesh_run(meshrun, snapdir, monkeypatch):
    from splosh import products, runs
    monkeypatch.setattr(plugins, "PRODUCTS", dict(plugins.PRODUCTS))
    monkeypatch.setattr(products, "PRODUCTS", plugins.PRODUCTS, raising=False)
    plugins.register_product("toy_particles", lambda path, params: {"scalars": {"x": 1.0}})
    plugins.register_product("toy_mesh", lambda path, params: {"scalars": {"x": 1.0}}, applies=lambda p: kinds.kind_of_file(p) == "meshref")
    mesh_paths, part_paths = runs.run_snapshots(str(meshrun)), runs.run_snapshots(snapdir)
    assert not products.applicable("toy_particles", mesh_paths) and products.applicable("toy_mesh", mesh_paths)
    assert products.applicable("toy_particles", part_paths) and not products.applicable("toy_mesh", part_paths)
