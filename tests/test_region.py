"""Partial loading (region.py): a chunk index per snapshot, region loads, and maps/tiles from a region equal to those from the whole snapshot."""
import json
import os
import threading
import urllib.parse
import urllib.request

import h5py
import numpy as np
import pytest

from conftest import write_snapshot
from splosh import View, load_snapshot, render_map
from splosh import region
from splosh.mapstore import MapStore
from splosh.server import make_server

ALL = ("Coordinates", "SmoothingLength", "Masses", "Density", "InternalEnergy", "Velocities")


def morton_sort(path):
    """reorder the gas along a Morton (Z-order) curve, as GIZMO's domain order along its space-filling curve does."""
    with h5py.File(path, "r+") as f:
        g = f["PartType0"]
        x = g["Coordinates"][...]
        q = ((x - x.min(0)) / (np.ptp(x, 0) + 1e-12) * 1023).astype(np.uint64)
        key = np.zeros(len(x), np.uint64)
        for bit in range(10):
            for d in range(3):
                key |= ((q[:, d] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit + d)
        o = np.argsort(key, kind="stable")
        for k in list(g.keys()):
            g[k][...] = g[k][...][o]


@pytest.fixture
def big(tmp_path, monkeypatch):
    monkeypatch.setattr(region, "CHUNK", 128)
    p = str(tmp_path / "snap_000.hdf5")
    write_snapshot(p, n_gas=40000, seed=5)
    with h5py.File(p, "r+") as f:                          # smaller kernels than the generator's (H ~ 0.4 on a +-2.3 box): like real runs, a window reaches few chunks
        f["PartType0/SmoothingLength"][...] = f["PartType0/SmoothingLength"][...] / 4
    morton_sort(p)
    return p


def test_index_and_plan(big, tmp_path):
    ix = region.get_index(big, str(tmp_path / "store"))
    assert ix["chunk"] == 128 and ix["n"] == 40000 and len(ix["lo"]) == -(-40000 // 128)
    fn = region.index_file(big, str(tmp_path / "store"))
    t = os.path.getmtime(fn)
    region.get_index(big, str(tmp_path / "store"))
    assert os.path.getmtime(fn) == t                                          # reused, not rebuilt
    s = load_snapshot(big, ALL)
    x = s["Coordinates"]
    for k in range(len(ix["lo"])):                                           # the boxes do bound their chunks
        c = x[k * 128:(k + 1) * 128]
        assert (c >= ix["lo"][k] - 1e-6).all() and (c <= ix["hi"][k] + 1e-6).all()
    from splosh.frames import make_frame
    v = View(center="bh", width=0.4, npix=64, cx=1.0)                      # off the dense centre of the synthetic cloud
    rows, bounds = region.plan(ix, make_frame(s, "bh"), v, margin=0.5)
    n = sum(b - a for a, b in rows)
    assert 0 < n < 0.4 * 40000 and np.allclose(bounds["a"], [0.6, 1.4])
    assert all(a < b for a, b in rows) and all(rows[i][1] < rows[i + 1][0] for i in range(len(rows) - 1))   # merged, sorted, disjoint


@pytest.mark.parametrize("view", [
    View(center="bh", width=0.4, npix=64, cx=1.0),
    View(center="bh", width=0.4, npix=64, cx=-1.0, cy=0.5, exact=True, field="u"),
    View(center="bh", align="bh", tilt="euler:20,50,-30", width=0.4, npix=48, cx=0.2, cy=-0.1),
    View(center="xyz:0.3,0.2,0.0", axes="xz", width=0.4, npix=48),
    View(center="bh", mode="slice", z0=0.05, width=0.5, npix=48, field="u"),
])
def test_region_map_equals_full_map(big, tmp_path, view):
    snap = region.load_region(big, view, tuple(sorted(set(ALL))), root=str(tmp_path / "store"), margin=0.5, max_fraction=0.9)
    assert snap is not None and snap["region"]["fraction"] < 0.9
    full = load_snapshot(big, ALL)
    a, ma = render_map(snap, view)
    b, mb = render_map(full, view)
    assert ma["n_particles"] == mb["n_particles"]                            # every particle that reaches the window is in the region
    assert np.allclose(a, b, rtol=1e-6, atol=0, equal_nan=True)
    assert region.covers(snap["region"], view) and region.covers(snap["region"], view.with_(cx=view.cx + 0.4 * view.width))
    assert not region.covers(snap["region"], view.with_(cx=view.cx + 2 * view.width)) and not region.covers(snap["region"], view.with_(width=3 * view.width))
    assert not region.covers(snap["region"], view.with_(axes="yz" if view.axes != "yz" else "xy"))


def test_rules_that_need_all_gas_load_everything(big, tmp_path):
    for v in (View(center="com", width=1.0), View(center="peak", width=1.0), View(center="bh", tilt="L:1", width=1.0), View(center="ids:999999", width=0.5)):
        assert region.load_region(big, v, ALL, root=str(tmp_path / "s")) is None
    assert region.load_region(big, View(center="bh", width=100.0), ALL, root=str(tmp_path / "s")) is None      # the whole box: no gain
    assert region.load_region(big, View(center="bh", width=0.5), ALL, root=str(tmp_path / "s"), min_particles=10 ** 6) is None


def test_server_uses_regions_and_gives_the_same_maps(big, tmp_path):
    d = os.path.dirname(big)
    out = {}
    for mode in ("on", "off"):
        srv, state = make_server([d], token="t", store=MapStore(str(tmp_path / f"store_{mode}")), defaults=View(center="bh", width=0.4, npix=64, cx=1.0), partial=mode)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        r = urllib.request.urlopen(base + "/api/map?" + urllib.parse.urlencode({"snap": 0, "token": "t", "exact": 1}), timeout=60)
        meta = json.loads(r.headers["X-Splosh-Meta"])
        img = np.frombuffer(r.read(), "<f4")
        t = urllib.request.urlopen(base + "/api/tile?" + urllib.parse.urlencode({"snap": 0, "token": "t", "level": 4, "ix": 2, "iy": 0, "tile": 64}), timeout=60)
        out[mode] = (meta, img, np.frombuffer(t.read(), "<f4"), len(state._regions))
        srv.shutdown(); srv.server_close()
    (mon, ion, ton, non), (moff, ioff, toff, noff) = out["on"], out["off"]
    assert "region" in mon and mon["region"]["fraction"] < 0.5 and "region" not in moff
    assert non >= 1 and noff == 0
    assert np.allclose(ion, ioff, rtol=1e-6) and np.allclose(ton, toff, rtol=1e-6)
