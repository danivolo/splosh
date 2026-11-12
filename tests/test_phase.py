"""Phase diagrams of the gas in a view's window, selections on them, and the map of a selection (phase.py, /api/phase, /api/map?sel=, /api/selection)."""
import json
import threading
import urllib.parse
import urllib.request

import numpy as np
import pytest

from splosh import View, load_snapshot, render_map
from splosh import phase
from splosh.mapstore import MapStore
from splosh.server import make_server

FIELDS = ("Coordinates", "SmoothingLength", "Masses", "Density", "InternalEnergy", "Velocities")


def test_histogram_conserves_the_gas_of_the_window(snapfile):
    s = load_snapshot(snapfile, gas_fields=FIELDS)
    v = View(center="bh", width=6.0, npix=64)
    grid, meta = phase.histogram(s, v, "density", "u", bins=64)
    assert grid.shape == (64, 64) and grid.dtype == np.float32
    assert np.isclose(grid.sum(), meta["total_in_range"], rtol=1e-5)
    keep, _ = phase._in_view(s, v, phase.make_frame(s, "bh"))
    assert meta["n_in_view"] == int(keep.sum()) and np.isclose(meta["total"], s["Masses"][keep].sum(), rtol=1e-6)
    g2, m2 = phase.histogram(s, v, "density", "u", bins=64, weight="count", xr=(-30, 30), yr=(-30, 30))
    assert g2.sum() == m2["n_used"] == m2["n_in_range"]                          # wide ranges: every particle with finite logs is counted once
    _, m3 = phase.histogram(s, v, "R", "vz", xlog=False, ylog=False, bins=32)
    assert m3["x"]["log"] is False and m3["x"]["label"] == "cylindrical radius"
    with pytest.raises(KeyError):
        phase.histogram(s, v, "rho", "u")                                       # rho is a map quantity, not a particle one
    with pytest.raises(ValueError):
        phase.histogram(s, v, "density", "u", bins=4)


def test_polygon_and_selection(snapfile):
    sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert list(phase._inside_polygon(np.array([0.5, 1.5, 0.5, -0.1]), np.array([0.5, 0.5, 0.99, 0.5]), sq)) == [True, False, True, False]
    with pytest.raises(ValueError):
        phase._inside_polygon(np.zeros(1), np.zeros(1), [[0, 0], [1, 1]])
    s = load_snapshot(snapfile, gas_fields=FIELDS)
    v = View(center="bh", width=6.0, npix=64)
    _, meta = phase.histogram(s, v, "density", "u")
    (x0, x1), (y0, y1) = meta["x"]["range"], meta["y"]["range"]
    xm = 0.5 * (x0 + x1)
    left = [[x0 - 1, y0 - 1], [xm, y0 - 1], [xm, y1 + 1], [x0 - 1, y1 + 1]]
    right = [[xm, y0 - 1], [x1 + 1, y0 - 1], [x1 + 1, y1 + 1], [xm, y1 + 1]]
    a, b = phase.select(s, v, "density", "u", left), phase.select(s, v, "density", "u", right)
    keep, _ = phase._in_view(s, v, phase.make_frame(s, "bh"))
    assert not (a & b).any() and ((a | b) == keep).all()                        # the two halves split the window's gas
    sub = phase.subset(s, a)
    assert len(sub["Masses"]) == a.sum() and sub["bh_pos"] is s["bh_pos"] and sub["header"] is s["header"]
    # maps are additive in the selection: map(left) + map(right) == map(all) for the window's gas
    everything = phase.select(s, v, "density", "u", [[x0 - 99, y0 - 99], [x1 + 99, y0 - 99], [x1 + 99, y1 + 99], [x0 - 99, y1 + 99]], window=False)
    ma, _ = render_map(phase.subset(s, phase.select(s, v, "density", "u", left, window=False)), v.with_(exact=True))
    mb, _ = render_map(phase.subset(s, phase.select(s, v, "density", "u", right, window=False)), v.with_(exact=True))
    mall, _ = render_map(phase.subset(s, everything), v.with_(exact=True))
    assert np.allclose(ma + mb, mall, rtol=1e-4, atol=1e-6 * mall.max())


@pytest.fixture
def server(snapdir, tmp_path):
    srv, state = make_server([snapdir], token="t", store=MapStore(str(tmp_path / "store")), defaults=View(center="bh", width=6.0, npix=64))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown(); srv.server_close()


def get(base, path, **q):
    return urllib.request.urlopen(base + path + "?" + urllib.parse.urlencode(dict(q, token="t")), timeout=60)


def test_phase_and_selection_endpoints(server):
    base, state = server
    st = json.load(get(base, "/api/state"))
    assert "density" in st["phase_quantities"] and "R" in st["phase_quantities"]
    r = get(base, "/api/phase", snap=0, x="density", y="u", bins=32)
    meta = json.loads(r.headers["X-Splosh-Meta"])
    grid = np.frombuffer(r.read(), "<f4").reshape(32, 32)
    assert np.isclose(grid.sum(), meta["total_in_range"], rtol=1e-5) and meta["x"]["log"] and meta["bins"] == 32
    (x0, x1), (y0, y1) = meta["x"]["range"], meta["y"]["range"]
    sel = json.dumps({"x": "density", "y": "u", "xlog": True, "ylog": True, "poly": [[x0 - 1, y0 - 1], [x1 + 1, y0 - 1], [x1 + 1, y1 + 1], [x0 - 1, y1 + 1]]})
    full = get(base, "/api/map", snap=0)
    whole = np.frombuffer(full.read(), "<f4")
    r = get(base, "/api/map", snap=0, sel=sel, exact=0)
    m = json.loads(r.headers["X-Splosh-Meta"])
    img = np.frombuffer(r.read(), "<f4")
    assert m["selection"]["n_selected"] > 0 and img.shape == whole.shape
    assert np.isclose(img.sum(), whole.sum(), rtol=1e-3)                     # everything selected: the same map
    pts = json.load(get(base, "/api/selection", snap=0, sel=sel))
    assert pts["n_in_window"] == len(pts["points"]) == meta["n_used"] and pts["mass_selected"] > 0
    empty = json.dumps({"x": "density", "y": "u", "poly": [[100, 100], [101, 100], [101, 101]]})
    e = get(base, "/api/map", snap=0, sel=empty)
    assert json.loads(e.headers["X-Splosh-Meta"])["selection"]["n_selected"] == 0 and not np.frombuffer(e.read(), "<f4").any()
    for bad in ({"x": "density"}, {"x": "nope", "y": "u", "poly": [[0, 0], [1, 0], [1, 1]]}):
        with pytest.raises(urllib.error.HTTPError) as err:
            get(base, "/api/map", snap=0, sel=json.dumps(bad))
        assert err.value.code == 400
