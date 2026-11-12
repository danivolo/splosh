"""Viewer page and server: things that need no browser (the browser tests are in test_viewer_browser.py)."""
import json
import threading
import urllib.request

import h5py
import numpy as np
import pytest

from splosh import View
from splosh.mapstore import MapStore
from splosh.server import make_server, VIEWER

from conftest import write_snapshot


@pytest.fixture
def server(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    write_snapshot(str(d / "snap_000.hdf5"), seed=1)
    write_snapshot(str(d / "snap_001.hdf5"), seed=1)
    with h5py.File(d / "snap_001.hdf5", "r+") as f:                 # the same particles with doubled masses
        f["PartType0/Masses"][...] = f["PartType0/Masses"][...] * 2
    srv, state = make_server([str(d)], token="tok", store=MapStore(str(tmp_path / "store")), defaults=View(center="bh", width=6.0, npix=48))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown(); srv.server_close()


def getmap(base, **kw):
    q = "&".join(f"{k}={v}" for k, v in kw.items())
    r = urllib.request.urlopen(f"{base}/api/map?token=tok&{q}", timeout=60)
    return np.frombuffer(r.read(), dtype="<f4").reshape(48, 48), json.loads(r.headers["X-Splosh-Meta"])


def test_page_has_the_viewer_features():
    html = open(VIEWER).read()
    for needle in ('id="help"', 'id="diff"', 'id="panes"', 'id="link"', 'id="together"', 'id="units"', 'id="setA"', 'id="blink"', "unitText", "RdBu_r",
                   "X-Splosh-Meta", "/api/map", "/api/bh", "/api/lut", "/api/state", 'id="ovbox"', 'id="insp"', 'canvas class="ovl"', '"/overlay"', '"/inspect"'):
        assert needle in html, needle
    for key in ("ArrowRight", '"?"', '" "', '"d"', '"l"', '"p"', '"Escape"'):          # the keys listed in the help are handled
        assert key in html, key
    assert html.count('id="view"') == 0 and "p.cv.id = \"view\"" in html     # the first pane's canvas keeps the id the smoke test uses


def test_doubled_masses_give_exactly_twice_the_column_density(server):
    """the premise of the browser test of the difference mode: A - B = -A and A / B = 1/2 for the same particles with doubled masses"""
    a, ma = getmap(server, snap=0, field="rho", width=5, npix=48)
    b, mb = getmap(server, snap=1, field="rho", width=5, npix=48)
    assert np.array_equal(b, np.float32(2) * a) and a.max() > 0
    assert ma["extent"] == mb["extent"]


def test_units_in_the_map_metadata_for_the_hover_read_out(server):
    """the read-out prints the value in the chosen system with quantity_unit['unit'] from the metadata (also slices and derived fields)"""
    for units in ("code", "cgs", "astro"):
        for field, mode in (("rho", "proj"), ("pressure", "slice"), ("mach", "slice")):
            img, m = getmap(server, snap=0, field=field, mode=mode, z0=0.0, width=5, npix=48, units=units)
            assert m["unit_system"] == units and isinstance(m["quantity_unit"]["unit"], str) and np.isfinite(img).any()
    _, mc = getmap(server, snap=0, field="rho", width=5, npix=48, units="code")
    img, ma = getmap(server, snap=0, field="rho", width=5, npix=48, units="astro")
    assert ma["quantity_unit"]["unit"] != mc["quantity_unit"]["unit"] and "Msun" in ma["quantity_unit"]["unit"]


def test_state_gives_the_tile_grid_and_tiles_carry_the_snapshot_name(server):
    """the page addresses tiles with the base width of /api/state (default width rounded up to a power of two); a tile's metadata names the snapshot like a map's"""
    st = json.loads(urllib.request.urlopen(f"{server}/api/state?token=tok", timeout=60).read())
    assert st["tile"] == {"base": 8.0, "size": 256}                        # default width 6
    r = urllib.request.urlopen(f"{server}/api/tile?token=tok&snap=1&level=2&ix=0&iy=0&base=8&field=rho", timeout=120)
    meta = json.loads(r.headers["X-Splosh-Meta"])
    assert meta["snapshot"] == "snap_001.hdf5" and meta["n"] == 256 and meta["tile"]["extent"] == [0.0, 2.0, 0.0, 2.0]
    assert np.frombuffer(r.read(), dtype="<f4").size == 256 * 256


def test_page_has_the_run_and_alignment_controls():
    html = open(VIEWER).read()
    for needle in ('id="run"', 'id="alignmode"', 'id="tol"', 'id="offsets"', 'class="clock"', 'class="nosnap"', "/api/runs", "alignedSnap", "no snapshot at this time", "snaps: {get"):
        assert needle in html or needle.replace('"', "'") in html, needle
