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
    for key in ("ArrowRight", '"?"', '" "', '"d"', '"l"', '"p"', '"Escape"', '"q"', "e.button === 3", '"contextmenu"', 'id="ctx"', 'id="snapghost"', "KEYMAP"):          # the keys listed in the help are handled
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


# ----------------------------------------------------------------------------- the radial / spectrum panes (GET /api/series?radial= | spectrum=)

def _family_run(tmp_path):
    """a run (one snapshot; P0 = 1 without a bound binary) with an analysis table of 64 rows: a family sig_<R> = R^-1 (x (1 + t / T)), dm_<R> = 2 sig_<R>, an oscillation at 0.5 / P0."""
    from splosh.server import ViewerState
    d = tmp_path / "frun"
    (d / "output").mkdir(parents=True)
    write_snapshot(str(d / "output" / "snapshot_000.hdf5"), seed=1)
    st = ViewerState([str(d / "output")], store=MapStore(str(tmp_path / "store")), jobs_dir=str(tmp_path / "jobs"))
    P = st.series(0)["period"] or 1.0                                               # no bound binary: x in code time, frequencies in 1 / code time
    t = np.linspace(0, 20 * P, 64)
    R = (0.5, 1.0, 2.0, 4.0, 8.0)
    an = d / "analysis"                                                              # the run's own analysis table
    an.mkdir()
    head = ["snapshot", "time"] + [f"sig_{r:g}" for r in R] + [f"dm_{r:g}" for r in R] + ["osc"]
    rows = [[f"s{k}", f"{tk:.17g}"] + [f"{(1 + tk / t[-1]) / r:.17g}" for r in R] + [f"{2 * (1 + tk / t[-1]) / r:.17g}" for r in R] + [f"{np.sin(2 * np.pi * 0.5 * tk / P):.17g}"]
            for k, tk in enumerate(t)]
    (an / "scalars.csv").write_text(",".join(head) + "\n" + "\n".join(",".join(r) for r in rows) + "\n")
    return st, P, R


def test_series_endpoint_radial_and_spectrum_panels(tmp_path):
    st, P, R = _family_run(tmp_path)
    q = lambda kind, **kw: st.series(0, {kind: [json.dumps(kw)]})
    r = q("radial", source="snapshots", prefix="sig_", windows=2)
    json.dumps(r, allow_nan=False)                                                   # plain JSON (NaN as null)
    assert r["kind"] == "radial" and r["xlog"] and len(r["curves"]) == 2 and r["curves"][0]["x"] == pytest.approx(R)
    early, late = (np.array(c["y"]) for c in r["curves"])
    assert np.all(late > early) and early * np.array(R) == pytest.approx(np.full(5, early[0] * R[0]))     # ~ 1/R, later windows higher
    b = r["styles"][0]["band"]
    assert len(b) == 2 and np.all(np.array(b[0]) <= early + 1e-12) and np.all(np.array(b[1]) >= early - 1e-12)
    assert r["styles"][0]["color"].startswith("#")
    s = q("radial", source="snapshots", prefix="sig_", slope=True, windows="last:5")
    assert len(s["curves"]) == 1 and np.allclose(s["curves"][0]["y"], -1.0, atol=1e-6) and "band" not in s["styles"][0]
    d = q("radial", source="snapshots", prefixes=["dm_"], divide="sig_", windows=[[0, 10], [10, 20]])
    assert len(d["curves"]) == 2 and np.allclose(d["curves"][0]["y"], 2.0)
    sp = q("spectrum", source="snapshots", columns=["osc"], fmax=1.5)
    f, pw = np.array(sp["curves"][0]["x"]), np.array(sp["curves"][0]["y"])
    assert sp["xlog"] and sp["ylog"] and abs(f[np.argmax(pw)] - 0.5) < 0.03 and f.max() == pytest.approx(1.5)
    for bad in ('{"source": "snapshots"}', "[1]", "nope", '{"source": "snapshots", "prefix": "zz_"}', '{"source": "snapshots", "prefix": "sig_", "windows": 99}'):
        with pytest.raises((ValueError, KeyError)):                                  # both are a 400 for the page
            st.series(0, {"radial": [bad]})


def test_series_endpoint_http_errors_are_400(server):
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server + "/api/series?token=tok&radial=" + urllib.request.quote('{"source": "snapshots"}'), timeout=60)
    assert e.value.code == 400 and b"prefix" in e.value.read()


def test_page_has_the_radial_and_spectrum_panes_and_the_zoom_keys():
    html = open(VIEWER).read()
    for needle in ('<option value="radial">', '<option value="spectrum">', "function drawRS", "function rsPanel", "zoomKeys", "restoreZoom", "xlim", "ylim", "mapBoxZoom", '"view.box"',
                   'id="ov_lv"', 'id="ov_lB"', "lic:v,", 'it.kind === "image"'):
        assert needle in html, needle
