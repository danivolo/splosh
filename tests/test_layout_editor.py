"""The layout editor's backend: /layout page, registry, files in the layouts directory, validation, save (minimal form), PNG preview (inline maps, products from a config, time series from scalars.csv)."""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from splosh import View, layout as lay
from splosh.mapstore import MapStore
from splosh.server import make_server


@pytest.fixture
def ed(snapdir, tmp_path):
    ldir, adir = tmp_path / "layouts", tmp_path / "analysis"
    ldir.mkdir(); adir.mkdir()
    (ldir / "one.toml").write_text('name = "one"\nncols = 2\n\n[[panel]]\nid = "m"\ntype = "map"\nfield = "rho"\nwidth = 6.0\nnpix = 48\ncenter = "com"\n\n[[panel]]\nid = "t"\ntype = "text"\ntext = "snap {snapshot}"\n')
    (adir / "scalars.csv").write_text("snapshot,time,time_myr,gas_mass\n" + "\n".join(f"snap_{i:03d},{0.1 * i},{i},{1 + i}" for i in (0, 1)) + "\n")     # snap_002 is not analysed
    cfg = tmp_path / "an.toml"
    cfg.write_text('[[map]]\nname = "dens"\ncenter = "com"\nwidth = 6.0\nnpix = 40\nfield = "rho"\n\n[[profile]]\nname = "prof"\ncenter = "com"\nquantities = ["u"]\nrmin = 0.2\nrmax = 4.0\nbins = 12\n')
    srv, state = make_server([snapdir], token="tok", store=MapStore(str(tmp_path / "store")), defaults=View(center="com", width=6.0, npix=48),
                             layouts=str(ldir), analysis=str(adir), config=str(cfg))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state, ldir
    srv.shutdown(); srv.server_close()


def call(base, path, body=None, token="tok", ctype="application/json"):
    h = {"X-Splosh-Token": token}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = ctype
    try:
        return urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=h), timeout=600)
    except urllib.error.HTTPError as e:
        return e


def test_page_registry_and_listing(ed):
    base, state, _ = ed
    page = call(base, "/layout").read().decode()
    assert "layout editor" in page and "/api/layout/preview" in page
    reg = json.load(call(base, "/api/layout/registry"))
    assert reg["layouts"] == [{"name": "one", "file": "one.toml"}] and reg["has_table"] and reg["has_config"]
    assert "map" in reg["panel_types"] and "cmap" in reg["panel_types"]["map"]
    raw = json.load(call(base, "/api/layout?name=one"))
    assert raw["raw"]["panel"][0]["id"] == "m" and "cmap" not in raw["raw"]["panel"][0]       # as written, no defaults filled in


def test_post_needs_token_json_and_a_plain_name(ed):
    base, _, ldir = ed
    assert call(base, "/api/layout/validate", {"layout": {}}, token="no").code == 403
    assert call(base, "/api/layout/validate", {"layout": {}}, ctype="text/plain").code == 415
    for bad in ("../evil", "a/b", "x.toml", "", ".hidden"):
        r = call(base, "/api/layout/save", {"name": bad, "layout": {"name": "n"}})
        assert r.code == 400, bad
    assert [p.name for p in ldir.iterdir()] == ["one.toml"]
    assert call(base, "/api/layout?name=../one").code == 400


def test_validate_reports_errors_and_places_panels(ed):
    base, _, _ = ed
    r = json.load(call(base, "/api/layout/validate", {"layout": {"name": "n", "panel": [{"id": "a", "type": "text"}, {"id": "a", "type": "text"}]}}))
    assert r["ok"] is False and "unique" in r["error"]
    r = json.load(call(base, "/api/layout/validate", {"layout": {"name": "n", "ncols": 1, "panel": [{"id": "a", "type": "text"}, {"id": "b", "type": "text"}]}}))
    assert r["ok"] and [(p["row"], p["col"]) for p in r["layout"]["panel"]] == [(0, 0), (1, 0)]


def test_save_writes_the_minimal_form_and_it_loads_back(ed):
    base, _, ldir = ed
    mine = {"name": "mine", "ncols": 3, "panel": [{"id": "m", "type": "map", "field": "u", "row": 0, "col": 0, "colspan": 2}, {"id": "x", "type": "text", "text": "hi", "x_note": 1}]}
    r = json.load(call(base, "/api/layout/save", {"name": "mine", "layout": mine}))
    assert r["file"] == "mine.toml"
    text = (ldir / "mine.toml").read_text()
    assert "cmap" not in text and "x_note" in text and 'schema = "splosh-layout-1"' in text
    back = lay.load_layout(str(ldir / "mine.toml"))
    assert back == lay.validate_layout(mine)
    assert json.load(call(base, "/api/layout/registry"))["layouts"][0]["name"] == "mine"
    r = call(base, "/api/layout/save", {"name": "bad", "layout": {"name": "bad", "ncols": 0}})
    assert r.code == 400 and not (ldir / "bad.toml").exists()                                  # an invalid layout is never written


def test_migrate_hook():
    assert lay.migrate({"schema": lay.SCHEMA, "name": "a"})["name"] == "a"
    with pytest.raises(lay.LayoutError, match="cannot be migrated"):
        lay.validate_layout({"schema": "splosh-layout-0", "name": "a"})
    lay.MIGRATIONS["splosh-layout-0"] = lambda d: dict(d, schema=lay.SCHEMA, title="migrated")
    try:
        assert lay.validate_layout({"schema": "splosh-layout-0", "name": "a"})["title"] == "migrated"
    finally:
        del lay.MIGRATIONS["splosh-layout-0"]


def _png(r):
    body = r.read()
    assert r.headers["Content-Type"] == "image/png" and body[:8] == b"\x89PNG\r\n\x1a\n", body[:200]
    return body, json.loads(r.headers["X-Splosh-Notes"])


def test_preview_inline_map_text_and_timeseries(ed):
    base, _, _ = ed
    layout = {"name": "p", "ncols": 2, "panel": [
        {"id": "m", "type": "map", "field": "rho", "width": 6.0, "npix": 48, "center": "com"},
        {"id": "ts", "type": "timeseries", "columns": ["gas_mass", "gas_mass * 2"]},
        {"id": "t", "type": "text", "text": "{snapshot}"}]}
    body, notes = _png(call(base, "/api/layout/preview", {"layout": layout, "snap": 1}))
    assert len(body) > 2000 and "global limits" in notes["m"] and "ts" not in notes
    body2, _ = _png(call(base, "/api/layout/preview", {"layout": layout, "snap": 1}))
    assert body2 == body                                                                       # deterministic (second one comes from the store)
    other, notes = _png(call(base, "/api/layout/preview", {"layout": layout, "snap": 2}))
    assert other != body and "not in scalars.csv" in notes["ts"]


def test_preview_marker_follows_the_snapshot_name_and_run_scope_has_none(snapdir, tmp_path):
    from splosh import layout_preview as lp
    from splosh.io import load_snapshot
    (tmp_path / "scalars.csv").write_text("snapshot,time,time_myr,x\nsnap_002,2,2,5\nsnap_001,1,1,4\n")
    table = lp.read_scalars_table(str(tmp_path))
    assert table["_snapshots"] == ["snap_002", "snap_001"] and list(table["x"]) == [5, 4] and list(table["index"]) == [0, 1]
    seen = []
    orig = lay.timeseries_data
    lay.timeseries_data = lambda p, t, cur=None, marker=None: seen.append(cur) or orig(p, t, cur=cur, marker=marker)
    try:
        path = f"{snapdir}/snap_001.hdf5"
        store = MapStore(str(tmp_path / "store"))
        load = lambda fields: load_snapshot(path, gas_fields=tuple(sorted(set(fields) | {"Coordinates", "Masses", "SmoothingLength", "Density"})))
        ts = {"id": "ts", "type": "timeseries", "columns": ["x"]}
        lp.preview({"name": "a", "panel": [ts]}, path, load, store, table=table)
        lp.preview({"name": "b", "scope": "run", "panel": [ts]}, path, load, store, table=table)
    finally:
        lay.timeseries_data = orig
    assert seen == [1, None]                                                  # row 1 is snap_001 (the file order of scalars.csv is not the run's)


def test_preview_caches_profiles_and_reports_a_broken_config(ed, monkeypatch):
    base, state, _ = ed
    from splosh import analyze
    calls = []
    orig = analyze._profile
    monkeypatch.setattr(analyze, "_profile", lambda snap, q: calls.append(q["name"]) or orig(snap, q))
    layout = {"name": "p", "panel": [{"id": "b", "type": "profile", "product": "prof"}]}
    for _ in range(2):
        _png(call(base, "/api/layout/preview", {"layout": layout, "snap": 0}))
    assert calls == ["prof"]
    with open(state.config, "a") as f:
        f.write("\n[[bogus]]\nname = 'x'\n")
    _, notes = _png(call(base, "/api/layout/preview", {"layout": layout, "snap": 0}))
    assert "unknown section" in notes["(config)"] and "--config" in notes["b"]


def test_preview_products_from_the_config_and_missing_ones(ed):
    base, _, _ = ed
    layout = {"name": "p", "ncols": 2, "panel": [{"id": "a", "type": "map", "product": "dens"}, {"id": "b", "type": "profile", "product": "prof"},
                                                 {"id": "c", "type": "map", "product": "nope"}, {"id": "d", "type": "timeseries", "columns": ["nocolumn"]}]}
    body, notes = _png(call(base, "/api/layout/preview", {"layout": layout, "snap": 0}))
    assert "a" in notes and "global" in notes["a"] and "b" not in notes
    assert "not a [[map]]" in notes["c"] and "unknown column" in notes["d"]


def test_preview_without_table_or_config_and_bad_requests(snapdir, tmp_path):
    srv, _ = make_server([snapdir], token="tok", store=MapStore(str(tmp_path / "s")), defaults=View(center="com", width=6.0, npix=32), layouts=str(tmp_path / "l"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        layout = {"name": "p", "panel": [{"id": "a", "type": "timeseries", "columns": ["time"]}, {"id": "b", "type": "profile", "product": "x"}]}
        _, notes = _png(call(base, "/api/layout/preview", {"layout": layout, "snap": 0}))
        assert "scalars.csv" in notes["a"] and "--config" in notes["b"]
        assert call(base, "/api/layout/preview", {"layout": layout, "snap": 99}).code == 404
        assert call(base, "/api/layout/preview", {"layout": {"name": "p", "ncols": 0}, "snap": 0}).code == 400
    finally:
        srv.shutdown(); srv.server_close()


def test_analyze_reads_a_json_config_like_the_toml_one(tmp_path):
    from splosh import analyze
    toml = tmp_path / "a.toml"
    toml.write_text('[[map]]\nname = "dens"\nwidth = 6.0\nnpix = 40\n\n[[profile]]\nname = "prof"\nquantities = ["u"]\n\n[[scalar]]\nname = "builtin"\n')
    js = tmp_path / "a.json"
    js.write_text(json.dumps(analyze.read_toml(str(toml))))
    assert analyze.read_config(str(js)) == analyze.read_config(str(toml))
    assert analyze.load_config(str(js)) == analyze.load_config(str(toml))
