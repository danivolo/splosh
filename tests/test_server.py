import gzip
import os
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from splosh import View
from splosh.mapstore import MapStore
from splosh.server import make_server, find_snapshots


@pytest.fixture
def server(snapdir, tmp_path):
    srv, state = make_server([snapdir], token="s3cret", store=MapStore(str(tmp_path / "store")), defaults=View(center="bh", align="bh", width=6.0, npix=64))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    port = srv.server_address[1]
    yield f"http://127.0.0.1:{port}", state, port
    srv.shutdown(); srv.server_close()


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=60)


def test_binds_localhost_only_and_rejects_other_hosts(snapdir):
    with pytest.raises(ValueError):
        make_server([snapdir], host="0.0.0.0")
    srv, _ = make_server([snapdir], token="t")
    assert srv.server_address[0] == "127.0.0.1"
    srv.server_close()


def test_token_required(server):
    base, state, port = server
    for url in (base + "/", base + "/api/state", base + "/api/map?snap=0", base + "/api/state?token=wrong"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(url)
        assert e.value.code == 403
    assert get(base + "/api/state?token=s3cret").status == 200
    assert get(base + "/api/state", {"X-Splosh-Token": "s3cret"}).status == 200            # header form
    with pytest.raises(urllib.error.HTTPError) as e:                                            # DNS-rebinding style Host header
        get(base + "/api/state?token=s3cret", {"Host": "evil.example.com"})
    assert e.value.code == 403


def test_viewer_page_and_state(server):
    base, state, port = server
    html = get(base + "/?token=s3cret").read().decode()
    assert "<canvas" in html and "/api/map" in html and "X-Splosh-Meta" in html
    st = json.load(get(base + "/api/state?token=s3cret"))
    assert [s["name"] for s in st["snapshots"]] == ["snap_000.hdf5", "snap_001.hdf5", "snap_002.hdf5"]
    assert st["snapshots"][0]["time"] == pytest.approx(0.25) and "rho" in st["fields"] and "viridis" in st["colormaps"] and st["defaults"]["npix"] == 64


def test_map_endpoint_cache_and_bh_overlay(server):
    base, state, port = server
    u = base + "/api/map?token=s3cret&snap=1&field=rho&mode=proj&width=5&npix=64"
    r = get(u)
    meta = json.loads(r.headers["X-Splosh-Meta"])
    data = np.frombuffer(r.read(), dtype="<f4").reshape(64, 64)
    assert meta["n"] == 64 and meta["cached"] is False and data.max() > 0 and meta["extent"][1] - meta["extent"][0] == pytest.approx(5.0)
    r2 = get(u)
    meta2 = json.loads(r2.headers["X-Splosh-Meta"])
    assert meta2["cached"] is True and np.array_equal(np.frombuffer(r2.read(), dtype="<f4").reshape(64, 64), data)
    rg = get(u, {"Accept-Encoding": "gzip"})
    assert rg.headers["Content-Encoding"] == "gzip" and np.array_equal(np.frombuffer(gzip.decompress(rg.read()), dtype="<f4").reshape(64, 64), data)
    # panning by less than a quarter pixel reuses the entry (quantised request)
    assert json.loads(get(u + "&cx=0.001").headers["X-Splosh-Meta"])["cached"] is True
    # a weighted slice and the png
    s = get(base + "/api/map?token=s3cret&snap=0&field=u&mode=slice&z0=0.02&width=5&npix=64")
    assert json.loads(s.headers["X-Splosh-Meta"])["view"]["mode"] == "slice"
    png = get(base + "/api/png?token=s3cret&snap=0&field=rho&width=5&npix=64&cmap=magma").read()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    bh = json.load(get(base + "/api/bh?token=s3cret&snap=2&width=5&npix=64"))
    ids = {m["id"]: m for m in bh["markers"]}
    assert set(ids) == {3, 7} and abs(ids[3]["y"]) < 1e-9 and ids[7]["x"] > ids[3]["x"] and bh["time"] == pytest.approx(0.25)     # aligned: 3 -> 7 along +x


def test_bad_requests(server):
    base, state, port = server
    for bad, code in (("/api/map?token=s3cret&snap=9", 404), ("/api/map?token=s3cret&snap=0&axes=zz", 400), ("/api/map?token=s3cret&snap=0&npix=1", 400),
                      ("/api/map?token=s3cret&snap=0&field=nonexistent", 400), ("/nope?token=s3cret", 404)):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(base + bad)
        assert e.value.code == code, bad


def test_find_snapshots_natural_order(tmp_path):
    for n in (2, 10, 1):
        (tmp_path / f"snapshot_{n}.hdf5").write_bytes(b"")
    assert [p.rsplit("_", 1)[1] for p in find_snapshots([str(tmp_path)])] == ["1.hdf5", "2.hdf5", "10.hdf5"]


def test_threads_endpoint_and_state(server):
    url, state, _ = server
    st = json.load(get(f"{url}/api/state?token=s3cret"))
    assert st["threads"] == state.threads and st["max_threads"] == state.max_threads >= 1
    def post(n):
        req = urllib.request.Request(f"{url}/api/threads", data=json.dumps({"threads": n}).encode(), headers={"X-Splosh-Token": "s3cret", "Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))
    assert post(1)["threads"] == 1 and state.threads == 1
    assert post(10 ** 6)["threads"] == state.max_threads                        # capped at the pool
    assert post(-3)["threads"] == 1
    import numba
    with state.work():
        assert numba.get_num_threads() == 1                                    # applied in the thread that renders


def test_watchdog_stops_on_idle_and_on_a_lost_parent(snapdir, tmp_path, monkeypatch):
    import os
    from splosh import server as S
    for kind in ("idle", "orphan"):
        srv, state = make_server([snapdir], token="t", store=MapStore(str(tmp_path / "s")))
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        if kind == "idle":
            state.last_request -= 3600
            w = S.watchdog(srv, state, idle_minutes=1, every=0.05)
        else:
            w = S.watchdog(srv, state, orphan=True, every=0.05)
            monkeypatch.setattr(os, "getppid", lambda: -1)
        w.join(20)
        t.join(20)
        monkeypatch.undo()
        assert not w.is_alive() and not t.is_alive(), kind
        srv.server_close()


def test_start_empty_browse_and_open_folders(tmp_path):
    import shutil
    from conftest import write_snapshot
    root = tmp_path / "home"
    (root / "proj" / "runA" / "output").mkdir(parents=True)
    (root / "proj" / "runB").mkdir(parents=True)
    (root / "proj" / ".hidden").mkdir()
    (root / "empty").mkdir()
    for k in range(2):
        write_snapshot(str(root / "proj" / "runA" / "output" / f"snapshot_{k:03d}.hdf5"), n_gas=300, seed=k)
    write_snapshot(str(root / "proj" / "runB" / "snap_000.hdf5"), n_gas=300, seed=5)
    os.symlink(str(tmp_path), str(root / "proj" / "loop"))                      # symlinks are not followed by the deep search
    srv, state = make_server([], token="t", store=MapStore(str(tmp_path / "s")), defaults=View(center="com", width=6.0, npix=32), root=str(root))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    H = {"X-Splosh-Token": "t"}
    try:
        st = json.load(get(f"{base}/api/state", H))
        assert st["runs"] == [] and st["snapshots"] == [] and st["root"] == os.path.realpath(root)
        with pytest.raises(urllib.error.HTTPError) as e:
            get(f"{base}/api/map?snap=0", H)
        assert e.value.code == 404 and b"no runs open" in e.value.read()
        top = json.load(get(f"{base}/api/browse?dir=", H))
        assert [x["name"] for x in top["entries"]] == ["empty", "proj"] and top["parent"] is None
        proj = json.load(get(f"{base}/api/browse?dir=proj", H))
        assert [(x["name"], x["n"]) for x in proj["entries"]] == [("loop", 0), ("runA", 2), ("runB", 1)] and proj["parent"] == ""   # runA opens its output/
        assert [x["output"] for x in proj["entries"]] == [False, True, False]
        deep = json.load(get(f"{base}/api/browse?dir=&deep=1", H))
        assert [(x["path"], x["n"], x["output"]) for x in deep["entries"]] == [("proj/runA", 2, True), ("proj/runB", 1, False)]   # the run, not its output/
        assert deep["entries"][0]["desc"].startswith("2 snapshots in output/, t = ")
        inside = json.load(get(f"{base}/api/browse?dir=proj/runA&deep=1", H))               # inside the run: its output/ is the only folder below
        assert [(x["path"], x["n"]) for x in inside["entries"]] == [("proj/runA/output", 2)] and inside["n"] == 2 and inside["output"]
        for bad in ("..", "../..", "/etc", "proj/loop"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(f"{base}/api/browse?dir={bad}", H)
            assert e.value.code == 400, bad
        def post(body):
            req = urllib.request.Request(f"{base}/api/open", data=json.dumps(body).encode(), headers={**H, "Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=60)
        r = json.load(post({"paths": ["proj/runA"]}))                         # a run directory: its output/ is found as on the command line
        assert r["runs"] == [{"label": "runA", "n": 2}]
        assert json.load(post({"paths": ["proj/runB"], "add": True}))["runs"][1] == {"label": "runB", "n": 1}
        assert get(f"{base}/api/map?snap=1&run=runA", H).status == 200
        for bad in ({"paths": ["empty"]}, {"paths": ["../x"]}, {"paths": []}):
            with pytest.raises(urllib.error.HTTPError) as e:
                post(bad)
            assert e.value.code == 400, bad
        assert [r.label for r in state.runs] == ["runA", "runB"]           # a failed open changes nothing
    finally:
        srv.shutdown(); srv.server_close()
