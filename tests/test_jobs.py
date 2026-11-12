"""Long jobs (jobs.py, prefetch.py, the server's /api/jobs): local runs, the free-node attempt with a fake srun (start / refuse), the fallback, cancel, outputs."""
import json
import os
import stat
import threading
import time
import urllib.error
import urllib.request

import pytest

from splosh import View
from splosh.jobs import JobManager
from splosh.mapstore import MapStore, snapshot_hash, view_key
from splosh.server import make_server

SRUN_OK = """#!/bin/bash
# fake srun: skip the options up to the command, run it with a CPU count like a job step
while [ "$1" != "bash" ]; do shift; done
SLURM_CPUS_PER_TASK=2 exec "$@"
"""
SRUN_BUSY = """#!/bin/bash
echo "srun: error: Unable to allocate resources: Immediate execution impossible, insufficient priority" >&2; exit 1
"""
SRUN_SLOW = """#!/bin/bash
echo "srun: job 7 queued and waiting for resources" >&2; sleep 60
"""


def fake_srun(tmp_path, body, monkeypatch):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    f = d / "srun"
    f.write_text(body)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{d}{os.pathsep}{os.environ['PATH']}")


def wait(mgr, jid, t=180):
    t0 = time.time()
    while time.time() - t0 < t:
        j = next(x for x in mgr.list() if x["id"] == jid)
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise AssertionError(f"job {jid} still {j['state']}: {j['log_tail']}")


def plan_for(snapdir):
    files = sorted(os.path.join(snapdir, f) for f in os.listdir(snapdir))
    view = View(center="com", width=6.0, npix=32).as_dict()
    return files, view, json.dumps([{"files": files, "view": view}])


@pytest.mark.parametrize("srun", [None, SRUN_OK, SRUN_BUSY])
def test_prefetch_job_local_on_a_node_or_falling_back(snapdir, tmp_path, monkeypatch, srun):
    if srun:
        fake_srun(tmp_path, srun, monkeypatch)
    store = MapStore(str(tmp_path / "store"))
    files, view, plan = plan_for(snapdir)
    mgr = JobManager(str(tmp_path / "jobs"), slurm="-c 2 -t 5" if srun else None, wait=1, local_threads=lambda: 2)
    j = mgr.submit("prefetch", ["prefetch", "--plan", "{dir}/plan.json", "--store", store.root, "--procs", "auto"], write={"plan.json": plan})
    j = wait(mgr, j["id"])
    assert j["state"] == "done", j["log_tail"]
    assert j["progress"] == [3, 3]
    if srun is SRUN_OK:
        assert j["where"] == os.uname().nodename                                 # the hostname the 'node' printed
    else:
        assert j["where"].startswith("this machine") and "2 threads" in j["where"]
        if srun is SRUN_BUSY:
            assert "no free node" in j["note"]
    v = View(**view).validate()
    assert all(store.get(view_key(snapshot_hash(f), v)) is not None for f in files)
    again = wait(mgr, mgr.submit("prefetch", ["prefetch", "--plan", "{dir}/plan.json", "--store", store.root], write={"plan.json": plan})["id"])
    assert "3 were cached" in " ".join(again["log_tail"])


def test_no_srun_falls_back_and_cancel_stops_a_waiting_job(snapdir, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")                                  # no srun at all
    mgr = JobManager(str(tmp_path / "jobs"), slurm="-c 2", wait=1, local_threads=lambda: 1)
    _, _, plan = plan_for(snapdir)
    j = wait(mgr, mgr.submit("prefetch", ["prefetch", "--plan", "{dir}/plan.json", "--store", str(tmp_path / "s")], write={"plan.json": plan})["id"])
    assert j["state"] == "done" and j["where"].startswith("this machine")
    fake_srun(tmp_path, SRUN_SLOW, monkeypatch)
    mgr2 = JobManager(str(tmp_path / "jobs2"), slurm="-c 2", wait=30)
    k = mgr2.submit("prefetch", ["prefetch", "--plan", "x"])
    time.sleep(1.0)
    mgr2.cancel(k["id"])
    k = wait(mgr2, k["id"], t=20)
    assert k["state"] == "cancelled"
    with pytest.raises(LookupError):
        mgr2.path(k["id"], "../../etc/passwd")


def test_movie_and_prefetch_through_the_server(snapdir, tmp_path):
    srv, state = make_server([snapdir], token="t", store=MapStore(str(tmp_path / "store")), defaults=View(center="com", width=6.0, npix=32), jobs_dir=str(tmp_path / "jobs"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    H = {"X-Splosh-Token": "t", "Content-Type": "application/json"}
    post = lambda path, body: json.load(urllib.request.urlopen(urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=H), timeout=60))
    pane = {"field": "rho", "mode": "proj", "axes": "xy", "center": "com", "align": "none", "npix": 32, "width": 6.0, "cx": 0, "cy": 0, "z0": 0, "units": "code", "snap": 1}
    try:
        m = post("/api/jobs", {"kind": "movie", "pane": pane, "fps": 4, "limits": "fixed", "vmin": 1e-4, "vmax": 1.0, "cmap": "magma", "scale": "log"})
        p = post("/api/jobs", {"kind": "prefetch", "panes": [pane, dict(pane, field="u")]})
        m, p = wait(state.jobs, m["id"]), wait(state.jobs, p["id"])
        assert p["state"] == "done" and p["progress"] == [3, 3] and "6 map(s)" in " ".join(p["log_tail"])
        assert m["state"] == "done", m["log_tail"]
        side = json.loads(urllib.request.urlopen(urllib.request.Request(f"{base}/api/jobs/file?id={m['id']}&name=movie.json", headers=H), timeout=60).read())
        assert side["colormap"] == "magma" and side["limits"]["vmin"] == 1e-4 and len(side["snapshots"]) == 3
        img, meta = state.get_map(1, state.view_from_query({k: [str(v)] for k, v in dict(pane, field="u").items()}))
        assert meta["cached"]                                                     # the prefetch filled the viewer's own cache keys
        listing = json.load(urllib.request.urlopen(urllib.request.Request(f"{base}/api/jobs", headers=H), timeout=60))
        assert [j["kind"] for j in listing["jobs"]] == ["prefetch", "movie"] and listing["slurm"] == ""
        for bad in ({"kind": "nope"}, {"kind": "movie", "pane": pane, "from": 2, "to": 2}):
            with pytest.raises(urllib.error.HTTPError) as e:
                post("/api/jobs", bad)
            assert e.value.code == 400
    finally:
        srv.shutdown(); srv.server_close()


def test_analyze_these_panes_and_the_editor_preview_of_its_layout(snapdir, tmp_path):
    srv, state = make_server([snapdir], token="t", store=MapStore(str(tmp_path / "store")), defaults=View(center="com", width=6.0, npix=32), jobs_dir=str(tmp_path / "jobs"),
                             layouts=str(tmp_path / "layouts"), root=str(tmp_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    H = {"X-Splosh-Token": "t", "Content-Type": "application/json"}
    post = lambda path, body: urllib.request.urlopen(urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=H), timeout=120)
    pane = {"field": "rho", "mode": "proj", "axes": "xy", "center": "com", "align": "none", "npix": 32, "width": 6.0, "cx": 0, "cy": 0, "z0": 0, "units": "code", "snap": 0}
    try:
        r = json.load(post("/api/jobs", {"kind": "analyze", "name": "viewer", "panes": [dict(pane, cmap="magma", scale="log"), dict(pane, field="u", scale="linear")]}))
        (job,) = r["jobs"]
        j = wait(state.jobs, job["id"], t=300)
        assert j["state"] == "done", j["log_tail"]
        out = os.path.join(snapdir, "analysis_viewer", "viewer")
        assert j["editor"]["analysis"] == out and set(j["files"]) >= {"out/scalars.csv", "out/viewer/snap_002.png", "config.json"}
        assert os.path.isfile(os.path.join(out, "p1_rho", "snap_002.png")) and "gas_mass" in open(os.path.join(out, "scalars.csv")).readline()
        assert j["progress"] and j["progress"][0] == j["progress"][1]
        ed = tmp_path / "layouts" / "run-viewer.json"
        assert json.load(open(ed))["x_viewer_job"]
        lay = json.load(urllib.request.urlopen(urllib.request.Request(f"{base}/api/layout?name=run-viewer", headers=H), timeout=60))["raw"]
        r = post("/api/layout/preview", {"layout": lay, "snap": 2, "analysis": j["editor"]["analysis"], "config": j["editor"]["config"]})
        notes = json.loads(r.headers["X-Splosh-Notes"])
        assert r.headers["Content-Type"] == "image/png" and not any("not a [[" in v or "scalars" in v for v in notes.values()), notes
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/api/layout/preview", {"layout": lay, "snap": 0, "config": "/etc/passwd"})
        assert e.value.code == 400
        again = wait(state.jobs, json.load(post("/api/jobs", {"kind": "analyze", "panes": [pane]}))["jobs"][0]["id"], t=300)
        assert again["state"] == "done"                                          # a second job on the same folder (incremental)
        m = json.load(post("/api/jobs", {"kind": "movie", "pane": pane, "fps": 4, "overlays": ["particles:bh"]}))
        m = wait(state.jobs, m["id"])
        side = json.loads(urllib.request.urlopen(urllib.request.Request(f"{base}/api/jobs/file?id={m['id']}&name=movie.json", headers=H), timeout=60).read())
        assert m["state"] == "done" and "particles:bh" in json.dumps(side)
    finally:
        srv.shutdown(); srv.server_close()
