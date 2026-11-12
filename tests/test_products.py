"""Snapshot products (products.py): computed once per snapshot, stored next to the run under a key (snapshot signature, version, parameters), listed when
missing, read back as the series source prod_<name>."""
import os

import numpy as np
import pytest

from splosh import plugins, products as P, series as SR
from test_server_fields import _snap


@pytest.fixture
def toy(tmp_path):
    calls = []

    def fn(path, params):
        calls.append(os.path.basename(path))
        return {"scalars": {"n_files": 1.0, "scale": params["k"] * 2.0, "bad": float("nan")}, "arrays": {"hist": np.arange(3)}}
    plugins.register_product("toy", fn, version=1, params={"k": 1.0}, defaults=["scale"])
    run = tmp_path / "run"; (run / "output").mkdir(parents=True)
    paths = [_snap(run / "output" / f"snapshot_{k:03d}.hdf5") for k in range(3)]
    yield run, paths, calls
    plugins.PRODUCTS.pop("toy", None); plugins.SERIES.pop("prod_toy", None)


def test_compute_store_and_reuse(toy):
    run, paths, calls = toy
    assert len(P.missing(str(run), ["toy"], paths)) == 3
    assert P.compute(str(run), ["toy"], paths, workers=1, log=lambda m: None) == 3
    assert P.missing(str(run), ["toy"], paths) == [] and len(calls) == 3
    r = P.load_result(str(run), "toy", paths[0])
    assert r["scalars"]["scale"] == 2.0 and r["scalars"]["bad"] is None and list(r["arrays"]["hist"]) == [0, 1, 2]
    assert P.compute(str(run), ["toy"], paths, workers=1, log=lambda m: None) == 0           # nothing new
    assert P.load_result(str(run), "toy", paths[0], {"k": 3.0}) is None                      # other parameters: not the stored result
    plugins.PRODUCTS["toy"]["version"] = 2
    assert len(P.missing(str(run), ["toy"], paths)) == 3                                      # a new version computes again


def test_product_as_series_source(toy):
    run, paths, calls = toy
    P.compute(str(run), ["toy"], paths[:2], workers=1, log=lambda m: None)
    ctx = SR.context(str(run), paths)
    assert "prod_toy" in [s["name"] for s in SR.available(ctx)]
    tab = SR.table("prod_toy", ctx, use_cache=False)
    assert len(tab) == 2 and list(tab.columns["scale"]) == [2.0, 2.0] and "2 of 3" in tab.note


def test_workers_respect_memory(tmp_path, monkeypatch):
    f = _snap(tmp_path / "s.hdf5")
    monkeypatch.setattr(P, "_mem_available", lambda: 10 * os.path.getsize(f))
    assert P.workers_for([f], max_workers=64) == 1                                            # 4x the snapshot each: room for one


def test_viewer_starts_one_products_job_per_run(toy, tmp_path, monkeypatch):
    """job_products: a job `splosh products RUN --names ...` when results are missing; the same job back while it runs; nothing when all is current;
    auto_products starts it from a thread when runs are opened."""
    from splosh.mapstore import MapStore
    from splosh.server import ViewerState
    run, paths, calls = toy
    st = ViewerState([str(run / "output")], store=MapStore(str(tmp_path / "store")), jobs_dir=str(tmp_path / "jobs"))
    submitted = []

    def fake_submit(kind, args, outputs=(), label="", write=None):
        job = {"id": f"j{len(submitted)}", "kind": kind, "label": label, "args": args, "outputs": [], "dir": str(tmp_path), "state": "running", "created": len(submitted)}
        st.jobs.jobs[job["id"]] = job
        submitted.append(job)
        return st.jobs.public(job)
    monkeypatch.setattr(st.jobs, "submit", fake_submit)
    names, n = st.products_status(0, ["toy"])
    assert names == ["toy"] and n == 3
    r = st.job_products({"run": "0", "names": ["toy"]})
    assert submitted[0]["args"][:2] == ["products", str(run)] and submitted[0]["args"][2:] == ["--names", "toy"] and len(r["jobs"]) == 1
    assert st.job_products({"run": "0", "names": ["toy"]})["note"] == "already computing" and len(submitted) == 1
    submitted[0]["state"] = "done"
    P.compute(str(run), ["toy"], paths, workers=1, log=lambda m: None)
    assert st.job_products({"run": "0", "names": ["toy"]}) == {"jobs": [], "note": "every product is current"}
    st.auto_products = True
    plugins.PRODUCTS["toy"]["version"] = 5                                                     # out of date again
    st.start_auto_products().join(30)
    assert len(submitted) == 2 and "[automatic]" in submitted[1]["label"]
