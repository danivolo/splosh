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
