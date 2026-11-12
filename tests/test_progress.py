"""The progress record behind GET /api/progress: block reads give the same arrays and count every byte; the time-based estimate stays below 1."""
import json
import time

import h5py
import numpy as np

from splosh import io, progress


def _snap(path, n=1000):
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        f.create_group("Header").attrs["Time"] = 0.25
        g = f.create_group("PartType0")
        for k, shape in (("Coordinates", (n, 3)), ("Velocities", (n, 3)), ("Masses", (n,)), ("SmoothingLength", (n,)), ("Density", (n,)), ("InternalEnergy", (n,))):
            g[k] = rng.random(shape)
    return str(path)


def test_block_reads_are_exact_and_counted(tmp_path, monkeypatch):
    p = _snap(tmp_path / "s.hdf5")
    whole = io.load_snapshot(p)
    monkeypatch.setattr(io, "BLOCK_BYTES", 1000)                 # many blocks per dataset
    progress.begin("test")
    try:
        blocks = io.load_snapshot(p)
        cur = progress.current()
    finally:
        progress.end()
    for k in ("Coordinates", "Velocities", "Masses", "SmoothingLength", "Density", "InternalEnergy"):
        assert np.array_equal(whole[k], blocks[k])
    assert cur["busy"] and cur["bytes"][0] == cur["bytes"][1] == 1000 * 8 * (3 + 3 + 4) and cur["fraction"] == 1.0
    rows = io.load_snapshot(p, rows=[(0, 10), (500, 520)])
    assert np.array_equal(rows["Masses"], np.concatenate([whole["Masses"][:10], whole["Masses"][500:520]]))
    assert progress.current() == {"busy": False}


def test_time_based_estimate_and_learning():
    progress.learn("unit-test", 1000, 2.0)
    assert abs(progress.expect("unit-test", 500) - 1.0) < 1e-12
    progress.begin("x")
    progress.stage("rendering", 1e-3)
    time.sleep(0.01)
    c = progress.current()
    progress.end()
    assert c["stage"] == "rendering" and 0.9 < c["fraction"] <= 0.97      # capped: compiled splatting has no real progress
    json.dumps(c)
