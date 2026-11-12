"""movie frames rendered in parallel processes are the same as one by one (and the limits pass too)."""
import glob
import os

import h5py
import numpy as np

from splosh import figures
from splosh.maps import View
from splosh.mapstore import MapStore


def _run(tmp_path, n=4):
    d = tmp_path / "run"; d.mkdir()
    rng = np.random.default_rng(0)
    for k in range(n):
        m = 3000
        x = rng.normal(size=(m, 3)) * (0.3 + 0.05 * k)
        with h5py.File(d / f"snapshot_{k:03d}.hdf5", "w") as f:
            h = f.create_group("Header").attrs
            h.update({"Time": 0.1 * k, "NumPart_ThisFile": np.array([m, 0, 0, 0, 0, 0]), "NumPart_Total": np.array([m, 0, 0, 0, 0, 0]), "MassTable": np.zeros(6)})
            g = f.create_group("PartType0")
            g["Coordinates"] = x; g["Velocities"] = np.cross(x, [0, 0, 1.0]); g["Masses"] = np.full(m, 1e-3); g["SmoothingLength"] = np.full(m, 0.1)
            g["Density"] = np.ones(m); g["InternalEnergy"] = np.ones(m)
    return str(d)


def test_parallel_frames_equal_sequential(tmp_path):
    run = _run(tmp_path)
    v = View(field="rho", width=3.0, npix=64, center="xyz:0,0,0")
    outs = {}
    for w in (1, 2):
        out = str(tmp_path / f"m{w}.mp4")
        r = figures.movie([run], out, v, store=MapStore(str(tmp_path / f"s{w}")), keep_frames=True, workers=w,
                          overlays=["contours:n=4"])
        fr = sorted(glob.glob(os.path.join(r["frames_dir"], "frame_*.png")))
        outs[w] = (r["limits"], [open(f, "rb").read() for f in fr])
    assert len(outs[1][1]) == 4 and outs[1][1] == outs[2][1]          # the same PNG bytes, frame by frame
    assert outs[1][0] == outs[2][0]                                   # the same colour limits


def test_movie_workers_default(monkeypatch):
    import splosh.cpus as C
    monkeypatch.setattr(C, "physical_cores", lambda: 1000)
    assert figures.movie_workers(16) == 4 and figures.movie_workers(2) == 1 and figures.movie_workers(64) == 6
    monkeypatch.setattr(C, "physical_cores", lambda: 8)          # 16 logical CPUs on 8 cores: planned from the 8 cores
    assert figures.movie_workers(16) == 2
