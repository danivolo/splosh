"""/api/state lists only the fields the open runs can show; a run's own analysis table feeds the editor's time series; NaN times are sent as null."""
import json
import os

import h5py
import numpy as np

from splosh.server import ViewerState, available_fields, _finite


def _snap(path, mhd=False, divb=False, n=50):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_group("Header").attrs["Time"] = 0.0
        g = f.create_group("PartType0")
        for k, shape in (("Coordinates", (n, 3)), ("Velocities", (n, 3)), ("Masses", (n,)), ("SmoothingLength", (n,)), ("Density", (n,)), ("InternalEnergy", (n,))):
            g[k] = rng.random(shape)
        if mhd:
            g["MagneticField"] = rng.random((n, 3))
        if divb:
            g["DivergenceOfMagneticField"] = rng.random(n)
    return str(path)


def test_fields_follow_the_snapshot(tmp_path):
    (tmp_path / "hydro" / "output").mkdir(parents=True); (tmp_path / "mhd").mkdir()
    h = _snap(tmp_path / "hydro" / "output" / "snapshot_000.hdf5")
    st = ViewerState([h], store=str(tmp_path / "store"))
    f = available_fields(st)
    assert "rho" in f and "cs" in f and "mach" in f
    assert not {"B", "beta", "divB", "T"} & set(f)                 # no MagneticField, no ElectronAbundance
    m = _snap(tmp_path / "mhd" / "snapshot_000.hdf5", mhd=True)
    f = available_fields(ViewerState([m], store=str(tmp_path / "store")))
    assert "B" in f and "beta" in f and "divB" not in f
    f = available_fields(ViewerState([_snap(tmp_path / "mhd" / "snapshot_001.hdf5", mhd=True, divb=True)], store=str(tmp_path / "store")))
    assert "divB" in f


def test_run_analysis_table_found_next_to_the_run(tmp_path):
    run = tmp_path / "run"; (run / "output").mkdir(parents=True)
    st = ViewerState([_snap(run / "output" / "snapshot_000.hdf5")], store=str(tmp_path / "store"))
    assert st.run_analysis(0) is None
    (run / "analysis").mkdir(); (run / "analysis" / "scalars.csv").write_text("snapshot,t\nsnapshot_000,0\n")
    assert st.run_analysis(0) == str(run / "analysis")
    d = run / "analysis_viewer" / "a"; d.mkdir(parents=True); (d / "scalars.csv").write_text("snapshot,t\n")
    os.utime(d / "scalars.csv", (2e9, 2e9))
    assert st.run_analysis(0) == str(d)                                # the newest table wins


def test_snapshot_datasets_are_fields(tmp_path):
    from splosh.maps import View, required_gas_fields
    p = str(tmp_path / "snapshot_000.hdf5"); _snap(p, mhd=True)
    n = 50
    with h5py.File(p, "a") as f:
        g = f["PartType0"]
        g["WindFraction"] = np.linspace(0, 1, n); g["Metallicity"] = np.tile(np.arange(11.0), (n, 1)); g["DivBcleaningFunctionGradPhi"] = np.tile([3.0, 4.0, 0.0], (n, 1))
        g["ParticleIDs"] = np.arange(n)
    f = available_fields(ViewerState([p], store=str(tmp_path / "store")))
    assert {"WindFraction", "Metallicity_0", "Metallicity_10", "DivBcleaningFunctionGradPhi", "Pmag"} <= set(f)
    assert "ParticleIDs" not in f and "Metallicity" not in f and "MagneticField" not in f
    assert required_gas_fields(View(field="WindFraction")) >= {"WindFraction"}
    from splosh.io import load_snapshot
    from splosh.frames import make_frame
    from splosh.maps import FieldContext, _field_values
    def val(name):
        v = View(field=name, center="xyz:0,0,0")
        s = load_snapshot(p, gas_fields=tuple(sorted(required_gas_fields(v))))
        return _field_values(name, FieldContext(s, make_frame(s, v.center), v))
    assert np.allclose(val("WindFraction"), np.linspace(0, 1, n))
    assert np.allclose(val("Metallicity_3"), 3.0)
    assert np.allclose(val("DivBcleaningFunctionGradPhi"), 5.0)          # magnitude of a 3-vector
    # beta = pressure / Pmag
    assert np.allclose(val("beta") * val("Pmag"), val("pressure"), rtol=1e-5)


def test_nan_time_is_null():
    assert _finite(float("nan")) is None and _finite(1.5) == 1.5
    assert json.loads(json.dumps({"t": _finite(float("inf"))})) == {"t": None}


def test_reversed_colour_maps():
    from splosh import colormaps as c
    assert (c.lut("viridis_r") == c.lut("viridis")[::-1]).all() and (c.lut("RdBu") == c.lut("RdBu_r")[::-1]).all()
    assert c.known("magma_r") and not c.known("nope")


def test_recent_folder_sets(tmp_path):
    from splosh import server
    a, b = tmp_path / "runA", tmp_path / "runB"
    for d in (a, b):
        d.mkdir(); _snap(d / "snapshot_000.hdf5")
    st = ViewerState([], store=str(tmp_path / "store"), root=str(tmp_path))
    st.open_runs(["runA"]); st.open_runs(["runB"]); st.open_runs(["runA"], add=True)
    rs = server.recent_sets()
    assert [len(r["paths"]) for r in rs] == [2, 1] and rs[0]["paths"][0].endswith("runB") and rs[1]["paths"][0].endswith("runA")
