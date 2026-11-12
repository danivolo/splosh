"""Time series of a run (series.py, GET /api/series): sources (snapshot scalars, GIZMO energy.txt and blackholes.txt), the cache, min/max thinning that keeps spikes
and gaps, the running mean, and the background job that builds the snapshot scalars."""
import json
import os

import numpy as np

from splosh import series as SR
from splosh.mapstore import MapStore
from splosh.server import ViewerState
from test_server_fields import _snap


def _run(tmp_path, n=3):
    run = tmp_path / "run"; (run / "output").mkdir(parents=True)
    return run, [_snap(run / "output" / f"snapshot_{k:03d}.hdf5") for k in range(n)]


def test_sources_listed_and_snapshot_scalars(tmp_path):
    run, paths = _run(tmp_path)
    st = ViewerState(paths, store=MapStore(str(tmp_path / "store")))
    assert [s["name"] for s in st.series(0)["sources"]] == []
    assert st.series(0, {"source": ["snapshots"]})["missing"] is True
    d = run / "analysis_viewer" / "series"; d.mkdir(parents=True)
    (d / "scalars.csv").write_text("snapshot,time,time_myr,bh_sep,empty\nsnapshot_000,0,0,1.5,nan\nsnapshot_002,2,0.2,nan,nan\n")
    assert "snapshots" in [s["name"] for s in st.series(0)["sources"]]
    desc = st.series(0, {"source": ["snapshots"]})
    assert list(desc["columns"]) == ["bh_sep"] and desc["t_range"] == [0.0, 2.0]          # an all-NaN column is not offered
    cv = st.series(0, {"source": ["snapshots"], "columns": ["bh_sep"]})["curves"]
    assert cv[0]["t"] == [0.0, 2.0] and cv[0]["y"] == [1.5, None]                          # NaN -> null (valid JSON)
    json.dumps(cv, allow_nan=False)


def test_energy_and_blackholes_logs(tmp_path):
    run, paths = _run(tmp_path, 1)
    rows = []
    for k, t in enumerate([0.0, 0.1, 0.2, 0.15, 0.2, 0.3]):                                  # a restart from t = 0.1 rewrote 0.15 .. 0.2
        rows.append(" ".join(str(v) for v in [t, 10 + k, 0, 5 + k] + [0] * 18 + ([1, 0, 0, 0, 0, 2] if k % 2 else [])))
    (run / "output" / "energy.txt").write_text("\n".join(rows) + "\n")
    (run / "output" / "blackholes.txt").write_text("".join(f"{t} 2 4307.9 99.7 0.0237 4328.6 1.07\n" for t in (1e-3, 2e-3, 3e-3)))
    ctx = SR.context(str(run), paths)
    assert {s["name"] for s in SR.available(ctx)} >= {"energy", "blackholes"}
    e = SR.table("energy", ctx)
    assert list(e.time) == [0.0, 0.1, 0.15, 0.2, 0.3]                                       # the stale 0.2 row of before the restart is gone
    assert "minus_E_pot" not in e.columns and e.note                                      # E_pot = 0: GIZMO did not compute it
    np.testing.assert_allclose(e.columns["E_tot"], e.columns["E_int"] + e.columns["E_kin"])
    assert e.columns["dE_rel"][0] == 0.0
    b = SR.table("blackholes", ctx)
    assert len(b) == 3 and b.columns["mdot_msun_yr"][0] == 0.0237
    assert os.path.isfile(os.path.join(SR.cache_dir(ctx), "series_energy.npz"))
    SR._memo.clear()
    assert len(SR.table("energy", ctx)) == 5                                               # from the npz cache


def test_thinning_keeps_spikes_and_gaps():
    t = np.linspace(0, 1, 100001)
    y = np.sin(20 * t); y[50000] = 40.0; y[70000:72000] = np.nan
    tt, yy = SR.thin_minmax(t, y, 500)
    assert len(tt) <= 1100 and np.nanmax(yy) == 40.0
    assert np.isnan(yy[(tt > 0.705) & (tt < 0.715)]).any()                                # the gap is still a gap
    assert np.all(np.diff(tt) >= 0)


def test_running_mean_matches_a_direct_window():
    t = np.linspace(0, 10, 1001); y = t ** 2
    ts, ym = SR.running_mean(t, y, 1.0)
    i = 500
    sel = (t >= t[i] - 0.5) & (t <= t[i] + 0.5)
    assert abs(ym[i] - y[sel].mean()) < 1e-9


def test_series_job_runs_builtin_scalars_into_the_run(tmp_path, monkeypatch):
    run, paths = _run(tmp_path, 1)
    st = ViewerState(paths, store=MapStore(str(tmp_path / "store")))
    got = {}

    def submit(kind, argv, outputs=(), label="", write=None):
        got.update(kind=kind, argv=argv, write=write)
        jd = tmp_path / "job"; jd.mkdir()
        return {"id": "j1", "dir": str(jd)}
    monkeypatch.setattr(st.jobs, "submit", submit)
    r = st.job_series({"run": "0"})
    out = str(run / "analysis_viewer" / "series")
    assert got["kind"] == "analyze" and got["argv"][1] == str(run) and got["argv"][got["argv"].index("--out") + 1] == out
    assert json.loads(got["write"]["config.json"]) == {"scalar": [{"name": "builtin"}]}
    assert os.path.realpath(os.path.join(r["jobs"][0]["dir"], "out")) == os.path.realpath(out)


def test_layout_timeseries_panel_from_a_source(tmp_path):
    """a layout timeseries panel with source = energy: curves, the t/P0-less fallback axis, the current-time marker, hlines, both renderers."""
    from splosh import layout as lay, layout_preview as lp
    run, paths = _run(tmp_path, 2)
    (run / "output" / "energy.txt").write_text("".join(f"{t} {10 + t} 0 {5 - t} " + " ".join(["0"] * 18) + "\n" for t in (0.0, 0.5, 1.0, 1.5)))
    L = {"schema": lay.SCHEMA, "name": "t", "ncols": 1, "panel": [
        {"id": "e", "type": "timeseries", "source": "energy", "columns": ["E_int", "E_kin"], "twin": ["dE_rel"], "hlines": [10], "xaxis": "time", "yscale": "lin"}]}
    v = lay.validate_layout(L)
    assert v["panel"][0]["twin"] == ["dE_rel"] and v["panel"][0]["source"] == "energy"
    from splosh.io import load_snapshot
    _, tiles, notes, header = lp.tiles_for(L, paths[1], lambda f: load_snapshot(paths[1], gas_fields=f), MapStore(str(tmp_path / "store")))
    t = tiles["e"]
    assert [c[0] for c in t["curves"]] == ["internal energy (gas)", "kinetic energy (all types)", "|E - E0| / |E0| (not conserved with cooling, accretion, feedback)"]
    assert [s["twin"] for s in t["styles"]] == [False, False, True] and t["hlines"] == [10.0] and t["xnow"] == 0.0
    from splosh import layout_mpl
    assert layout_mpl.render_bytes(v, tiles, header, fmt="png")[:4] == b"\x89PNG"
    rgb = lay.render_layout(v, tiles, header)
    assert rgb.ndim == 3


def test_layout_rejects_expressions_for_sources():
    from splosh import layout as lay
    import pytest
    with pytest.raises(lay.LayoutError):
        lay.validate_layout({"schema": lay.SCHEMA, "name": "t", "panel": [{"id": "e", "type": "timeseries", "source": "energy", "columns": ["E_int / 2"]}]})


def test_profile_panel_of_its_own_map(tmp_path):
    """a profile panel without `product`: the azimuthal statistics of the map of its own view (the viewer's radial-profile pane), both renderers."""
    from splosh import layout as lay, layout_preview as lp, layout_mpl
    from splosh.io import load_snapshot
    f = _snap(tmp_path / "snapshot_000.hdf5", n=2000)
    L = {"schema": lay.SCHEMA, "name": "t", "ncols": 2, "panel": [{"id": "p", "type": "profile", "field": "rho", "center": "com", "width": 1.0, "npix": 64, "bins": 20}]}
    lo, tiles, notes, h = lp.tiles_for(L, f, lambda g: load_snapshot(f, gas_fields=g), MapStore(str(tmp_path / "store")))
    names = [c[0] for c in tiles["p"]["curves"]]
    assert names[0] == "rho (azimuthal mean)" and len(names) == 3
    r = np.asarray(tiles["p"]["curves"][0][1])
    assert len(r) > 5 and np.all(np.diff(r) > 0)
    assert layout_mpl.render_bytes(lo, tiles, h)[:4] == b"\x89PNG" and lay.render_layout(lo, tiles, h).ndim == 3
    import pytest
    with pytest.raises(lay.LayoutError):
        lay.validate_layout({"schema": lay.SCHEMA, "name": "t", "panel": [{"id": "p", "type": "profile", "mode": "bogus"}]})


def test_run_directory_with_initial_conditions_next_to_output(tmp_path):
    """a run directory holding its initial conditions (a snapshot-like file) next to output/: the run's snapshots are those of output/, also for the
    context of a layout figure (the binary period P0 is read from output/'s first snapshot, not from the initial conditions)."""
    from splosh.runs import run_snapshots
    run, paths = _run(tmp_path, 3)
    _snap(run / "ic_snapshot_005.hdf5")
    assert run_snapshots(str(run)) == paths
    SR._RUNCTX.clear()
    assert SR.context_for_snapshot(paths[1])["snapshots"] == paths


def test_heatmap_panel_from_indexed_columns(tmp_path):
    """a heatmap panel: the columns occ_<bin> of timebin.txt against time, contiguous integer rows, both renderers."""
    from splosh import layout as lay, layout_preview as lp, layout_mpl
    from splosh.io import load_snapshot
    run, paths = _run(tmp_path, 1)
    txt = ""
    for k, t in enumerate([0.0, 0.1, 0.2, 0.3]):
        txt += f"Sync-Point {k}, Time: {t}, Systemstep: 0.1\nOccupied timebins: non-cells     cells       dt                 cumulative A D    avg-time  cpu-frac\n"
        txt += f" X  bin=40               0       {100 + k}     0.1   1 *   0.00  0.0%\n    bin=37               0       5     0.01   1   0.00  0.0%\n"
        txt += "               ------------------------\nTotal active:            0      10    Sum:     105\n\n"
    (run / "output" / "timebin.txt").write_text(txt)
    L = {"schema": lay.SCHEMA, "name": "t", "ncols": 1, "panel": [{"id": "h", "type": "heatmap", "source": "timebin", "prefix": "occ_", "overlay": ["bin_min"]}]}
    lo, tiles, notes, h = lp.tiles_for(L, paths[0], lambda g: load_snapshot(paths[0], gas_fields=g), MapStore(str(tmp_path / "store")))
    hm = tiles["h"]["heat"]
    assert list(hm["y"]) == [37.0, 38.0, 39.0, 40.0]               # contiguous rows, the empty 38 and 39 included
    assert np.isnan(hm["z"][1]).all() and np.nanmax(hm["z"][3]) == 103.0
    assert layout_mpl.render_bytes(lo, tiles, h)[:4] == b"\x89PNG" and lay.render_layout(lo, tiles, h).ndim == 3
