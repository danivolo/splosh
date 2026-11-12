"""splosh analyze: a synthetic multi-snapshot run; incremental work counts, parallel == serial, partial snapshots, colour limits, digest, config reader, dry run."""
import json
import os
import shutil
import sys

import h5py
import numpy as np
import pytest

from splosh import analyze, cli, plugins
from splosh.mapstore import MapStore, view_key, snapshot_hash
from splosh.maps import View
from conftest import write_snapshot

CONFIG = {
    "map": [{"name": "rho", "center": "com", "width": 4.0, "npix": 48, "limits": "global", "pct": [1, 99.9]},
            {"name": "temp", "center": "com", "width": 4.0, "npix": 48, "field": "u", "limits": "frame", "cmap": "inferno"}],
    "profile": [{"name": "prof", "center": "com", "quantities": ["u", "density"], "rmin": 0.1, "rmax": 3.0, "bins": 8}],
    "scalar": [{"name": "builtin"}],
    "dashboard": [{"name": "dash", "panels": ["rho", "temp"], "labels": ["column density", "u"], "scalars": ["gas_mass", "bh_sep"]}],
    "movie": [{"name": "mov", "source": "dash", "fps": 4}],
}


def make_run(root, n=3, **kw):
    d = root / "run" / "output"
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        p = d / f"snapshot_{k:03d}.hdf5"
        write_snapshot(str(p), n_gas=800, seed=1 + k, bh_pos=((1.0 + 0.1 * k, 0.5, 0.1), (-1.0 - 0.1 * k, -0.5, 0.0)), **kw)
        with h5py.File(p, "r+") as f:
            f["Header"].attrs["Time"] = 0.1 * k
    return root / "run"


def go(run, cfg=CONFIG, **kw):
    kw.setdefault("out", str(run / "analysis"))
    return analyze.run(str(run), cfg, **kw)


def stages(r):
    return dict(sorted(r.counts.items()))


@pytest.fixture
def rundir(tmp_path):
    return make_run(tmp_path)


def test_first_run_does_everything_second_does_nothing(rundir):
    r = go(rundir)
    n = 3
    assert stages(r)["render"] == 2 * n and stages(r)["png"] == 2 * n and stages(r)["profile"] == n and stages(r)["scalar"] == n and stages(r)["dashboard"] == n
    assert not r.failures and not r.skipped
    out = rundir / "analysis"
    for f in ("rho/snapshot_000.png", "rho/snapshot_000.json", "temp/snapshot_002.png", "prof/snapshot_001.npz", "prof/snapshot_001.csv", "dash/snapshot_002.png", "scalars.csv", "digest.txt", "manifest.json"):
        assert (out / f).exists(), f
    if shutil.which("ffmpeg"):
        assert (out / "movies" / "mov.mp4").exists() and stages(r)["movie"] == 1
    r2 = go(rundir)
    assert r2.total == 0 and not r2.failures
    m = json.load(open(out / "manifest.json"))
    u = m["products"]["rho"]["units"]["snapshot_000"]
    assert u["outputs"] and u["wall_render"] >= 0 and m["git_commit"] is not None or m["git_commit"] is None
    assert m["schema"] == analyze.MANIFEST_SCHEMA and m["splosh_version"] and m["config_hash"]
    assert len(m["products"]["dash"]["units"]) == 3


def test_map_is_in_the_mapstore(rundir):
    go(rundir)
    snap = str(rundir / "output" / "snapshot_001.hdf5")
    store = MapStore(str(rundir / "analysis" / ".maps"))
    img = store.get(view_key(snapshot_hash(snap), View(center="com", width=4.0, npix=48)))
    assert img is not None and img[0].dtype == np.float32 and img[0].shape == (48, 48)


def test_touch_one_snapshot_recomputes_only_its_products(rundir):
    go(rundir)
    p = rundir / "output" / "snapshot_001.hdf5"
    write_snapshot(str(p), n_gas=800, seed=77)
    with h5py.File(p, "r+") as f:
        f["Header"].attrs["Time"] = 0.1
    r = go(rundir)
    assert {w[1] for w in r.work if w[2] in ("render", "profile", "scalar")} == {"snapshot_001"}
    assert stages(r)["render"] == 2 and stages(r)["profile"] == 1 and stages(r)["scalar"] == 1 and stages(r)["dashboard"] >= 1   # (other dashboards: their series / limits changed)
    # global limits may move with the new data: only the png stage of `rho` may touch other snapshots
    assert all(w[0] == "rho" for w in r.work if w[2] == "png" and w[1] != "snapshot_001")
    assert go(rundir).total == 0


def test_config_change_recomputes_only_that_product(rundir):
    go(rundir)
    cfg = json.loads(json.dumps(CONFIG))
    cfg["map"][1]["cmap"] = "magma"                  # style only: png of `temp`, then the dashboard and movie that use it; no render
    r = go(rundir, cfg)
    assert {w[0] for w in r.work} <= {"temp", "dash", "mov"} and "render" not in r.counts and r.counts["png"] == 3
    cfg["profile"][0]["bins"] = 12
    r = go(rundir, cfg)
    assert {w[0] for w in r.work} == {"prof"} and r.counts == {"profile": 3}
    cfg["map"][0]["width"] = 5.0                     # view change: render + png + dashboard for `rho`
    r = go(rundir, cfg)
    assert r.counts["render"] == 3 and {w[0] for w in r.work} <= {"rho", "dash", "mov"}
    r = go(rundir, cfg, force=True)
    assert r.counts["render"] == 6 and r.counts["profile"] == 3


def test_only_and_every(rundir):
    r = go(rundir, only="prof", every=2)
    assert r.counts == {"profile": 2}
    r = go(rundir, only=["dash"])                      # dashboards bring their inputs (maps, scalars)
    assert r.counts["dashboard"] == 3 and "profile" not in r.counts
    with pytest.raises(ValueError):
        go(rundir, only="nope")


def test_dry_run_writes_nothing(rundir):
    r = go(rundir, dry_run=True)
    assert "render" in r.plan and "prof" in r.plan and not (rundir / "analysis").exists()
    go(rundir)
    assert "nothing to do" in go(rundir, dry_run=True).plan


def test_parallel_equals_serial(rundir):
    a, b = rundir / "a", rundir / "b"
    cfg = {k: v for k, v in CONFIG.items() if k != "movie"}
    go(rundir, cfg, out=str(a))
    go(rundir, cfg, out=str(b), jobs=2)
    for k in range(3):
        stem = f"snapshot_{k:03d}"
        for prod in ("rho", "temp", "dash"):
            assert (a / prod / f"{stem}.png").read_bytes() == (b / prod / f"{stem}.png").read_bytes(), (prod, stem)
        pa, pb = np.load(a / "prof" / f"{stem}.npz"), np.load(b / "prof" / f"{stem}.npz")
        for key in pa.files:
            if key != "meta":
                assert np.array_equal(pa[key], pb[key], equal_nan=True)
    assert (a / "scalars.csv").read_text() == (b / "scalars.csv").read_text()
    sa = MapStore(str(a / ".maps")); sb = MapStore(str(b / ".maps"))
    snap = str(rundir / "output" / "snapshot_002.hdf5")
    key = view_key(snapshot_hash(snap), View(center="com", width=4.0, npix=48))
    assert np.array_equal(sa.get(key)[0], sb.get(key)[0])


def test_partial_snapshot_is_skipped_then_picked_up(rundir):
    bad = rundir / "output" / "snapshot_003.hdf5"
    bad.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 100)             # being written: not a valid file yet
    r = go(rundir)
    assert list(r.skipped) == ["snapshot_003"] and r.counts["scalar"] == 3
    assert "snapshot_003" in r.digest and "unreadable" in r.digest
    assert not (rundir / "analysis" / "rho" / "snapshot_003.png").exists()
    write_snapshot(str(bad), n_gas=800, seed=9)
    with h5py.File(bad, "r+") as f:
        f["Header"].attrs["Time"] = 0.3
    r = go(rundir)
    assert not r.skipped and {w[1] for w in r.work if w[2] in ("render", "profile", "scalar")} == {"snapshot_003"}
    assert (rundir / "analysis" / "rho" / "snapshot_003.png").exists()
    assert len((rundir / "analysis" / "scalars.csv").read_text().splitlines()) == 5


def test_failed_item_is_reported_and_retried(rundir):
    bad = rundir / "output" / "snapshot_001.hdf5"
    with h5py.File(bad, "r+") as f:
        del f["PartType0/InternalEnergy"]                          # opens fine, the field needs a dataset that is not there
    r = go(rundir)
    assert any(f[0] == "snapshot_001" and f[1] in ("temp", "prof", "builtin") for f in r.failures)
    assert (rundir / "analysis" / "rho" / "snapshot_001.png").exists()
    assert "snapshot_001" in r.digest and "failures:" in r.digest
    assert go(rundir).failures                                      # retried, still failing: nothing was recorded as done


def test_global_limits_identical_across_frames_and_frame_limits_differ(rundir):
    go(rundir)
    m = json.load(open(rundir / "analysis" / "manifest.json"))
    lims = [m["products"]["rho"]["units"][f"snapshot_{k:03d}"]["lim"] for k in range(3)]
    assert lims[0] == lims[1] == lims[2]
    side = [json.load(open(rundir / "analysis" / "rho" / f"snapshot_{k:03d}.json"))["limits"] for k in range(3)]
    assert len({(s["vmin"], s["vmax"]) for s in side}) == 1
    tl = [m["products"]["temp"]["units"][f"snapshot_{k:03d}"]["lim"] for k in range(3)]
    assert len({tuple(x) for x in tl}) > 1
    # growing the run moves the global limits only for the png stage
    make_run(rundir.parent, n=4)
    r = go(rundir)
    assert r.counts["render"] == 2 and r.counts["scalar"] == 1


def test_viewport_and_fixed_limits(rundir):
    cfg = {"map": [{"name": "v", "center": "com", "width": 4.0, "npix": 48, "limits": "viewport", "viewport": [0.4, 0.4, 0.6, 0.6]},
                   {"name": "f", "center": "com", "width": 4.0, "npix": 48, "limits": "fixed", "vmin": 0.01, "vmax": 5.0}]}
    go(rundir, cfg)
    m = json.load(open(rundir / "analysis" / "manifest.json"))
    assert m["products"]["f"]["units"]["snapshot_000"]["lim"] == [0.01, 5.0]
    lv = m["products"]["v"]["units"]["snapshot_000"]["lim"]
    assert lv[0] < lv[1]
    with pytest.raises(ValueError):
        analyze.load_config({"map": [{"name": "x", "limits": "fixed"}]})


def test_digest_content(rundir):
    r = go(rundir)
    lines = r.digest.splitlines()
    assert len(lines) < 60 and r.digest.isascii()
    txt = (rundir / "analysis" / "digest.txt").read_text()
    assert txt == r.digest
    assert "last snapshot: snapshot_002" in txt and "snapshots: 3 readable, 0 unreadable" in txt
    assert "gas_mass" in txt and "bh_sep" in txt and "failures: none" in txt and "this run:" in txt
    assert "nothing was out of date" in go(rundir).digest


def test_scalars_table(rundir):
    go(rundir)
    rows = [l.split(",") for l in (rundir / "analysis" / "scalars.csv").read_text().splitlines()]
    head = rows[0]
    assert head[:2] == ["snapshot", "time"] and len(rows) == 4
    col = lambda name: [float(r[head.index(name)]) for r in rows[1:]]
    assert col("n_gas") == [800.0] * 3 and col("n_bh") == [2.0] * 3 and np.allclose(col("time"), [0, 0.1, 0.2])
    assert np.allclose(col("gas_mass"), 0.8) and col("bh_sep")[0] < col("bh_sep")[2] and np.isnan(col("T_mw")).all()
    assert np.allclose(col("bh_mtot"), 3.6)


def test_plugin_scalar(rundir):
    plugins.register_scalar("rho_min", lambda snap, scale=1.0: {"rho_min": float(np.min(snap["Density"])) * scale})
    try:
        r = go(rundir, {"scalar": [{"name": "rho_min", "params": {"scale": 2.0}}, {"names": ["gas_mass"], "label": "m"}]})
        head, *rows = [l.split(",") for l in (rundir / "analysis" / "scalars.csv").read_text().splitlines()]
        assert "rho_min" in head and "gas_mass" in head and float(rows[0][head.index("rho_min")]) > 0
        with pytest.raises(KeyError, match="unknown scalar"):
            analyze.scalar_functions(["nope"])
    finally:
        plugins.SCALARS.pop("rho_min", None)


def test_default_config_and_cli(rundir, capsys):
    cli.main(["analyze", str(rundir / "output"), "--jobs", "1"])
    out = capsys.readouterr().out
    assert "snapshots: 3 readable" in out and "density" in out
    assert (rundir / "analysis" / "density" / "snapshot_000.png").exists() and (rundir / "analysis" / "radial" / "snapshot_000.csv").exists()
    cli.main(["analyze", str(rundir), "--dry-run"])
    assert "nothing to do" in capsys.readouterr().out
    cli.main(["analyze", str(rundir), "--force", "--only", "profile"])
    assert "profile 3" in capsys.readouterr().out


def test_config_validation(tmp_path):
    for bad in ({"map": [{"name": "a", "bogus": 1}]}, {"map": [{"name": "a"}, {"name": "a"}]}, {"dashboard": [{"name": "d", "panels": ["zz"]}]},
                {"movie": [{"name": "m", "source": "zz"}]}, {"nope": []}, {"map": [{"field": "rho"}]}):
        with pytest.raises(ValueError):
            analyze.load_config(bad)


EXAMPLE = """# comment
[analysis]
store = "maps"   # trailing comment

[[map]]
name = "a"
center = 'com'
width = 4.0
pct = [1, 99.9]
npix = 64
scalebar = false

[[dashboard]]
name = "d"
panels = [
  "a",   # one
]
scalars = ["gas_mass", "bh_sep"]
"""


def test_toml_subset_reader_matches_tomllib(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text(EXAMPLE)
    mine = analyze.parse_toml_subset(EXAMPLE)
    assert mine["map"][0]["pct"] == [1, 99.9] and mine["dashboard"][0]["panels"] == ["a"] and mine["map"][0]["scalebar"] is False
    if sys.version_info >= (3, 11):
        import tomllib
        assert tomllib.loads(EXAMPLE) == mine
    assert analyze.read_toml(str(p)) == mine


def test_docs_example_config_is_valid():
    here = os.path.join(os.path.dirname(__file__), "..", "docs", "analyze.example.toml")
    prods, _ = analyze.load_config(here)
    assert {p["kind"] for p in prods} == {"map", "profile", "scalar", "dashboard", "timeseries", "movie", "compare"}
    with open(here) as f:
        assert analyze.parse_toml_subset(f.read()).keys() == analyze.read_toml(here).keys()
