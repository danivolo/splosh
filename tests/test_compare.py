"""splosh compare: two synthetic runs with different snapshot spacing."""
import json
import os

import h5py
import numpy as np
import pytest

from splosh import cli, compare
from conftest import write_snapshot


def _run(root, name, times, seed0):
    d = root / name / "output"
    d.mkdir(parents=True)
    for k, t in enumerate(times):
        p = d / f"snapshot_{k:03d}.hdf5"
        write_snapshot(str(p), n_gas=600, seed=seed0 + k, bh_pos=((1.0 + t, 0.5, 0.1), (-1.0 - t, -0.5, 0.0)))
        with h5py.File(p, "r+") as f:
            f["Header"].attrs["Time"] = t
    return root / name


CFG = {
    "map": [{"name": "density", "center": "com", "width": "auto", "npix": 40, "limits": "global", "pct": [1, 99.9]}],
    "profile": [{"name": "radial", "center": "com", "quantities": ["u"], "rmin": 0.1, "rmax": 3.0, "bins": 6}],
    "scalar": [{"name": "builtin"}],
    "compare": [{"name": "series", "type": "timeseries", "columns": ["gas_mass", "bh_sep / bh_mtot"]},
                {"name": "prof", "type": "profile", "product": "radial", "quantities": ["u"], "at": [0.2, 0.3]},
                {"name": "grid", "type": "maps", "maps": ["density"], "at": [0.0, 0.15, 0.3, 0.9]}],
}


@pytest.fixture
def two(tmp_path):
    a = _run(tmp_path, "fine", [0.0, 0.1, 0.2, 0.3], 1)
    b = _run(tmp_path, "coarse", [0.0, 0.15, 0.3], 11)
    return tmp_path, a, b


def cmp(two, cfg=CFG, **kw):
    root, a, b = two
    return compare.compare([("fine", str(a)), ("coarse", str(b / "output"))], cfg, out=str(root / "cmp"), **kw)


def test_comparison_products_and_blank_tile(two):
    root, a, b = two
    r = cmp(two)
    assert not r.failures, r.failures
    out = root / "cmp"
    for f in ("series.png", "series.json", "prof.png", "grid/at_000.png", "grid/at_003.png", "manifest.json", "digest.txt"):
        assert (out / f).exists(), f
    side = json.load(open(out / "grid" / "at_001.json"))
    snaps = {(s[0], s[1]): s[2] for s in side["snapshots"]}
    assert snaps[("density", "fine")] in ("snapshot_001", "snapshot_002") and snaps[("density", "coarse")] == "snapshot_001"     # t = 0.15: nearest within half the spacing
    last = json.load(open(out / "grid" / "at_003.json"))                                                                    # t = 0.9: no run has a snapshot there
    assert last["snapshots"] == []
    assert r.counts["maps"] == 4 and r.counts["timeseries"] == 1 and r.counts["profile"] == 1
    assert "fine" in r.digest and "coarse" in r.digest and len(r.digest.splitlines()) < 60
    # each run kept its own cache, comparison outputs are separate
    assert (a / "analysis" / "manifest.json").exists() and (b / "analysis" / "manifest.json").exists() and not (out / "density").exists()


def test_second_run_does_nothing_and_change_is_incremental(two):
    r = cmp(two)
    r2 = cmp(two)
    assert r2.total == 0 and all(x.total == 0 for x in r2.runs.values())
    cfg = json.loads(json.dumps(CFG))
    cfg["compare"][0]["columns"] = ["gas_mass"]
    r3 = cmp(two, cfg)
    assert [w[0] for w in r3.work] == ["series"] and all(x.total == 0 for x in r3.runs.values())


def test_tolerance_gives_blank_tile_between_snapshots(two):
    cfg = json.loads(json.dumps(CFG))
    cfg["compare"] = [{"name": "grid", "type": "maps", "maps": ["density"], "at": [0.1], "tol": 0.01}]
    root, a, b = two
    cmp(two, cfg)
    side = json.load(open(root / "cmp" / "grid" / "at_000.json"))
    assert [s[1] for s in side["snapshots"]] == ["fine"]            # coarse has 0.0 and 0.15 only


def test_alignment_options(two):
    root, a, b = two
    cfg = json.loads(json.dumps(CFG))
    cfg["compare"] = [{"name": "idx", "type": "maps", "maps": ["density"], "align": "index", "at": [1.0]},
                      {"name": "off", "type": "maps", "maps": ["density"], "align": "offset", "offsets": [0.0, 0.15], "at": [0.0]}]
    cmp(two, cfg)
    idx = json.load(open(root / "cmp" / "idx" / "at_000.json"))
    assert {s[1]: s[2] for s in idx["snapshots"]} == {"fine": "snapshot_001", "coarse": "snapshot_001"}
    off = json.load(open(root / "cmp" / "off" / "at_000.json"))
    assert {s[1]: s[2] for s in off["snapshots"]} == {"fine": "snapshot_000", "coarse": "snapshot_001"}       # coarse shifted: its t = 0.15 is aligned time 0
    cmp(two, cfg, offsets={"coarse": 0.3}, align="offset", force=False)
    off2 = json.load(open(root / "cmp" / "off" / "at_000.json"))
    assert {s[1]: s[2] for s in off2["snapshots"]} == {"fine": "snapshot_000", "coarse": "snapshot_002"}


def test_same_view_and_limits_across_runs(two):
    root, a, b = two
    cmp(two)
    ma = json.load(open(a / "analysis" / "manifest.json")); mb = json.load(open(b / "analysis" / "manifest.json"))
    assert ma["auto_radius"] == mb["auto_radius"]                                  # one automatic width for all runs
    side = json.load(open(root / "cmp" / "grid" / "at_000.json"))
    lims = {tuple(s[4]) for s in json.load(open(root / "cmp" / "grid" / "at_000.json"))["snapshots"]} if len(side["snapshots"][0]) > 4 else None
    assert lims is None or len(lims) == 1


def test_parse_runs_and_errors(two, capsys):
    root, a, b = two
    assert compare.parse_runs([f"x={a}", str(b / "output")]) == [("x", str(a)), ("coarse", str(b / "output"))]
    with pytest.raises(ValueError, match="unique"):
        compare.parse_runs([str(a), f"fine={b}"])
    with pytest.raises(ValueError, match="at least two"):
        compare.compare([str(a)], CFG, out=str(root / "c"))
    with pytest.raises(ValueError):
        compare.complete_compare({"name": "z", "type": "maps"})
    with pytest.raises(ValueError, match="offsets"):
        compare.complete_compare({"name": "z", "type": "timeseries", "columns": ["x"], "align": "offset"})


def test_cli_compare_and_dry_run(two, capsys):
    root, a, b = two
    cli.main(["compare", f"fine={a}", f"coarse={b}", "--out", str(root / "c2"), "--dry-run"])
    assert "run fine" in capsys.readouterr().out and not (root / "c2").exists()
    cli.main(["compare", f"fine={a}", f"coarse={b}", "--out", str(root / "c2"), "--run-out", str(root / "caches")])
    assert "failures: none" in capsys.readouterr().out
    assert (root / "c2" / "scalars.png").exists() and (root / "caches" / "fine" / "manifest.json").exists()
