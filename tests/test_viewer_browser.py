"""Drives the viewer in a real headless browser through tools/browser_smoke.py. Skipped when Playwright or its browser is not available.

Set SPLOSH_BROWSER_PYTHON to the python of an environment that has Playwright with its chromium installed (PLAYWRIGHT_BROWSERS_PATH as needed);
without it the tests use the current python if it can import playwright. Screenshots go to $SPLOSH_BROWSER_SHOTS when that is set."""
import os
import subprocess
import sys

import h5py
import pytest

from conftest import write_snapshot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, "tools", "browser_smoke.py")
PROBE = "from playwright.sync_api import sync_playwright as s\np = s().start()\nb = p.chromium.launch()\nb.close()\np.stop()\n"


def browser_python():
    py = os.environ.get("SPLOSH_BROWSER_PYTHON") or sys.executable
    try:
        r = subprocess.run([py, "-c", PROBE], capture_output=True, text=True, timeout=300)          # generous: on a loaded machine a 120 s probe skipped the whole browser suite
    except (OSError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"no usable browser python: {e}")
    if r.returncode != 0:
        pytest.skip("playwright or its chromium is not available (set SPLOSH_BROWSER_PYTHON): " + r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "playwright missing")
    return py


@pytest.fixture(scope="module")
def smoke_output(tmp_path_factory):
    py = browser_python()
    tmp = tmp_path_factory.mktemp("browser")
    d = tmp / "run"
    d.mkdir()
    write_snapshot(str(d / "snap_000.hdf5"), seed=1, n_star=300, n_dm=300, bfield=(0.0, 1e-6, 0.0), time=0.0)
    write_snapshot(str(d / "snap_001.hdf5"), seed=1, n_star=300, n_dm=300, bfield=(0.0, 1e-6, 0.0), time=1.0)
    with h5py.File(d / "snap_001.hdf5", "r+") as f:                 # same particles, doubled masses: a known difference
        f["PartType0/Masses"][...] = f["PartType0/Masses"][...] * 2
    write_snapshot(str(d / "snap_002.hdf5"), seed=3, bh_pos=((1.3, 0.5, 0.1), (-1.3, -0.5, 0.0)), time=2.0)
    slow = tmp / "slow"                                             # a second run with another snapshot spacing (1.5 against 1)
    slow.mkdir()
    for k, t in enumerate((0.0, 1.5, 3.0, 4.5)):
        write_snapshot(str(slow / f"snap_{k:03d}.hdf5"), n_gas=1500, seed=30 + k, time=t)
    ld = tmp / "layouts"
    ld.mkdir()
    (ld / "seed.toml").write_text('name = "seed"\nncols = 4\n\n[[panel]]\nid = "m"\ntype = "map"\nfield = "rho"\nwidth = 6.0\nnpix = 48\ncenter = "com"\n\n[[panel]]\nid = "t"\ntype = "text"\ntext = "snap {snapshot}"\n')
    (d / "analysis").mkdir()                                        # the run's own snapshot scalars: the time series, heatmap and radial panes (a family sig_<R> = (1 + k) / R)
    (d / "analysis" / "scalars.csv").write_text("snapshot,time,time_myr,gas_mass,sig_1,sig_2,sig_4,sig_8\n" +
                                                "\n".join(f"snap_{k:03d},{k},{k},{1 + k}," + ",".join(f"{(1 + k) / r:.12g}" for r in (1, 2, 4, 8)) for k in range(3)) + "\n")
    an = tmp / "analysis"
    an.mkdir()
    (an / "scalars.csv").write_text("snapshot,time,time_myr,gas_mass\n" + "\n".join(f"snap_{k:03d},{k},{k},{1 + k}" for k in range(3)) + "\n")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), SPLOSH_STORE=str(tmp / "store"))
    cmd = [py, SMOKE, str(d), "--server-python", sys.executable, "--synthetic", "--extra-run", str(slow), "--layouts", str(ld), "--analysis", str(an), "--serve-args", "--center bh --align bh --width 6 --npix 96"]
    if os.environ.get("SPLOSH_BROWSER_SHOTS"):
        cmd += ["--shots", os.environ["SPLOSH_BROWSER_SHOTS"]]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    return r.returncode, r.stdout + r.stderr


def lines(out, word):
    return [l for l in out.splitlines() if word in l]


def test_whole_smoke_passes(smoke_output):
    code, out = smoke_output
    assert code == 0 and out.rstrip().endswith("PASSED"), out


@pytest.mark.parametrize("word, n", [("blink", 4), ("A - A", 1), ("A - B recovers", 1), ("log10(A/B)", 1), ("linked panes keep the same window", 1), ("hover rho projection", 1),
                                     ("slice of the derived field pressure", 1), ("slice of the derived field mach", 1), ("keyboard help", 1), ("no console errors", 1), ("narrow window", 1),
                                     ("layout editor", 12), ("centre rule menu", 1), ("overlay", 6), ("phase diagram of the window", 1), ("the lasso became", 1), ("highlight:", 1), ("map of only the selection", 1), ("clear selection", 1), ("GPU colouring", 3), ("volume mode", 1), ("pixel histogram", 1), ("dragging the upper handle", 1), ("asinh F", 1), ("valueAtT", 1), ("histeq", 1), ("contrast map", 1), ("post=resid", 1), ("no tiles while a transform", 1), ("contours overlay items", 1), ("gradient map", 1), ("restores the colours", 1), ("inspect", 3), ("Esc clears", 1), ("double click centres", 1), ("no B field", 1),
                                     ("tiled composite equals", 1), ("panning by one tile", 1), ("placeholder", 1), ("prefetch", 1), ("difference mode never", 1), ("tiny window", 1), ("seam", 1),
                                     ("step all panes", 1), ("clocks", 1), ("no snapshot at this time", 1), ("offset 1.5", 1), ("page's nearest-snapshot rule", 1), ("index alignment", 2), ("a pane can show the other run", 1), ("independent /api/map?run=", 1),
                                     ("mouse forward / back", 1), ("scans the snapshots without loading", 1), ("releasing shift loads", 1), ("the page does not zoom", 1), ("Esc cancels a scan", 1),
                                     ("plain wheel still zooms", 1), ("middle click", 2), ("right click opens the map menu", 1), ("shift+right click keeps", 1), ("hides and shows the side panel", 1),
                                     ("stackOf", 1), ("stack option goes into its layout panel", 1), ("signed field vx", 1),
                                     ("plot zoom:", 10), ("phase pane:", 9), ("align orbit:", 1), ("folder picker", 2), ("map box zoom", 1), ("lic overlay", 2), ("radial pane", 4), ("spectrum pane", 1), ("zoom saved", 3)])
def test_group_ran_and_passed(smoke_output, word, n):
    _, out = smoke_output
    got = lines(out, word)
    assert len(got) >= n, out
    assert all(l.startswith("ok") for l in got), "\n".join(got)
