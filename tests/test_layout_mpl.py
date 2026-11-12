"""The publication renderer of layouts: same tiles as the bitmap dashboards, [figure] options from fixed lists, shared colour bars, PDF output."""
import h5py
import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib                                     # noqa: E402
matplotlib.use("Agg")

from splosh import layout as lay, layout_preview as lp, layout_mpl      # noqa: E402
from splosh.io import load_snapshot                                      # noqa: E402
from splosh.mapstore import MapStore                                     # noqa: E402


def _snap(path, n=3000):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 3)) * 0.4
    with h5py.File(path, "w") as f:
        h = f.create_group("Header").attrs
        h.update({"Time": 0.25, "NumPart_ThisFile": np.array([n, 0, 0, 0, 0, 0]), "NumPart_Total": np.array([n, 0, 0, 0, 0, 0]), "MassTable": np.zeros(6)})
        g = f.create_group("PartType0")
        g["Coordinates"] = x; g["Velocities"] = np.cross(x, [0, 0, 1.0]); g["Masses"] = np.full(n, 1e-3); g["SmoothingLength"] = np.full(n, 0.15)
        g["Density"] = np.exp(-(x ** 2).sum(1)); g["InternalEnergy"] = np.ones(n)
    return str(path)


def _layout(colorbar="row", **fig):
    m = dict(type="map", field="rho", center="xyz:0,0,0", width=3.0, npix=48, limits="fixed", vmin=1e-3, vmax=1.0)
    return {"schema": lay.SCHEMA, "name": "t", "ncols": 2, "figure": {"colorbar": colorbar, **fig},
            "panel": [dict(m, id="a", overlays=["contours:n=4", "vectors:v,grid=8"]), dict(m, id="b", axes="xz"),
                      {"id": "u", "type": "map", "field": "u", "center": "xyz:0,0,0", "width": 3.0, "npix": 48, "limits": "frame"},
                      {"id": "note", "type": "text", "text": "snapshot {snapshot}"}]}


def _tiles(tmp_path, L):
    p = _snap(tmp_path / "snapshot_000.hdf5")
    return lp.tiles_for(L, p, lambda f: load_snapshot(p, gas_fields=tuple(sorted(f))), MapStore(str(tmp_path / "store")))


def test_figure_options_are_checked():
    fo = lay.validate_layout(_layout())["figure"]
    assert {k: fo[k] for k in ("size", "colorbar", "letters", "time", "style", "dpi")} == {"size": "wide", "colorbar": "row", "letters": False, "time": "panel", "style": "default", "dpi": 200}
    assert fo["panel_in"] is None and fo["colorbar_ticks"] == "splosh" and fo["header"] == "none" and fo["nodata"] == "dark"
    with pytest.raises(lay.LayoutError, match="panel_in"):
        lay.validate_layout(_layout(panel_in=0.1))
    with pytest.raises(lay.LayoutError, match="figure.colorbar"):
        lay.validate_layout(_layout(colorbar="sideways"))
    with pytest.raises(lay.LayoutError, match="overlay"):
        lay.validate_layout({**_layout(), "panel": [dict(_layout()["panel"][0], overlays=["nonsense:"])]})
    assert "figure = {" in lay.layout_to_toml(lay.validate_layout(_layout()))          # written back as an inline table


def test_render_shared_bar_overlays_and_formats(tmp_path):
    lo, tiles, notes, header = _tiles(tmp_path, _layout(letters=True))
    assert tiles["a"]["overlays"] and not notes.get("a")
    fig = layout_mpl.render(lo, tiles, header)
    cbars = [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert len(cbars) == 2                       # row 0: a and b share one bar (same limits); row 1: u has its own
    letters = [t.get_text() for a in fig.axes for t in a.texts if t.get_text().startswith("(")]
    assert letters[:3] == ["(a)", "(b)", "(c)"]
    pdf = layout_mpl.render_bytes(lo, tiles, header, fmt="pdf")
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000
    out = layout_mpl.save(lo, tiles, str(tmp_path / "f.png"), header)
    assert open(out, "rb").read(8) == b"\x89PNG\r\n\x1a\n"
    # the bitmap renderer draws the same tiles, overlays included
    rgb = lay.render_layout(lo, tiles, header)
    assert rgb.ndim == 3 and rgb.shape[2] == 3


def test_panel_colorbars_and_none(tmp_path):
    for mode, n in (("panel", 3), ("none", 0)):
        L = lay.validate_layout(_layout(colorbar=mode))
        lo, tiles, _, header = _tiles(tmp_path, L)
        fig = layout_mpl.render(lo, tiles, header)
        assert len([a for a in fig.axes if a.get_label() == "<colorbar>"]) == n


def test_viewer_pane_becomes_a_layout_panel(tmp_path):
    """POST /api/layout/addpane: a new layout from a pane, then a second panel; both validate; overlays and fixed limits kept."""
    import json
    import threading
    import urllib.request
    from splosh.server import make_server
    p = _snap(tmp_path / "snapshot_000.hdf5")
    srv, state = make_server([p], token="tok", store=MapStore(str(tmp_path / "store")), layouts=str(tmp_path / "layouts"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/layout/addpane"
    def post(body):
        r = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"X-Splosh-Token": "tok", "Content-Type": "application/json"}, method="POST")
        return json.loads(urllib.request.urlopen(r).read())
    try:
        pane = {"snap": 0, "field": "rho", "mode": "proj", "axes": "xy", "z0": 0, "center": "xyz:0,0,0", "align": "none", "tilt": "none", "npix": 64, "units": "code",
                "width": 3.0, "cx": 0.1, "cy": 0.0}
        a = post({"name": "fig1", "pane": pane, "cmap": "inferno", "scale": "log", "vmin": 1e-3, "vmax": 2.0, "overlays": ["contours:n=5"], "label": "rho"})
        b = post({"name": "fig1", "pane": dict(pane, field="u"), "cmap": "magma", "scale": "linear", "label": "rho"})
        L = lay.validate_layout(lay.load_layout(str(tmp_path / "layouts" / "fig1.toml")))
        assert [q["id"] for q in L["panel"]] == ["rho", "rho_2"] and a["panel"] == "rho" and b["panel"] == "rho_2"
        assert L["panel"][0]["overlays"] == ["contours:n=5"] and L["panel"][0]["limits"] == "fixed" and L["panel"][0]["cmap"] == "inferno"
        assert L["panel"][1]["field"] == "u" and L["panel"][1]["limits"] == "frame"
    finally:
        srv.shutdown()


def test_layout_movie_global_limits_and_frames(tmp_path):
    from splosh import layout_movie
    run = tmp_path / "run"; run.mkdir()
    for k in range(3):
        _snap(run / f"snapshot_{k:03d}.hdf5")
    L = _layout(colorbar="row")
    L["panel"][0].update(limits="global"); L["panel"][0].pop("vmin"); L["panel"][0].pop("vmax")
    (tmp_path / "l.toml").write_text(lay.layout_to_toml(lay.validate_layout(L)))
    lines = []
    res = layout_movie.make(str(run), str(tmp_path / "l.toml"), str(tmp_path / "m.mp4"), fps=4, dpi=60, store=MapStore(str(tmp_path / "s")),
                            keep_frames=True, ffmpeg="/nonexistent-ffmpeg-for-the-test", log=lines.append)
    assert "a" in res.get("limits", {"a": 1}) or not res["encoded"]
    assert lines[:3] == ["pass 1/3", "pass 2/3", "pass 3/3"] and any(l.startswith("frame 3/3") for l in lines)
    import glob, os
    frames = sorted(glob.glob(str(tmp_path / "m_frames" / "frame_*.png"))) or sorted(glob.glob(str(tmp_path / ".frames_*" / "frame_*.png")))
    assert len(frames) == 3


def test_style_problem_is_reported(monkeypatch):
    import builtins
    real = builtins.__import__
    def no_astroplot(name, *a, **k):
        if name.startswith("astroplot"):
            raise ImportError("no astroplot")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_astroplot)
    assert "astroplot" in layout_mpl.style_problem("pgplot") and layout_mpl.style_problem("default") is None


def test_inside_bars_no_titles_and_touching_maps(tmp_path):
    L = _layout(colorbar="inside", titles=False, gap="none", panel_in=3.0, tick_pt=7.0, label_pt=8.0)
    lo, tiles, notes, header = _tiles(tmp_path, lay.validate_layout(L))
    fig = layout_mpl.render(lo, tiles, header)
    maps = [a for a in fig.axes if a.get_images() and a.get_label() != "<colorbar>"]
    assert all(a.get_title() == "" for a in maps)
    insets = [c for m in maps for c in m.child_axes]                                   # the bars are child axes of their maps
    assert len(insets) == 3 and not [a for a in fig.axes if a.get_label() == "<colorbar>"]
    assert fig.subplotpars.wspace == 0.0 and fig.subplotpars.hspace == 0.0
    assert maps[0].xaxis.label.get_size() == 8.0 or maps[0].yaxis.label.get_size() == 8.0


def test_layout_movie_parallel_equals_sequential(tmp_path):
    import glob
    from splosh import layout_movie
    run = tmp_path / "run"; run.mkdir()
    for k in range(4):
        _snap(run / f"snapshot_{k:03d}.hdf5")
    L = _layout(colorbar="row")
    L["panel"][0].update(limits="global"); L["panel"][0].pop("vmin"); L["panel"][0].pop("vmax")
    (tmp_path / "l.toml").write_text(lay.layout_to_toml(lay.validate_layout(L)))
    frames, lims = {}, {}
    for w in (1, 2):
        out = tmp_path / f"m{w}.mp4"
        res = layout_movie.make(str(run), str(tmp_path / "l.toml"), str(out), fps=4, dpi=50, store=MapStore(str(tmp_path / f"s{w}")),
                                keep_frames=True, log=lambda m: None, workers=w)
        lims[w] = res.get("limits")
        frames[w] = [open(f, "rb").read() for f in sorted(glob.glob(str(tmp_path / f"m{w}_frames" / "frame_*.png")))]
    assert len(frames[1]) == 4 and frames[1] == frames[2] and lims[1] == lims[2]


def test_format_file_under_the_layout(tmp_path):
    d = tmp_path / "layouts"; (d / "formats").mkdir(parents=True)
    (d / "formats" / "look.toml").write_text('[figure]\nstyle = "pgplot"\ncolorbar_ticks = "matplotlib"\ndpi = 120\ntitles = false\n')
    L = _layout(colorbar="inside"); L["format"] = "look"; L["figure"] = {"colorbar": "inside", "dpi": 300}
    (d / "a.toml").write_text(lay.layout_to_toml(L))
    fo = lay.load_layout(str(d / "a.toml"))["figure"]
    assert fo["style"] == "pgplot" and fo["colorbar_ticks"] == "matplotlib" and fo["titles"] is False      # from the format
    assert fo["dpi"] == 300 and fo["colorbar"] == "inside"                                                  # the layout's own win
    assert lay.list_formats(str(d)) == ["look"]
    L["format"] = "missing"
    with pytest.raises(lay.LayoutError, match="missing"):
        lay.apply_format(L, lay.format_dirs(str(d)))
