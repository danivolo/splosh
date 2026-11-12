"""layouts as data: validation errors, round trip through TOML / JSON, safe expressions, registry, rendering, and a pipeline run driven by layout files."""
import json
import os

import numpy as np
import pytest

from splosh import analyze, layout as lay, plugins
from test_analyze import make_run, go

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def _base():
    return {"name": "t", "ncols": 2, "panel": [{"id": "a", "type": "map", "product": "m"}, {"id": "b", "type": "timeseries", "columns": ["x"]}]}


def test_expressions_are_whitelisted():
    cols = {"a": np.array([1.0, 2.0, 4.0]), "b": np.array([2.0, 2.0, 2.0])}
    assert np.allclose(lay.eval_expr("a / b + 1", cols), [1.5, 2, 3])
    assert np.allclose(lay.eval_expr("-a ** 2 + log10(b) * 0 + abs(-1)", cols), [0, -3, -15])
    for bad in ("__import__('os')", "a.real", "a[0]", "open('x')", "a if b else 1", "lambda: 1", "a ** b", "(1, 2)", "'s'", "a; b", "", "exp(a, b)", "x" * 300):
        with pytest.raises(lay.LayoutError):
            lay.eval_expr(bad, cols)
    with pytest.raises(lay.LayoutError, match="unknown column"):
        lay.eval_expr("a / nope", cols)


def test_validate_fills_defaults_and_places_panels():
    v = lay.validate_layout(_base())
    assert v["schema"] == lay.SCHEMA and v["scope"] == "snapshot"
    assert [(p["row"], p["col"]) for p in v["panel"]] == [(0, 0), (0, 1)]
    w = _base()
    w["panel"].append({"id": "c", "type": "text", "text": "hi", "colspan": 2})
    assert [(p["row"], p["col"]) for p in lay.validate_layout(w)["panel"]][2] == (1, 0)


@pytest.mark.parametrize("edit,msg", [
    (lambda d: d["panel"][0].update(bogus=1), "unknown key"),
    (lambda d: d["panel"][1].update(id="a"), "unique"),
    (lambda d: d["panel"][0].update(type="pie"), "type must be"),
    (lambda d: d["panel"][0].update(field="u"), "exclusive"),
    (lambda d: d["panel"][1].update(columns=["__import__('x')"]), "not allowed|may be called"),
    (lambda d: d["panel"][1].update(xaxis="wall"), "xaxis"),
    (lambda d: d["panel"][0].update(row=0, col=0) or d["panel"][1].update(row=0, col=0), "overlap"),
    (lambda d: d["panel"][1].update(row=0, col=1, colspan=2), "fit"),
    (lambda d: d.update(scope="galaxy"), "scope"),
    (lambda d: d.update(schema="splosh-layout-9"), "schema"),
    (lambda d: d["panel"][0].update(row=1), "both row and col"),
    (lambda d: d.update(derived=[{"name": "q", "expr": "a b"}]), "syntax"),
])
def test_validation_errors_are_clear(edit, msg):
    d = _base()
    edit(d)
    with pytest.raises(lay.LayoutError, match=msg):
        lay.validate_layout(d)


def test_inline_map_panel_is_validated_against_view():
    d = {"name": "t", "panel": [{"id": "a", "type": "map", "width": -1.0}]}
    with pytest.raises(lay.LayoutError, match="width"):
        lay.validate_layout(d)
    d["panel"][0]["width"] = 3.0
    assert lay.validate_layout(d)["panel"][0]["scale"] == "log"


@pytest.mark.parametrize("ext", ["toml", "json"])
def test_round_trip_is_lossless(tmp_path, ext):
    v = lay.load_layout(os.path.join(DOCS, "layouts", "overview.toml"))
    p = tmp_path / f"x.{ext}"
    lay.save_layout(v, str(p))
    again = lay.load_layout(str(p))
    assert again == v
    assert [q for q in again["panel"] if q["id"] == "ts"][0]["x_gui_note"] == "free for a GUI"
    lay.save_layout(again, str(tmp_path / f"y.{ext}"))
    assert (tmp_path / f"y.{ext}").read_text() == p.read_text()


def test_load_layout_error_names_the_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"name": "n", "panel": [{"id": "a", "type": "pie"}]}))
    with pytest.raises(lay.LayoutError, match="bad.json"):
        lay.load_layout(str(p))
    with pytest.raises(lay.LayoutError, match="no such file"):
        lay.find_layout("nothere", [str(tmp_path)])


def test_registry_lists_ingredients():
    plugins.register_scalar("zz_test_scalar", lambda snap: 1.0)
    try:
        r = lay.registry()
    finally:
        plugins.SCALARS.pop("zz_test_scalar", None)
    assert {"map", "profile", "timeseries", "heatmap", "text"} == set(r["panel_types"])
    assert "rho" in [f["name"] for f in r["fields"]] and "T" in [f["name"] for f in r["fields"]]
    names = {s["name"]: s for s in r["scalars"]}
    assert "gas_mass" in names and names["zz_test_scalar"]["builtin"] is False and "bh_sep" in names["black_holes"]["columns"]
    assert "density" in r["profile_quantities"] and "inferno" in r["colormaps"] and r["xaxes"] == ["time", "time_myr", "index", "orb"]
    json.dumps(r)                                                  # a GUI gets it as JSON


def test_render_layout_draws_every_panel_type():
    L = lay.validate_layout({"name": "r", "ncols": 2, "cell": [260, 200], "panel": [
        {"id": "t", "type": "timeseries", "columns": ["a", "a * 2"]}, {"id": "p", "type": "profile", "product": "q"}, {"id": "x", "type": "text", "text": "hello"}, {"id": "m", "type": "map", "product": "mm"}]})
    xs = np.linspace(0, 1, 20)
    tiles = {"t": {"curves": [("a", xs, xs ** 2), ("b", xs, xs)], "xnow": 0.5, "xlabel": "time", "xlog": False, "ylog": False}, "p": {"curves": [("u", xs + 0.1, xs + 1)], "xlog": True, "ylog": True},
             "x": {"text": "hello"}, "m": {"blank": "no snapshot within 0.05"}}
    rgb = lay.render_layout(L, tiles, "t = 1")
    assert rgb.dtype == np.uint8 and rgb.shape[2] == 3 and rgb.shape[1] >= 2 * 260
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 5
    rgb2 = lay.render_layout(L, {k: v for k, v in tiles.items() if k != "t"}, "t = 1")    # a panel without data: a blank tile, no error
    assert rgb2.shape == rgb.shape


CFG = {
    "map": [{"name": "density", "center": "com", "width": 4.0, "npix": 48}],
    "profile": [{"name": "radial", "center": "com", "quantities": ["u"], "rmin": 0.1, "rmax": 3.0, "bins": 8}],
    "scalar": [{"name": "builtin"}],
    "dashboard": [{"name": "board", "layout": {"name": "b", "ncols": 2, "cell": [200, 200], "derived": [{"name": "ratio", "expr": "bh_m1 / bh_m2"}], "panel": [
        {"id": "d", "type": "map", "product": "density"},
        {"id": "u", "type": "map", "field": "u", "center": "com", "width": 4.0, "npix": 48, "limits": "frame"},
        {"id": "pr", "type": "profile", "product": "radial"},
        {"id": "ts", "type": "timeseries", "columns": ["ratio", "gas_mass * 2"], "series": "past"},
        {"id": "n", "type": "text", "text": "{snapshot} t={time}", "colspan": 2}]}}],
    "timeseries": [{"name": "hist", "columns": ["gas_mass", "bh_sep / bh_mtot"], "xaxis": "time_myr"}],
}


def test_pipeline_with_layouts(tmp_path):
    run = make_run(tmp_path)
    r = go(run, CFG)
    assert not r.failures, r.failures
    out = run / "analysis"
    assert r.counts["dashboard"] == 3 and r.counts["timeseries"] == 1 and r.counts["render"] == 6      # the inline map panel became a product `board.u`
    for f in ("board/snapshot_001.png", "board.u/snapshot_001.png", "figures/hist.png", "figures/hist.json"):
        assert (out / f).exists(), f
    assert go(run, CFG).total == 0
    cfg = json.loads(json.dumps(CFG))
    cfg["dashboard"][0]["layout"]["panel"][4]["text"] = "changed"
    r = go(run, cfg)
    assert r.counts == {"dashboard": 3}                                                                  # a layout edit redoes the dashboards only
    cfg["timeseries"][0]["xaxis"] = "index"
    assert go(run, cfg).counts == {"timeseries": 1}


def test_layout_files_by_name_and_bad_references(tmp_path):
    run = make_run(tmp_path)
    cfgdir = tmp_path / "cfg"
    (cfgdir / "layouts").mkdir(parents=True)
    lay.save_layout({"name": "mine", "ncols": 1, "panel": [{"id": "d", "type": "map", "product": "density"}]}, str(cfgdir / "layouts" / "mine.toml"))
    (cfgdir / "a.toml").write_text('[[map]]\nname = "density"\ncenter = "com"\nwidth = 4.0\nnpix = 48\n\n[[dashboard]]\nname = "b"\nlayout = "mine"\n')
    r = analyze.run(str(run), str(cfgdir / "a.toml"), out=str(run / "o"))
    assert r.counts["dashboard"] == 3
    with pytest.raises(ValueError, match="not a \\[\\[profile\\]\\]"):
        analyze.load_config({"map": [{"name": "m"}], "dashboard": [{"name": "d", "layout": {"name": "l", "panel": [{"id": "p", "type": "profile", "product": "m"}]}}]})
    with pytest.raises(ValueError, match="scope"):
        analyze.load_config({"timeseries": [{"name": "t", "layout": {"name": "l", "panel": [{"id": "p", "type": "timeseries", "columns": ["x"]}]}}]})
    r = go(run, {"scalar": [{"name": "builtin"}], "timeseries": [{"name": "bad", "columns": ["nope_column"]}]})
    assert any("nope_column" in f[3] for f in r.failures)


def _parts():
    """three parts of a total (a circumbinary disc and two minidiscs) on the snapshot table; the total grows (accretion onto the system)"""
    t = np.linspace(0.0, 10.0, 11)
    return {"time": t, "time_myr": t, "index": np.arange(11.0), "cbd": 8.0 + 0.0 * t, "md1": 1.0 + 0.1 * t, "md2": 1.0 + 0.2 * t}


@pytest.mark.parametrize("given,kept", [(True, True), ("stack", True), ("fraction", "fraction"), (False, False), ("none", False)])
def test_stack_key_is_validated(given, kept):
    d = _base()
    d["panel"][1]["stack"] = given
    assert lay.validate_layout(d)["panel"][1]["stack"] == kept
    assert lay.validate_layout(_base())["panel"][1]["stack"] is False


@pytest.mark.parametrize("edit,msg", [(dict(stack="pie"), "stack must be"), (dict(stack=True, ylog=True), "linear y axis"), (dict(stack="fraction", yscale="symlog"), "linear y axis")])
def test_stack_errors(edit, msg):
    d = _base()
    d["panel"][1].update(edit)
    with pytest.raises(lay.LayoutError, match=msg):
        lay.validate_layout(d)


def test_stacked_zones_are_cumulative_sums_and_the_total_shows_when_it_varies():
    L = lay.validate_layout({"name": "s", "panel": [{"id": "m", "type": "timeseries", "columns": ["cbd", "md1", "md2"], "stack": True}]})
    t = lay.timeseries_data(L["panel"][0], _parts(), cur=4)
    st = t["stack"]
    assert st["mode"] == "sum" and [b[0] for b in st["bands"]] == ["cbd", "md1", "md2"] and t["curves"] == []
    tb = _parts()
    assert np.allclose(st["bands"][0][1], 0) and np.allclose(st["bands"][0][2], tb["cbd"])
    assert np.allclose(st["bands"][1][1], tb["cbd"]) and np.allclose(st["bands"][2][2], tb["cbd"] + tb["md1"] + tb["md2"])
    assert np.allclose(st["total"], tb["cbd"] + tb["md1"] + tb["md2"]) and t["xnow"] == 4.0          # the marker is kept
    F = lay.timeseries_data(dict(L["panel"][0], stack="fraction"), tb)["stack"]
    assert F["mode"] == "fraction" and np.allclose(F["bands"][-1][2], 1.0) and np.allclose(F["bands"][0][2], 8.0 / (10.0 + 0.3 * tb["time"]))
    assert np.isclose(np.max(F["total"]), 1.0) and "total / max" in F["total_label"]
    zero = dict(tb, md2=2.0 - 0.1 * tb["time"], cbd=8.0 - 0.0 * tb["time"])                         # a zero-sum exchange: constant total, no total line
    assert lay.timeseries_data(L["panel"][0], zero)["stack"]["total"] is None
    plain = lay.timeseries_data(dict(L["panel"][0], stack=False), tb)
    assert "stack" not in plain and len(plain["curves"]) == 3


def test_stack_interpolates_parts_on_other_grids_and_keeps_twin_and_faint_curves():
    x1, x2 = np.linspace(0, 1, 5), np.linspace(0, 1, 9)
    tile = {"curves": [("a", x1, np.ones(5)), ("a raw", x1, np.ones(5)), ("b", x2, 2 * x2), ("r", x1, x1)],
            "styles": [{"color": "#112233"}, {"legend": False, "alpha": 0.35}, {}, {"twin": True}], "ylog": False, "yscale": "auto"}
    out = lay.stack_tile(tile, True)
    assert [b[0] for b in out["stack"]["bands"]] == ["a", "b"] and out["stack"]["bands"][0][3] == "#112233"
    assert np.allclose(out["stack"]["bands"][1][2], 1 + 2 * x1)                                     # b interpolated onto a's x
    assert [c[0] for c in out["curves"]] == ["r"] and out["styles"] == [{"twin": True}] and out["yscale"] == "lin"


def test_stacked_panel_renders_in_both_renderers():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from splosh import layout_mpl
    L = lay.validate_layout({"name": "s", "ncols": 2, "cell": [260, 200], "panel": [{"id": "m", "type": "timeseries", "columns": ["cbd", "md1", "md2"], "stack": True},
                                                                                   {"id": "f", "type": "timeseries", "columns": ["cbd", "md1", "md2"], "stack": "fraction"}]})
    tiles = {p["id"]: lay.timeseries_data(p, _parts(), cur=3) for p in L["panel"]}
    rgb = lay.render_layout(L, tiles, "t")
    plain = lay.render_layout(L, {k: lay.timeseries_data(dict(p, stack=False), _parts(), cur=3) for k, p in zip(tiles, L["panel"])}, "t")
    assert rgb.shape == plain.shape and len(np.unique(rgb.reshape(-1, 3), axis=0)) > len(np.unique(plain.reshape(-1, 3), axis=0))     # the shaded zones add colours
    fig = layout_mpl.render(L, tiles)
    ax = fig.axes[0]
    assert len(ax.collections) == 3 and ax.get_ylim()[0] == 0 and any(l.get_label() == "total" for l in ax.lines)
    assert fig.axes[1].get_ylim() == (0.0, 1.05) and fig.axes[1].get_ylabel() == "fraction of the total"


def test_heatmap_symlog_and_log_y():
    d = {"name": "h", "panel": [{"id": "h", "type": "heatmap", "source": "s", "prefix": "dT_", "zscale": "symlog", "yscale": "log"}]}
    p = lay.validate_layout(d)["panel"][0]
    assert p["zscale"] == "symlog" and p["yscale"] == "log"
    with pytest.raises(lay.LayoutError, match="yscale"):
        lay.validate_layout({"name": "h", "panel": [dict(d["panel"][0], yscale="sqrt")]})
    assert lay.heat_cmap({"zscale": "symlog", "cmap": "magma"}) == "RdBu_r" and lay.heat_cmap({"zscale": "symlog", "cmap": "PuOr"}) == "PuOr"
    y = np.geomspace(0.1, 10, 6)
    z = np.outer(np.sin(y), np.linspace(-1, 1, 8))
    tile = {"heat": {"x": np.linspace(0, 1, 9), "y": y, "z": z, "zscale": "symlog", "cmap": "magma", "norm": "none"}, "curves": [], "styles": [], "xnow": 0.5, "yscale": "log"}
    L = lay.validate_layout(d)
    rgb = lay.render_layout(L, {"h": tile}, "t")
    assert rgb.dtype == np.uint8
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mc
    from splosh import layout_mpl
    fig = layout_mpl.render(L, {"h": tile})
    ax = fig.axes[0]
    assert ax.get_yscale() == "log" and isinstance(ax.collections[0].norm, mc.SymLogNorm) and ax.collections[0].norm.vmin == -ax.collections[0].norm.vmax
    lo, hi = ax.get_ylim()
    assert np.isclose(lo, y[0] ** 1.5 / y[1] ** 0.5) and np.isclose(hi, y[-1] ** 1.5 / y[-2] ** 0.5)
