import json
import os

import numpy as np
import pytest

from splosh import View, cli
from splosh.info import snapshot_info, format_info
from splosh.mapstore import MapStore, snapshot_hash, view_key
from splosh import imaging, colormaps


def test_info_counts_fields_units_and_json(snapfile):
    info = snapshot_info(snapfile)
    assert info["counts"] == {"gas": 3000, "bh": 2} and info["n_total"] == 3002
    assert info["time"] == pytest.approx(0.25) and info["periodic"] is False
    assert info["units"]["time_s"] == pytest.approx(3.0857e18 / 1e5)
    names = {d["name"] for d in info["fields"]["gas"]}
    assert {"Coordinates", "SmoothingLength", "Density", "ParticleIDs"} <= names
    g = info["gas"]
    assert g["mass"]["total"] == pytest.approx(3.0, rel=1e-5) and g["H"]["percentiles"]["50"] > 0.15 and "density_peak" in g
    assert info["bh"]["ids"] == [3, 7] and info["bh"]["mass"] == [1.0, 3.0]                       # sorted by id
    assert np.allclose(info["bh"]["center_of_mass"], (3 * np.array([1, .5, .1]) + np.array([-1, -.5, 0])) / 4)
    assert info["warnings"] == []
    json.dumps(info)                                                                              # machine readable
    txt = format_info(info)
    assert "gas 3000" in txt and "kernel support radius" in txt and "black holes: 2" in txt


def test_info_warns_on_split_snapshot(tmp_path):
    import h5py
    from conftest import write_snapshot
    p = str(tmp_path / "s.hdf5")
    write_snapshot(p)
    with h5py.File(p, "r+") as f:
        f["Header"].attrs["NumFilesPerSnapshot"] = 4
    assert any("split into 4 files" in w for w in snapshot_info(p)["warnings"])


def test_info_plugin_section(snapfile, monkeypatch):
    from splosh import plugins
    monkeypatch.setattr(plugins, "INFO_SECTIONS", [lambda path, info: ("demo", {"n": info["n_total"], "text": "hello"})])
    info = snapshot_info(snapfile)
    assert info["extra"]["demo"]["n"] == 3002 and "[demo]" in format_info(info)


def test_snapshot_hash_is_content_based(snapfile, tmp_path):
    import shutil
    h1 = snapshot_hash(snapfile)
    cp = str(tmp_path / "copy_elsewhere.hdf5")
    shutil.copy(snapfile, cp)
    assert snapshot_hash(cp) == h1                                  # same content, other name and mtime
    with open(cp, "r+b") as f:                                      # damage a sampled block
        f.seek(os.path.getsize(cp) // 2); f.write(b"\x01" * 70000)
    assert snapshot_hash(cp, full=True) != snapshot_hash(snapfile, full=True)


def test_mapstore_hit_miss_and_key(snapfile, tmp_path):
    st = MapStore(str(tmp_path / "store"))
    v = View(center="bh", width=5.0, npix=64)
    img, meta, hit = st.get_or_render(snapfile, v)
    assert not hit and img.shape == (64, 64) and meta["time"] == pytest.approx(0.25) and meta["render_seconds"] >= 0
    img2, meta2, hit2 = st.get_or_render(snapfile, v)
    assert hit2 and np.array_equal(img, img2, equal_nan=True) and meta2["key"] == meta["key"]
    _, _, hit3 = st.get_or_render(snapfile, v.with_(width=5.0000001))       # same key to 6 significant digits
    assert hit3
    _, _, hit4 = st.get_or_render(snapfile, v.with_(field="u"))
    assert not hit4
    s = st.stats()
    assert s["maps"] == 2 and s["hits"] == 2 and s["misses"] == 2
    assert view_key("a", v) != view_key("b", v) and view_key("a", v) != view_key("a", v.with_(npix=128))
    st.clear()
    assert st.stats()["maps"] == 0


def test_png_and_colormaps(tmp_path):
    img = np.linspace(1, 100, 64 * 64, dtype=np.float32).reshape(64, 64)
    img[0, 0] = np.nan
    p = str(tmp_path / "a.png")
    imaging.write_png(p, img, cmap="inferno", scale="log")
    data = open(p, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 100
    rgb, (lo, hi) = imaging.colorize(img, "gray", "linear", 1.0, 100.0)
    assert rgb.shape == (64, 64, 3)
    assert rgb[0, 1, 0] > rgb[-1, 1, 0]                             # rows are flipped: +y on top; the map's values increase with its row index
    for n in colormaps.names():
        assert colormaps.lut(n).shape == (256, 3)
    z = imaging.stretch(np.array([0.0, 1.0, 10.0, np.nan]), 1.0, 10.0, "log")
    assert np.isnan(z[0]) and z[1] == 0 and z[2] == 1 and np.isnan(z[3])


def test_cli_render_info_and_view_parsing(snapfile, tmp_path, capsys):
    store = str(tmp_path / "st")
    out = str(tmp_path / "m.png")
    cli.main(["render", snapfile, "--view", "width=6,npix=64,mode=slice", "--field", "u", "--center", "bh", "--align", "bh", "--out", out, "--store", store])
    assert os.path.getsize(out) > 100 and "rendered" in capsys.readouterr().out
    cli.main(["render", snapfile, "--view", "width=6,npix=64,mode=slice", "--field", "u", "--center", "bh", "--align", "bh", "--out", str(tmp_path / "m.npz"), "--store", store])
    assert "cache hit" in capsys.readouterr().out
    d = np.load(str(tmp_path / "m.npz"))
    assert d["img"].shape == (64, 64) and json.loads(str(d["meta"]))["view"]["mode"] == "slice"
    cli.main(["info", snapfile, "--json"])
    assert json.loads(capsys.readouterr().out)["counts"]["gas"] == 3000
    with pytest.raises(SystemExit):
        cli.main(["render", snapfile, "--view", "nokey=1", "--store", store])
