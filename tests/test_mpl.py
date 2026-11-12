"""splosh.mpl: maps drawn on matplotlib Axes with the viewer's colours, a shared colour bar, overlays as vector graphics, PDF output."""
import h5py
import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from splosh import colormaps, imaging             # noqa: E402
from splosh import mpl as avm                     # noqa: E402
from splosh.maps import View                      # noqa: E402
from splosh.mapstore import MapStore              # noqa: E402


def _snap(path, t=0.0, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)) * 0.4
    with h5py.File(path, "w") as f:
        h = f.create_group("Header").attrs
        h.update({"Time": t, "NumPart_ThisFile": np.array([n, 0, 0, 0, 0, 1]), "NumPart_Total": np.array([n, 0, 0, 0, 0, 1]), "MassTable": np.zeros(6)})
        g = f.create_group("PartType0")
        g["Coordinates"] = x; g["Velocities"] = np.cross(x, [0, 0, 1.0]); g["Masses"] = np.full(n, 1e-3); g["SmoothingLength"] = np.full(n, 0.15)
        g["Density"] = np.exp(-(x ** 2).sum(1)); g["InternalEnergy"] = np.ones(n)
        b = f.create_group("PartType5")
        b["Coordinates"] = np.array([[0.3, 0.1, 0.0]]); b["Masses"] = np.ones(1); b["Velocities"] = np.zeros((1, 3)); b["ParticleIDs"] = np.array([10 ** 7])
    return str(path)


def test_two_panels_shared_colorbar_and_overlays(tmp_path):
    a, b = _snap(tmp_path / "snapshot_000.hdf5"), _snap(tmp_path / "snapshot_001.hdf5", t=0.5, seed=2)
    store = MapStore(str(tmp_path / "store"))
    v = View(field="rho", width=3.0, npix=64, center="xyz:0,0,0")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    pa = avm.draw_map(axes[0], a, v, cmap="inferno", store=store, overlays=["contours:n=4", "vectors:v,grid=8", "particles:bh"])
    pb = avm.draw_map(axes[1], b, v, cmap="inferno", store=store, vmin=pa.limits[0], vmax=pa.limits[1])
    assert pb.limits == pa.limits and pa.extent == [-1.5, 1.5, -1.5, 1.5]
    # the drawn image is the viewer's stretch of the map, with the viewer's colour table
    assert np.allclose(np.asarray(pa.image.get_array()), imaging.stretch(pa.img, *pa.limits, pa.scale), equal_nan=True)
    assert np.allclose(pa.image.get_cmap()(1.0)[:3], colormaps.lut("inferno")[-1] / 255.0, atol=1 / 255)
    cb = avm.add_colorbar(pb, ax=list(axes))
    ticks = cb.get_ticks()
    assert len(ticks) >= 2 and all(0.0 <= t <= 1.0 for t in ticks)
    avm.add_time(pa); avm.add_scalebar(pa)
    assert len(axes[0].lines) + len(axes[0].collections) + len(axes[0].texts) > 3          # contours, BH, arrows, labels drawn as artists
    out = tmp_path / "pair.pdf"
    fig.savefig(out)
    assert out.stat().st_size > 2000 and out.read_bytes()[:4] == b"%PDF"
    plt.close(fig)


def test_reversed_colormap_and_nan_background(tmp_path):
    p = _snap(tmp_path / "snapshot_000.hdf5")
    fig, ax = plt.subplots()
    pan = avm.draw_map(ax, p, View(field="rho", width=20.0, npix=32, center="xyz:0,0,0"), cmap="viridis_r", store=MapStore(str(tmp_path / "s")))
    assert np.allclose(pan.image.get_cmap()(0.0)[:3], colormaps.lut("viridis")[-1] / 255.0, atol=1 / 255)
    plt.close(fig)
