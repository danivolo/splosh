"""overlay disc: the plane from the gas's angular momentum, the constant aspect ratio H/R and the inclination, on a synthetic tilted disc."""
import re

import h5py
import numpy as np

from splosh import overlays as O
from splosh.maps import View


def _disc(path, inc_deg=40.0, hr=0.1, n=20000):
    rng = np.random.default_rng(5)
    R = np.sqrt(rng.random(n)) * 1.0
    ph = rng.random(n) * 2 * np.pi
    z = rng.normal(size=n) * hr * R                      # thickness proportional to radius: H/R = hr
    x, y = R * np.cos(ph), R * np.sin(ph)
    v = np.stack([-np.sin(ph), np.cos(ph), np.zeros(n)], 1)        # rotation about +z
    i = np.radians(inc_deg); Rx = np.array([[1, 0, 0], [0, np.cos(i), -np.sin(i)], [0, np.sin(i), np.cos(i)]])
    P = np.stack([x, y, z], 1) @ Rx.T; V = v @ Rx.T
    with h5py.File(path, "w") as f:
        h = f.create_group("Header").attrs
        h.update({"Time": 0.0, "NumPart_ThisFile": np.array([n, 0, 0, 0, 0, 0]), "NumPart_Total": np.array([n, 0, 0, 0, 0, 0]), "MassTable": np.zeros(6)})
        g = f.create_group("PartType0")
        g["Coordinates"] = P; g["Velocities"] = V; g["Masses"] = np.full(n, 1e-4); g["SmoothingLength"] = np.full(n, 0.05)
        g["Density"] = np.ones(n); g["InternalEnergy"] = np.ones(n)
    return str(path)


def test_disc_plane_aspect_and_inclination(tmp_path):
    p = _disc(tmp_path / "snapshot_000.hdf5")
    res = O.compute_for_path(p, View(width=3.0, npix=64, center="xyz:0,0,0"), ["disc:R=1.0"])
    kinds = [i["kind"] for i in res["items"]]
    assert "polygon" in kinds and kinds.count("polyline") >= 3                       # midplane, two rims, outline
    leg = [i["text"] for i in res["items"] if i["kind"] == "legend"][0]
    hr = float(re.search(r"H/R = ([\d.]+)", leg).group(1)); inc = float(re.search(r"inclination (\d+)", leg).group(1))
    assert abs(hr - 0.1) < 0.02 and abs(inc - 40) <= 2
    poly = np.array([i for i in res["items"] if i["kind"] == "polygon"][0]["points"])
    assert abs(np.ptp(poly[:, 0]) - 2.0) < 0.05 and abs(np.ptp(poly[:, 1]) - 2.0 * np.cos(np.radians(40))) < 0.05   # the projected circle: an ellipse


def test_polygon_rasterised():
    rgb = np.zeros((50, 50, 3), np.uint8)
    O.draw_items(rgb, [{"kind": "polygon", "points": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], "color": [255, 255, 255], "alpha": 1.0}], [-1, 1, -1, 1])
    assert rgb[25, 25, 0] == 255 and rgb[2, 2, 0] == 0
