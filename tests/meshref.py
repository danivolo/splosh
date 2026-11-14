"""A tiny generic HDF5 mesh format and its kind (tests only): what a plugin for block-structured mesh data looks like (docs/KINDS.md).

File `*.mesh.h5`:
    /Header           attrs Time, Points (N x 3, optional: point masses), Markers (M x 3, optional)
    /Level_LL         attrs Dx, Origin (3), Centring ("vertex" | "cell")
    /Level_LL/Patch_PPPPP   attrs Lo, Hi (inclusive index range); one dataset per variable, stored [k][j][i] (z slowest)
A folder holding `*.mesh.h5` files is a run of kind "meshref"; other snapshot files next to them (particle dummies) are not its snapshots.
Importing this module registers the kind (SPLOSH_PLUGINS=meshref with this folder on PYTHONPATH does the same for a server)."""
import glob
import os

import h5py
import numpy as np

from splosh import plugins
from splosh.mesh import Level

EXT = ".mesh.h5"


def lin(x, y, z):
    return 1.0 + 2.0 * x - 3.0 * y + 0.5 * z


def pot(x, y, z):
    return 1.0 - 1.0 / (2.0 * np.sqrt(x * x + y * y + z * z + 0.1))


# level: (dx, origin, centring, [patches (lo, hi)]); level 1 has two boxes, its first box is split into two patches along x (a seam), level 0 along z
LEVELS = [
    (1.0, (-8.0, -8.0, -8.0), "vertex", [((0, 0, 0), (16, 16, 7)), ((0, 0, 8), (16, 16, 16))]),
    (0.5, (-8.0, -8.0, -8.0), "vertex", [((12, 12, 12), (15, 20, 20)), ((16, 12, 12), (20, 20, 20)), ((22, 14, 14), (26, 18, 18))]),
    (0.25, (-8.0, -8.0, -8.0), "vertex", [((28, 28, 28), (36, 36, 36))]),
]


def write_mesh(path, time=0.0, levels=LEVELS, shift=(0.0, 0.0, 0.0), points=((0.5, 0.25, 0.0),), markers=((1.0, 1.0, 0.0), (-1.0, 0.5, 0.0))):
    """variables: lin, pot (analytic, shifted by `shift`), level (the level index: shows which level a pixel took). Coarse data under finer levels are stored."""
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["Time"] = float(time)
        h.attrs["NumLevels"] = len(levels)
        if points:
            h.attrs["Points"] = np.asarray(points, float)
        if markers:
            h.attrs["Markers"] = np.asarray(markers, float)
        for li, (dx, origin, centring, patches) in enumerate(levels):
            g = f.create_group(f"Level_{li:02d}")
            g.attrs["Dx"] = dx
            g.attrs["Origin"] = np.asarray(origin, float)
            g.attrs["Centring"] = centring
            s = 0.0 if centring == "vertex" else 0.5
            for pi, (lo, hi) in enumerate(patches):
                p = g.create_group(f"Patch_{pi:05d}")
                p.attrs["Lo"], p.attrs["Hi"] = np.asarray(lo, np.int32), np.asarray(hi, np.int32)
                ax = [origin[a] + (np.arange(lo[a], hi[a] + 1) + s) * dx - shift[a] for a in range(3)]
                Z, Y, X = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")          # [k][j][i]
                p["lin"] = lin(X, Y, Z)
                p["pot"] = pot(X, Y, Z)
                p["level"] = np.full(X.shape, float(li))


class MeshRef:
    """the opened file: the data model of splosh.mesh (time, levels, variables, read)."""

    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, "r")
        hd = self.f["Header"].attrs
        self.time = float(hd["Time"])
        self.header = {"Time": self.time}
        self.levels, self._names = [], []
        for li in range(int(hd["NumLevels"])):
            g = self.f[f"Level_{li:02d}"]
            names = sorted(k for k in g if k.startswith("Patch_"))
            c = g.attrs["Centring"]
            self.levels.append(Level(dx=float(g.attrs["Dx"]), origin=tuple(g.attrs["Origin"]), centring=c.decode() if isinstance(c, bytes) else str(c),
                                     patches=[(tuple(g[n].attrs["Lo"]), tuple(g[n].attrs["Hi"])) for n in names]))
            self._names.append(names)
        first = self.f[f"Level_00/{self._names[0][0]}"]
        self.variables = sorted(first.keys())
        self.reads = []                        # what was read (tests check that a slice reads only what it needs)

    def read(self, var, level, patch, lo, hi):
        plo = self.levels[level].patches[patch][0]
        ds = self.f[f"Level_{level:02d}/{self._names[level][patch]}/{var}"]
        i0, j0, k0 = (lo[a] - plo[a] for a in range(3))
        i1, j1, k1 = (hi[a] - plo[a] + 1 for a in range(3))
        self.reads.append((var, level, patch, tuple(lo), tuple(hi)))
        return ds[k0:k1, j0:j1, i0:i1].transpose(2, 1, 0)          # partial read, axes to (x, y, z)

    def points(self):
        hd = self.f["Header"].attrs
        return {"pos": np.asarray(hd["Points"]).reshape(-1, 3)} if "Points" in hd else None

    def markers(self):
        hd = self.f["Header"].attrs
        return [{"name": "marks", "pos": np.asarray(hd["Markers"]).reshape(-1, 3), "labels": ["A", "B"], "color": "red"}] if "Markers" in hd else []

    def close(self):
        self.f.close()


def detect(folder):
    return bool(glob.glob(os.path.join(folder, "*" + EXT)))


def snapshots(folder):
    return glob.glob(os.path.join(folder, "*" + EXT))


def snap_time(path):
    with h5py.File(path, "r") as f:
        return float(f["Header"].attrs["Time"])


FIELDS = {
    "lin": {"needs": ("lin",), "label": "linear test field", "unit": ""},
    "pot": {"needs": ("pot",), "label": "1 - 1/(2r)", "unit": ""},
    "lin2": {"needs": ("lin", "pot"), "fn": lambda v: 2.0 * v["lin"] + 0.0 * v["pot"] + 1.0, "label": "2 lin + 1", "unit": "M"},
    "radius": {"needs": ("lin",), "fn": lambda v: np.sqrt(v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2) + 0.0 * v["lin"], "label": "distance from the origin", "unit": "M"},
}

plugins.register_kind("meshref", detect, snapshots, MeshRef, fields=FIELDS, units={"length": "M", "time": "M"},
                      default_view={"center": "origin", "align": "none", "mode": "slice", "field": "pot", "width": 8.0, "npix": 64}, label="mesh test",
                      time=snap_time, doc="the tests' reference mesh format")
