import os

# keep pytest's temporary files where the user says (SPLOSH_TMP), not in the default RAM-backed temp directory
if os.environ.get("SPLOSH_TMP"):
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", os.environ["SPLOSH_TMP"])

import h5py
import numpy as np
import pytest


def write_snapshot(path, n_gas=3000, seed=1, bh_ids=(7, 3), bh_pos=((1.0, 0.5, 0.1), (-1.0, -0.5, 0.0)), bh_mass=(3.0, 1.0), bh_vel=((0.0, 1.0, 0.0), (0.0, -3.0, 0.0)), box=0.0, bh_mass_field=None, n_dm=0, n_star=0, bfield=None, time=0.25, comoving=0, units=True):
    """A small GIZMO-like snapshot: gas cloud (Gaussian, rho ~ mass / H^3) plus two black holes (ids 7 and 3 are NOT in ascending order in the file)."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n_gas, 3)) * [1.0, 1.0, 0.3]
    m = np.full(n_gas, 1e-3)
    H = (0.15 + 0.5 * rng.random(n_gas))
    rho = m / (H**3 * 0.1)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["Time"] = time
        h.attrs["BoxSize"] = box
        h.attrs["Redshift"] = 0.0
        h.attrs["ComovingIntegrationOn"] = comoving
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["NumPart_ThisFile"] = np.array([n_gas, 0, 0, 0, 0, len(bh_ids)])
        if units:
            h.attrs["UnitLength_In_CGS"] = 3.0857e18
            h.attrs["UnitMass_In_CGS"] = 4.0e35
            h.attrs["UnitVelocity_In_CGS"] = 1e5
        h.attrs["Kernel_Function_ID"] = 3
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Masses"] = m.astype(np.float32)
        g["SmoothingLength"] = H.astype(np.float32)
        g["Density"] = rho.astype(np.float32)
        g["InternalEnergy"] = (2.0 + pos[:, 0] ** 2 + 0.5 * pos[:, 1]).astype(np.float32)
        g["Velocities"] = np.column_stack([np.ones(n_gas), 0.5 * pos[:, 0], np.zeros(n_gas)]).astype(np.float32)
        g["ParticleIDs"] = np.arange(1000, 1000 + n_gas).astype(np.uint32)
        if bfield is not None:
            g["MagneticField"] = np.tile(np.asarray(bfield, dtype=np.float32), (n_gas, 1))
        for t, n, off in ((1, n_dm, 100000), (4, n_star, 200000)):
            if n:
                r2 = np.random.default_rng(10 + t)
                q = f.create_group(f"PartType{t}")
                q["Coordinates"] = r2.normal(size=(n, 3)) * [1.5, 1.5, 0.5]
                q["Velocities"] = r2.normal(size=(n, 3)).astype(np.float32)
                q["Masses"] = np.full(n, 0.01, np.float32)
                q["ParticleIDs"] = np.arange(off, off + n).astype(np.uint32)
        b = f.create_group("PartType5")
        b["Coordinates"] = np.array(bh_pos, dtype=np.float64)
        b["Velocities"] = np.array(bh_vel, dtype=np.float32)
        b["Masses"] = np.array(bh_mass, dtype=np.float32)
        b["BH_Mass"] = np.array(bh_mass if bh_mass_field is None else bh_mass_field, dtype=np.float32) * (0.9 if bh_mass_field is None else 1.0)
        b["ParticleIDs"] = np.array(bh_ids, dtype=np.uint32)


@pytest.fixture
def snapfile(tmp_path):
    p = tmp_path / "snap_000.hdf5"
    write_snapshot(str(p))
    return str(p)


@pytest.fixture
def snapdir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    for k in range(3):
        write_snapshot(str(d / f"snap_{k:03d}.hdf5"), seed=1 + k, bh_pos=((1.0 + 0.1 * k, 0.5, 0.1), (-1.0 - 0.1 * k, -0.5, 0.0)))
    return str(d)


def lattice_snapshot(vel, n=48, half=3.0, nz=4, zhalf=0.3, B=None, ne=None):
    """A snapshot dict (what io.load_snapshot returns, no file): a regular lattice of equal-mass gas particles filling [-half, half]^2 x [-zhalf, zhalf], kernel H = 2.5 lattice spacings,
    density = mass / cell volume, velocity vel(x, y) -> (vx, vy) (vz = 0), optional uniform magnetic field B = (Bx, By, Bz), ElectronAbundance ne. Centre it with the frame rule 'xyz:0,0,0'."""
    d = 2.0 * half / n
    g = (np.arange(n) + 0.5) * d - half
    x, y, z = np.meshgrid(g, g, (np.arange(nz) + 0.5) * (2 * zhalf / nz) - zhalf, indexing="ij")
    pos = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    N = len(pos)
    v = np.zeros((N, 3))
    v[:, 0], v[:, 1] = vel(pos[:, 0], pos[:, 1])
    cellvol = d * d * (2 * zhalf / nz)
    snap = {"time": 0.5, "path": "synthetic", "header": {"BoxSize": 0.0, "UnitLength_In_CGS": 3.0857e18, "UnitMass_In_CGS": 1.989e33, "UnitVelocity_In_CGS": 1e5},
            "Coordinates": pos, "Velocities": v.astype(np.float32), "Masses": np.full(N, 1.0, np.float32), "SmoothingLength": np.full(N, 2.5 * d, np.float32),
            "Density": np.full(N, 1.0 / cellvol, np.float32), "InternalEnergy": np.full(N, 3.0, np.float32), "ParticleIDs": np.arange(N, dtype=np.uint32) + 5000}
    if B is not None:
        snap["MagneticField"] = np.tile(np.asarray(B, np.float32), (N, 1))
    if ne is not None:
        snap["ElectronAbundance"] = np.full(N, ne, np.float32)
    return snap


@pytest.fixture(autouse=True)
def _private_recent(tmp_path, monkeypatch):
    """the server's list of recently opened folders goes to the test's folder, never to the user's own file"""
    monkeypatch.setenv("SPLOSH_RECENT", str(tmp_path / "recent.json"))
    try:
        from splosh import server
        monkeypatch.setattr(server, "RECENT_FILE", str(tmp_path / "recent.json"))
    except ImportError:
        pass
