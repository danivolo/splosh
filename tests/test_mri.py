"""The core product `mri` (butterfly, dynamo columns, Maxwell alpha per annulus) and the field spectra of `modes`, on synthetic discs with known answers."""
import h5py
import numpy as np
import pytest

from splosh import plugins, products as P
from test_disc import kepler_disc, write


@pytest.fixture
def run(tmp_path):
    (tmp_path / "output").mkdir()
    return tmp_path


def b_code_factor(path):
    """|B| in code units (mu_0 = 1) per unit of the MagneticField dataset, for the test header."""
    from splosh.io import load_snapshot
    from splosh.mri import b_code
    s = load_snapshot(path, gas_fields=("Coordinates", "Masses", "MagneticField"))
    return float(b_code(s)[0, 0] / np.asarray(s["MagneticField"])[0, 0])


def test_butterfly_antisymmetric_bphi(run):
    """B = B0 sign(z) (e_phi - 0.5 e_R): B_phi(z) flips sign at the midplane, the upper and lower means are +-B0, the mean over the annulus ~ 0, and
    alpha_M = 0.5 B0^2 / <P> (P = rho c_s^2 / gamma with c_s = 0.1 v_K)."""
    pos, vel, m, rho, u = kepler_disc(n=100_000)
    R = np.hypot(pos[:, 0], pos[:, 1])
    s = np.sign(pos[:, 2])
    eR = np.stack([pos[:, 0] / R, pos[:, 1] / R, 0 * R], 1)
    ephi = np.stack([-pos[:, 1] / R, pos[:, 0] / R, 0 * R], 1)
    B0 = 1e-3
    p = write(run / "output" / "snapshot_000.hdf5", pos, vel, m, rho, u, [0, 0, 0], [0, 0, 0], 1.0, B=B0 * s[:, None] * (ephi - 0.5 * eR))
    plugins.load_plugins()
    out = P.PRODUCTS["mri"]["fn"](p, P.params_of("mri", {"radii": [4.0], "nz": 20}))["scalars"]
    b = B0 * b_code_factor(p)
    # the halves are split at the annulus' mean height (not exactly 0 for a random disc): cells right at the midplane may switch side
    assert out["Bphiup4"] == pytest.approx(b, rel=0.03) and out["Bphilo4"] == pytest.approx(-b, rel=0.03)
    assert abs(out["Bphi4"]) < 0.15 * b                                  # volume weights: the few cells near 3 H dominate the noise
    assert out["BR4"] == pytest.approx(0.0, abs=0.1 * b)
    zs = sorted((float(k[len("Bphi4z_"):]), v) for k, v in out.items() if k.startswith("Bphi4z_"))
    assert len(zs) == 20 and all(np.isnan(v) or abs(z) < 0.5 or (np.sign(v) == np.sign(z) and abs(abs(v) - b) < 1e-4 * b) for z, v in zs)
    assert out["H4"] == pytest.approx(0.1, rel=0.05)                     # rms height / R of the Gaussian disc (H/R = 0.1)
    sel = (R > 3.6) & (R < 4.4) & (np.abs(pos[:, 2]) < 3 * out["H4"] * 4)
    V = m / rho
    Pg = (2 / 3) * rho * u
    assert out["aM4"] == pytest.approx(0.5 * b * b * V[sel].sum() / (Pg * V)[sel].sum(), rel=0.02)
    assert out["beta4"] == pytest.approx((Pg * V)[sel].sum() / (0.5 * 1.25 * b * b * V[sel].sum()), rel=0.02)


def test_butterfly_hydro_run_is_empty(run):
    pos, vel, m, rho, u = kepler_disc(n=5_000)
    p = write(run / "output" / "snapshot_000.hdf5", pos, vel, m, rho, u, [0, 0, 0], [0, 0, 0], 1.0)
    assert not P.PRODUCTS["mri"]["applies"](p)
    assert P.PRODUCTS["mri"]["fn"](p, P.params_of("mri")) == {}


def test_modes_field_spectra(run):
    """B_phi = B0 (1 + 0.3 cos 2 phi): |c_0| = B0, |c_2| = 0.15 B0, the other m at the noise level; the existing columns are unchanged by `fields`."""
    pos, vel, m, rho, u = kepler_disc(n=200_000)
    R = np.hypot(pos[:, 0], pos[:, 1])
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    ephi = np.stack([-pos[:, 1] / R, pos[:, 0] / R, 0 * R], 1)
    B0 = 1e-3
    rho = np.ones_like(rho)                       # equal volumes: the volume weights are the particle counts
    p = write(run / "output" / "snapshot_000.hdf5", pos, vel, m, rho, u, [0, 0, 0], [0, 0, 0], 1.0, B=B0 * (1 + 0.3 * np.cos(2 * phi))[:, None] * ephi)
    base = P.PRODUCTS["modes"]["fn"](p, P.params_of("modes", {"nbins": 6, "r_min": 1.0, "r_max": 10.0}))["scalars"]
    out = P.PRODUCTS["modes"]["fn"](p, P.params_of("modes", {"nbins": 6, "r_min": 1.0, "r_max": 10.0, "fields": ["Bphi", "rho", "vphi"]}))["scalars"]
    assert all(out[k] == v or (np.isnan(v) and np.isnan(out[k])) for k, v in base.items())
    b = B0 * b_code_factor(p)
    assert out["Bphim0"] == pytest.approx(b, rel=0.01)
    assert out["Bphim2"] == pytest.approx(0.15 * b, rel=0.05)
    assert max(out[f"Bphim{k}"] for k in (1, 3, 4, 5)) < 0.02 * b
    shells = sorted(k for k in out if k.startswith("Bphim2_"))
    assert len(shells) == 6 and all(abs(out[k] / (0.15 * b) - 1) < 0.1 for k in shells)
    assert out["rhom0"] == pytest.approx(1.0) and out["rhom2"] < 0.01          # noise of the random azimuths ~ 1 / sqrt(N)
    assert out["vphim1"] < 0.01 * out["vphim0"]                  # pure rotation: no m = 1 in v_phi
    with pytest.raises(ValueError):
        P.PRODUCTS["modes"]["fn"](p, P.params_of("modes", {"fields": ["nonsense"]}))


def test_family_ignores_underscored_suffixes():
    """the family A (A1 .. A8 against m) does not take the shell columns A1_4.5 (float() would read '1_4.5' as 14.5)."""
    from splosh.series import heat_columns

    class T:
        columns = {"A1": 0, "A2": 0, "A1_4.5": 0, "A2_10": 0}
    assert heat_columns(T, "A") == [(1.0, "A1"), (2.0, "A2")]
