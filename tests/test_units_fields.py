"""Units layer and built-in derived fields, against hand-computed analytic values on small synthetic snapshots."""
import numpy as np
import h5py
import pytest

from splosh import plugins
from splosh.fields import mean_molecular_weight, format_fields, DEFAULT_GAMMA
from splosh.frames import make_frame
from splosh.info import snapshot_info, format_info
from splosh.io import load_snapshot
from splosh.maps import FieldContext, View, render_map, required_gas_fields, _field_values
from splosh.units import Units, PROTONMASS, BOLTZMANN, PC, MSUN, MYR, COLUMN, VELOCITY, TEMPERATURE

HDR = {"UnitLength_In_CGS": 3.0857e18, "UnitMass_In_CGS": 4.0e35, "UnitVelocity_In_CGS": 1e5}


def write(path, pos, vel=None, u=1.5, xe=1.0, rho=2.0, B=None, header=HDR, extra=None, hsml=0.4, mass=1e-3):
    n = len(pos)
    vel = np.zeros((n, 3)) if vel is None else vel
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        h.attrs["Time"] = 0.5
        for k, v in header.items():
            h.attrs[k] = v
        g = f.create_group("PartType0")
        g["Coordinates"] = np.asarray(pos, dtype=np.float64)
        g["Velocities"] = np.asarray(vel, dtype=np.float64)
        g["Masses"] = np.full(n, mass)
        g["SmoothingLength"] = np.full(n, hsml)
        g["Density"] = np.broadcast_to(np.asarray(rho, dtype=np.float64), (n,)).copy()
        g["InternalEnergy"] = np.broadcast_to(np.asarray(u, dtype=np.float64), (n,)).copy()
        if xe is not None:
            g["ElectronAbundance"] = np.broadcast_to(np.asarray(xe, dtype=np.float64), (n,)).copy()
        if B is not None:
            g["MagneticField"] = np.asarray(B, dtype=np.float64)
        for k, v in (extra or {}).items():
            g[k] = v
    return str(path)


def cloud(n=400, seed=3):
    return np.random.default_rng(seed).uniform(-1, 1, size=(n, 3))


def values(path, field, **kw):
    view = View(center="xyz:0,0,0", field=field, **kw)
    snap = load_snapshot(path, gas_fields=tuple(sorted(required_gas_fields(view))))
    return _field_values(field, FieldContext(snap, make_frame(snap, view.center, view.align), view)), snap


# ---- units -----------------------------------------------------------------------------------------------------------------
def test_units_from_header_both_spellings_and_defaults():
    a = Units.from_header({"UnitLength_In_CGS": 2.0, "UnitMass_In_CGS": 3.0, "UnitVelocity_In_CGS": 5.0, "HubbleParam": 0.7})
    b = Units.from_header({"UnitLength_in_cm": 2.0, "UnitMass_in_g": 3.0, "UnitVelocity_in_cm_per_s": 5.0})
    assert (a.length_cm, a.mass_g, a.velocity_cm_s, a.hubble) == (2.0, 3.0, 5.0, 0.7) and (b.length_cm, b.mass_g, b.velocity_cm_s) == (2.0, 3.0, 5.0)
    assert a.time_s == pytest.approx(0.4) and a.density_cgs == pytest.approx(3.0 / 8) and a.column_density_cgs == pytest.approx(3.0 / 4)
    assert a.energy_per_mass_cgs == 25.0 and a.pressure_cgs == pytest.approx(3.0 / 8 * 25)
    d = Units.from_header({})
    assert d.length_cm == 3.085678e21 and set(d.defaulted) == {"length_cm", "mass_g", "velocity_cm_s"} and not d.comoving
    assert Units.from_header({"ComovingIntegrationOn": 1, "Time": 0.5}).scale_factor == 0.5


def test_quantity_units_strings_and_factors():
    u = Units(length_cm=PC, mass_g=MSUN, velocity_cm_s=KM_S if (KM_S := 1e5) else 0)
    assert u.quantity_unit(COLUMN, "cgs") == ("g cm^-2", pytest.approx(MSUN / PC ** 2))
    s, f = u.quantity_unit(COLUMN, "astro")
    assert s == "Msun pc^-2" and f == pytest.approx(1.0)                 # code = pc, Msun, km/s: identity
    assert u.quantity_unit(VELOCITY, "astro") == ("km s^-1", pytest.approx(1.0)) and u.quantity_unit(VELOCITY, "cgs") == ("cm s^-1", pytest.approx(1e5))
    s, f = u.time_unit("astro")
    assert s == "Myr" and f == pytest.approx(PC / 1e5 / MYR) and u.time_unit("cgs")[0] == "s"
    assert u.quantity_unit(TEMPERATURE, "astro") == ("K", 1.0)
    assert u.quantity_unit(COLUMN, "code")[1] == 1.0
    with pytest.raises(ValueError):
        u.quantity_unit(COLUMN, "si")


# ---- temperature, mu ---------------------------------------------------------------------------------------------------------
def test_mu_formula():
    assert mean_molecular_weight(0.0) == pytest.approx(4 / (1 + 3 * 0.76))            # neutral
    assert mean_molecular_weight(1.0) == pytest.approx(4 / (1 + 3 * 0.76 + 4 * 0.76))    # H fully ionised
    y = (1 - 0.76) / (4 * 0.76)                                                     # same through the helium number fraction
    assert mean_molecular_weight(0.3) == pytest.approx((1 + 4 * y) / (1 + y + 0.3))
    assert mean_molecular_weight(1.0, xh=1.0) == pytest.approx(0.5)                   # pure ionised hydrogen


def test_temperature_from_u_mu(tmp_path):
    p = write(tmp_path / "s.h5", cloud(), u=1.5, xe=1.0)
    T, _ = values(p, "T")
    mu = 4 / (1 + 3 * 0.76 + 4 * 0.76)
    expect = (5 / 3 - 1) * mu * PROTONMASS / BOLTZMANN * 1.5 * 1e10
    assert np.allclose(T, expect, rtol=2e-6)
    assert 70 < expect < 80                                                          # u = 1.5 (km/s)^2 = 1.5e10 erg/g: T ~ 77 K for a fully ionised gas, sanity of the magnitude
    T2, _ = values(p, "T", gamma=1.4)
    assert np.allclose(T2, expect * 0.4 / (2 / 3), rtol=2e-6)
    T3, _ = values(p, "T", mu=0.5)
    assert np.allclose(T3, (2 / 3) * 0.5 * PROTONMASS / BOLTZMANN * 1.5 * 1e10, rtol=2e-6)


def test_temperature_per_particle_electron_abundance_and_velocity_unit(tmp_path):
    xe = np.linspace(0.0, 1.2, 50)
    p = write(tmp_path / "s.h5", cloud(50), u=2.0, xe=xe, header={**HDR, "UnitVelocity_In_CGS": 2e5})
    T, _ = values(p, "T")
    expect = (2 / 3) * mean_molecular_weight(xe) * PROTONMASS / BOLTZMANN * 2.0 * (2e5) ** 2
    assert np.allclose(T, expect, rtol=2e-6)


def test_missing_dataset_errors_name_the_dataset(tmp_path):
    p = write(tmp_path / "s.h5", cloud(20), xe=None)
    with pytest.raises(KeyError, match="ElectronAbundance"):
        load_snapshot(p, gas_fields=tuple(sorted(required_gas_fields(View(center="xyz:0,0,0", field="T")))))
    # with mu given, ElectronAbundance is not needed and not requested
    v = View(center="xyz:0,0,0", field="T", mu=0.6)
    assert "ElectronAbundance" not in required_gas_fields(v)
    T, _ = values(p, "T", mu=0.6)
    assert np.all(T > 0)
    p2 = write(tmp_path / "t.h5", cloud(20))
    with pytest.raises(KeyError, match="MagneticField"):
        load_snapshot(p2, gas_fields=tuple(sorted(required_gas_fields(View(center="xyz:0,0,0", field="beta")))))
    snap = load_snapshot(p2, gas_fields=("Coordinates", "Velocities", "Masses", "SmoothingLength", "Density", "InternalEnergy"))
    with pytest.raises(KeyError, match="MagneticField"):                              # library use without the loader's check
        _field_values("beta", FieldContext(snap, make_frame(snap, "xyz:0,0,0"), View(field="beta")))


def test_loader_fetches_only_what_the_field_needs(tmp_path):
    p = write(tmp_path / "s.h5", cloud(20), extra={"Junk": np.zeros(20)})
    need = required_gas_fields(View(center="xyz:0,0,0", field="cs"))
    assert need == {"Coordinates", "SmoothingLength", "Masses", "InternalEnergy"}
    s = load_snapshot(p, gas_fields=tuple(sorted(need)))
    assert "ElectronAbundance" not in s and "Velocities" not in s
    assert required_gas_fields(View(center="xyz:0,0,0", field="T")) >= {"InternalEnergy", "ElectronAbundance"}
    assert required_gas_fields(View(center="xyz:0,0,0", field="mach", weight="volume")) >= {"Velocities", "InternalEnergy", "Density"}


# ---- sound speed, pressure, entropy, Mach --------------------------------------------------------------------------------------
def test_cs_pressure_entropy_mach_uniform_flow(tmp_path):
    u, rho, vflow = 1.5, 2.0, np.array([3.0, 0.0, 4.0])
    n = 100
    p = write(tmp_path / "s.h5", cloud(n), vel=np.tile(vflow, (n, 1)), u=u, rho=rho)
    g = 5 / 3
    cs = np.sqrt(g * (g - 1) * u)
    assert cs == pytest.approx(np.sqrt(1.5 * 10 / 9))
    assert np.allclose(values(p, "cs")[0], cs, rtol=1e-6)
    assert np.allclose(values(p, "pressure")[0], (g - 1) * rho * u, rtol=1e-6)
    assert np.allclose(values(p, "pressure")[0], 2.0)
    assert np.allclose(values(p, "entropy")[0], 2.0 / rho ** g, rtol=1e-6)         # P / rho^gamma
    assert np.allclose(values(p, "entropy", gamma=1.4)[0], 0.4 * u * rho ** -0.4, rtol=1e-6)
    # fixed centre: the frame velocity is 0, so the flow speed is 5
    assert np.allclose(values(p, "mach")[0], 5.0 / cs, rtol=1e-6)
    # centred on the mean velocity (the gas itself): relative Mach number 0
    v = View(center="mass-weighted", field="mach")
    snap = load_snapshot(p, gas_fields=tuple(sorted(required_gas_fields(v))))
    assert np.allclose(_field_values("mach", FieldContext(snap, make_frame(snap, "mass-weighted"), v)), 0.0, atol=1e-6)
    # an explicit frame velocity: centre 'xyz' has velocity 0, so use a rotated frame to check the rotation does not change |v|
    assert np.allclose(values(p, "mach", align="vec:1,1,0")[0], 5.0 / cs, rtol=1e-6)


# ---- angular momentum ---------------------------------------------------------------------------------------------------------------
def rigid(n=300, omega=0.7, c=(0.0, 0.0, 0.0), seed=5):
    r = cloud(n, seed)
    c = np.asarray(c, dtype=float)
    d = r - c
    v = omega * np.column_stack([-d[:, 1], d[:, 0], np.zeros(n)])
    return r, v, d


def test_jz_and_azimuthal_velocity_of_rigid_rotation(tmp_path):
    omega = 0.7
    r, v, d = rigid(omega=omega)
    p = write(tmp_path / "s.h5", r, vel=v)
    R2 = d[:, 0] ** 2 + d[:, 1] ** 2
    assert np.allclose(values(p, "jz")[0], omega * R2, rtol=1e-5, atol=1e-6)
    assert np.allclose(values(p, "vphi")[0], omega * np.sqrt(R2), rtol=1e-5)
    assert np.allclose(values(p, "vR")[0], 0.0, atol=1e-6)
    z = d[:, 2]
    jabs = omega * np.sqrt(R2 * (z ** 2 + R2))              # r x v = (-z vy, z vx, x vy - y vx), v = omega (-y, x, 0)
    assert np.allclose(values(p, "j")[0], jabs, rtol=1e-5, atol=1e-6)
    # invariance under the alignment rotation about z
    for fld in ("jz", "j", "vphi", "vR"):
        assert np.allclose(values(p, fld, align="vec:1,2,0")[0], values(p, fld)[0], rtol=1e-4, atol=1e-5)


def test_jz_about_an_offset_centre_and_radial_flow(tmp_path):
    c = (0.3, -0.2, 0.1)
    r, v, d = rigid(c=c)
    p = write(tmp_path / "s.h5", r, vel=v)
    view = View(center="xyz:0.3,-0.2,0.1", field="jz")
    snap = load_snapshot(p, gas_fields=tuple(sorted(required_gas_fields(view))))
    out = _field_values("jz", FieldContext(snap, make_frame(snap, view.center), view))
    assert np.allclose(out, 0.7 * (d[:, 0] ** 2 + d[:, 1] ** 2), rtol=1e-5, atol=1e-6)
    # pure radial flow v = k (x, y, 0): vR = k R, vphi = 0, jz = 0
    r = cloud(100, 9)
    p = write(tmp_path / "r.h5", r, vel=1.3 * np.column_stack([r[:, 0], r[:, 1], np.zeros(100)]))
    R = np.hypot(r[:, 0], r[:, 1])
    assert np.allclose(values(p, "vR")[0], 1.3 * R, rtol=1e-5) and np.allclose(values(p, "vphi")[0], 0, atol=1e-5) and np.allclose(values(p, "jz")[0], 0, atol=1e-5)


def test_radial_velocity_nan_on_the_axis(tmp_path):
    r = cloud(10)
    r[0] = (0.0, 0.0, 0.5)
    p = write(tmp_path / "s.h5", r, vel=np.ones((10, 3)))
    assert np.isnan(values(p, "vR")[0][0]) and np.isfinite(values(p, "vR")[0][1:]).all()


# ---- plasma beta --------------------------------------------------------------------------------------------------------------------
HDRB = {"UnitLength_in_cm": 3.085678e21, "UnitMass_in_g": 1.989e43, "UnitVelocity_in_cm_per_s": 1.0e5}


def _p_unit():
    return 1.989e43 / 3.085678e21 ** 3 * (1.0e5) ** 2          # code pressure unit in cgs (g cm^-1 s^-2), computed here independently of splosh


def _beta_cgs(b_out, bunit, p_code):
    """beta from the Gaussian-units definition: P_cgs / (B_gauss^2 / 8 pi), with B_gauss = B_out * UnitMagneticField_in_gauss (gizmo/io.c: B_out = B_int * UNIT_B_IN_GAUSS / bunit)."""
    return p_code * _p_unit() / ((b_out * bunit) ** 2 / (8.0 * np.pi))


def test_plasma_beta_gizmo_convention(tmp_path):
    n = 60
    B = np.tile([0.0, 3.0, 4.0], (n, 1))                                           # |B_out| = 5
    p = write(tmp_path / "s.h5", cloud(n), B=B, u=1.5, rho=2.0, header=HDRB)
    p_code = (5.0 / 3.0 - 1.0) * 2.0 * 1.5
    assert np.allclose(values(p, "beta", bunit=1.0)[0], _beta_cgs(5.0, 1.0, p_code), rtol=1e-6)      # run with UnitMagneticField_in_gauss = 1
    assert np.allclose(values(p, "beta", bunit=2.5)[0], _beta_cgs(5.0, 2.5, p_code), rtol=1e-6)      # the factor enters squared
    # GIZMO's own internal convention: B_int = B_out * bunit / sqrt(4 pi P_unit), P_B = B_int^2 / 2 in code units
    b_int = 5.0 * 1.0 / np.sqrt(4 * np.pi * _p_unit())
    assert np.allclose(values(p, "beta", bunit=1.0)[0], p_code / (0.5 * b_int ** 2), rtol=1e-6)
    assert "MagneticField" in required_gas_fields(View(field="beta"))


def test_beta_reads_the_unit_from_the_parameter_file_then_defaults(tmp_path):
    n = 20
    B = np.tile([0.0, 3.0, 4.0], (n, 1))
    p_code = (5.0 / 3.0 - 1.0) * 2.0 * 1.5
    p = write(tmp_path / "s.h5", cloud(n), B=B, u=1.5, rho=2.0, header=HDRB)
    assert np.allclose(values(p, "beta")[0], _beta_cgs(5.0, np.sqrt(4 * np.pi), p_code), rtol=1e-6)     # no file: GIZMO's default sqrt(4 pi)
    (tmp_path / "gizmo.params").write_text("Foo 3  % x\nUnitMagneticField_in_gauss  1.0   % 1 gauss\n")
    assert np.allclose(values(p, "beta")[0], _beta_cgs(5.0, 1.0, p_code), rtol=1e-6)                 # the parameter file next to the snapshot
    (tmp_path / "run.params-usedvalues").write_text("UnitMagneticField_in_gauss  4.0\n")
    assert np.allclose(values(p, "beta")[0], _beta_cgs(5.0, 4.0, p_code), rtol=1e-6)                 # usedvalues wins over gizmo.params
    assert np.allclose(values(p, "beta", bunit=1.0)[0], _beta_cgs(5.0, 1.0, p_code), rtol=1e-6)      # an explicit option wins over both


def test_bfield_strength_in_gauss_and_divergence_error(tmp_path):
    n = 40
    B = np.tile([0.0, 3.0, 4.0], (n, 1))                                           # |B_out| = 5
    divb = np.linspace(-1.0, 1.0, n)
    p = write(tmp_path / "s.h5", cloud(n), B=B, header=HDRB, hsml=0.4, extra={"DivergenceOfMagneticField": divb})
    assert np.allclose(values(p, "B", bunit=2.0)[0], 10.0)                         # gauss = |B_out| * UnitMagneticField_in_gauss
    assert np.allclose(values(p, "divB")[0], 0.4 * np.abs(divb) / 5.0)              # H |div B| / |B|, independent of the unit
    assert Units().quantity_unit("G", "cgs") == ("G", 1.0) and Units().quantity_unit("G", "astro") == ("G", 1.0)
    q = write(tmp_path / "t.h5", cloud(n), B=B, header=HDRB)                       # built without OUTPUT_BFIELD_DIVCLEAN_INFO
    with pytest.raises(KeyError, match="DivergenceOfMagneticField"):
        values(q, "divB")


def test_hydrogen_fraction_from_metallicity(tmp_path):
    n = 30
    Z = np.zeros((n, 11)); Z[:, 0] = 0.02; Z[:, 1] = 0.28                          # total metals 0.02, helium 0.28 -> X = 0.70
    p = write(tmp_path / "s.h5", cloud(n), u=2.0, rho=1.0, xe=0.0, extra={"Metallicity": Z}, header=HDRB)
    t = values(p, "T")[0]
    mu_x = lambda x: 4.0 / (1.0 + 3.0 * x)                                         # neutral gas: x_e = 0
    t_of = lambda mu: (5.0 / 3.0 - 1.0) * mu * PROTONMASS / BOLTZMANN * 2.0 * 1.0e10
    assert np.allclose(t, t_of(mu_x(0.70)), rtol=1e-9)                             # automatic: from Metallicity
    assert np.allclose(values(p, "T", xh=0.76)[0], t_of(mu_x(0.76)), rtol=1e-9)    # an explicit xh wins
    q = write(tmp_path / "s2.h5", cloud(n), u=2.0, rho=1.0, xe=0.0, header=HDRB)    # no Metallicity: 0.76
    assert np.allclose(values(q, "T")[0], t_of(mu_x(0.76)), rtol=1e-9)


# ---- maps carry units ---------------------------------------------------------------------------------------------------------------
def render(path, **kw):
    v = View(center="xyz:0,0,0", width=3.0, npix=16, **kw)
    return render_map(load_snapshot(path, gas_fields=tuple(sorted(required_gas_fields(v)))), v)


def test_map_unit_metadata_and_conversion(tmp_path):
    p = write(tmp_path / "s.h5", cloud(3000), u=1.5, xe=1.0, mass=1e-3, hsml=0.5)
    us = Units.from_header(HDR)
    img0, m0 = render(p, field="rho")
    assert m0["quantity_unit"]["unit"] == "code_mass code_length^-2" and m0["quantity_unit"]["factor"] == 1.0
    assert m0["quantity_unit"]["cgs_unit"] == "g cm^-2" and m0["quantity_unit"]["cgs_factor"] == pytest.approx(us.column_density_cgs)
    for system in ("cgs", "astro"):
        img, m = render(p, field="rho", units=system)
        q = m["quantity_unit"]
        assert q["system"] == system and np.allclose(img, img0 * q["factor"], rtol=1e-6, equal_nan=True)
    img, m = render(p, field="rho", units="cgs")
    assert m["quantity_unit"]["unit"] == "g cm^-2" and m["quantity_unit"]["factor"] == pytest.approx(4e35 / 3.0857e18 ** 2)
    img, m = render(p, field="rho", units="astro")
    assert m["quantity_unit"]["unit"] == "Msun pc^-2" and m["quantity_unit"]["factor"] == pytest.approx(4e35 / 3.0857e18 ** 2 / (MSUN / PC ** 2))
    sl, ms = render(p, field="rho", mode="slice", units="cgs")
    assert ms["quantity_unit"]["unit"] == "g cm^-3"
    v, mv = render(p, field="vx", units="astro")
    assert mv["quantity_unit"]["unit"] == "km s^-1" and mv["quantity_unit"]["factor"] == pytest.approx(1.0)       # 1e5 cm/s is 1 km/s
    assert mv["length_unit"]["unit"] == "pc" and mv["time_unit"]["unit"] == "Myr" and mv["time_physical"] == pytest.approx(0.5 * mv["time_unit"]["factor"])
    assert mv["extent_physical"][1] - mv["extent_physical"][0] == pytest.approx(3.0 * mv["length_unit"]["factor"])
    uimg, mu_ = render(p, field="u", units="cgs")
    assert mu_["quantity_unit"]["unit"] == "cm^2 s^-2" and np.nanmax(uimg) == pytest.approx(1.5e10, rel=1e-5)


def test_temperature_map_is_in_kelvin_in_every_system(tmp_path):
    p = write(tmp_path / "s.h5", cloud(3000), u=1.5, xe=1.0, hsml=0.5)
    expect = (2 / 3) * mean_molecular_weight(1.0) * PROTONMASS / BOLTZMANN * 1.5e10
    for system in ("code", "cgs", "astro"):
        img, m = render(p, field="T", units=system)
        assert m["quantity_unit"]["unit"] == "K" and m["quantity_unit"]["factor"] == 1.0
        assert np.nanmedian(img) == pytest.approx(expect, rel=1e-5)


def test_entropy_unit_follows_gamma(tmp_path):
    p = write(tmp_path / "s.h5", cloud(500), hsml=0.5)
    _, m = render(p, field="entropy", gamma=1.4, units="cgs")
    q = m["quantity_unit"]
    us = Units.from_header(HDR)
    assert q["cgs_factor"] == pytest.approx(us.cgs_factor((-3 * (1 - 1.4), 1 - 1.4, 2.0)))


def test_unknown_plugin_field_has_no_unit_and_is_not_converted(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins, "FIELDS", dict(plugins.FIELDS))
    monkeypatch.setattr(plugins, "FIELD_META", dict(plugins.FIELD_META))
    plugins.register_field("one_unitless", lambda ctx: np.ones(len(ctx.snap["Masses"])))
    p = write(tmp_path / "s.h5", cloud(1000), hsml=0.5)
    img, m = render(p, field="one_unitless", units="cgs")
    assert m["quantity_unit"]["known"] is False and np.nanmax(img) == pytest.approx(1.0, rel=1e-5)
    plugins.register_field("with_unit", lambda ctx: np.ones(len(ctx.snap["Masses"])), unit=VELOCITY, needs=lambda view: ("Velocities",))
    assert "Velocities" in required_gas_fields(View(field="with_unit"))
    img, m = render(p, field="with_unit", units="cgs")
    assert m["quantity_unit"]["unit"] == "cm s^-1" and np.nanmax(img) == pytest.approx(1e5, rel=1e-5)


def test_view_validation():
    for bad in ({"units": "si"}, {"gamma": 1.0}, {"xh": -0.1}, {"mu": -1.0}):
        with pytest.raises(ValueError):
            View(**bad).validate()
    assert View().units == "code" and View().gamma == pytest.approx(DEFAULT_GAMMA)


# ---- info, fields listing, CLI -------------------------------------------------------------------------------------------------------
def test_info_unit_system_and_temperature(tmp_path):
    p = write(tmp_path / "s.h5", cloud(200), u=1.5, xe=1.0)
    info = snapshot_info(p, units="astro")
    us = info["unit_system"]
    assert us["length_cm"] == 3.0857e18 and us["time_s"] == pytest.approx(3.0857e13) and us["defaulted"] == [] and us["display"] == "astro"
    expect = (2 / 3) * mean_molecular_weight(1.0) * PROTONMASS / BOLTZMANN * 1.5e10
    assert info["gas"]["T"]["percentiles"]["50"] == pytest.approx(expect, rel=1e-5)
    assert info["gas"]["in_units"]["density"]["unit"] == "Msun pc^-3"
    txt = format_info(info)
    assert "unit system:" in txt and "in astro:" in txt
    bare = write(tmp_path / "b.h5", cloud(20), header={})
    ib = snapshot_info(bare)
    assert ib["unit_system"]["defaulted"] and "WARNING: header has no" in format_info(ib)


def test_fields_listing():
    txt = format_fields()
    for name in ("T", "cs", "pressure", "entropy", "mach", "jz", "j", "vR", "vphi", "beta"):
        assert any(line.split()[0] == name for line in txt.splitlines())
    line = [l for l in txt.splitlines() if l.split()[0] == "beta"][0]
    assert "MagneticField" in line and "InternalEnergy" in line
    assert "ElectronAbundance" in [l for l in txt.splitlines() if l.split()[0] == "T"][0]


def test_cli_units_fields_info(tmp_path, capsys):
    from splosh.cli import main
    p = write(tmp_path / "s.h5", cloud(500), u=1.5, xe=1.0, hsml=0.5)
    out = str(tmp_path / "m.npz")
    main(["render", p, "--field", "T", "--units", "cgs", "--center", "xyz:0,0,0", "--width", "3", "--npix", "16", "--out", out, "--store", str(tmp_path / "st")])
    z = np.load(out)
    import json
    meta = json.loads(str(z["meta"]))
    assert meta["quantity_unit"]["unit"] == "K" and meta["view"]["units"] == "cgs"
    main(["fields"])
    assert "MagneticField" in capsys.readouterr().out
    main(["info", p, "--units", "cgs"])
    assert "unit system:" in capsys.readouterr().out
