"""Unit system of a snapshot: code units from the header, derived factors, and the unit of every map quantity.

Header keys read (case-insensitive, both spellings): UnitLength_in_cm | UnitLength_In_CGS, UnitMass_in_g | UnitMass_In_CGS,
UnitVelocity_in_cm_per_s | UnitVelocity_In_CGS, HubbleParam, ComovingIntegrationOn, Time (the scale factor when comoving).
Defaults when a key is missing (GIZMO/GADGET customary values, flagged by `Units.defaulted`): 1 kpc/h is NOT assumed; we use
length 3.085678e21 cm (1 kpc), mass 1.989e43 g (1e10 Msun), velocity 1e5 cm/s (1 km/s).

A quantity's dimension is a tuple of exponents (a, b, c) of (code length, code mass, code velocity); code time is length/velocity.
Unit systems:  'code' (factor 1, unit written in code units), 'cgs' (cm, g, cm/s, s), 'astro' (pc, Msun, km/s, Myr; column density Msun pc^-2).
Temperature has no code unit: it is always in K (it needs mu, gamma, so it is computed in cgs; factor 1 in every system).
Comoving snapshots: the flags and a, h are kept in the Units object but NO a/h factor is applied to any quantity (the factors are those of the stored code units)."""
from dataclasses import dataclass, field

# physical constants (cgs); the values GIZMO uses for the temperature conversion
PROTONMASS = 1.67262178e-24
BOLTZMANN = 1.38065e-16
PC = 3.0856775814913673e18
MSUN = 1.98841e33
MYR = 3.15576e13          # Julian year * 1e6
KM = 1.0e5

DEFAULTS = {"length_cm": 3.085678e21, "mass_g": 1.989e43, "velocity_cm_s": 1.0e5}
_KEYS = {"length_cm": ("UnitLength_in_cm", "UnitLength_In_CGS"), "mass_g": ("UnitMass_in_g", "UnitMass_In_CGS"),
         "velocity_cm_s": ("UnitVelocity_in_cm_per_s", "UnitVelocity_In_CGS")}
SYSTEMS = ("code", "cgs", "astro")

# named dimensions (exponents of code length, mass, velocity)
LENGTH, MASS, VELOCITY, TIME = (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, -1)
COLUMN, DENSITY, ENERGY_PER_MASS, PRESSURE, DIMLESS = (-2, 1, 0), (-3, 1, 0), (0, 0, 2), (-3, 1, 2), (0, 0, 0)
SPECIFIC_J = (1, 0, 1)
TEMPERATURE = "K"       # special: not a dimension
GAUSS = "G"             # special: magnetic field strength in gauss (not an integer power of length, mass, velocity)
FIXED = (TEMPERATURE, GAUSS)   # units that are the same in every system


def _lookup(header, names):
    low = {str(k).lower(): v for k, v in header.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] is not None:
            return float(low[n.lower()])
    return None


def _num(x):
    return f"{x:g}"


def _compose(symbols):
    """[(symbol, exponent)] -> 'g cm^-2' (zero exponents dropped, order as given)."""
    parts = []
    for s, e in symbols:
        if abs(e) < 1e-12:
            continue
        parts.append(s if abs(e - 1) < 1e-12 else f"{s}^{_num(e)}")
    return " ".join(parts) or "1"


@dataclass(frozen=True)
class Units:
    length_cm: float = DEFAULTS["length_cm"]
    mass_g: float = DEFAULTS["mass_g"]
    velocity_cm_s: float = DEFAULTS["velocity_cm_s"]
    hubble: float = None            # HubbleParam if present
    comoving: bool = False
    scale_factor: float = None      # header Time when comoving
    defaulted: tuple = ()           # names of the unit scales that were not in the header

    @classmethod
    def from_header(cls, header):
        header = header or {}
        vals, dflt = {}, []
        for k, names in _KEYS.items():
            v = _lookup(header, names)
            if v is None or not v > 0:
                v = DEFAULTS[k]; dflt.append(k)
            vals[k] = v
        comoving = bool(_lookup(header, ("ComovingIntegrationOn",)) or 0)
        return cls(hubble=_lookup(header, ("HubbleParam",)), comoving=comoving, scale_factor=_lookup(header, ("Time",)) if comoving else None,
                   defaulted=tuple(dflt), **vals)

    # ---- derived scales (code -> cgs) -------------------------------------------------------------------------------------
    @property
    def time_s(self):
        return self.length_cm / self.velocity_cm_s

    @property
    def density_cgs(self):
        return self.mass_g / self.length_cm ** 3

    @property
    def column_density_cgs(self):
        return self.mass_g / self.length_cm ** 2

    @property
    def energy_per_mass_cgs(self):
        return self.velocity_cm_s ** 2

    @property
    def pressure_cgs(self):
        return self.mass_g / self.length_cm ** 3 * self.velocity_cm_s ** 2

    def temperature_scale(self, mu, gamma=5.0 / 3.0):
        """K per unit of code specific internal energy: T = (gamma - 1) * mu * m_p / k_B * u * UnitVelocity^2."""
        return (gamma - 1.0) * mu * PROTONMASS / BOLTZMANN * self.velocity_cm_s ** 2

    # ---- quantities -------------------------------------------------------------------------------------------------------
    def cgs_factor(self, dim):
        a, b, c = dim
        return self.length_cm ** a * self.mass_g ** b * self.velocity_cm_s ** c

    def quantity_unit(self, dim, system="code"):
        """(unit string, factor) of a quantity of dimension `dim` (or 'K', 'G') in `system`: value_in_system = factor * value_in_code_units."""
        if system not in SYSTEMS:
            raise ValueError(f"units must be one of {SYSTEMS}")
        if dim is None:
            return "", 1.0
        if dim in FIXED:
            return dim, 1.0
        a, b, c = (float(x) for x in dim)
        if system == "code":
            return _compose([("code_mass", b), ("code_length", a), ("code_velocity", c)]), 1.0
        cgs = self.cgs_factor((a, b, c))
        if system == "cgs":
            return _compose([("g", b), ("cm", a + c), ("s", -c)]), cgs
        if (a, b, c) == (1.0, 0.0, -1.0):
            return "Myr", cgs / MYR
        unit = PC ** a * MSUN ** b * KM ** c
        return _compose([("Msun", b), ("pc", a), ("km", c), ("s", -c)]), cgs / unit

    def time_unit(self, system="code"):
        return self.quantity_unit(TIME, system)

    def length_unit(self, system="code"):
        return self.quantity_unit(LENGTH, system)

    def describe(self):
        return {"length_cm": self.length_cm, "mass_g": self.mass_g, "velocity_cm_s": self.velocity_cm_s, "time_s": self.time_s,
                "density_cgs": self.density_cgs, "column_density_cgs": self.column_density_cgs, "energy_per_mass_cgs": self.energy_per_mass_cgs,
                "pressure_cgs": self.pressure_cgs, "hubble_param": self.hubble, "comoving": self.comoving, "scale_factor": self.scale_factor,
                "defaulted": list(self.defaulted),
                "note": "comoving: stored code units, no a/h factor applied" if self.comoving else ""}


def quantity_meta(units, dim, quantity, system):
    """The metadata block every map carries for its quantity."""
    s, f = units.quantity_unit(dim, system)
    scode, _ = units.quantity_unit(dim, "code")
    if dim is None:
        scgs, fcgs = "", 1.0
    elif dim in FIXED:
        scgs, fcgs = dim, 1.0
    else:
        scgs, fcgs = units.quantity_unit(dim, "cgs")
    return {"quantity": quantity, "unit": s, "factor": f, "system": system, "code_unit": scode, "cgs_unit": scgs, "cgs_factor": fcgs,
            "known": dim is not None}
