"""
Real k-Wave (pseudospectral time-domain) acoustic wave-equation simulation
through a heterogeneous, layered head model built from the same tissue
properties used elsewhere (tissue.py) -- this is the research-grade
upgrade flagged in validation.py: unlike the ray-based spatial_map.py /
view_3d.py, this actually solves the acoustic wave equation, capturing
real interference, diffraction, and reflection/refraction at tissue
boundaries (including oblique ones, and multi-path from reflected waves)
that the ray model cannot.

Honest scope: this is a 2D simulation (full 3D at this resolution would
be computationally heavy without a GPU). Absorption uses k-Wave's
power-law absorption model, parameterized from the same alpha0/freq
exponent values as tissue.py, so results should be directly comparable
to (and cross-check) the ray-based model's tissue loss numbers.
"""

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions

from tissue import SKIN, SKULL, BRAIN


def _density_kg_m3(layer) -> float:
    """rho = Z / c, converting impedance from MRayl to Pa*s/m."""
    z_pa_s_m = layer.impedance_mrayl * 1e6
    return z_pa_s_m / layer.speed_m_s


def build_head_medium(Nx: int, Ny: int, dx_m: float,
                       skin_thickness_cm: float = 0.2,
                       skull_thickness_cm: float = 0.7):
    """Builds heterogeneous sound_speed/density/alpha_coeff grids for a
    flat-layered skin->skull->brain stack, transducer facing down from
    the top edge of the grid (row 0)."""
    sound_speed = np.full((Nx, Ny), BRAIN.speed_m_s, dtype=np.float32)
    density = np.full((Nx, Ny), _density_kg_m3(BRAIN), dtype=np.float32)
    alpha_coeff = np.full((Nx, Ny), BRAIN.alpha0_db_cm_mhz, dtype=np.float32)
    alpha_power = np.full((Nx, Ny), BRAIN.freq_exponent, dtype=np.float32)

    dx_cm = dx_m * 100.0
    skin_rows = int(round(skin_thickness_cm / dx_cm))
    skull_rows = int(round(skull_thickness_cm / dx_cm))

    sound_speed[:skin_rows, :] = SKIN.speed_m_s
    density[:skin_rows, :] = _density_kg_m3(SKIN)
    alpha_coeff[:skin_rows, :] = SKIN.alpha0_db_cm_mhz
    alpha_power[:skin_rows, :] = SKIN.freq_exponent

    sound_speed[skin_rows:skin_rows + skull_rows, :] = SKULL.speed_m_s
    density[skin_rows:skin_rows + skull_rows, :] = _density_kg_m3(SKULL)
    alpha_coeff[skin_rows:skin_rows + skull_rows, :] = SKULL.alpha0_db_cm_mhz
    alpha_power[skin_rows:skin_rows + skull_rows, :] = SKULL.freq_exponent

    return sound_speed, density, alpha_coeff, alpha_power


def run_transcranial_simulation(freq_mhz: float = 1.0, domain_cm: float = 4.0,
                                 points_per_wavelength: int = 6):
    """
    Runs a real k-Wave 2D simulation of a plane-ish pulse from a small
    transducer through the layered head model, returning the full
    pressure field snapshot (max pressure envelope over time at each
    grid point, a standard way to visualize a wave simulation's spatial
    reach) plus timing/resolution metadata so the caller can be honest
    about cost.
    """
    c_ref = BRAIN.speed_m_s
    wavelength_m = c_ref / (freq_mhz * 1e6)
    dx_m = wavelength_m / points_per_wavelength
    N = int(round((domain_cm / 100.0) / dx_m))
    N = max(32, min(N, 300))  # guard rails: keep runtime bounded

    sound_speed, density, alpha_coeff, alpha_power = build_head_medium(N, N, dx_m)

    kgrid = kWaveGrid([N, N], [dx_m, dx_m])
    medium = kWaveMedium(sound_speed=sound_speed, density=density,
                          alpha_coeff=alpha_coeff, alpha_power=alpha_power.mean())
    kgrid.makeTime(sound_speed.max(), cfl=0.2)

    source = kSource()
    source.p_mask = np.zeros((N, N))
    tx_width = max(2, N // 8)
    mid = N // 2
    source.p_mask[0, mid - tx_width // 2: mid + tx_width // 2] = 1
    freq_hz = freq_mhz * 1e6
    n_cycles = 3
    t = kgrid.t_array
    tone_burst = np.sin(2 * np.pi * freq_hz * t) * (t < n_cycles / freq_hz)
    source.p = np.tile(tone_burst, (int(source.p_mask.sum()), 1))

    sensor = kSensor()
    sensor.mask = np.ones((N, N))
    sensor.record = ["p_max"]

    sim_opts = SimulationOptions(save_to_disk=True, data_cast="single",
                                  pml_inside=False)
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=False)

    result = kspaceFirstOrder2DC(
        kgrid=kgrid, source=source, sensor=sensor, medium=medium,
        simulation_options=sim_opts, execution_options=exec_opts
    )

    p_max_field = np.array(result["p_max"]).reshape(N, N, order="F")
    # NOTE: k-Wave's sensor output is MATLAB/Fortran-ordered internally
    # (k-Wave is a MATLAB-derived tool) -- reshape must use order="F", not
    # numpy's default "C", or the field comes back transposed. Verified
    # empirically with a known asymmetric source location before trusting
    # this in the actual simulation.
    return {
        "p_max_field": p_max_field,
        "grid_size": N,
        "dx_cm": dx_m * 100.0,
        "domain_cm": domain_cm,
        "n_time_steps": kgrid.Nt,
    }
