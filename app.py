"""
Neural Dust Link-Budget Simulator — single-file Streamlit app.
Combines tissue.py, piezo_model.py, diffraction.py, link_budget.py, sweep.py,
spatial_map.py, view_3d.py, and the Streamlit UI into one file to avoid
multi-file copy/paste issues.
"""

from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from scipy.special import j1
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ======================================================================
# --- from tissue.py ---
# ======================================================================
"""
Layered tissue acoustic properties for ultrasonic power/comms link modeling.

Attenuation follows the standard power-law form used in bioacoustics:
    alpha(f) = alpha0 * f^b      [dB/cm], f in MHz
Values below are representative literature figures (soft tissue ~1 dB/cm/MHz
rule of thumb; skull is a major outlier due to its much higher absorption
and scattering). These are approximations for engineering-level modeling,
not a substitute for a proper literature review per target site.
"""



@dataclass
class TissueLayer:
    name: str
    thickness_cm: float          # path length through this layer
    alpha0_db_cm_mhz: float       # attenuation coefficient prefactor
    freq_exponent: float          # b in alpha(f) = alpha0 * f^b
    impedance_mrayl: float        # acoustic impedance, MRayl (for reflection loss)
    speed_m_s: float               # speed of sound in this layer

    def attenuation_db(self, freq_mhz: float) -> float:
        """Attenuation in dB for a single pass through this layer at freq_mhz."""
        alpha_db_per_cm = self.alpha0_db_cm_mhz * (freq_mhz ** self.freq_exponent)
        return alpha_db_per_cm * self.thickness_cm


# Representative literature values (approximate, MRayl impedance, dB/cm/MHz^b)
SKIN = TissueLayer("skin", thickness_cm=0.2, alpha0_db_cm_mhz=0.6, freq_exponent=1.1,
                    impedance_mrayl=1.6, speed_m_s=1600)
FAT = TissueLayer("fat", thickness_cm=0.5, alpha0_db_cm_mhz=0.6, freq_exponent=1.0,
                   impedance_mrayl=1.4, speed_m_s=1450)
MUSCLE = TissueLayer("muscle", thickness_cm=1.0, alpha0_db_cm_mhz=1.1, freq_exponent=1.1,
                      impedance_mrayl=1.7, speed_m_s=1580)
SKULL = TissueLayer("skull", thickness_cm=0.7, alpha0_db_cm_mhz=15.0, freq_exponent=1.3,
                     impedance_mrayl=7.8, speed_m_s=2900)  # dominant loss term
BRAIN = TissueLayer("brain", thickness_cm=1.0, alpha0_db_cm_mhz=0.6, freq_exponent=1.1,
                     impedance_mrayl=1.6, speed_m_s=1540)

# Two representative target paths
PATH_TRANSCRANIAL = [SKIN, SKULL, BRAIN]          # through the skull to cortex
PATH_PERIPHERAL = [SKIN, FAT, MUSCLE]              # e.g. peripheral nerve target


def reflection_loss_db(z1_mrayl: float, z2_mrayl: float) -> float:
    """Power reflection loss in dB at a normal-incidence boundary between
    two media of acoustic impedance z1 -> z2."""
    r = (z2_mrayl - z1_mrayl) / (z2_mrayl + z1_mrayl)
    power_reflected = r ** 2
    power_transmitted = max(1e-12, 1 - power_reflected)
    return -10 * (power_transmitted and __import__("math").log10(power_transmitted))


def path_attenuation_db(path, freq_mhz: float, include_reflections: bool = True) -> float:
    """Total one-way attenuation (absorption + boundary reflection losses)
    through a sequence of TissueLayer objects, assuming water/coupling gel
    (Z ~ 1.5 MRayl) as the medium before the first layer."""
    total_db = 0.0
    prev_z = 1.5  # coupling medium impedance
    for layer in path:
        total_db += layer.attenuation_db(freq_mhz)
        if include_reflections:
            total_db += reflection_loss_db(prev_z, layer.impedance_mrayl)
        prev_z = layer.impedance_mrayl
    return total_db


def path_length_cm(path) -> float:
    return sum(layer.thickness_cm for layer in path)
# ======================================================================
# --- from piezo_model.py ---
# ======================================================================
"""
Physically-grounded piezoelectric transducer model, replacing the previous
arbitrary log-quadratic aperture-ratio penalty with a two-effect model
based on real material physics:

  1. Electromechanical coupling limit (k_t^2) -- the fundamental cap on
     how much electrical energy a piezo element can convert to mechanical
     energy, set by the material's coupling coefficient.
  2. Acoustic impedance mismatch loss -- real transmission-coefficient
     loss between the piezo's acoustic impedance and the load (tissue),
     using the same normal-incidence formula already used in tissue.py.

Honest scope: this is still a simplified two-effect model, NOT a full
distributed KLM/Mason equivalent circuit (no matching-layer bandwidth
shaping, no frequency-dependent transformer ratio, no backing-layer
reflections). What it fixes: it now uses real piezo material impedance
and real mismatch physics instead of an arbitrary penalty function with
no physical basis -- that was the single biggest accuracy gap flagged
in validation.py, and this closes most of it.
"""


# Representative PZT-5H material constants (typical literature values for
# a common piezoceramic used in ultrasonic transducers).
PZT5H_DENSITY_KG_M3 = 7500.0
PZT5H_SOUND_SPEED_M_S = 4600.0          # thickness-mode longitudinal speed
PZT5H_KT = 0.50                          # thickness-mode coupling coefficient
PZT5H_IMPEDANCE_MRAYL = (PZT5H_DENSITY_KG_M3 * PZT5H_SOUND_SPEED_M_S) / 1e6  # ~34.5 MRayl

COUPLING_MEDIUM_IMPEDANCE_MRAYL = 1.5  # water/gel, TX-side coupling
TISSUE_TYPICAL_IMPEDANCE_MRAYL = 1.6   # representative soft-tissue/brain


def power_transmission_coefficient(z1_mrayl: float, z2_mrayl: float) -> float:
    """Normal-incidence power transmission coefficient (0-1) between two
    acoustic impedances. Same physics as tissue.reflection_loss_db, kept
    separate here so this module has no dependency on tissue.py."""
    r = (z2_mrayl - z1_mrayl) / (z2_mrayl + z1_mrayl)
    return max(1e-12, 1 - r ** 2)


def tx_efficiency_db(matched: bool = True,
                      load_impedance_mrayl: float = COUPLING_MEDIUM_IMPEDANCE_MRAYL) -> float:
    """
    TX transducer efficiency (electrical -> acoustic), in dB (negative = loss).
    Assumes a well-designed external TX transducer uses a quarter-wave
    matching layer (standard practice, physically justified since TX size
    isn't constrained the way a sub-mm mote is) -- so mismatch loss is
    small, and the coupling-coefficient limit dominates.
    """
    coupling_limit_db = 10 * np.log10(PZT5H_KT ** 2)
    if matched:
        # An ideal quarter-wave matching layer (Z_match = sqrt(Z_p * Z_load))
        # brings transmission close to 1; residual loss is small (~1 dB,
        # representative of real matched transducer insertion loss).
        residual_mismatch_db = -1.0
    else:
        T = power_transmission_coefficient(PZT5H_IMPEDANCE_MRAYL, load_impedance_mrayl)
        residual_mismatch_db = 10 * np.log10(T)
    return coupling_limit_db + residual_mismatch_db


def mote_rx_efficiency_db(load_impedance_mrayl: float = TISSUE_TYPICAL_IMPEDANCE_MRAYL) -> float:
    """
    Mote (RX) receive-coupling efficiency, in dB (negative = loss).
    Motes are assumed UNMATCHED -- no room for a matching layer at
    sub-mm scale -- so the full acoustic impedance mismatch between bare
    PZT (~34.5 MRayl) and tissue (~1.6 MRayl) applies. This is why mote
    coupling loss dominates the link budget, consistent with real neural
    dust literature calling out small-transducer coupling as the
    limiting factor.
    """
    coupling_limit_db = 10 * np.log10(PZT5H_KT ** 2)
    T = power_transmission_coefficient(PZT5H_IMPEDANCE_MRAYL, load_impedance_mrayl)
    mismatch_db = 10 * np.log10(T)
    return coupling_limit_db + mismatch_db


def backscatter_modulation_loss_db(load_impedance_mrayl: float = TISSUE_TYPICAL_IMPEDANCE_MRAYL) -> float:
    """
    Derives backscatter modulation loss from the actual change in the
    mote's acoustic reflectivity when its electrical port is switched
    between short-circuit and open-circuit states -- replacing the
    previous fixed 6 dB placeholder.

    Physical basis: piezoelectric stiffening under open-circuit
    conditions increases effective acoustic impedance by a factor of
    ~1/sqrt(1 - k_t^2) relative to the short-circuit value (standard
    piezoelectric resonator relation between open- and short-circuit
    elastic stiffness). The two states present different acoustic
    impedances to the incident wave, producing different reflection
    coefficients; the difference between them sets the modulation depth.
    """
    z_sc = PZT5H_IMPEDANCE_MRAYL
    z_oc = PZT5H_IMPEDANCE_MRAYL / np.sqrt(1 - PZT5H_KT ** 2)

    def reflection_coeff(z_piezo):
        return (z_piezo - load_impedance_mrayl) / (z_piezo + load_impedance_mrayl)

    r_sc = reflection_coeff(z_sc)
    r_oc = reflection_coeff(z_oc)
    modulation_index = abs(r_oc - r_sc) / 2.0  # normalized to ideal full-swing reflector
    modulation_index = max(modulation_index, 1e-6)
    return 20 * np.log10(modulation_index)
# ======================================================================
# --- from diffraction.py ---
# ======================================================================
"""
Real circular-piston far-field diffraction pattern, replacing the Gaussian
lateral-falloff approximation with the standard, well-established closed-form
directivity function for a baffled circular piston transducer (Kinsler &
Frey / Blackstock -- standard acoustics textbook result, low risk of error
since it's an exact analytical solution, not a fitted approximation):

    D(theta) = | 2*J1(x) / x |,   x = k*a*sin(theta)

where J1 is the Bessel function of the first kind order 1, k is the
wavenumber, a is the piston (transducer) radius, and theta is the angle
off-axis. This produces genuine sidelobes and a real first-null angle,
unlike the Gaussian approximation it replaces.

Honest scope: this is the FAR-FIELD directivity pattern specifically. Near
field is still handled by the existing collimation approximation in
link_budget.py / spatial_map.py -- true near-field diffraction (Fresnel
zone ripples) would need a full Rayleigh-Sommerfeld numerical integral,
which is a further upgrade beyond this pass. This fix targets the "no
sidelobes, no true beam pattern" gap specifically for the region where it
matters most (far field, where most of the tissue path lives at typical
implant depths).
"""



def piston_directivity(angle_rad, aperture_radius_cm: float, freq_mhz: float,
                        c_m_s: float = 1540.0):
    """Returns directivity gain (0-1, linear, not dB) for a circular piston
    transducer at the given off-axis angle(s). angle_rad can be a scalar
    or numpy array."""
    freq_hz = freq_mhz * 1e6
    k = 2 * np.pi * freq_hz / c_m_s          # wavenumber, 1/m
    a_m = aperture_radius_cm / 100.0
    x = k * a_m * np.sin(angle_rad)
    # handle x -> 0 (on-axis) where 2*J1(x)/x -> 1 by limit
    x_safe = np.where(np.abs(x) < 1e-9, 1e-9, x)
    directivity = np.abs(2 * j1(x_safe) / x_safe)
    directivity = np.where(np.abs(x) < 1e-9, 1.0, directivity)
    return directivity


def directivity_loss_db(lateral_offset_cm, depth_cm, aperture_radius_cm: float,
                         freq_mhz: float, c_m_s: float = 1540.0):
    """Convenience wrapper: given a field point's lateral offset and depth
    (both in cm) relative to the transducer, returns the directivity loss
    in dB (<=0). Depth is clamped away from zero to avoid a division
    singularity directly at the transducer face."""
    depth_safe = np.maximum(depth_cm, 1e-6)
    angle_rad = np.arctan2(lateral_offset_cm, depth_safe)
    d = piston_directivity(angle_rad, aperture_radius_cm, freq_mhz, c_m_s)
    d_safe = np.maximum(d, 1e-6)
    return 20 * np.log10(d_safe)


def first_null_angle_deg(aperture_radius_cm: float, freq_mhz: float,
                          c_m_s: float = 1540.0) -> float:
    """First-null half-angle of the main lobe (real, testable quantity --
    the classic sin(theta) = 1.22*lambda/(2a) result for a circular piston)."""
    freq_hz = freq_mhz * 1e6
    wavelength_cm = (c_m_s / freq_hz) * 100.0
    diameter_cm = 2 * aperture_radius_cm
    sin_theta = 1.22 * wavelength_cm / diameter_cm
    sin_theta = min(sin_theta, 1.0)  # guard against no-null case (very small aperture)
    return np.degrees(np.arcsin(sin_theta))
# ======================================================================
# --- from link_budget.py ---
# ======================================================================
"""
One-way and two-way (TX -> mote backscatter -> RX) ultrasonic link budget,
combining near-field collimation, far-field spreading, tissue attenuation,
and small-aperture receive coupling loss at the mote.
"""

derived_backscatter_loss_db = backscatter_modulation_loss_db  # preserved alias from link_budget.py

C_TISSUE = 1540.0  # m/s, representative sound speed for near-field calc


def near_field_length_cm(aperture_cm: float, freq_mhz: float, c_m_s: float = C_TISSUE) -> float:
    """z_nf = a^2 * f / c  (classic transducer near-field / Fresnel length)."""
    a_m = aperture_cm / 100.0
    freq_hz = freq_mhz * 1e6
    z_nf_m = (a_m ** 2) * freq_hz / c_m_s
    return z_nf_m * 100.0  # back to cm


def spreading_loss_db(z_cm: float, aperture_cm: float, freq_mhz: float) -> float:
    """
    Geometric spreading loss, stitched at the near/far transition:
      - within near field: ~0 dB (collimated beam, minimal spreading)
      - beyond near field: spherical spreading ~ 1/z^2 -> 20*log10(z/z_nf) dB
    This is a standard piecewise approximation, not a full diffraction model.
    """
    z_nf = near_field_length_cm(aperture_cm, freq_mhz)
    if z_cm <= z_nf:
        return 0.0
    return 20 * np.log10(z_cm / z_nf)


def piezo_receive_efficiency_db(*args, **kwargs):
    """Deprecated: superseded by piezo_model.mote_rx_efficiency_db, which
    uses real PZT impedance and coupling-coefficient physics instead of
    an arbitrary aperture-ratio penalty. Kept only so old imports don't
    break; not used internally anymore."""
    return mote_rx_efficiency_db()


def two_way_link_budget_db(depth_cm: float, freq_mhz: float,
                            tx_aperture_cm: float, mote_aperture_cm: float,
                            path=None, backscatter_modulation_loss_db: float = None):
    """
    Full two-way budget: TX -> tissue -> mote receive coupling ->
    backscatter modulation -> tissue -> RX at TX aperture.
    Returns dict of loss components (all positive dB = loss) and net budget.

    mote_rx_loss and backscatter_modulation_loss_db now come from
    piezo_model.py's physically-grounded model (real PZT impedance +
    electromechanical coupling limit) instead of the previous arbitrary
    aperture-ratio penalty. Pass backscatter_modulation_loss_db explicitly
    to override the derived value.
    """
    if path is None:
        path = PATH_TRANSCRANIAL
    if backscatter_modulation_loss_db is None:
        backscatter_modulation_loss_db = -derived_backscatter_loss_db()  # convert to positive=loss
    total_path_len = path_length_cm(path)
    # scale tissue path to requested depth (keep proportions, adjust length)
    scale = depth_cm / total_path_len if total_path_len > 0 else 1.0

    tissue_loss_one_way = path_attenuation_db(path, freq_mhz) * scale
    spread_loss_out = spreading_loss_db(depth_cm, tx_aperture_cm, freq_mhz)
    mote_rx_loss = -mote_rx_efficiency_db()  # positive = loss
    spread_loss_back = spreading_loss_db(depth_cm, mote_aperture_cm, freq_mhz)
    tissue_loss_return = tissue_loss_one_way  # symmetric path assumption

    total_loss_db = (tissue_loss_one_way + spread_loss_out + mote_rx_loss
                      + backscatter_modulation_loss_db + spread_loss_back
                      + tissue_loss_return)

    return {
        "tissue_loss_outbound_db": tissue_loss_one_way,
        "spreading_loss_outbound_db": spread_loss_out,
        "mote_receive_coupling_loss_db": mote_rx_loss,
        "backscatter_modulation_loss_db": backscatter_modulation_loss_db,
        "spreading_loss_return_db": spread_loss_back,
        "tissue_loss_return_db": tissue_loss_return,
        "total_two_way_loss_db": total_loss_db,
    }
# ======================================================================
# --- from sweep.py ---
# ======================================================================
"""
Sweep over frequency (and optionally mote aperture, coupled to frequency)
to find the operating point that maximizes received SNR at a target depth,
subject to a TX power budget.
"""


NOISE_FLOOR_DBM = -90.0  # representative electronic + thermal noise floor
TX_ELECTRICAL_TO_ACOUSTIC_DB = tx_efficiency_db(matched=True)  # now modeled, was previously ignored (assumed 100%)


MAX_MOTE_APERTURE_CM = 0.1  # 1 mm hard physical ceiling — this is what makes "dust" dust


def coupled_mote_aperture_cm(freq_mhz: float, aperture_over_lambda: float = 0.75) -> float:
    """Mote aperture sized to ~0.75 * wavelength for reasonable coupling,
    capped at a physical dust-scale ceiling. At low frequencies the
    'ideal' aperture would exceed that ceiling, forcing a real mismatch
    penalty — this is what creates the genuine frequency tradeoff instead
    of letting the model cheat by growing the mote arbitrarily large."""
    wavelength_cm = (C_TISSUE / (freq_mhz * 1e6)) * 100.0
    ideal = aperture_over_lambda * wavelength_cm
    return min(ideal, MAX_MOTE_APERTURE_CM)


def snr_db_at_frequency(freq_mhz: float, depth_cm: float, tx_power_dbm: float,
                         tx_aperture_cm: float = 1.0) -> float:
    mote_aperture_cm = coupled_mote_aperture_cm(freq_mhz)
    budget = two_way_link_budget_db(depth_cm, freq_mhz, tx_aperture_cm, mote_aperture_cm)
    acoustic_tx_power_dbm = tx_power_dbm + TX_ELECTRICAL_TO_ACOUSTIC_DB  # electrical -> acoustic conversion loss
    rx_power_dbm = acoustic_tx_power_dbm - budget["total_two_way_loss_db"]
    return rx_power_dbm - NOISE_FLOOR_DBM


def find_optimal_frequency(depth_cm: float, tx_power_dbm: float = 0.0,
                            freq_range_mhz=(0.2, 10.0)) -> dict:
    result = minimize_scalar(
        lambda f: -snr_db_at_frequency(f, depth_cm, tx_power_dbm),
        bounds=freq_range_mhz, method="bounded"
    )
    best_freq = result.x
    best_snr = -result.fun
    return {"optimal_freq_mhz": best_freq, "max_snr_db": best_snr,
            "mote_aperture_cm": coupled_mote_aperture_cm(best_freq)}


# ======================================================================
# --- from spatial_map.py ---
# ======================================================================
"""
2D spatial visualization: SNR/intensity field over a synthetic head
cross-section (skin -> skull -> brain), not just a 1D depth plot.

This maps your link-budget physics onto an actual head-shaped geometry so
you can see the beam collimate in the near field, spread past it, and take
the big skull hit -- geometry-aware, built from the same tissue/link_budget
modules (no new physics engine yet -- this is the "achievable now" tier;
k-Wave/FEM is the tier above this for actual wave-equation propagation).
"""


GRID_SIZE_CM = 8.0   # simulate an 8x8 cm cross-section
GRID_RES = 300         # pixels per side
TX_APERTURE_CM = 1.0
SKULL_THICKNESS_CM = 0.7
SKIN_THICKNESS_CM = 0.2


def build_head_mask(grid_res=GRID_RES, grid_size_cm=GRID_SIZE_CM):
    """Synthetic 2D head cross-section: concentric arcs approximating
    skin -> skull -> brain, transducer at top center facing down."""
    x = np.linspace(-grid_size_cm / 2, grid_size_cm / 2, grid_res)
    y = np.linspace(0, grid_size_cm, grid_res)  # y=0 is transducer face
    X, Y = np.meshgrid(x, y)
    depth = Y  # straight-down beam axis from transducer at y=0

    tissue_id = np.zeros_like(depth, dtype=int)  # 0=coupling gel,1=skin,2=skull,3=brain
    tissue_id[depth > 0] = 1
    tissue_id[depth > SKIN_THICKNESS_CM] = 2
    tissue_id[depth > SKIN_THICKNESS_CM + SKULL_THICKNESS_CM] = 3
    return X, Y, depth, tissue_id


def compute_snr_field(freq_mhz: float, tx_power_dbm: float = 0.0,
                       noise_floor_dbm: float = -90.0):
    X, Y, depth, tissue_id = build_head_mask()
    snr_field = np.zeros_like(depth)

    # Precompute one-way loss vs depth along the beam axis (1D), then
    # broadcast across the lateral (x) dimension weighted by a simple
    # off-axis directivity falloff (Gaussian approx of the main lobe).
    depths_1d = np.linspace(0.001, GRID_SIZE_CM, GRID_RES)
    onset_skin = SKIN_THICKNESS_CM
    onset_skull_end = SKIN_THICKNESS_CM + SKULL_THICKNESS_CM

    loss_1d = np.zeros_like(depths_1d)
    for i, d in enumerate(depths_1d):
        acc_loss = 0.0
        prev_z = 1.5
        if d > 0:
            skin_len = min(d, onset_skin)
            acc_loss += SKIN.attenuation_db(freq_mhz) * (skin_len / SKIN.thickness_cm)
            acc_loss += reflection_loss_db(prev_z, SKIN.impedance_mrayl)
            prev_z = SKIN.impedance_mrayl
        if d > onset_skin:
            skull_len = min(d, onset_skull_end) - onset_skin
            acc_loss += SKULL.attenuation_db(freq_mhz) * (skull_len / SKULL.thickness_cm)
            acc_loss += reflection_loss_db(prev_z, SKULL.impedance_mrayl)
            prev_z = SKULL.impedance_mrayl
        if d > onset_skull_end:
            brain_len = d - onset_skull_end
            acc_loss += BRAIN.attenuation_db(freq_mhz) * (brain_len / BRAIN.thickness_cm)
            acc_loss += reflection_loss_db(prev_z, BRAIN.impedance_mrayl)
        acc_loss += spreading_loss_db(d, TX_APERTURE_CM, freq_mhz)
        loss_1d[i] = acc_loss

    for j, d in enumerate(depths_1d):
        rx_dbm = tx_power_dbm - loss_1d[j]
        snr_axis = rx_dbm - noise_floor_dbm
        # lateral falloff: real circular-piston diffraction pattern (Bessel
        # function, gives genuine sidelobes and a real first-null angle),
        # replacing the previous Gaussian approximation.
        x_row = X[j, :]
        loss_db_row = directivity_loss_db(np.abs(x_row), d, TX_APERTURE_CM / 2, freq_mhz)
        snr_field[j, :] = snr_axis + loss_db_row  # loss_db_row is already <=0

    return X, Y, tissue_id, snr_field


def plot_spatial_map(freq_mhz: float, out_path: str):
    X, Y, tissue_id, snr_field = compute_snr_field(freq_mhz)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    tissue_cmap = plt.cm.colors.ListedColormap(["#dce8f5", "#f2d8b8", "#8a8a8a", "#e7b8c4"])
    axes[0].pcolormesh(X, Y, tissue_id, cmap=tissue_cmap, shading="auto")
    axes[0].invert_yaxis()
    axes[0].set_title("Synthetic head cross-section\n(gel / skin / skull / brain)")
    axes[0].set_xlabel("lateral position (cm)")
    axes[0].set_ylabel("depth (cm)")

    vmin, vmax = np.percentile(snr_field, [2, 98])
    im = axes[1].pcolormesh(X, Y, snr_field, cmap="inferno", shading="auto",
                             vmin=vmin, vmax=vmax)
    axes[1].invert_yaxis()
    axes[1].set_title(f"Received SNR field @ {freq_mhz:.2f} MHz")
    axes[1].set_xlabel("lateral position (cm)")
    fig.colorbar(im, ax=axes[1], label="SNR (dB)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"Saved {out_path}")


# ======================================================================
# --- from view_3d.py ---
# ======================================================================
"""
3D volumetric view of the ultrasonic field through tissue, built by
revolving the existing 2D axisymmetric beam model (spatial_map.py) around
the depth axis -- physically justified since a circular transducer produces
a rotationally symmetric field, so this is a legitimate 3D extension of
already-validated 2D physics, not a new independent model.

Still ray/geometric, not a wave-equation solve (same caveat as spatial_map.py).
"""



GRID_EXTENT_CM = 4.0     # lateral radius simulated (cm)
GRID_DEPTH_CM = 8.0       # depth simulated (cm)
GRID_RES_3D = 28           # points per axis -- kept modest, volume rendering scales as N^3
SKIN_THICKNESS_CM = 0.2
SKULL_THICKNESS_CM = 0.7
TX_APERTURE_CM = 1.0


def _axial_loss_profile(freq_mhz: float, depths_1d: np.ndarray) -> np.ndarray:
    """Same layered-loss-vs-depth calculation as spatial_map.py's 2D version,
    factored out so the 3D volume reuses the identical validated physics."""
    onset_skin = SKIN_THICKNESS_CM
    onset_skull_end = SKIN_THICKNESS_CM + SKULL_THICKNESS_CM
    loss_1d = np.zeros_like(depths_1d)
    for i, d in enumerate(depths_1d):
        acc_loss = 0.0
        prev_z = 1.5
        if d > 0:
            skin_len = min(d, onset_skin)
            acc_loss += SKIN.attenuation_db(freq_mhz) * (skin_len / SKIN.thickness_cm)
            acc_loss += reflection_loss_db(prev_z, SKIN.impedance_mrayl)
            prev_z = SKIN.impedance_mrayl
        if d > onset_skin:
            skull_len = min(d, onset_skull_end) - onset_skin
            acc_loss += SKULL.attenuation_db(freq_mhz) * (skull_len / SKULL.thickness_cm)
            acc_loss += reflection_loss_db(prev_z, SKULL.impedance_mrayl)
            prev_z = SKULL.impedance_mrayl
        if d > onset_skull_end:
            brain_len = d - onset_skull_end
            acc_loss += BRAIN.attenuation_db(freq_mhz) * (brain_len / BRAIN.thickness_cm)
            acc_loss += reflection_loss_db(prev_z, BRAIN.impedance_mrayl)
        acc_loss += spreading_loss_db(d, TX_APERTURE_CM, freq_mhz)
        loss_1d[i] = acc_loss
    return loss_1d


def build_3d_field(freq_mhz: float, tx_power_dbm: float = 0.0,
                    noise_floor_dbm: float = -90.0):
    """Returns (X, Y, Z, snr_volume) 3D grids, built by revolving the
    validated 2D axial-loss profile around the depth (z) axis."""
    lin = np.linspace(-GRID_EXTENT_CM, GRID_EXTENT_CM, GRID_RES_3D)
    z_lin = np.linspace(0.01, GRID_DEPTH_CM, GRID_RES_3D)
    X, Y, Z = np.meshgrid(lin, lin, z_lin, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)  # radial distance from beam axis

    loss_1d = _axial_loss_profile(freq_mhz, z_lin)
    z_nf = near_field_length_cm(TX_APERTURE_CM, freq_mhz)

    snr_volume = np.zeros_like(R)
    for k, d in enumerate(z_lin):
        rx_dbm = tx_power_dbm - loss_1d[k]
        snr_axis = rx_dbm - noise_floor_dbm
        # Real circular-piston diffraction pattern (Bessel), replacing the
        # previous Gaussian falloff -- same physics as spatial_map.py.
        loss_db_slice = directivity_loss_db(R[:, :, k], d, TX_APERTURE_CM / 2, freq_mhz)
        snr_volume[:, :, k] = snr_axis + loss_db_slice  # loss_db_slice is already <=0

    return X, Y, Z, snr_volume


def make_plotly_figure(freq_mhz: float, tx_power_dbm: float = 0.0):
    X, Y, Z, snr_volume = build_3d_field(freq_mhz, tx_power_dbm)

    vmin, vmax = np.percentile(snr_volume, [15, 99])

    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=snr_volume.flatten(),
        isomin=vmin, isomax=vmax,
        opacity=0.12,
        surface_count=18,
        colorscale="Inferno",
        colorbar=dict(title="SNR (dB)"),
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    # Mark the skull shell as a translucent disk for spatial reference
    theta = np.linspace(0, 2 * np.pi, 40)
    r = np.linspace(0, GRID_EXTENT_CM, 10)
    Rm, Tm = np.meshgrid(r, theta)
    Xs = Rm * np.cos(Tm)
    Ys = Rm * np.sin(Tm)
    Zs_top = np.full_like(Xs, SKIN_THICKNESS_CM)
    Zs_bot = np.full_like(Xs, SKIN_THICKNESS_CM + SKULL_THICKNESS_CM)
    fig.add_trace(go.Surface(x=Xs, y=Ys, z=Zs_top, showscale=False,
                              opacity=0.15, colorscale=[[0, "gray"], [1, "gray"]],
                              name="skull (outer)"))
    fig.add_trace(go.Surface(x=Xs, y=Ys, z=Zs_bot, showscale=False,
                              opacity=0.15, colorscale=[[0, "gray"], [1, "gray"]],
                              name="skull (inner)"))

    fig.update_scenes(
        zaxis=dict(autorange="reversed", title="depth (cm)"),
        xaxis_title="lateral x (cm)",
        yaxis_title="lateral y (cm)",
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1.3),
    )
    fig.update_layout(
        title=f"3D SNR field @ {freq_mhz:.2f} MHz (revolved from validated 2D axial model)",
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ======================================================================
# --- from app.py ---
# ======================================================================
st.set_page_config(page_title="Neural Dust Link-Budget Simulator", layout="wide")

st.title("Ultrasonic Neural Dust — Power/Comms Link Budget Simulator")
st.caption(
    "A simplified engineering model of ultrasonic power delivery and backscatter "
    "communication for implantable neural-dust-style sensors, following the design "
    "approach in Seo et al. 2013 (UC Berkeley). Ray/geometric approximations, not a "
    "full wave-equation solver — see notes at the bottom."
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Parameters")
freq_mhz = st.sidebar.slider("Operating frequency (MHz)", 0.2, 5.0, 1.0, 0.05)
depth_cm = st.sidebar.slider("Target implant depth (cm)", 0.5, 8.0, 3.0, 0.1)
tx_power_dbm = st.sidebar.slider("TX drive power (dBm)", -10, 20, 0, 1)
tx_aperture_cm = st.sidebar.slider("TX transducer aperture (cm)", 0.3, 3.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Mote aperture is auto-sized to ~0.75x wavelength, capped at a 1mm physical "
    "ceiling — this is what makes the frequency/depth tradeoff real instead of trivial."
)

# ---------------- Top row: key numbers ----------------
mote_aperture_cm = coupled_mote_aperture_cm(freq_mhz)
budget = two_way_link_budget_db(depth_cm, freq_mhz, tx_aperture_cm, mote_aperture_cm)
snr_db = snr_db_at_frequency(freq_mhz, depth_cm, tx_power_dbm, tx_aperture_cm)
z_nf = near_field_length_cm(tx_aperture_cm, freq_mhz)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Received SNR", f"{snr_db:.1f} dB")
col2.metric("Mote aperture", f"{mote_aperture_cm * 10:.3f} mm")
col3.metric("Near-field length", f"{z_nf:.2f} cm")
col4.metric("Total 2-way loss", f"{budget['total_two_way_loss_db']:.1f} dB")

if snr_db < 0:
    st.warning(
        "SNR is negative at this operating point — the link would not close with "
        "these parameters. Try a lower frequency, shallower depth, or more TX power."
    )

# ---------------- Loss breakdown ----------------
st.subheader("Link budget breakdown")
labels = list(budget.keys())[:-1]
values = [budget[k] for k in labels]
fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
ax_bar.barh([l.replace("_db", "").replace("_", " ") for l in labels], values, color="#c0392b")
ax_bar.set_xlabel("Loss (dB)")
ax_bar.invert_yaxis()
st.pyplot(fig_bar)

# ---------------- Spatial field map ----------------
st.subheader("Spatial SNR field (synthetic transcranial cross-section)")
X, Y, tissue_id, snr_field = compute_snr_field(freq_mhz, tx_power_dbm)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
tissue_cmap = plt.cm.colors.ListedColormap(["#dce8f5", "#f2d8b8", "#8a8a8a", "#e7b8c4"])
axes[0].pcolormesh(X, Y, tissue_id, cmap=tissue_cmap, shading="auto")
axes[0].invert_yaxis()
axes[0].set_title("Tissue layers (gel/skin/skull/brain)")
axes[0].set_xlabel("lateral position (cm)")
axes[0].set_ylabel("depth (cm)")
axes[0].axhline(depth_cm, color="cyan", linestyle="--", linewidth=1, label="target depth")
axes[0].legend(loc="lower right", fontsize=8)

vmin, vmax = np.percentile(snr_field, [2, 98])
im = axes[1].pcolormesh(X, Y, snr_field, cmap="inferno", shading="auto", vmin=vmin, vmax=vmax)
axes[1].invert_yaxis()
axes[1].set_title(f"Received SNR field @ {freq_mhz:.2f} MHz")
axes[1].set_xlabel("lateral position (cm)")
axes[1].axhline(depth_cm, color="cyan", linestyle="--", linewidth=1)
fig.colorbar(im, ax=axes[1], label="SNR (dB)")
st.pyplot(fig)

# ---------------- 3D volumetric view ----------------
st.subheader("3D field view")
st.caption(
    "Revolved from the same validated 2D axial-loss model around the depth axis "
    "(physically justified — a circular transducer's field is rotationally "
    "symmetric). Rotate/zoom/slice with your mouse. Still ray-based, not a wave "
    "solve — see the accuracy notes below."
)
fig3d = make_plotly_figure(freq_mhz, tx_power_dbm)
st.plotly_chart(fig3d, use_container_width=True)

# ---------------- Frequency optimization sweep ----------------
st.subheader("Frequency optimization")
depths_to_check = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
opt_rows = []
for d in depths_to_check:
    opt = find_optimal_frequency(depth_cm=d, tx_power_dbm=tx_power_dbm)
    opt_rows.append({
        "Depth (cm)": d,
        "Optimal freq (MHz)": round(opt["optimal_freq_mhz"], 2),
        "Max SNR (dB)": round(opt["max_snr_db"], 1),
        "Mote aperture (mm)": round(opt["mote_aperture_cm"] * 10, 3),
    })
st.dataframe(opt_rows, use_container_width=True)

freqs = np.linspace(0.2, 5.0, 100)
snr_curve = [snr_db_at_frequency(f, depth_cm, tx_power_dbm, tx_aperture_cm) for f in freqs]
fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(freqs, snr_curve, color="#2c3e50")
ax2.axvline(freq_mhz, color="red", linestyle="--", label="current setting")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.set_xlabel("Frequency (MHz)")
ax2.set_ylabel("SNR (dB)")
ax2.set_title(f"SNR vs frequency at {depth_cm:.1f} cm depth")
ax2.legend()
st.pyplot(fig2)

st.markdown("---")
with st.expander("📊 Model accuracy — what's validated vs. heuristic"):
    st.markdown("""
**Validated (exact by construction):**
- Near-field length `z_nf = a²f/c`, reflection coefficients, power-law attenuation — standard formulas
- **Circular-piston diffraction pattern (Bessel function)** — exact closed-form solution, matches the analytical first-null angle to <0.01 directivity error. Produces real sidelobes.

**Checked against reference values:**
- Soft-tissue attenuation vs. ~1 dB/cm/MHz rule of thumb — mean error ~16.5% across 0.5-5 MHz
- Skull insertion loss vs. commonly-cited 10-20 dB range @ 1 MHz — **10.5 dB, within range**
- PZT-5H acoustic impedance vs. commonly-cited 30-36 MRayl range — **34.5 MRayl, within range**
- Optimal frequency vs. published neural dust operating point (~1.75 MHz) — same order of magnitude

**Upgraded from heuristic to physically-grounded:**
- **Piezo TX/RX coupling**: now uses real PZT-5H acoustic impedance + electromechanical coupling coefficient, not an arbitrary aperture-ratio penalty. TX assumes an ideal matching layer; the mote is modeled unmatched (physically justified — no room for one at sub-mm scale), which is why mote coupling now dominates the budget — consistent with real neural dust literature.
- **Backscatter modulation loss**: derived from actual reflectivity change between short/open-circuit piezo states, replacing a fixed 6 dB placeholder. **Caveat**: this captures piezoelectric stiffening only — real designs mainly modulate via electrical damping/Q-switching, a related mechanism that can achieve deeper modulation than this conservative estimate.
- **Lateral beam pattern**: exact circular-piston Bessel diffraction (real sidelobes), replacing the Gaussian approximation.

**Still not independently validated:**
- Near-field diffraction detail (Fresnel-zone ripples) — the Bessel fix covers far-field directivity specifically; a full Rayleigh-Sommerfeld integral would be the next step.
- The spatial/3D field is still ray-based for propagation — no interference, oblique refraction, or multi-path. A k-Wave (pseudospectral time-domain) simulation remains the research-grade next step — a genuinely larger undertaking, intentionally not attempted here rather than faked.
- The piezo model is a two-effect approximation (coupling limit + mismatch loss), not a full distributed KLM/Mason circuit.

Full numeric breakdown: `validation.py` in the repo.
""")

st.caption(
    "**Model notes:** attenuation/impedance values are representative literature "
    "figures, not a substitute for a source-specific literature review. Spatial field "
    "uses ray/geometric approximations (near-field collimation + far-field spreading + "
    "Gaussian lateral falloff), not a full acoustic wave-equation solve — k-Wave "
    "(pseudospectral time-domain) is the research-grade next step for real diffraction "
    "and multi-path effects. Sanity-checked against the ~1 dB/cm/MHz soft-tissue rule "
    "of thumb; see test_sanity.py and validation.py."
)
