"""
Neural Dust Link-Budget Simulator — single-file Streamlit app.
Combines tissue.py, link_budget.py, sweep.py, spatial_map.py, view_3d.py,
and the Streamlit UI into one file to avoid multi-file copy/paste issues.
"""

from dataclasses import dataclass
from scipy.optimize import minimize_scalar
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
# --- from link_budget.py ---
# ======================================================================
"""
One-way and two-way (TX -> mote backscatter -> RX) ultrasonic link budget,
combining near-field collimation, far-field spreading, tissue attenuation,
and small-aperture receive coupling loss at the mote.
"""


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


def piezo_receive_efficiency_db(aperture_cm: float, freq_mhz: float,
                                 optimal_aperture_over_lambda: float = 0.75) -> float:
    """
    Small-aperture receive coupling loss. Real efficiency depends on matching
    layers / KLM circuit details; this is a simplified penalty model:
    efficiency peaks when aperture ~ lambda/2 to lambda, and degrades for
    apertures much smaller than a wavelength (the regime sub-mm motes are in).
    """
    c = C_TISSUE
    wavelength_cm = (c / (freq_mhz * 1e6)) * 100.0
    ratio = aperture_cm / wavelength_cm
    optimal = optimal_aperture_over_lambda
    # Penalize deviation from the optimal aperture/wavelength ratio (log-quadratic)
    penalty_db = 20 * (np.log10(ratio / optimal)) ** 2
    return -penalty_db  # negative = loss


def two_way_link_budget_db(depth_cm: float, freq_mhz: float,
                            tx_aperture_cm: float, mote_aperture_cm: float,
                            path=None, backscatter_modulation_loss_db: float = 6.0):
    """
    Full two-way budget: TX -> tissue -> mote receive coupling ->
    backscatter modulation -> tissue -> RX at TX aperture.
    Returns dict of loss components (all positive dB = loss) and net budget.
    """
    if path is None:
        path = PATH_TRANSCRANIAL
    total_path_len = path_length_cm(path)
    # scale tissue path to requested depth (keep proportions, adjust length)
    scale = depth_cm / total_path_len if total_path_len > 0 else 1.0

    tissue_loss_one_way = path_attenuation_db(path, freq_mhz) * scale
    spread_loss_out = spreading_loss_db(depth_cm, tx_aperture_cm, freq_mhz)
    mote_rx_loss = -piezo_receive_efficiency_db(mote_aperture_cm, freq_mhz)  # positive = loss
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
    rx_power_dbm = tx_power_dbm - budget["total_two_way_loss_db"]
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
        # lateral falloff: narrower beam at higher freq (approx via near-field length)
        z_nf = near_field_length_cm(TX_APERTURE_CM, freq_mhz)
        beam_width_cm = max(TX_APERTURE_CM, TX_APERTURE_CM * (1 + d / max(z_nf, 0.1)))
        x_row = X[j, :]
        lateral_falloff_db = 20 * (x_row / (beam_width_cm)) ** 2
        snr_field[j, :] = snr_axis - lateral_falloff_db

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
        beam_width_cm = max(TX_APERTURE_CM, TX_APERTURE_CM * (1 + d / max(z_nf, 0.1)))
        lateral_falloff_db = 20 * (R[:, :, k] / beam_width_cm) ** 2
        snr_volume[:, :, k] = snr_axis - lateral_falloff_db

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
**Validated (formula-level, exact by construction):**
- Near-field length `z_nf = a²f/c` — standard transducer physics formula
- Reflection coefficient at boundaries — standard normal-incidence formula
- Power-law attenuation form `α(f) = α₀·f^b` — standard bioacoustics form

**Checked against reference values (quantified error):**
- Soft-tissue attenuation vs. the ~1 dB/cm/MHz rule of thumb — mean error ~16.5% across 0.5-5 MHz (grows with frequency since the rule of thumb is linear and this model uses a 1.1 exponent)
- Skull insertion loss vs. the commonly-cited 10-20 dB range @ 1 MHz — **10.5 dB, within range**
- Optimal frequency vs. the published neural dust operating point (~1.75 MHz, Seo et al. 2013) — model lands 0.4-0.6x that value, same order of magnitude, given a different aperture/depth regime and a simplified (not derived) piezo coupling model

**Not independently validated — heuristic/illustrative only:**
- **Piezo receive-coupling efficiency**: a simplified log-quadratic penalty, not a derived KLM/Mason equivalent-circuit model. This is the single biggest accuracy gap in the simulator.
- **Lateral beam falloff**: a Gaussian approximation, not a real diffraction integral — no sidelobes, no true beam pattern.
- **Backscatter modulation loss**: a fixed 6 dB placeholder, not derived from an actual impedance-switching circuit.
- **The spatial field itself**: ray-based, ignoring real wave effects — interference, oblique-boundary refraction, and multi-path are all absent. A k-Wave (pseudospectral time-domain) simulation is the research-grade next step that closes this gap.

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
