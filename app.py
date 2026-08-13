import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from tissue import path_attenuation_db, PATH_PERIPHERAL, MUSCLE
from link_budget import two_way_link_budget_db, near_field_length_cm, C_TISSUE
from sweep import snr_db_at_frequency, find_optimal_frequency, coupled_mote_aperture_cm
from spatial_map import compute_snr_field, SKIN_THICKNESS_CM, SKULL_THICKNESS_CM, GRID_SIZE_CM
from view_3d import make_plotly_figure

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
