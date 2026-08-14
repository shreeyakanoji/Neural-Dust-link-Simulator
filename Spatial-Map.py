"""
2D spatial visualization: SNR/intensity field over a synthetic head
cross-section (skin -> skull -> brain).

"""

import numpy as np
import matplotlib.pyplot as plt
from tissue import SKIN, SKULL, BRAIN, path_attenuation_db, reflection_loss_db
from link_budget import spreading_loss_db, near_field_length_cm

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


if __name__ == "__main__":
    plot_spatial_map(freq_mhz=1.0, out_path="head_field_1p0MHz.png")
    plot_spatial_map(freq_mhz=0.5, out_path="head_field_0p5MHz.png")
