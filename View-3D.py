import numpy as np
import plotly.graph_objects as go

from tissue import SKIN, SKULL, BRAIN, reflection_loss_db
from link_budget import spreading_loss_db, near_field_length_cm

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


if __name__ == "__main__":
    fig = make_plotly_figure(freq_mhz=1.0)
    fig.write_html("field_3d_preview.html")
    print("wrote field_3d_preview.html")
