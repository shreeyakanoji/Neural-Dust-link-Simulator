"""
Plotting helper for kwave_sim.py output; (since k-Wave depends on a system
library, libhdf5, that may not be present on every deployment target).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_kwave_field(result: dict, freq_mhz: float, skin_thickness_cm=0.2,
                      skull_thickness_cm=0.7):
    p_max = result["p_max_field"]
    N = result["grid_size"]
    dx_cm = result["dx_cm"]
    extent_cm = N * dx_cm

    fig, ax = plt.subplots(figsize=(6, 6))
    p_db = 20 * np.log10(np.maximum(p_max, p_max.max() * 1e-4) / p_max.max())
    im = ax.imshow(p_db, cmap="inferno", origin="upper",
                    extent=[-extent_cm / 2, extent_cm / 2, extent_cm, 0],
                    vmin=-40, vmax=0, aspect="equal")
    ax.axhline(skin_thickness_cm, color="cyan", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(skin_thickness_cm + skull_thickness_cm, color="cyan", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.set_xlabel("lateral position (cm)")
    ax.set_ylabel("depth (cm)")
    ax.set_title(f"k-Wave field @ {freq_mhz:.2f} MHz\nreal wave-equation solve", fontsize=10)
    fig.colorbar(im, ax=ax, label="relative pressure (dB)")
    plt.tight_layout()
    return fig
