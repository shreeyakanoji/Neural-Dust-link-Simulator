"""
Sweep over frequency (and optionally mote aperture, coupled to frequency)
to find the operating point that maximizes received SNR at a target depth,
subject to a TX power budget.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from link_budget import two_way_link_budget_db, C_TISSUE

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


if __name__ == "__main__":
    for depth in [1.0, 2.0, 3.0, 5.0]:
        opt = find_optimal_frequency(depth_cm=depth)
        print(f"Depth {depth} cm -> optimal freq {opt['optimal_freq_mhz']:.2f} MHz, "
              f"SNR {opt['max_snr_db']:.1f} dB, "
              f"mote aperture {opt['mote_aperture_cm']*10:.3f} mm")
