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


