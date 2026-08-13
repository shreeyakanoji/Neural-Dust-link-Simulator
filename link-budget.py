"""
One-way and two-way (TX -> mote backscatter -> RX) ultrasonic link budget,
combining near-field collimation, far-field spreading, tissue attenuation,
and small-aperture receive coupling loss at the mote.
"""

import numpy as np
from tissue import path_attenuation_db, path_length_cm, PATH_TRANSCRANIAL

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
