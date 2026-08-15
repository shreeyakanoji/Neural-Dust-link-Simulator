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

"""

import numpy as np

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
