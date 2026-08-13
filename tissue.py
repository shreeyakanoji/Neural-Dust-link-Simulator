"""
Layered tissue acoustic properties for ultrasonic power/comms link modeling.

Attenuation follows the standard power-law form used in bioacoustics:
    alpha(f) = alpha0 * f^b      [dB/cm], f in MHz
Values below are representative literature figures (soft tissue ~1 dB/cm/MHz
rule of thumb; skull is a major outlier due to its much higher absorption
and scattering). These are approximations for engineering-level modeling,
not a substitute for a proper literature review per target site.
"""

from dataclasses import dataclass


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
