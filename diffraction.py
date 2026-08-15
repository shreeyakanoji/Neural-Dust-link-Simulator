"""
Real circular-piston far-field diffraction pattern, replacing the Gaussian
lateral-falloff approximation with the standard, well-established closed-form
directivity function for a baffled circular piston transducer (Kinsler &
Frey / Blackstock -- standard acoustics textbook result, low risk of error
since it's an exact analytical solution, not a fitted approximation):

    D(theta) = | 2*J1(x) / x |,   x = k*a*sin(theta)

where J1 is the Bessel function of the first kind order 1, k is the
wavenumber, a is the piston (transducer) radius, and theta is the angle
off-axis. This produces genuine sidelobes and a real first-null angle,
unlike the Gaussian approximation it replaces.

Honest scope: this is the FAR-FIELD directivity pattern specifically. Near
field is still handled by the existing collimation approximation in
link_budget.py / spatial_map.py -- true near-field diffraction (Fresnel
zone ripples) would need a full Rayleigh-Sommerfeld numerical integral,
which is a further upgrade beyond this pass. This fix targets the "no
sidelobes, no true beam pattern" gap specifically for the region where it
matters most (far field, where most of the tissue path lives at typical
implant depths).
"""

import numpy as np
from scipy.special import j1


def piston_directivity(angle_rad, aperture_radius_cm: float, freq_mhz: float,
                        c_m_s: float = 1540.0):
    """Returns directivity gain (0-1, linear, not dB) for a circular piston
    transducer at the given off-axis angle(s). angle_rad can be a scalar
    or numpy array."""
    freq_hz = freq_mhz * 1e6
    k = 2 * np.pi * freq_hz / c_m_s          # wavenumber, 1/m
    a_m = aperture_radius_cm / 100.0
    x = k * a_m * np.sin(angle_rad)
    # handle x -> 0 (on-axis) where 2*J1(x)/x -> 1 by limit
    x_safe = np.where(np.abs(x) < 1e-9, 1e-9, x)
    directivity = np.abs(2 * j1(x_safe) / x_safe)
    directivity = np.where(np.abs(x) < 1e-9, 1.0, directivity)
    return directivity


def directivity_loss_db(lateral_offset_cm, depth_cm, aperture_radius_cm: float,
                         freq_mhz: float, c_m_s: float = 1540.0):
    """Convenience wrapper: given a field point's lateral offset and depth
    (both in cm) relative to the transducer, returns the directivity loss
    in dB (<=0). Depth is clamped away from zero to avoid a division
    singularity directly at the transducer face."""
    depth_safe = np.maximum(depth_cm, 1e-6)
    angle_rad = np.arctan2(lateral_offset_cm, depth_safe)
    d = piston_directivity(angle_rad, aperture_radius_cm, freq_mhz, c_m_s)
    d_safe = np.maximum(d, 1e-6)
    return 20 * np.log10(d_safe)


def first_null_angle_deg(aperture_radius_cm: float, freq_mhz: float,
                          c_m_s: float = 1540.0) -> float:
    """First-null half-angle of the main lobe (real, testable quantity --
    the classic sin(theta) = 1.22*lambda/(2a) result for a circular piston)."""
    freq_hz = freq_mhz * 1e6
    wavelength_cm = (c_m_s / freq_hz) * 100.0
    diameter_cm = 2 * aperture_radius_cm
    sin_theta = 1.22 * wavelength_cm / diameter_cm
    sin_theta = min(sin_theta, 1.0)  # guard against no-null case (very small aperture)
    return np.degrees(np.arcsin(sin_theta))
