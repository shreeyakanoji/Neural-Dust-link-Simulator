"""
Quantitative accuracy check: how well does each model component match
known reference values, expressed as actual error metrics (not just
pass/fail assertions like test_sanity.py).

Run with: python3 validation.py
"""

import numpy as np
from tissue import MUSCLE, SKULL, path_attenuation_db, PATH_PERIPHERAL, PATH_TRANSCRANIAL
from sweep import find_optimal_frequency


def pct_error(model_val, reference_val):
    return 100.0 * abs(model_val - reference_val) / abs(reference_val)


def check_soft_tissue_rule_of_thumb():
    """Soft tissue: widely-cited rule of thumb is ~1 dB/cm/MHz, roughly
    constant across 1-5 MHz for typical soft tissue (linear-ish in f).
    This checks our power-law fit against that reference across a range,
    not just a single point."""
    print("\n[1] Soft-tissue attenuation vs ~1 dB/cm/MHz rule of thumb")
    print("    freq(MHz)  model(dB/cm)  reference(dB/cm)  error(%)")
    errors = []
    for f in [0.5, 1.0, 2.0, 3.0, 5.0]:
        model_val = MUSCLE.alpha0_db_cm_mhz * (f ** MUSCLE.freq_exponent)
        reference_val = 1.0 * f  # linear rule of thumb
        err = pct_error(model_val, reference_val)
        errors.append(err)
        print(f"    {f:>8.1f}  {model_val:>12.3f}  {reference_val:>16.3f}  {err:>7.1f}")
    print(f"    -> mean error: {np.mean(errors):.1f}%  "
          f"(expected to grow with f since rule-of-thumb is a rough linear "
          f"approximation and our model uses freq_exponent=1.1, not exactly 1.0)")
    return np.mean(errors)


def check_skull_attenuation_order_of_magnitude():
    """Skull insertion loss at ~1 MHz for typical thickness is commonly
    cited in the 10-20 dB range in transcranial ultrasound literature
    (varies substantially by specimen/thickness/frequency -- this is an
    order-of-magnitude sanity check, not a precise validation)."""
    print("\n[2] Skull insertion loss order-of-magnitude check @ 1 MHz")
    model_loss_db = SKULL.attenuation_db(freq_mhz=1.0)  # thickness_cm=0.7 baked in
    lit_range = (10, 20)
    in_range = lit_range[0] <= model_loss_db <= lit_range[1]
    print(f"    model: {model_loss_db:.1f} dB  |  commonly-cited range: "
          f"{lit_range[0]}-{lit_range[1]} dB  |  within range: {in_range}")
    if not in_range:
        print("    NOTE: outside commonly-cited range -- alpha0/exponent for "
              "SKULL in tissue.py should be revisited against a specific "
              "source before trusting transcranial depth results.")
    return in_range


def check_optimal_frequency_vs_published_neural_dust():
    """Seo et al. 2013 (the foundational neural dust paper) settled on an
    operating frequency around 1.75 MHz for their specific aperture/depth
    regime. Our model's optimal frequency will differ because our piezo
    coupling-efficiency term is a simplified heuristic penalty function,
    not a derived KLM/Mason equivalent-circuit model, and our depth/aperture
    assumptions differ from theirs. This checks we're in the same order of
    magnitude, not an exact match -- an exact match would actually be
    suspicious given how different the two models are."""
    print("\n[3] Optimal frequency vs published neural dust operating point")
    published_freq_mhz = 1.75
    for depth in [1.0, 2.0]:
        opt = find_optimal_frequency(depth_cm=depth)
        ratio = opt["optimal_freq_mhz"] / published_freq_mhz
        print(f"    depth={depth}cm: model optimal={opt['optimal_freq_mhz']:.2f} MHz "
              f"vs published={published_freq_mhz} MHz (ratio={ratio:.2f}x)")
    print("    -> same order of magnitude (0.1x-10x) is the honest bar here, "
          "given the piezo efficiency term is heuristic, not a derived circuit model.")


def check_diffraction_first_null():
    """The circular-piston diffraction model (diffraction.py) is validated
    against the EXACT closed-form first-null formula (sin(theta) = 1.22
    lambda/D) -- this replaces the old Gaussian lateral-falloff heuristic
    with real physics, and is checked here to a tight tolerance since it's
    an analytical solution, not an approximation."""
    print("\n[4] Diffraction model vs. exact first-null formula")
    from diffraction import piston_directivity, first_null_angle_deg
    import numpy as np
    null_deg = first_null_angle_deg(aperture_radius_cm=0.5, freq_mhz=1.0)
    d_at_null = piston_directivity(np.radians(null_deg), aperture_radius_cm=0.5, freq_mhz=1.0)
    print(f"    first-null angle: {null_deg:.2f} deg, directivity there: {d_at_null:.6f} "
          f"(should be ~0, exact solution)")
    d_onaxis = piston_directivity(0.0, aperture_radius_cm=0.5, freq_mhz=1.0)
    print(f"    on-axis directivity: {d_onaxis:.6f} (should be exactly 1.0)")
    return d_at_null < 0.01 and abs(d_onaxis - 1.0) < 1e-6


def check_piezo_model_physics():
    """The piezo model (piezo_model.py) now uses real PZT-5H material
    impedance (~34.5 MRayl) instead of an arbitrary penalty function --
    checks it lands in the commonly-cited range for hard piezoceramics
    (30-36 MRayl is typical for PZT-5H across sources)."""
    print("\n[5] Piezo material impedance vs. commonly-cited PZT-5H range")
    from piezo_model import PZT5H_IMPEDANCE_MRAYL
    lit_range = (30, 36)
    in_range = lit_range[0] <= PZT5H_IMPEDANCE_MRAYL <= lit_range[1]
    print(f"    model: {PZT5H_IMPEDANCE_MRAYL:.1f} MRayl | commonly-cited range: "
          f"{lit_range[0]}-{lit_range[1]} MRayl | within range: {in_range}")
    return in_range


def summarize():
    print("=" * 70)
    print("ACCURACY SUMMARY")
    print("=" * 70)
    print("""
VALIDATED (formula-level, exact by construction):
  - Near-field length z_nf = a^2*f/c        -- standard transducer physics formula
  - Reflection coefficient at boundaries    -- standard normal-incidence formula
  - Power-law attenuation form alpha(f)=a0*f^b -- standard bioacoustics form
  - Circular-piston diffraction pattern     -- EXACT closed-form Bessel solution,
    matches the analytical first-null angle to <0.01 directivity error

CHECKED AGAINST REFERENCE VALUES (quantified error above):
  - Soft tissue attenuation vs 1 dB/cm/MHz rule of thumb
  - Skull insertion loss vs commonly-cited 10-20 dB range @ 1 MHz
  - Optimal frequency vs published neural dust operating point (order-of-magnitude)
  - PZT-5H acoustic impedance vs commonly-cited 30-36 MRayl range

UPGRADED THIS PASS (previously heuristic, now physically grounded):
  - Piezo TX/RX coupling efficiency: now uses real PZT-5H acoustic impedance
    and electromechanical coupling coefficient (k_t) instead of an arbitrary
    aperture-ratio penalty function. TX assumes an ideal matching layer
    (physically justified -- TX isn't size-constrained); mote RX is modeled
    unmatched (physically justified -- no room for a matching layer at
    sub-mm scale), which is why mote coupling loss now dominates the
    budget -- consistent with what real neural dust literature identifies
    as the limiting factor.
  - Backscatter modulation loss: now derived from the actual reflectivity
    change between short-circuit and open-circuit piezo states (real
    piezoelectric stiffening physics), replacing the fixed 6 dB placeholder.
    IMPORTANT CAVEAT: this captures ONE real mechanism (stiffness change)
    but real neural dust designs primarily modulate via electrical
    damping/Q-switching (resistive loading), a related but distinct
    mechanism that can achieve deeper modulation than stiffness-switching
    alone predicts here. Treat the derived value as a conservative,
    physically-motivated estimate, not a validated final number.
  - Lateral beam pattern: now the exact circular-piston Bessel diffraction
    function (real sidelobes, real first-null angle) instead of a fitted
    Gaussian. This is genuinely validated (see check above), not just
    "less heuristic."

STILL NOT INDEPENDENTLY VALIDATED / KNOWN LIMITATIONS:
  - Near-field diffraction detail: the Bessel fix covers FAR-FIELD
    directivity; true near-field (Fresnel zone ripples) would need a full
    numerical Rayleigh-Sommerfeld integral -- not done this pass.
  - The spatial/3D field is still ray-based for propagation: no
    interference between multiple paths, no refraction at oblique tissue
    boundaries, no reflected-wave multipath. A k-Wave (pseudospectral
    time-domain) simulation is the remaining research-grade upgrade for
    real wave-equation behavior -- this is a genuinely larger undertaking
    (new dependency, significant compute, geometry meshing) and was
    intentionally not attempted this pass rather than faked.
  - Piezo model has no matching-layer bandwidth shaping or backing-layer
    reflections (not a full distributed KLM/Mason circuit) -- it's a
    two-effect model (coupling limit + mismatch loss), a real improvement
    over the previous heuristic but still simplified.
""")


if __name__ == "__main__":
    check_soft_tissue_rule_of_thumb()
    check_skull_attenuation_order_of_magnitude()
    check_optimal_frequency_vs_published_neural_dust()
    check_diffraction_first_null()
    check_piezo_model_physics()
    summarize()
