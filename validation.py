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


def summarize():
    print("=" * 70)
    print("ACCURACY SUMMARY")
    print("=" * 70)
    print("""
VALIDATED (formula-level, exact by construction):
  - Near-field length z_nf = a^2*f/c        -- standard transducer physics formula
  - Reflection coefficient at boundaries    -- standard normal-incidence formula
  - Power-law attenuation form alpha(f)=a0*f^b -- standard bioacoustics form

(quantified error above):
  - Soft tissue attenuation vs 1 dB/cm/MHz rule of thumb
  - Skull insertion loss vs commonly-cited 10-20 dB range @ 1 MHz
  - Optimal frequency vs published neural dust operating point (order-of-magnitude)

NOT INDEPENDENTLY VALIDATED (heuristic / illustrative only -- treat with caution):
  - Piezo receive-coupling efficiency: a simplified log-quadratic penalty
    function, NOT a derived KLM/Mason equivalent-circuit model. This is the
    single biggest accuracy gap in the whole simulator -- replacing it with
    a real KLM model (per your own earlier spec) is the highest-value next
    upgrade.
  - Lateral beam falloff in spatial_map.py: a Gaussian-shaped approximation,
    not a real diffraction integral (no sidelobes, no true beam pattern).
  - Backscatter modulation loss: a fixed 6 dB placeholder, not derived from
    an actual impedance-switching circuit model.
  - Ray-based spatial field: ignores real wave effects entirely --
    interference, refraction at oblique boundaries, and multi-path are
    all absent. This is the gap a k-Wave-based simulation would close.
""")


if __name__ == "__main__":
    check_soft_tissue_rule_of_thumb()
    check_skull_attenuation_order_of_magnitude()
    check_optimal_frequency_vs_published_neural_dust()
    summarize()
