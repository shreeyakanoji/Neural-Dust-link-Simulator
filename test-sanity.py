"""
Sanity checks against known published reference numbers.
Run with: python3 test_sanity.py
"""

from tissue import MUSCLE, path_attenuation_db, PATH_PERIPHERAL


def test_soft_tissue_rule_of_thumb():
    """Soft tissue attenuation should land near the classic ~1 dB/cm/MHz
    rule of thumb at 1 MHz (within a reasonable engineering margin)."""
    atten_1mhz = MUSCLE.attenuation_db(freq_mhz=1.0)  # thickness_cm=1.0 baked in
    print(f"Muscle attenuation @ 1 MHz, 1 cm path: {atten_1mhz:.2f} dB "
          f"(rule of thumb: ~1 dB)")
    assert 0.5 < atten_1mhz < 2.0, "Should be within ~2x of the 1 dB/cm/MHz rule of thumb"

