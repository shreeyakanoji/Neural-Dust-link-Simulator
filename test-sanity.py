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


def test_peripheral_path_reasonable():
    """A shallow peripheral path at 2 MHz shouldn't produce absurd (near-zero
    or triple-digit) total loss for a ~1.7 cm path."""
    loss_db = path_attenuation_db(PATH_PERIPHERAL, freq_mhz=2.0)
    print(f"Peripheral path total one-way loss @ 2 MHz: {loss_db:.2f} dB")
    assert 1 < loss_db < 30, "Loss should be modest for a shallow soft-tissue path"


if __name__ == "__main__":
    test_soft_tissue_rule_of_thumb()
    test_peripheral_path_reasonable()
    print("\nAll sanity checks passed.")
