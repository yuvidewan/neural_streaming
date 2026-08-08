"""Raw tensor storage arithmetic for latents vs. source frames.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS MODULE.

Everything here is a plain "how many values times how many bits each"
calculation. It is NOT a codec bitrate and NOT a compression ratio:

- No entropy coding is applied (that is a later milestone). Real codecs
  spend far fewer than `bits` bits per symbol on a skewed distribution.
- The quantization scale/zero_point metadata is not counted.
- Sub-byte widths (6-bit, 4-bit) are counted as their theoretical bit cost.
  Nothing in this project actually bit-packs them yet; in memory they
  currently sit in int32 tensors.
- The "image" side is raw uncompressed RGB, not the PNG/JPEG the frame was
  loaded from - so this is not a fair comparison against a real image codec
  either.

The honest label for these figures is "theoretical raw tensor storage",
which is exactly how they are reported.
"""

from __future__ import annotations

import math

_BITS_PER_BYTE = 8
_FLOAT32_BITS = 32
_UINT8_BITS = 8


def tensor_bit_cost(shape: tuple[int, ...], bits_per_value: int) -> dict[str, float]:
    """Theoretical raw storage for a tensor of `shape` at `bits_per_value`."""
    num_values = math.prod(shape)
    total_bits = num_values * bits_per_value
    return {
        "shape": list(shape),
        "num_values": num_values,
        "bits_per_value": bits_per_value,
        "total_bits": total_bits,
        "total_bytes": total_bits / _BITS_PER_BYTE,
    }


def latent_storage_analysis(
    image_shape: tuple[int, int, int],
    latent_shape: tuple[int, int, int],
    bit_widths: tuple[int, ...] = (8, 6, 4),
) -> dict:
    """Compare raw storage of one frame against its latent at several widths.

    `image_shape` and `latent_shape` are per-sample [C, H, W] (no batch).
    The image baseline is raw uint8 RGB.
    """
    image = tensor_bit_cost(image_shape, _UINT8_BITS)
    entries = {"float32": tensor_bit_cost(latent_shape, _FLOAT32_BITS)}
    for bits in bit_widths:
        entries[f"{bits}_bit"] = tensor_bit_cost(latent_shape, bits)

    for entry in entries.values():
        # Labeled "raw_size_ratio_vs_uint8_image", never "compression ratio":
        # no entropy coding, no metadata, raw-RGB baseline.
        entry["raw_size_ratio_vs_uint8_image"] = image["total_bits"] / entry["total_bits"]

    return {
        "note": (
            "Theoretical raw tensor storage only. Not a codec bitrate and not a "
            "compression ratio: no entropy coding, no quantization metadata "
            "(scale/zero_point) counted, sub-byte widths not actually bit-packed, "
            "and the image baseline is raw uint8 RGB rather than a real image codec."
        ),
        "image_raw_uint8": image,
        "latent": entries,
    }
