"""Quantization, entropy coding, and .nvc binary serialization.

- quantization:     UniformQuantizer (uniform scalar affine), global and
                    per-channel modes, plus latent-space error metrics
- calibration:      fixed quantization parameters derived from the TRAINING
                    split only, saved to a shared calibration file
- entropy_model:    EmpiricalEntropyModel, static per-channel symbol
                    frequency tables (no learned/context/hyperprior model)
- range_coder:      arithmetic coder implemented from first principles
- nvc_format:       the .nvc container - NVCHeader / NVCWriter / NVCReader -
                    plus the additive .nvcs stream container (NVCStreamHeader /
                    NVCStreamWriter / NVCStreamReader) for multiple frames
                    sharing one model/calibration without repeating the
                    per-frame quantization-parameter block
- codec:            end-to-end frame <-> .nvc encode and decode, plus the
                    stream-payload variants (encode_frame_payload /
                    decode_frame_payload / build_stream_header)
- storage_analysis: theoretical raw tensor storage arithmetic

Learned entropy models, hyperpriors, context/autoregressive models, and
inter-frame prediction are NOT implemented - the entropy model here is a
static table counted from calibration data.
"""

from .calibration import (
    CALIBRATION_METHOD,
    allocate_bits_per_channel,
    calibrate_quantization_params,
    collect_calibration_latents,
    load_calibration,
    save_calibration,
)
from .codec import (
    EncodeResult,
    build_stream_header,
    channel_table_index,
    decode_frame,
    decode_frame_payload,
    decode_latent,
    decode_payload_to_latent,
    encode_frame,
    encode_frame_payload,
    encode_latent,
    encode_latent_to_payload,
    latent_to_symbols,
    symbols_to_latent,
    verify_lossless_roundtrip,
)
from .entropy_model import (
    EmpiricalEntropyModel,
    empirical_entropy,
    symbol_distribution_summary,
)
from .nvc_format import (
    NVCFormatError,
    NVCHeader,
    NVCReader,
    NVCStreamHeader,
    NVCStreamReader,
    NVCStreamWriter,
    NVCWriter,
)
from .quantization import (
    QUANTIZATION_MODES,
    QuantizationParams,
    UniformQuantizer,
    count_clipped,
    quantization_error,
)
from .range_coder import decode_symbols, encode_symbols
from .storage_analysis import latent_storage_analysis, tensor_bit_cost

__all__ = [
    # quantization
    "UniformQuantizer",
    "QuantizationParams",
    "QUANTIZATION_MODES",
    "quantization_error",
    "count_clipped",
    # calibration
    "calibrate_quantization_params",
    "allocate_bits_per_channel",
    "collect_calibration_latents",
    "save_calibration",
    "load_calibration",
    "CALIBRATION_METHOD",
    # entropy model
    "EmpiricalEntropyModel",
    "empirical_entropy",
    "symbol_distribution_summary",
    # entropy coder
    "encode_symbols",
    "decode_symbols",
    # .nvc container
    "NVCHeader",
    "NVCWriter",
    "NVCReader",
    "NVCFormatError",
    # .nvcs stream container
    "NVCStreamHeader",
    "NVCStreamWriter",
    "NVCStreamReader",
    # end-to-end codec
    "encode_frame",
    "decode_frame",
    "encode_latent",
    "decode_latent",
    "encode_latent_to_payload",
    "decode_payload_to_latent",
    "encode_frame_payload",
    "decode_frame_payload",
    "build_stream_header",
    "latent_to_symbols",
    "symbols_to_latent",
    "channel_table_index",
    "verify_lossless_roundtrip",
    "EncodeResult",
    # storage
    "latent_storage_analysis",
    "tensor_bit_cost",
]
