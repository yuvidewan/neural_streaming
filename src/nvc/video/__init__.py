"""The learned video codec.

    from nvc.video import VideoCodec
    codec = VideoCodec.from_file("outputs/codec_bundles/nvc_m22_3bit.pt")
    encoded = codec.encode(frames)          # frames: [N, 3, H, W] float in [0, 1]
    frames_hat = codec.decode(encoded.data)

Or, without holding a codec object, `nvc.encode(frames, bundle)` and
`nvc.decode(data, bundle)`.

Modules:
- motion:    block motion estimation, integer warping, motion-payload coding
- container: the `.nvct` v2 stream format
- entropy:   causal channel-context entropy model, shared codebook, residual coding
- bundle:    `CodecBundle`, the frozen, self-verifying state of one operating point
- codec:     `VideoCodec`, the closed-loop encoder and decoder

Promoted from the research scripts (M10H, M10L, M11, M13, M14, M22), which remain
in `scripts/` unchanged as the record of how each piece was derived.
"""

from nvc.video.bundle import BundleError, CodecBundle
from nvc.video.codec import EncodedVideo, VideoCodec
from nvc.video.container import TemporalFormatError

__all__ = ["BundleError", "CodecBundle", "EncodedVideo", "TemporalFormatError", "VideoCodec"]
