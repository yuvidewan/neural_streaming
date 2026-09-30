"""Where does nvc.video spend its time? Per-stage timing of encode and decode.

    python docs/c_rewrite_prototypes/profile_codec.py                 # CPU (what the report used)
    python docs/c_rewrite_prototypes/profile_codec.py --device cuda   # your GPU

Builds a realistic-size codec (256x256 frames, 64-channel latent, 4-bit, GOP 10, block 16,
search range 16, channel-group size 16, 512-entry codebook) with RANDOM weights - that is
fine for timing, since the cost of every stage here depends on tensor shapes, not on what
the weights are. It does NOT need the codec bundles (*.pt), which are gitignored.

Frames are real DAVIS frames from data/external/DAVIS if present, else synthetic.

The CUDA path is UNTESTED by the author (no GPU on the machine it was written on). It calls
torch.cuda.synchronize() around each timed stage so asynchronous kernel launches are not
mis-attributed; if numbers look odd on GPU, check that first.
"""
from __future__ import annotations

import argparse
import collections
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nvc.compression.calibration import calibrate_quantization_params  # noqa: E402
from nvc.compression.codec import latent_to_symbols  # noqa: E402
from nvc.compression.entropy_model import EmpiricalEntropyModel  # noqa: E402
from nvc.models.autoencoder import BaselineAutoencoder  # noqa: E402
from nvc.video import CodecBundle, VideoCodec  # noqa: E402
from nvc.video import codec as vcodec  # noqa: E402
from nvc.video import entropy as ventropy  # noqa: E402
from nvc.video import motion as vmotion  # noqa: E402
from nvc.video.motion import motion_alphabet_bits  # noqa: E402

BITS, GOP, BLOCK, RANGE = 4, 10, 16, 16
LATENT, GROUP, K, SIZE = 64, 16, 512, 256


def load_frames(count: int) -> tuple[torch.Tensor, str]:
    root = ROOT / "data/external/DAVIS/JPEGImages/480p"
    if root.is_dir():
        import cv2
        seq = sorted(root.iterdir())[0]
        frames = []
        for path in sorted(seq.glob("*.jpg"))[:count]:
            img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        if len(frames) == count:
            return torch.stack(frames).contiguous(), f"DAVIS/{seq.name}"
    g = torch.Generator().manual_seed(0)
    base = torch.nn.functional.interpolate(torch.rand(1, 3, 16, 16, generator=g), size=(SIZE * 2, SIZE * 2),
                                           mode="bilinear", align_corners=False)[0]
    frames = [base[:, 4 * t:4 * t + SIZE, 2 * t:2 * t + SIZE] for t in range(count)]
    return torch.stack(frames).clamp(0, 1).contiguous(), "synthetic"


def table(seed: int, tables: int, bits: int) -> EmpiricalEntropyModel:
    rng = np.random.default_rng(seed)
    return EmpiricalEntropyModel.from_symbols(rng.integers(0, 2 ** bits, size=(400, tables)),
                                              bits=bits, num_tables=tables)


def build_bundle(frames: torch.Tensor) -> CodecBundle:
    torch.manual_seed(0)
    ae = BaselineAutoencoder(latent_channels=LATENT, base_channels=32).eval()
    with torch.no_grad():
        latents = ae.encode(frames)
    intra_params = calibrate_quantization_params(latents, bits=BITS, mode="per_channel")
    intra_symbols = np.stack([latent_to_symbols(latents[i:i + 1], intra_params).reshape(LATENT, -1)
                              for i in range(latents.shape[0])])
    intra_model = EmpiricalEntropyModel.from_symbols(intra_symbols, bits=BITS, num_tables=LATENT)
    residual_params = calibrate_quantization_params(latents[1:] - latents[:-1], bits=BITS, mode="per_channel")
    torch.manual_seed(1)
    context = ventropy.ChannelContextEntropyModel(LATENT, 2 ** BITS, hidden=32, group_size=GROUP).eval()
    assign, coding = table(11, K, BITS).frequencies, table(12, K, BITS).frequencies
    motion = table(3, 2, motion_alphabet_bits(RANGE))
    signature = ventropy.calibration_signature(residual_params, bits=BITS, calibration_frames=20)
    identity = ventropy.model_identity(context, m10k_identity=b"\x00" * 8, calibration_signature=signature,
                                       bits=BITS, codebook=ventropy.SharedCodebook(coding, bits=BITS))
    return CodecBundle.build(
        name="profile", bits=BITS, gop_size=GOP, block_size=BLOCK, search_range=RANGE,
        calibration_frames=20, autoencoder=ae, intra_params=intra_params, intra_entropy_model=intra_model,
        residual_params=residual_params, context_model=context,
        assign_codebook_frequencies=assign, assign_codebook_metric="code_length",
        coding_codebook_frequencies=coding, motion_entropy_model=motion, m10k_identity="00" * 8,
        calibration_signature=signature, residual_entropy_model_id=identity.hex(),
        provenance={"source": "docs/c_rewrite_prototypes/profile_codec.py (random weights)"})


ACC: dict[str, float] = collections.defaultdict(float)
CNT: collections.Counter = collections.Counter()
SYNC = lambda: None  # replaced with torch.cuda.synchronize on CUDA


def timed(name, fn):
    def wrapper(*a, **k):
        SYNC()
        t = time.perf_counter()
        result = fn(*a, **k)
        SYNC()
        ACC[name] += time.perf_counter() - t
        CNT[name] += 1
        return result
    return wrapper


def isolated(codec: VideoCodec, frames: torch.Tensor) -> None:
    from nvc.compression.range_coder import decode_symbols, encode_symbols

    def med(fn, n):
        fn()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts) * 1000

    prev, cur = frames[3:4], frames[4:5]
    print("\n--- isolated stages, median ms (CPU only) ---")
    print(f"estimate_block_motion          {med(lambda: vmotion.estimate_block_motion(prev, cur, block_size=BLOCK, search_range=RANGE), 5):9.2f}")
    mv = vmotion.estimate_block_motion(prev, cur, block_size=BLOCK, search_range=RANGE)
    print(f"warp_blocks                    {med(lambda: vmotion.warp_blocks(prev, mv, block_size=BLOCK), 30):9.2f}")
    ae = codec.autoencoder
    with torch.no_grad():
        z = ae.encode(cur)
        print(f"autoencoder.encode             {med(lambda: ae.encode(cur), 20):9.2f}")
        print(f"autoencoder.decode             {med(lambda: ae.decode(z), 20):9.2f}")
        symbols = latent_to_symbols(z, codec.residual_params).reshape(tuple(z.shape[1:]))
        target = torch.from_numpy(symbols.astype(np.int64))[None]
        planes = codec.context_model.planes(target, codec.zero)
        print(f"context_planes                 {med(lambda: codec.context_model.planes(target, codec.zero), 20):9.2f}")
        print(f"context_model.log_probs (64ch) {med(lambda: codec.context_model.log_probabilities(z, planes), 10):9.2f}")
        rows = ventropy._rows(codec.context_model.log_probabilities(z, planes))
        print(f"assign_tensor (16384 rows)     {med(lambda: codec.assign_codebook.assign_tensor(rows), 10):9.2f}")
        table_index = codec.assign_codebook.assign_tensor(rows)
        flat = symbols.reshape(-1).astype(np.int64)
        cumulative = codec.coding_codebook.cumulative
        print(f"encode_symbols (16384, C)      {med(lambda: encode_symbols(flat, cumulative, table_index), 30):9.2f}")
        payload = encode_symbols(flat, cumulative, table_index)
        print(f"decode_symbols (16384, C)      {med(lambda: decode_symbols(payload, flat.size, cumulative, table_index), 30):9.2f}")
        print(f"latent_to_symbols              {med(lambda: latent_to_symbols(z, codec.residual_params), 30):9.2f}")


def main() -> None:
    global SYNC
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--frames", type=int, default=10, help="1 I-frame then P-frames (GOP 10)")
    args = parser.parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        SYNC = torch.cuda.synchronize

    frames, source = load_frames(args.frames)
    print(f"frames: {source} {tuple(frames.shape)}   device: {args.device}   torch threads: {torch.get_num_threads()}")
    codec = VideoCodec(build_bundle(frames), device=args.device)
    if args.device == "cpu":
        isolated(codec, frames)

    ae = codec.autoencoder
    vcodec.estimate_block_motion = timed("motion.estimate_block_motion", vcodec.estimate_block_motion)
    vcodec.encode_motion_payload = timed("motion.encode_payload", vcodec.encode_motion_payload)
    vcodec.decode_motion_payload = timed("motion.decode_payload", vcodec.decode_motion_payload)
    vcodec.warp_blocks = timed("motion.warp_blocks", vcodec.warp_blocks)
    vcodec.encode_residual_frame = timed("entropy.encode_residual_frame (total)", vcodec.encode_residual_frame)
    vcodec.decode_residual_frame = timed("entropy.decode_residual_frame (total)", vcodec.decode_residual_frame)
    vcodec.latent_to_symbols = timed("quant.latent_to_symbols", vcodec.latent_to_symbols)
    vcodec.symbols_to_latent = timed("quant.symbols_to_latent", vcodec.symbols_to_latent)
    ae.encode = timed("net.autoencoder.encode", ae.encode)
    ae.decode = timed("net.autoencoder.decode", ae.decode)
    codec.context_model.log_probabilities = timed("net.context_model.log_probs", codec.context_model.log_probabilities)
    codec.context_model.planes = timed("ctx.planes", codec.context_model.planes)
    codec.assign_codebook.assign_tensor = timed("entropy.assign_tensor", codec.assign_codebook.assign_tensor)
    decoder_cls = ventropy.ResumableDecoder
    decoder_cls.decode_group = timed("coder.decode_group (C)", decoder_cls.decode_group)
    ventropy.encode_symbols = timed("coder.encode_symbols (C)", ventropy.encode_symbols)

    codec.encode(frames[:2])                     # warm-up, untimed
    data = None
    for phase in ("encode", "decode"):
        ACC.clear()
        CNT.clear()
        t0 = time.perf_counter()
        if phase == "encode":
            data = codec.encode(frames).data
        else:
            codec.decode(data)
        total = time.perf_counter() - t0
        n = frames.shape[0]
        print(f"\n=== {phase.upper()} {n} frames: total {total * 1000:.0f} ms = {total * 1000 / n:.0f} ms/frame ===")
        for name, seconds in sorted(ACC.items(), key=lambda kv: -kv[1]):
            print(f"  {name:42s} {seconds * 1000:9.1f} ms  ({100 * seconds / total:5.1f}%)  calls={CNT[name]}")
        print("  (nested timers overlap: entropy.*_residual_frame contains assign_tensor, log_probs, coder)")


if __name__ == "__main__":
    main()
