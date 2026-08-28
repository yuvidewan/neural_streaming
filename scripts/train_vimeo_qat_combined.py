"""Fully local equivalent of colab_train_vimeo.ipynb's Section 9 - no Drive,
no Colab, nothing leaves this machine.

Downloads one Vimeo-90K chunk from Kaggle, trains BOTH the QAT and control
models on it (each with its own per-chunk early stopping), deletes the
chunk to free disk space, moves to the next one. Checkpoints, progress, and
history are all read from and written to a plain local directory
(--output-dir, defaults to <repo>/outputs/qat_combined/) - nothing here
talks to Google Drive or the internet except Kaggle downloads.

Safe to interrupt and re-run: it always resumes from --output-dir's own
checkpoints_qat_noise/latest.pt and checkpoints_qat_control/latest.pt plus
their progress_*.json (which chunks are already done), so stopping and
restarting the script picks up exactly where it left off. The very first
time either run starts (no latest.pt yet), it bootstraps from
--bootstrap-checkpoint instead and begins its own epoch count at 1.

PREREQUISITES (once, on the machine running this script):
    1. This repo cloned, its venv created and activated, and
       `pip install -e .` (or `pip install -r requirements.txt`) run - same
       as any other script in this project.
    2. `pip install kaggle`, plus a Kaggle API token at
       ~/.kaggle/kaggle.json (Linux/Mac) or
       %USERPROFILE%\\.kaggle\\kaggle.json (Windows). Get one from
       kaggle.com -> Account -> Create New API Token.
    3. A calibration file already generated from the TRAIN split (see
       scripts\\calibrate_quantizer.py) and a starting checkpoint to
       fine-tune from - both default to files already in this repo's
       outputs/ folder (outputs/calibration/vimeo_epoch17_4bit.json and
       outputs/checkpoints/vimeo_epoch17_best.pt); override with
       --calibration-path / --bootstrap-checkpoint if yours live elsewhere.

Example usage (all defaults, just run it):

    python scripts\\train_vimeo_qat_combined.py

Everything else (chunk range, epochs-per-chunk ceiling, early-stopping
patience, batch size, crop size, seed) defaults to match the notebook
exactly; override any of them with the flags below. Run with --help for
the full option list.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import timedelta
from pathlib import Path

import torch

from nvc.data.loaders import create_sequence_train_loader, create_sequence_test_loader
from nvc.data.vimeo import build_sequence_manifest
from nvc.models import BaselineAutoencoder
from nvc.training import (
    QuantizationNoise,
    resume_training_state,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

KAGGLE_DATASET_OWNER = "wangsally"
KAGGLE_DATASET_PREFIX = "vimeo-90k"  # -> wangsally/vimeo-90k-1, vimeo-90k-2, ...


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fully local equivalent of colab_train_vimeo.ipynb's Section 9: trains the "
            "QAT and control models together, one Vimeo-90K chunk at a time, "
            "reading/writing checkpoints in a plain local output directory - no Drive, no Colab."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Local directory to read/write checkpoints_qat_noise/, checkpoints_qat_control/, "
            "and progress_*.json for both runs. Re-running with the same --output-dir resumes "
            "automatically. Defaults to <repo>/outputs/qat_combined."
        ),
    )
    parser.add_argument(
        "--bootstrap-checkpoint", type=Path, default=None,
        help=(
            "Checkpoint each run fine-tunes from the FIRST time it starts (ignored once "
            "--output-dir has its own checkpoints_qat_noise/latest.pt or "
            "checkpoints_qat_control/latest.pt to resume from instead). "
            "Defaults to <repo>/outputs/checkpoints/vimeo_epoch17_best.pt."
        ),
    )
    parser.add_argument(
        "--calibration-path", type=Path, default=None,
        help=(
            "Calibration file supplying the QAT training-noise scale, generated from the "
            "TRAIN split only (see scripts\\calibrate_quantizer.py). Required to exist before "
            "this script runs. Defaults to "
            "<repo>/outputs/calibration/vimeo_epoch17_<qat-bits>bit.json."
        ),
    )
    parser.add_argument(
        "--scratch-dir", type=Path, default=None,
        help="Local staging directory for one chunk's raw download/extract "
             "(wiped and recreated per chunk). Defaults to <repo>/data/external/_vimeo_scratch.",
    )
    parser.add_argument(
        "--vimeo-root", type=Path, default=None,
        help="Directory where this script rebuilds the sequences/ symlink tree (plus "
             "sep_trainlist.txt/sep_testlist.txt) each chunk. Defaults to "
             "<repo>/data/external/vimeo_septuplet. Like --scratch-dir, point this outside any "
             "cloud-synced folder (OneDrive/Google Drive) if you hit FileExistsError here - a "
             "synced folder can hold a sustained lock on a file inside it while re-uploading "
             "the previous chunk's contents, which a few retries won't outlast.",
    )
    parser.add_argument(
        "--chunks", type=int, nargs="+", default=list(range(1, 11)),
        help="Which Kaggle chunk numbers to cycle through (wangsally/vimeo-90k-1..10).",
    )
    parser.add_argument("--epochs-per-chunk-max", type=int, default=10,
                         help="Ceiling per chunk; early stopping usually stops sooner.")
    parser.add_argument("--early-stop-patience", type=int, default=2,
                         help="Consecutive non-improving epochs (within one chunk) before stopping early.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5,
                         help="Minimum val_loss improvement required to reset patience.")
    parser.add_argument("--batch-size", type=int, default=32, help="Matches the notebook's BATCH_SIZE.")
    parser.add_argument("--crop-size", type=int, default=256, help="Must be divisible by 16.")
    parser.add_argument("--latent-channels", type=int, default=defaults.latent_channels)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="0 is safe on every OS. If you're on Linux/Mac (fork-based multiprocessing), "
             "2-4 is faster - see nvc.data.loaders' Windows-safety note.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument(
        "--keep-chunk", action="store_true",
        help="Keep each chunk's downloaded/extracted files instead of deleting them after training on it.",
    )
    parser.add_argument("--kaggle-dataset-owner", default=KAGGLE_DATASET_OWNER)
    parser.add_argument("--kaggle-dataset-prefix", default=KAGGLE_DATASET_PREFIX)
    parser.add_argument(
        "--add-defender-exclusion", action="store_true",
        help=(
            "Windows only, opt-in, off by default: before the chunk loop starts, try to add "
            "a Windows Defender exclusion for the scratch/vimeo data folder, via "
            "'Add-MpPreference -ExclusionPath'. This needs Administrator privileges - if the "
            "terminal isn't elevated it will just print a message and continue (never fatal). "
            "Reduces the odds of the transient FileExistsError extraction issue and speeds up "
            "extraction/training I/O. Nothing is changed unless you pass this flag."
        ),
    )
    return parser


def _try_add_defender_exclusion(target_dir: Path) -> None:
    """Best-effort, non-fatal: ask Windows Defender to stop scanning
    target_dir (Add-MpPreference needs Administrator privileges - if the
    current process isn't elevated this fails gracefully with a message
    instead of blocking the run). No-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        print("[setup] --add-defender-exclusion is Windows-only - skipping.")
        return
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        print("[setup] Could not find powershell/pwsh on PATH - skipping Defender exclusion.", file=sys.stderr)
        return

    escaped = str(target_dir).replace("'", "''")
    try:
        result = subprocess.run(
            [powershell, "-NoProfile", "-Command", f"Add-MpPreference -ExclusionPath '{escaped}'"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception as exc:  # subprocess itself failing to launch, timeout, etc.
        print(f"[setup] Could not add Windows Defender exclusion automatically: {exc!r}", file=sys.stderr)
        return

    if result.returncode == 0:
        print(f"[setup] Added Windows Defender exclusion for {target_dir}")
    else:
        print(
            "[setup] Could not add Windows Defender exclusion automatically (this usually means "
            "the terminal isn't running as Administrator). Training will continue without it - "
            "add it manually if you want to: Windows Security -> Virus & threat protection -> "
            f"Manage settings -> Add or remove exclusions -> Folder -> {target_dir}\n"
            f"         Details: {result.stderr.strip()}",
            file=sys.stderr,
        )


# --- Chunk download/extraction/relinking - ported verbatim from
# colab_train_vimeo.ipynb Section 5, parameterized instead of relying on
# notebook globals. Keep this in sync with the notebook if either changes. ---


def _find_sequences_source_root(chunk_dir: Path) -> Path:
    """Locate the folder whose children are Vimeo "<group>" directories, by
    finding the first im1.png anywhere under chunk_dir and walking up two
    levels (im1.png -> <clip>/ -> <group>/ -> the folder holding all groups).
    """
    for im1 in chunk_dir.rglob("im1.png"):
        clip_dir = im1.parent
        group_dir = clip_dir.parent
        return group_dir.parent
    raise RuntimeError(f"No im1.png found anywhere under {chunk_dir} - unexpected chunk layout")


def download_and_extract_chunk(
    chunk_number: int, scratch_dir: Path, *, dataset_owner: str, dataset_prefix: str,
) -> Path:
    """Download one <dataset_owner>/<dataset_prefix>-N Kaggle dataset and extract it.
    Returns the folder whose direct children are Vimeo "<group>" folders.
    """
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True)

    slug = f"{dataset_owner}/{dataset_prefix}-{chunk_number}"
    print(f"[chunk {chunk_number}] downloading {slug} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(scratch_dir)], check=True)

    zips = list(scratch_dir.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"[chunk {chunk_number}] no .zip downloaded into {scratch_dir}")
    print(f"[chunk {chunk_number}] extracting {zips[0].name} ...")
    _extract_with_retry(zips[0], scratch_dir, chunk_number)
    zips[0].unlink()

    return _find_sequences_source_root(scratch_dir)


def _extract_with_retry(zip_path: Path, dest_dir: Path, chunk_number: int, *, attempts: int = 5) -> None:
    """zipfile.extractall() can intermittently raise FileExistsError partway
    through a large extraction on Windows (WinError 183, "Cannot create a
    file when that file already exists") when antivirus real-time scanning
    locks a just-created file or directory mid-write - a known Windows-side
    race with bulk small-file extraction, not a corrupt archive. Re-running
    extractall from scratch is safe (it just overwrites whatever partial
    files already landed), so retry a few times with a short backoff before
    giving up.
    """
    last_exc: FileExistsError | None = None
    for attempt in range(1, attempts + 1):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_dir)
            return
        except FileExistsError as exc:
            last_exc = exc
            print(
                f"[chunk {chunk_number}] extraction hit a transient FileExistsError "
                f"(attempt {attempt}/{attempts}): {exc} - retrying...",
                file=sys.stderr,
            )
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"[chunk {chunk_number}] extraction failed {attempts} times in a row with "
        "FileExistsError. This is almost always Windows Defender (or another antivirus) "
        "real-time-scanning and locking files during a large bulk extraction, not a bad "
        f"download. Add a Windows Defender exclusion for {dest_dir.parent} (Windows Security "
        "-> Virus & threat protection -> Manage settings -> Add or remove exclusions -> "
        "Folder), then re-run this script - it will retry this chunk automatically."
    ) from last_exc


def _reset_dir_with_retry(path: Path, *, attempts: int = 5) -> None:
    """Delete path (if present) and recreate it empty, retrying with
    backoff. shutil.rmtree(path, ignore_errors=True) can silently fail to
    fully remove path - e.g. when path is under a synced folder (OneDrive,
    Google Drive) that's mid-upload and holding a lock on a file inside it,
    or Windows Defender is scanning one - and the subsequent mkdir() then
    crashes with FileExistsError because the old directory is still there.
    Same root cause and fix shape as _extract_with_retry above.
    """
    last_exc: OSError | None = None
    for attempt in range(1, attempts + 1):
        shutil.rmtree(path, ignore_errors=True)
        try:
            path.mkdir(parents=True)
            return
        except FileExistsError as exc:
            last_exc = exc
            print(
                f"[setup] Could not fully clear {path} (attempt {attempt}/{attempts}): "
                f"{exc} - retrying...",
                file=sys.stderr,
            )
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"Could not clear and recreate {path} after {attempts} attempts. This usually means a "
        "synced folder (OneDrive/Google Drive) or antivirus real-time scanning is holding a "
        f"lock on something inside it. Either add a Defender exclusion for {path.parent} "
        "(Windows Security -> Virus & threat protection -> Manage settings -> Add or remove "
        "exclusions -> Folder), or move this project's data out of any cloud-synced folder, "
        "then re-run this script."
    ) from last_exc


def relink_sequences_to_chunk(chunk_group_root: Path, vimeo_root: Path) -> list[str]:
    """Point vimeo_root/sequences at exactly this chunk's group folders via
    symlinks - no copying, so this costs no extra disk beyond the chunk
    itself. Returns the list of group directory names now available.

    Requires symlink permission: on Windows this needs either Developer
    Mode enabled (Settings -> Privacy & security -> For developers) or
    running as Administrator; Linux/Mac need no special setup.
    """
    sequences_dir = vimeo_root / "sequences"
    _reset_dir_with_retry(sequences_dir)

    group_names = []
    for group_dir in sorted(p for p in chunk_group_root.iterdir() if p.is_dir()):
        (sequences_dir / group_dir.name).symlink_to(group_dir, target_is_directory=True)
        group_names.append(group_dir.name)
    return group_names


def discover_complete_sequence_ids(sequences_dir: Path) -> list[str]:
    """List "<group>/<clip>" ids that have all 7 im*.png frames present."""
    ids = []
    for group_dir in sorted(p for p in sequences_dir.iterdir() if p.is_dir()):
        for clip_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if all((clip_dir / f"im{i}.png").is_file() for i in range(1, 8)):
                ids.append(f"{group_dir.name}/{clip_dir.name}")
    return ids


def write_chunk_split_lists(
    vimeo_root: Path, sequence_ids: list[str], seed: int, test_fraction: float = 0.1,
) -> None:
    """(Re)write sep_trainlist.txt / sep_testlist.txt scoped to exactly the
    sequence ids available from the chunk currently linked into
    vimeo_root/sequences. A per-chunk *training-progress* split, not a
    persistent benchmark - discarded with the chunk. Does NOT reproduce the
    official Vimeo-90K train/test split. DAVIS's existing test split is what
    the actual final model comparison uses.
    """
    ordered = sorted(sequence_ids)
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_ids = sorted(shuffled[:n_test])
    train_ids = sorted(shuffled[n_test:])

    (vimeo_root / "sep_trainlist.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (vimeo_root / "sep_testlist.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    print(f"[split] {len(train_ids)} train / {len(test_ids)} test sequence ids for this chunk")


def build_chunk_manifests(vimeo_root: Path, vimeo_manifest_path: Path, seed: int) -> tuple[Path, Path]:
    train_path = vimeo_manifest_path.with_stem(vimeo_manifest_path.stem + "_train")
    test_path = vimeo_manifest_path.with_stem(vimeo_manifest_path.stem + "_test")
    build_sequence_manifest(vimeo_root, train_path, split="train", max_sequences=None, seed=seed, validate=True)
    build_sequence_manifest(vimeo_root, test_path, split="test", max_sequences=None, seed=seed, validate=True)
    return train_path, test_path


# --- Progress / bootstrap / per-chunk early stopping - ported verbatim
# from colab_train_vimeo.ipynb Section 9. Keep in sync with the notebook. ---


def _load_or_init_progress(progress_path: Path) -> dict:
    if progress_path.is_file():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"completed_chunks": [], "best_val_loss": None}


def _save_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _bootstrap_run(run_name, model, optimizer, checkpoint_dir, bootstrap_from, device):
    """Resume this run's own latest.pt if present, else bootstrap weights
    from `bootstrap_from` and restart epoch numbering at 1, else random init.
    """
    latest_ckpt = checkpoint_dir / "latest.pt"
    if latest_ckpt.is_file():
        epoch, history = resume_training_state(latest_ckpt, model=model, optimizer=optimizer, map_location=device)
        print(f"[{run_name}] resumed from {latest_ckpt} - next epoch {epoch}, {len(history)} prior record(s)")
        return epoch, history
    if bootstrap_from is not None and bootstrap_from.is_file():
        resume_training_state(bootstrap_from, model=model, optimizer=optimizer, map_location=device)
        print(f"[{run_name}] bootstrapped from {bootstrap_from}, starting fresh at epoch 1")
        return 1, []
    print(f"[{run_name}] no existing checkpoint - starting from random init at epoch 1")
    return 1, []


def _train_one_chunk_with_early_stopping(
    run_name, model, optimizer, train_loader, test_loader, device, *, start_epoch,
    max_epochs, patience, min_delta, chunk_number, history, checkpoint_dir,
    model_config, best_val_loss, run_type, qat_enabled, qat_bits, qat_mode,
):
    """Up to max_epochs epochs on one chunk, stopping early if val_loss
    hasn't improved by min_delta for `patience` consecutive epochs.
    Returns (next_epoch, updated_best_val_loss). best_val_loss/checkpointing
    is GLOBAL (across all chunks) - only the early-stopping decision itself
    is scoped to this one chunk.
    """
    epoch = start_epoch
    best_chunk_val_loss = float("inf")
    epochs_without_improvement = 0

    for step in range(max_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            progress_desc=f"[{run_name}] chunk {chunk_number} epoch {epoch} train",
        )
        val_metrics = validate_one_epoch(
            model, test_loader, device,
            progress_desc=f"[{run_name}] chunk {chunk_number} epoch {epoch} val",
        )
        train_loss, val_loss, val_psnr = train_metrics["loss"], val_metrics["loss"], val_metrics["psnr"]

        history.append({
            "epoch": epoch, "chunk": chunk_number,
            "train_loss": train_loss, "val_loss": val_loss, "val_psnr": val_psnr,
            "run_type": run_type, "qat_enabled": qat_enabled,
            "qat_bits": qat_bits if qat_enabled else None,
            "qat_mode": qat_mode if qat_enabled else None,
        })
        print(f"  [{run_name}] chunk {chunk_number} epoch {epoch} (step {step + 1}/{max_epochs}): "
              f"train_mse={train_loss:.6f} val_mse={val_loss:.6f} val_psnr={val_psnr:.2f} dB")

        save_checkpoint(checkpoint_dir / "latest.pt", model=model, optimizer=optimizer,
                         epoch=epoch, history=history, model_config=model_config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", model=model, optimizer=optimizer,
                             epoch=epoch, history=history, model_config=model_config)
            print(f'    [{run_name}] new global-best val MSE {best_val_loss:.6f} -> {checkpoint_dir / "best.pt"}')
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        epoch += 1
        if val_loss < best_chunk_val_loss - min_delta:
            best_chunk_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  [{run_name}] early stop on chunk {chunk_number} - no improvement for "
                      f"{patience} epoch(s) (used {step + 1}/{max_epochs})")
                break

    return epoch, best_val_loss


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if shutil.which("kaggle") is None:
        print(
            "[ERROR] The 'kaggle' CLI was not found on PATH. Install it with "
            "'pip install kaggle' and place your API token at "
            "~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\kaggle.json on Windows).",
            file=sys.stderr,
        )
        return 1

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    output_dir = args.output_dir or (defaults.checkpoint_dir.parent / "qat_combined")
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = args.scratch_dir or (defaults.raw_data_dir.parent / "external" / "_vimeo_scratch")
    vimeo_root = args.vimeo_root or defaults.vimeo_root
    vimeo_root.mkdir(parents=True, exist_ok=True)

    if args.add_defender_exclusion:
        # scratch_dir's parent (data/external/) covers both the raw chunk
        # scratch space and vimeo_root - the two places under heavy
        # small-file I/O during extraction and training.
        _try_add_defender_exclusion(scratch_dir.parent)

    qat_resume_from = args.bootstrap_checkpoint or (defaults.checkpoint_dir / "vimeo_epoch17_best.pt")
    qat_calibration_path = args.calibration_path or (
        defaults.checkpoint_dir.parent / "calibration" / f"vimeo_epoch17_{args.qat_bits}bit.json"
    )
    print(f"[paths] output dir:          {output_dir}")
    print(f"[paths] bootstrap checkpoint: {qat_resume_from}"
          f"{' (missing - will fall back to random init)' if not qat_resume_from.is_file() else ''}")
    print(f"[paths] calibration file:    {qat_calibration_path}")
    if not qat_calibration_path.is_file():
        print(
            f"[ERROR] {qat_calibration_path} not found.\n"
            "        Generate it first (scripts\\calibrate_quantizer.py --checkpoint "
            f"<the base checkpoint> --bits {args.qat_bits} --mode {args.qat_mode} "
            "--output <this path>) or pass an existing one via --calibration-path "
            "before running this script.",
            file=sys.stderr,
        )
        return 1

    latent_channels = args.latent_channels
    learning_rate = args.learning_rate

    quantization_noise = QuantizationNoise.from_calibration(
        qat_calibration_path, bits=args.qat_bits, mode=args.qat_mode,
    )
    print(f"[qat] noise scale loaded - {quantization_noise.bits}-bit / {quantization_noise.mode}")

    runs = {}
    for run_type, noise in (("qat", quantization_noise), ("qat_control", None)):
        checkpoint_dir = output_dir / ("checkpoints_qat_noise" if run_type == "qat" else "checkpoints_qat_control")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / ("progress_qat_noise.json" if run_type == "qat" else "progress_qat_control.json")

        model = BaselineAutoencoder(latent_channels=latent_channels, quantization_noise=noise).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        epoch, history = _bootstrap_run(run_type, model, optimizer, checkpoint_dir, qat_resume_from, device)
        progress = _load_or_init_progress(progress_path)

        runs[run_type] = {
            "model": model, "optimizer": optimizer, "model_config": model.config_dict(),
            "checkpoint_dir": checkpoint_dir, "progress_path": progress_path,
            "epoch": epoch, "history": history, "progress": progress,
            "best_val_loss": progress["best_val_loss"] if progress["best_val_loss"] is not None else float("inf"),
            "qat_enabled": run_type == "qat",
        }

    run_start = time.time()
    chunk_durations: list[float] = []
    total_chunks = len(args.chunks)

    for chunk_index, chunk_number in enumerate(args.chunks, start=1):
        pending = [name for name, run in runs.items() if chunk_number not in run["progress"]["completed_chunks"]]
        if not pending:
            print(f"[chunk {chunk_number}] already completed by both runs - skipping")
            continue

        chunk_start = time.time()
        sep = "=" * 60
        print()
        print(sep)
        print(f"[chunk {chunk_number}] starting ({chunk_index}/{total_chunks}, pending: {pending})")
        if chunk_durations:
            # Empirical ETA from chunks actually completed so far - never a
            # guess made before any real timing exists. Early stopping and
            # per-chunk download size both vary, so this refines as it goes
            # rather than being fixed at the start.
            avg = sum(chunk_durations) / len(chunk_durations)
            remaining = total_chunks - chunk_index + 1
            print(f"[chunk {chunk_number}] avg {timedelta(seconds=round(avg))}/chunk so far "
                  f"({len(chunk_durations)} completed) - "
                  f"estimated {timedelta(seconds=round(avg * remaining))} remaining "
                  f"({remaining} chunk(s) left)")
        print(sep)

        try:
            chunk_group_root = download_and_extract_chunk(
                chunk_number, scratch_dir,
                dataset_owner=args.kaggle_dataset_owner, dataset_prefix=args.kaggle_dataset_prefix,
            )
            group_names = relink_sequences_to_chunk(chunk_group_root, vimeo_root)
            print(f"[chunk {chunk_number}] linked {len(group_names)} group folder(s)")

            sequence_ids = discover_complete_sequence_ids(vimeo_root / "sequences")
            print(f"[chunk {chunk_number}] {len(sequence_ids)} complete 7-frame sequences found")
            if not sequence_ids:
                raise RuntimeError("no complete sequences found in this chunk")

            write_chunk_split_lists(vimeo_root, sequence_ids, seed=args.seed)
            train_manifest, test_manifest = build_chunk_manifests(
                vimeo_root, defaults.vimeo_manifest_path, seed=args.seed,
            )

            train_loader = create_sequence_train_loader(
                train_manifest, batch_size=args.batch_size, num_workers=args.num_workers,
                seed=args.seed, crop_size=args.crop_size,
            )
            test_loader = create_sequence_test_loader(
                test_manifest, batch_size=args.batch_size, num_workers=args.num_workers,
                crop_size=args.crop_size,
            )

            for name in pending:
                run = runs[name]
                run["epoch"], run["best_val_loss"] = _train_one_chunk_with_early_stopping(
                    name, run["model"], run["optimizer"], train_loader, test_loader, device,
                    start_epoch=run["epoch"], max_epochs=args.epochs_per_chunk_max,
                    patience=args.early_stop_patience, min_delta=args.early_stop_min_delta,
                    chunk_number=chunk_number, history=run["history"],
                    checkpoint_dir=run["checkpoint_dir"], model_config=run["model_config"],
                    best_val_loss=run["best_val_loss"], run_type=name, qat_enabled=run["qat_enabled"],
                    qat_bits=args.qat_bits, qat_mode=args.qat_mode,
                )
                run["progress"]["completed_chunks"].append(chunk_number)
                run["progress"]["best_val_loss"] = run["best_val_loss"]
                _save_progress(run["progress_path"], run["progress"])
                print(f"[chunk {chunk_number}] [{name}] done, progress saved")

        except Exception as exc:
            print(f"[chunk {chunk_number}] FAILED: {exc!r} - re-run this script to retry it", file=sys.stderr)
            traceback.print_exc()
        finally:
            if not args.keep_chunk and scratch_dir.exists():
                shutil.rmtree(scratch_dir)
            chunk_durations.append(time.time() - chunk_start)
            print(f"[chunk {chunk_number}] took {timedelta(seconds=round(chunk_durations[-1]))}")

    print()
    print(f"All requested chunks processed (or already were) for both runs "
          f"- total elapsed {timedelta(seconds=round(time.time() - run_start))}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
