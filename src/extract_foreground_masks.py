"""foreground extraction using src.inference.blur.pixel_threshold.

Example usage:
    python -m src.extract_foreground_masks \
        --input-dir /path/to/input_volumes \
        --output-dir /path/to/output_masks

This script iterates over all ``.npy`` files inside the input directory, runs
``pixel_threshold`` to obtain a foreground mask (or rescaled volume) and stores
results as ``.npy`` files under the output directory.
"""
from __future__ import annotations

import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.inference.blur import pixel_threshold


def iter_npy_files(directory: Path, recursive: bool, pattern: str) -> Iterable[Path]:
    search_pattern = "**/*.npy" if recursive else "*.npy"
    if pattern:
        search_pattern = pattern
    yield from sorted(directory.glob(search_pattern))


def process_file(
    input_path: Path,
    output_path: Path,
    device: torch.device,
    gaussian_k: int,
    maxpool_k: int,
    bg_t_r: float,
    rescale_p: float,
    only_scale: bool,
) -> None:
    volume = np.load(input_path)
    if volume.dtype == np.uint16:
        volume = volume.astype(np.float32)
    tensor = torch.from_numpy(volume).to(device, dtype=torch.float32)

    with torch.no_grad():
        result = pixel_threshold(
            tensor,
            gaussian_k=gaussian_k,
            maxpool_k=maxpool_k,
            bg_t_r=bg_t_r,
            rescale_p=rescale_p,
            only_scale=only_scale,
        )

    result_tensor = result.detach().to("cpu")
    result_array = result_tensor.numpy()
    if only_scale:
        result_array = result_array.astype(np.float32, copy=False)
    else:
        result_array = result_array.astype(np.bool_, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result_array)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pixel_threshold on .npy volumes.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with input .npy files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store processed .npy files.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_mask",
        help="Suffix appended to the original file stem for the output filename.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="",
        help=(
            "Optional glob pattern relative to --input-dir. "
            "If omitted, all .npy files are processed."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively traverse --input-dir when collecting .npy files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run the computation on. 'cuda' requires GPU support.",
    )
    parser.add_argument("--gaussian-k", type=int, default=9, help="Gaussian kernel size.")
    parser.add_argument("--maxpool-k", type=int, default=75, help="MaxPool kernel size.")
    parser.add_argument("--bg-t-r", type=float, default=1.15, help="Background threshold ratio.")
    parser.add_argument(
        "--rescale-p",
        type=float,
        default=0.97,
        help="Percentile used for rescaling foreground intensities.",
    )
    parser.add_argument(
        "--only-scale",
        action="store_true",
        help="Return rescaled volume instead of a binary mask.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if args.pattern and (".npy" not in args.pattern and "*" not in args.pattern):
        print("[WARN] --pattern does not appear to be a glob; ensure it ends with .npy or uses wildcards.")

    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        print(f"[ERROR] Input directory {input_dir} does not exist.")
        return 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[ERROR] CUDA device requested but torch.cuda.is_available() is False.")
        return 1
    if device.type != "cuda":
        print(
            "[WARN] Running on CPU. automatic_foreground_process expects CUDA and may raise an error. "
            "Consider using --device cuda if a GPU is available."
        )

    files = list(iter_npy_files(input_dir, args.recursive, args.pattern))
    if not files:
        print(f"[INFO] No .npy files found under {input_dir}.")
        return 0

    for idx, input_path in enumerate(files, start=1):
        relative = input_path.relative_to(input_dir)
        output_path = (output_dir / relative).with_suffix("")
        output_path = output_path.with_name(output_path.name + args.suffix + ".npy")

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {relative} -> output exists (use --overwrite to force).")
            continue

        print(f"[PROCESS] ({idx}/{len(files)}) {relative} -> {output_path.relative_to(output_dir)}")
        process_file(
            input_path,
            output_path,
            device=device,
            gaussian_k=args.gaussian_k,
            maxpool_k=args.maxpool_k,
            bg_t_r=args.bg_t_r,
            rescale_p=args.rescale_p,
            only_scale=args.only_scale,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
