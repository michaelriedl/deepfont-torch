"""Script to generate a parquet manifest file for DeepFont datasets.

The manifest pre-computes image metadata (BCF byte offsets, real image paths,
labels) so that dataset initialization no longer needs to scan the filesystem
at training time. This is especially valuable on cloud compute where network
filesystem operations are slow.

All paths stored in the manifest are relative to the manifest file's parent
directory, making the manifest portable across machines and mount points.

Usage:
    python scripts/create_manifest.py \\
        --bcf-file data/deepfont_data/BCF\\ format/VFR_syn_train/train.bcf \\
        --label-file data/deepfont_data/BCF\\ format/VFR_syn_train/train.label \\
        --real-image-dir data/deepfont_data/Raw\\ Image/VFR_real_u/scrape-wtf-new \\
        --output data/manifests/pretrain.parquet

    # Finetune manifest (BCF + labels only, no real images):
    python scripts/create_manifest.py \\
        --bcf-file data/deepfont_data/BCF\\ format/VFR_syn_train/train.bcf \\
        --label-file data/deepfont_data/BCF\\ format/VFR_syn_train/train.label \\
        --output data/manifests/finetune.parquet

    # Real images only:
    python scripts/create_manifest.py \\
        --real-image-dir data/deepfont_data/Raw\\ Image/VFR_real_u/scrape-wtf-new \\
        --output data/manifests/real_only.parquet
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REAL_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif")


def _make_relative(path: Path, base: Path) -> str:
    """Return path relative to base, falling back to absolute if not under base."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        # Path is not under base (e.g. different drive on Windows) — store absolute.
        return str(path)


def _build_synthetic_rows(
    bcf_file: Path,
    label_file: Path | None,
    manifest_dir: Path,
) -> pa.Table:
    """Read a BCF file and optional label file; return a PyArrow Table."""
    with open(bcf_file, "rb") as f:
        count = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        sizes = np.frombuffer(f.read(8 * count), dtype=np.uint64)

    # Absolute byte offset of image i from start of file:
    #   header = 8 (count) + count*8 (size index) = (count+1)*8
    #   data offset[i] = header + sum(sizes[0..i-1])
    header_size = np.uint64((count + 1) * 8)
    cumsum = np.concatenate([[np.uint64(0)], np.cumsum(sizes[:-1])]) if count > 0 else np.array([], dtype=np.uint64)
    abs_offsets = header_size + cumsum  # shape (count,)

    bcf_rel = _make_relative(bcf_file.resolve(), manifest_dir)

    arrays = {
        "image_type": pa.array(["synthetic"] * count, type=pa.string()),
        "bcf_file": pa.array([bcf_rel] * count, type=pa.string()),
        "bcf_index": pa.array(np.arange(count, dtype=np.int64), type=pa.int64()),
        "bcf_offset": pa.array(abs_offsets.astype(np.int64), type=pa.int64()),
        "bcf_size": pa.array(sizes.astype(np.int64), type=pa.int64()),
        "filepath": pa.array([None] * count, type=pa.string()),
        "label": pa.array([None] * count, type=pa.int32()),
    }

    if label_file is not None:
        with open(label_file, "rb") as f:
            labels = np.frombuffer(f.read(), dtype=np.uint32)
        if len(labels) != count:
            raise ValueError(
                f"Label count ({len(labels)}) does not match BCF image count ({count}). "
                f"BCF: {bcf_file}, labels: {label_file}"
            )
        arrays["label"] = pa.array(labels.astype(np.int32), type=pa.int32())

    return pa.table(arrays)


def _build_real_rows(real_image_dir: Path, manifest_dir: Path) -> pa.Table:
    """Scan a directory for image files; return a PyArrow Table."""
    names = [
        x for x in os.listdir(real_image_dir) if x.lower().endswith(_REAL_IMAGE_EXTENSIONS)
    ]
    rel_paths = [
        _make_relative((real_image_dir / name).resolve(), manifest_dir) for name in names
    ]
    count = len(rel_paths)

    return pa.table(
        {
            "image_type": pa.array(["real"] * count, type=pa.string()),
            "bcf_file": pa.array([None] * count, type=pa.string()),
            "bcf_index": pa.array([None] * count, type=pa.int64()),
            "bcf_offset": pa.array([None] * count, type=pa.int64()),
            "bcf_size": pa.array([None] * count, type=pa.int64()),
            "filepath": pa.array(rel_paths, type=pa.string()),
            "label": pa.array([None] * count, type=pa.int32()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a parquet manifest for DeepFont datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bcf-file", metavar="PATH", help="Path to BCF store file.")
    parser.add_argument(
        "--label-file",
        metavar="PATH",
        help="Path to binary uint32 label file (requires --bcf-file).",
    )
    parser.add_argument(
        "--real-image-dir",
        metavar="PATH",
        help="Directory of real images (.png/.jpg/.jpeg/.gif).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        required=True,
        help="Destination path for the output parquet file.",
    )
    args = parser.parse_args()

    if args.bcf_file is None and args.real_image_dir is None:
        parser.error("At least one of --bcf-file or --real-image-dir must be provided.")

    if args.label_file is not None and args.bcf_file is None:
        parser.error("--label-file requires --bcf-file.")

    output_path = Path(args.output).resolve()
    manifest_dir = output_path.parent
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tables: list[pa.Table] = []

    if args.bcf_file is not None:
        bcf_file = Path(args.bcf_file).resolve()
        label_file = Path(args.label_file).resolve() if args.label_file else None
        print(f"Reading BCF file: {bcf_file}")
        syn_table = _build_synthetic_rows(bcf_file, label_file, manifest_dir)
        tables.append(syn_table)
        print(f"  Synthetic images: {len(syn_table):,}")
        if label_file:
            print(f"  Labels loaded from: {label_file}")

    if args.real_image_dir is not None:
        real_dir = Path(args.real_image_dir).resolve()
        print(f"Scanning real image directory: {real_dir}")
        real_table = _build_real_rows(real_dir, manifest_dir)
        tables.append(real_table)
        print(f"  Real images found: {len(real_table):,}")

    if not tables:
        print("No data to write. Exiting.", file=sys.stderr)
        sys.exit(1)

    combined = pa.concat_tables(tables)
    pq.write_table(combined, str(output_path), compression="snappy")
    print(f"\nManifest written to: {output_path}")
    print(f"Total rows: {len(combined):,}")


if __name__ == "__main__":
    main()
