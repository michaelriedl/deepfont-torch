"""Benchmark DataLoader num_workers to find optimal throughput.

Tests different worker counts with the PretrainData dataset and reports
images/sec for each configuration. Optionally tests with and without caching.

Usage:
    python scripts/benchmark_workers.py
    python scripts/benchmark_workers.py --num_images_to_cache 100000
    python scripts/benchmark_workers.py --workers 0 2 4 8 16 --batches 50
"""

import os
import sys
import time
import argparse

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PROJECT_ROOT", os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from deepfont.data.config import PretrainDataConfig
from deepfont.data.datasets import PretrainData, bcf_worker_init_fn


def benchmark_workers(
    dataset: PretrainData,
    num_workers: int,
    batch_size: int,
    num_batches: int,
    pin_memory: bool,
) -> dict:
    """Time a DataLoader with a given number of workers.

    Returns a dict with throughput stats.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=bcf_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # Warmup: iterate a few batches to let workers spin up
    warmup = min(5, num_batches // 2)
    it = iter(loader)
    for _ in range(warmup):
        next(it)

    # Timed run
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for i in range(num_batches):
        batch = next(it)
        # Simulate minimal GPU transfer if pin_memory is on
        if pin_memory and torch.cuda.is_available():
            batch = batch.cuda(non_blocking=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0

    images_per_sec = (num_batches * batch_size) / elapsed
    sec_per_batch = elapsed / num_batches

    del loader
    return {
        "num_workers": num_workers,
        "images_per_sec": images_per_sec,
        "sec_per_batch": sec_per_batch,
        "total_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DataLoader workers")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 12, 16, 20, 24, 32],
        help="Worker counts to test",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=100, help="Batches to time (after warmup)")
    parser.add_argument("--num_images_to_cache", type=int, default=0)
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument(
        "--synthetic_bcf_file",
        type=str,
        default=None,
        help="Override BCF file path",
    )
    parser.add_argument(
        "--real_image_dir",
        type=str,
        default=None,
        help="Override real image dir path",
    )
    args = parser.parse_args()

    # Resolve default paths
    project_root = os.environ.get(
        "PROJECT_ROOT", os.path.join(os.path.dirname(__file__), "..")
    )
    data_dir = os.path.join(project_root, "data")
    bcf_file = args.synthetic_bcf_file or os.path.join(
        data_dir, "deepfont_data", "BCF format", "VFR_syn_train", "train.bcf"
    )
    real_dir = args.real_image_dir or os.path.join(
        data_dir, "deepfont_data", "Raw Image", "VFR_real_u", "scrape-wtf-new"
    )

    print("=" * 60)
    print("DataLoader Worker Benchmark")
    print("=" * 60)
    print(f"Batch size:         {args.batch_size}")
    print(f"Batches to time:    {args.batches}")
    print(f"Images to cache:    {args.num_images_to_cache}")
    print(f"Pin memory:         {not args.no_pin_memory}")
    print(f"Workers to test:    {args.workers}")
    print(f"BCF file:           {bcf_file}")
    print(f"Real image dir:     {real_dir}")
    print()

    # Build the dataset once
    config = PretrainDataConfig(
        synthetic_bcf_file=bcf_file,
        real_image_dir=real_dir,
        aug_prob=0.5,
        image_normalization="0to1",
    )
    dataset = PretrainData(config)
    train_set, _ = dataset.split_data_random(train_ratio=0.8)
    train_set.upsample_real_images()
    print(f"Dataset size:       {len(train_set)} images")

    if args.num_images_to_cache > 0:
        print(f"Caching {args.num_images_to_cache} images...")
        t0 = time.perf_counter()
        train_set.cache_images(args.num_images_to_cache)
        cache_time = time.perf_counter() - t0
        print(f"Caching took {cache_time:.1f}s")

    print()

    # Ensure we have enough batches in the dataset
    min_images_needed = (args.batches + 5) * args.batch_size  # +5 for warmup
    if len(train_set) < min_images_needed:
        print(f"WARNING: dataset ({len(train_set)}) < needed images ({min_images_needed})")
        print("Reducing batch count.")
        args.batches = max(10, len(train_set) // args.batch_size - 10)

    pin_memory = not args.no_pin_memory and torch.cuda.is_available()

    results = []
    best = None

    for nw in args.workers:
        print(f"Testing num_workers={nw:>2d} ... ", end="", flush=True)
        try:
            result = benchmark_workers(
                train_set, nw, args.batch_size, args.batches, pin_memory
            )
            results.append(result)
            print(
                f"{result['images_per_sec']:>8.0f} img/s  "
                f"({result['sec_per_batch']*1000:>6.1f} ms/batch)"
            )
            if best is None or result["images_per_sec"] > best["images_per_sec"]:
                best = result
        except Exception as e:
            print(f"FAILED: {e}")

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Workers':>8s}  {'img/s':>10s}  {'ms/batch':>10s}  {'speedup':>8s}")
    print("-" * 42)
    baseline = results[0]["images_per_sec"] if results else 1
    for r in results:
        speedup = r["images_per_sec"] / baseline
        print(
            f"{r['num_workers']:>8d}  "
            f"{r['images_per_sec']:>10.0f}  "
            f"{r['sec_per_batch']*1000:>10.1f}  "
            f"{speedup:>7.2f}x"
        )

    if best:
        print()
        print(f">>> Best: num_workers={best['num_workers']} "
              f"({best['images_per_sec']:.0f} img/s)")


if __name__ == "__main__":
    main()
