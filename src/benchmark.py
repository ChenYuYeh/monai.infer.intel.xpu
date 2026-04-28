"""
benchmark.py — Inference Benchmarking on Intel XPU / CPU
=========================================================
Measures throughput, latency percentiles, and memory usage for
the MONAI UNet inference pipeline.  Implements the benchmarking
concerns from diagram nodes **D1** and **F2** (Clara Validation).

Supports:
- Intel XPU (via ``intel_extension_for_pytorch`` + Triton)
- CUDA (via standard PyTorch)
- CPU baseline

Usage
-----
    python -m src.benchmark                           # default from config
    python -m src.benchmark --device xpu --warmup 3 --runs 10
    python -m src.benchmark --device cpu --runs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.pipeline import (
    InferenceOperator,
    PreProcessingOperator,
    load_config,
    resolve_device,
)

# Suppress MONAI deprecation warning for non-tuple indexing (fixed in newer MONAI)
warnings.filterwarnings(
    "ignore",
    message="Using a non-tuple sequence for multidimensional indexing",
    category=UserWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("benchmark")


# ────────────────────────────────────────────────────────────
# Benchmark Runner
# ────────────────────────────────────────────────────────────

def benchmark_inference(
    config_path: str = "config.yaml",
    device_override: Optional[str] = None,
    warmup_runs: int = 3,
    timed_runs: int = 10,
    synthetic: bool = False,
) -> Dict:
    """Run inference N times and collect timing statistics.

    Parameters
    ----------
    config_path : path to config YAML
    device_override : force a specific device string
    warmup_runs : number of untimed warm-up iterations
    timed_runs : number of timed iterations
    synthetic : if True, use random input instead of real data

    Returns
    -------
    dict with latency stats, throughput, and device info.
    """
    cfg = load_config(config_path)
    if device_override:
        cfg["inference"]["device"] = device_override

    device = resolve_device(cfg["inference"]["device"])
    logger.info("Benchmark device: %s", device)

    # Build operators
    preprocess_op = PreProcessingOperator(cfg, device)
    inference_op = InferenceOperator(cfg, device)

    # Prepare input tensor
    if synthetic:
        spatial = cfg["preprocessing"]["spatial_size"]
        volume = np.random.randn(*spatial).astype(np.float32)
        logger.info("Using synthetic volume %s", spatial)
    else:
        root = cfg["data"]["root_dir"]
        cid = cfg["data"]["case_ids"][0]
        nifti_path = Path(root) / cid / cfg["data"]["file_pattern"]
        if not nifti_path.exists():
            logger.warning("Real data not found — falling back to synthetic input")
            spatial = cfg["preprocessing"]["spatial_size"]
            volume = np.random.randn(*spatial).astype(np.float32)
        else:
            import nibabel as nib
            volume = np.asarray(nib.load(str(nifti_path)).dataobj, dtype=np.float32)
            logger.info("Loaded %s  shape=%s", cid, volume.shape)

    tensor = preprocess_op(volume)
    logger.info("Input tensor: %s on %s", tensor.shape, tensor.device)

    # ── warm-up ──
    logger.info("Warm-up: %d runs …", warmup_runs)
    for _ in range(warmup_runs):
        _ = inference_op(tensor)
    _sync(device)

    # ── timed runs ──
    logger.info("Timed: %d runs …", timed_runs)
    latencies: List[float] = []
    for i in range(timed_runs):
        _sync(device)
        t0 = time.perf_counter()
        _ = inference_op(tensor)
        _sync(device)
        latencies.append(time.perf_counter() - t0)

    latencies_ms = [t * 1000.0 for t in latencies]
    arr = np.array(latencies_ms)

    # ── memory info ──
    mem_info = _get_memory_info(device)

    report = {
        "device": str(device),
        "triton_backend": cfg["inference"].get("triton_backend", False),
        "input_shape": list(tensor.shape),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "latency_ms": {
            "mean": round(float(arr.mean()), 2),
            "std": round(float(arr.std()), 2),
            "min": round(float(arr.min()), 2),
            "max": round(float(arr.max()), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
        },
        "throughput_fps": round(1000.0 / float(arr.mean()), 2) if arr.mean() > 0 else 0,
        "memory": mem_info,
    }
    return report


# ────────────────────────────────────────────────────────────
# Device-comparison benchmark
# ────────────────────────────────────────────────────────────

def compare_devices(
    config_path: str = "config.yaml",
    devices: Optional[List[str]] = None,
    warmup_runs: int = 3,
    timed_runs: int = 10,
) -> List[Dict]:
    """Run benchmarks on multiple devices and return a comparison table."""
    if devices is None:
        devices = ["cpu"]
        try:
            import intel_extension_for_pytorch  # noqa: F401
            if torch.xpu.is_available():
                devices.append("xpu")
        except ImportError:
            pass
        if torch.cuda.is_available():
            devices.append("cuda")

    results = []
    for dev in devices:
        logger.info("══════ Benchmarking %s ══════", dev)
        r = benchmark_inference(
            config_path=config_path,
            device_override=dev,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
            synthetic=True,
        )
        results.append(r)
    return results


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _sync(device: torch.device) -> None:
    """Synchronise the device to ensure accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        try:
            torch.xpu.synchronize()
        except Exception:
            pass


def _get_memory_info(device: torch.device) -> Dict[str, str]:
    info: Dict[str, str] = {}
    if device.type == "cuda":
        info["allocated_MB"] = f"{torch.cuda.memory_allocated(device) / 1e6:.1f}"
        info["reserved_MB"] = f"{torch.cuda.memory_reserved(device) / 1e6:.1f}"
    elif device.type == "xpu":
        try:
            info["allocated_MB"] = f"{torch.xpu.memory_allocated(device) / 1e6:.1f}"
            info["reserved_MB"] = f"{torch.xpu.memory_reserved(device) / 1e6:.1f}"
        except Exception:
            info["note"] = "XPU memory stats not available"
    else:
        try:
            import psutil
            vm = psutil.virtual_memory()
            info["system_total_MB"] = f"{vm.total / 1e6:.0f}"
            info["system_used_MB"] = f"{vm.used / 1e6:.0f}"
        except ImportError:
            info["note"] = "Install psutil for CPU memory stats"
    return info


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MONAI inference on Intel XPU")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="Device to benchmark (xpu|cpu|cuda)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true", help="Use random input tensor")
    parser.add_argument("--compare", action="store_true", help="Benchmark all available devices")
    args = parser.parse_args()

    if args.compare:
        results = compare_devices(
            config_path=args.config,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
        )
    else:
        results = [benchmark_inference(
            config_path=args.config,
            device_override=args.device,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
            synthetic=args.synthetic,
        )]

    # Save
    os.makedirs("output/benchmarks", exist_ok=True)
    out_path = "output/benchmarks/benchmark_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved → %s", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
