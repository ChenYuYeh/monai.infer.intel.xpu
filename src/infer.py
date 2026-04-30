"""
infer.py — Standalone Inference Entry-Point
=============================================
Loads a pre-trained MONAI UNet, runs sliding-window inference on
KiTS19 NIfTI volumes, and saves predictions.

Acceleration: CUDA or Intel XPU via upstream PyTorch, with optional
Triton JIT kernels (``torch.compile(backend='inductor')``).

Usage
-----
    python -m src.infer                          # defaults from config.yaml
    python -m src.infer --config config.yaml --case case_00002
    python -m src.infer --input kits19/data      # scan all case_* folders
    python -m src.infer --device cpu             # force CPU
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ── project imports ──────────────────────────────────────────
from src.pipeline import (
    ClaraSegmentationModule,
    InputOperator,
    InferenceOperator,
    OutputOperator,
    PostProcessingOperator,
    PreProcessingOperator,
    discover_case_ids,
    load_config,
    resolve_device,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

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
logger = logging.getLogger("infer")


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def run_inference(
    config_path: str = "config.yaml",
    case_id: Optional[str] = None,
    input_dir: Optional[str] = None,
    device_override: Optional[str] = None,
) -> List[Dict]:
    """Run inference on one or more KiTS19 cases.

    Returns a list of per-case result dicts containing paths and timing.
    """
    cfg = load_config(config_path)

    if device_override:
        cfg["inference"]["device"] = device_override
    if input_dir:
        cfg["data"]["root_dir"] = input_dir
        cfg["data"]["case_ids"] = discover_case_ids(
            input_dir, cfg["data"]["file_pattern"]
        )

    device = resolve_device(cfg["inference"]["device"])
    logger.info("Device: %s", device)

    # Instantiate operators (mirrors the pipeline diagram)
    input_op = InputOperator(cfg)
    preprocess_op = PreProcessingOperator(cfg, device)
    inference_op = InferenceOperator(cfg, device)
    postprocess_op = PostProcessingOperator(cfg)
    clara_module = ClaraSegmentationModule(cfg)
    output_op = OutputOperator(cfg)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║  Medical AI Pipeline — MONAI + Clara-style Validation   ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info("║  [B] Input Reader            : NIfTI / Frame Reader     ║")
    logger.info("║  [C] Pre-Processing          : CT Window + Normalize    ║")
    logger.info("║  [D] MONAI Inference         : UNet + SlidingWindow     ║")
    logger.info("║  [E] Post-Processing         : Argmax + Label Map       ║")
    logger.info("║  [F] Clara-style Validation  : Dice / Metrics           ║")
    logger.info("║  [G] Output & Integration    : Predictions + Reports    ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info("║  Device: %-47s ║", str(device))
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # Resolve cases
    cases = [case_id] if case_id else cfg["data"]["case_ids"]
    root = cfg["data"]["root_dir"]
    pattern = cfg["data"]["file_pattern"]

    results: List[Dict] = []
    for cid in cases:
        nifti_path = Path(root) / cid / pattern
        if not nifti_path.exists():
            logger.warning("Skipping %s — file not found: %s", cid, nifti_path)
            continue

        logger.info("═══ %s ═══", cid)

        # B — Input read (nibabel-backed NIfTI reader)
        logger.info("  [B] Input:     Reading NIfTI volume …")
        volume = input_op.read_nifti(nifti_path)
        logger.info("  [B] Input:     Volume shape: %s", volume.shape)

        try:
            # C — Pre-process (Python/PyTorch operator)
            logger.info("  [C] Pre-process: CT window → normalize → device transfer …")
            tensor = preprocess_op(volume)
            logger.info("  [C] Pre-process: Tensor %s on %s", tensor.shape, tensor.device)

            # D — MONAI Inference (UNet + sliding window)
            logger.info("  [D] MONAI:    Running sliding-window inference (UNet) …")
            t0 = time.perf_counter()
            logits = inference_op(tensor)
            infer_sec = time.perf_counter() - t0
        except Exception as exc:
            if device.type == "cpu":
                raise
            logger.warning(
                "  [D] MONAI:    %s failed (%s); retrying this case on CPU",
                device.type.upper(),
                exc,
            )
            device = torch.device("cpu")
            preprocess_op = PreProcessingOperator(cfg, device)
            inference_op = InferenceOperator(cfg, device)
            tensor = preprocess_op(volume)
            logger.info("  [C] Pre-process: Tensor %s on %s", tensor.shape, tensor.device)
            t0 = time.perf_counter()
            logits = inference_op(tensor)
            infer_sec = time.perf_counter() - t0
        logger.info("  [D] MONAI:    Inference complete — %.3f s", infer_sec)

        # E — Post-process
        logger.info("  [E] Post:     Argmax + label mapping …")
        prediction = postprocess_op(logits, foreground_mask=volume > 0)
        unique, counts = np.unique(prediction, return_counts=True)
        logger.info("  [E] Post:     Labels found: %s", dict(zip(unique.tolist(), counts.tolist())))

        # F — Clara-style validation (self-comparison when no separate GT)
        logger.info("  [F] Clara-style: Computing validation metrics (Dice) …")
        validation = clara_module.validate(prediction, prediction)
        logger.info("  [F] Clara-style: Validation results: %s", validation)

        # G — Output
        logger.info("  [G] Output:   Saving prediction + report …")
        pred_path = output_op.save_prediction(prediction, cid)
        report = {
            "case_id": cid,
            "volume_shape": list(volume.shape),
            "inference_time_s": round(infer_sec, 4),
            "prediction_path": pred_path,
            "labels": dict(zip(unique.tolist(), counts.tolist())),
            "clara_validation": validation,
        }
        output_op.save_report(report, cid)
        results.append(report)

        del volume, tensor, logits, prediction
        if device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        elif (
            device.type == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()
        gc.collect()

    return results


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MONAI KiTS19 inference")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config YAML")
    parser.add_argument("--case", default=None, help="Single case ID (e.g. case_00002)")
    parser.add_argument("--input", default=None, help="Folder containing case_* subfolders")
    parser.add_argument("--device", default=None, help="Override device (auto|cpu|cuda|xpu|mps)")
    args = parser.parse_args()

    results = run_inference(
        config_path=args.config,
        case_id=args.case,
        input_dir=args.input,
        device_override=args.device,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
